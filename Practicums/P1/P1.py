import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from power_grid_model import (
    PowerGridModel,
    ComponentType,
    DatasetType,
    LoadGenType,
    initialize_array,
    CalculationMethod
)

# Robuuste import voor validatie
try:
    from power_grid_model.validation import assert_valid_input_data
except ImportError:
    assert_valid_input_data = None

def run_practicum_final():
    # --- 1. DATA LADEN ---
    print("Bezig met laden van data...")
    df_topo = pd.read_csv('data/grid_topology.csv')
    df_cons = pd.read_csv('data/consumption_one_year_15min.csv', index_col='date')
    df_pv   = pd.read_csv('data/pv_one_year_15min.csv', index_col='date')
    df_ev   = pd.read_csv('data/ev_one_year_15min.csv', index_col='date')

    # --- 2. CONFIGURATIE PV & EV (Opdracht D) ---
    # Hier kun je exact instellen wie wat heeft
    # Standaard: iedereen heeft PV en EV
    pv_customers = list(range(1, 96)) 
    ev_customers = [i for i in range(1, 96) if i not in [10, 20]] # Iedereen behalve 10 en 20

    print("Bezig met combineren van profielen...")
    df_combined = df_cons.copy()
    
    for i in range(1, 96):
        col = f"Customer_{i} (kW)"
        
        # PV meerekenen indien de klant PV heeft
        if i in pv_customers:
            df_combined[col] = df_combined[col] - df_pv[col]
            
        # EV meerekenen indien de klant een lader heeft
        if i in ev_customers:
            df_combined[col] = df_combined[col] + df_ev[col]
    
    # Sla de gecombineerde CSV op zoals gevraagd 
    df_combined.to_csv('data/combined_profile_year.csv')
    print("Gecombineerd profiel opgeslagen als 'data/combined_profile_year.csv'.")

    # --- 3. POWER FLOW FUNCTIE ---
    def build_and_run(p_values_kw):
        unique_nodes = np.unique(df_topo[['FROM', 'TO']].values)
        
        # Nodes
        node = initialize_array(DatasetType.input, ComponentType.node, len(unique_nodes))
        node["id"] = unique_nodes
        node["u_rated"] = 400.0

        # Lines (ID's 300+)
        line = initialize_array(DatasetType.input, ComponentType.line, len(df_topo))
        line["id"] = np.arange(300, 300 + len(df_topo))
        line["from_node"] = df_topo["FROM"].values
        line["to_node"] = df_topo["TO"].values
        line["from_status"] = 1
        line["to_status"] = 1
        line["r1"] = df_topo["Raa"].values
        line["x1"] = df_topo["Xaa"].values
        line["c1"] = 0.0    
        line["tan1"] = 0.0  
        line["i_n"] = df_topo["Imax"].values

        # Source (ID 100)
        source = initialize_array(DatasetType.input, ComponentType.source, 1)
        source["id"] = 100
        source["node"] = 1
        source["status"] = 1 
        source["u_ref"] = 1.0

        # Loads (ID's 200+) - 0.95 leading [cite: 87]
        p_w = p_values_kw.values * 1000.0
        q_w = - p_w * np.tan(np.arccos(0.95))

        sym_load = initialize_array(DatasetType.input, ComponentType.sym_load, 95)
        sym_load["id"] = np.arange(200, 200 + 95)
        sym_load["node"] = np.arange(1, 96) 
        sym_load["status"] = 1
        sym_load["type"] = LoadGenType.const_power
        sym_load["p_specified"] = p_w
        sym_load["q_specified"] = q_w

        input_data = {
            ComponentType.node: node,
            ComponentType.line: line,
            ComponentType.sym_load: sym_load,
            ComponentType.source: source
        }

        if assert_valid_input_data:
            assert_valid_input_data(input_data)

        model = PowerGridModel(input_data)
        return model.calculate_power_flow(calculation_method=CalculationMethod.newton_raphson)

    # --- 4. EXECUTIE & VISUALISATIE ---
    target_time = '2020-06-21 18:00:00'
    res_c = build_and_run(df_cons.loc[target_time])      # Zonder PV/EV
    res_d = build_and_run(df_combined.loc[target_time])  # Met PV/EV

    nodes_c, nodes_d = pd.DataFrame(res_c[ComponentType.node]), pd.DataFrame(res_d[ComponentType.node])
    lines_c, lines_d = pd.DataFrame(res_c[ComponentType.line]), pd.DataFrame(res_d[ComponentType.line])

    v_col = 'u_pu' if 'u_pu' in nodes_c.columns else 'u_mag'

    # Vergelijkingsplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Voltages [cite: 98]
    ax1.plot(nodes_c['id'], nodes_c[v_col], 'b-o', label='Case C (Basis)', markersize=3)
    ax1.plot(nodes_d['id'], nodes_d[v_col], 'r-s', label='Case D (PV/EV)', markersize=3)
    ax1.axhline(1.0, color='black', linestyle='--', alpha=0.3)
    ax1.set_title(f"Voltage Profiel Vergelijking\n{target_time}")
    ax1.set_xlabel("Node ID")
    ax1.set_ylabel("Voltage [p.u.]")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Line Loadings [cite: 104]
    x = np.arange(len(lines_c))
    width = 0.35
    ax2.bar(x - width/2, lines_c['loading'], width, label='Case C', color='blue', alpha=0.6)
    ax2.bar(x + width/2, lines_d['loading'], width, label='Case D', color='red', alpha=0.6)
    ax2.set_title(f"Line Loading Vergelijking\n{target_time}")
    ax2.set_xlabel("Line Index")
    ax2.set_ylabel("Loading [%]")
    ax2.legend()
    ax2.grid(True, axis='y')

    plt.tight_layout()
    plt.show()

    # Terminal Output [cite: 89, 104]
    volt_dict = {f"Customer {int(row['id'])}": round(float(row[v_col]), 6) for _, row in nodes_d.iterrows()}
    print(f"\nTime step: {target_time}, Voltages Case D: {volt_dict}")
    print(f"\nMax Loading Case C: {lines_c['loading'].max():.2f}%")
    print(f"Max Loading Case D: {lines_d['loading'].max():.2f}%")

if __name__ == "__main__":
    run_practicum_final()
import pandapower as pp
import pandapower.networks as pn

# Load base case
net = pn.case4gs()

# Modify load at bus 2: 200 MW -> 80 MW
net.load.loc[net.load.bus == 2, 'p_mw'] = 80
net.load.loc[net.load.bus == 2, 'q_mvar'] = 80 * 0.6199  # Keep same power factor

# Run power flow
pp.runpp(net)

print("=== MODIFIED CASE (Bus 2: 80 MW) ===")
print("\n=== SLACK BUS ===")
print(net.res_ext_grid)

print("\n=== BUS VOLTAGES ===")
print(net.res_bus[['vm_pu', 'va_degree']])

print("\n=== LINE LOADING ===")
print(net.res_line[['loading_percent']])

print("\n=== GENERATOR ===")
print(net.res_gen[['p_mw', 'q_mvar']])
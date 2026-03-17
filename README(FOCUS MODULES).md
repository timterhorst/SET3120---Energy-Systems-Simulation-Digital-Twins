# FM-c: Electric Vehicle with Vehicle-to-Grid (V2G)

Focus module extending the base co-simulation with a bidirectional EV charger connected at the same grid node (`Customer_95`) as the heat pump (HP). The EV can both **absorb** power during overvoltage (smart charging) and **inject** power during undervoltage (V2G), subject to an SOC reserve.

## What this module adds

The base implementation simulates a smart house with a heat pump on a 95-bus LV network. FM-c adds three EV-related components and an extended manager:

| Component | Type | File | Purpose |
|---|---|---|---|
| **Driver Model** | Discrete event | `src/driver_model.py` | Daily departure/arrival schedule + stochastic driving SOC depletion + occupancy flag |
| **EV Battery** | Dynamic (ODE) | `src/ev_battery.py` | SOC integration with asymmetric charge/discharge efficiency |
| **Charging Controller** | Quasi-static | `src/charging_controller.py` | Smart charging + V2G logic based on voltage, SOC, and occupancy |
| **EVManager** | Orchestrator | `src/ev_cosim_framework.py` | 7-step synchronization loop and logging of EV signals |

## Running (config 1 nominal / config 99 stressed)

From the project root:

```bash
# Base (no EV), nominal loading (config 1)
python -m src.run_co_simulation 1

# With EV/V2G, nominal loading (config 1)
python -m src.run_ev_co_simulation 1

# Base (no EV), nominal loading (config 1) using forecasted passive consumers
python -m src.run_co_simulation 1 --use-forecasted

# With EV/V2G, nominal loading (config 1) using forecasted passive consumers
python -m src.run_ev_co_simulation 1 --use-forecasted

# Base (no EV), stressed loading (config 99, 3× passive loads)
python -m src.run_co_simulation 99

# With EV/V2G, stressed loading (config 99, 3× passive loads)
python -m src.run_ev_co_simulation 99

# Analyse saved .npz results and print LaTeX-ready tables
python -m src.analyze_results
```

### Outputs

**Figures** (saved in the project root):
- `results_config{id}.png` (base run)
- `ev_results_base_overview_config{id}.png`
- `ev_results_full_overview_config{id}.png`
- `ev_results_zoom_config{id}.png`

**Saved time-series** (saved in `results/` as compressed `.npz`):
- `results/config1_base.npz`, `results/config1_ev.npz`
- `results/config1_base_forecasted.npz`, `results/config_ev_forecasted.npz` (config 1 with `--use-forecasted`)
- `results/config99_base.npz`, `results/config99_ev.npz`

The `.npz` files contain the full-year arrays (15-minute resolution) used by `src/analyze_results.py`.

### Forecasted passive-consumer dataset

Both `src/run_co_simulation.py` and `src/run_ev_co_simulation.py` support `--use-forecasted`.
When enabled, the simulation uses `data/combined_active_power_forecasted.csv` as the passive-consumer time-series (instead of the default dataset configured in `configurations/config*.yaml`).

## Synchronization sequence (EV run)

The base uses a 4-step loop: Grid → Heat Pump → Room → Controller. FM-c preserves the “measure-then-act” causality (controllers see **current-step** voltage) and extends it to 7 steps:

```
Per timestep k:
  1. Driver Model            → is_home[k], temp_min[k], soc_depletion[k]
  2. Electric Grid           → V_grid[k]        (uses P_hp[k-1] + P_charge[k-1])
  3. Heat Pump               → Q_hp[k]          (uses P_hp[k-1])
  4. Room                    → T_room[k]        (uses Q_hp[k])
  5. EV Battery              → SOC[k]           (uses P_charge[k-1], soc_depletion[k])
  6. Thermostat Controller   → P_hp[k]          (uses V_grid[k], T_room[k], temp_min[k])
  7. Charging Controller     → P_charge[k]      (uses V_grid[k], SOC[k], is_home[k])
```

Both controller outputs (`P_hp[k]`, `P_charge[k]`) affect the grid at step \(k+1\).

## Controller logic (charging/V2G)

The charging controller (`src/charging_controller.py`) prioritizes voltage support:

- **Not home** → `P_charge = 0`
- **Undervoltage** (V < 0.98) → V2G **discharge** at `-3700 W` if `SOC > 0.40`, else `0`
- **Overvoltage** (V > 1.02) → **charge** at `+7400 W` if `SOC < 1.0`, else `0`
- **Normal voltage** → SOC-tiered charging (7400/5550/3700 W) until full

## Configuration

### Base configs

- `configurations/config1.yaml`: nominal scenario (`load_scale: 1`)
- `configurations/config99.yaml`: stressed scenario (`load_scale: 3`)

`load_scale` multiplies **passive** consumer loads (from `data/combined_profiles_one_year.csv`). HP and EV actions are added on top at `Customer_95`.

### EV config

All EV parameters live in `configurations/ev_config.yaml` (battery size, efficiencies, max charge/discharge power, SOC reserve, voltage thresholds, and the driver schedule/seed).

## Analysis

`src/analyze_results.py` loads the `.npz` files and prints:

- **Table 1**: Nominal conditions (config 1), sectioned into “Grid voltage” and “Node power”
- **Table 2**: Stressed grid (config 99), sectioned into “Grid voltage” and “Node power”
- **Table 3**: SOC constraint validation (config 99)
- A **diagnostic block** verifying overvoltage controller behaviour under config 1

## File overview (current)

```
src/
├── run_co_simulation.py        # Base entry point (supports config id + --use-forecasted)
├── run_ev_co_simulation.py     # EV/V2G entry point (supports config id + --use-forecasted)
├── analyze_results.py          # Loads results/*.npz and prints tables/diagnostics
├── cosim_framework.py          # Base Manager: supports load_scale, saves results/config{id}_base.npz
├── ev_cosim_framework.py       # EVManager: supports load_scale, saves results/config{id}_ev.npz
├── controller.py               # HP controller (+ temp_min_override, prints muted for batch runs)
├── charging_controller.py      # EV charging/V2G controller
├── driver_model.py             # Driver schedule + is_home + SOC depletion
├── ev_battery.py               # SOC dynamics (explicit Euler)
├── grid.py
├── heat_pump.py
├── room.py
└── load_configurations.py

configurations/
├── config1.yaml
├── config99.yaml
├── controller_config.yaml
└── ev_config.yaml
```

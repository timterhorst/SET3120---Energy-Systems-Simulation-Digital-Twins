# FM-c: Electric Vehicle with Vehicle-to-Grid (V2G)

Focus Module extending the base co-simulation with a bidirectional EV charger connected to the same grid node (`Customer_95`) as the heat pump.

## What this module adds

The base implementation simulates a smart house with a heat pump on a 95-bus distribution network. The heat pump adjusts its power based on room temperature and grid voltage.

FM-c introduces three new components:

| Component | Type | File | Purpose |
|-----------|------|------|---------|
| **Driver Model** | Discrete event | `src/driver_model.py` | Daily departure/arrival schedule + stochastic driving consumption |
| **EV Battery** | Dynamic (ODE) | `src/ev_battery.py` | State of Charge integration with asymmetric charge/discharge efficiency |
| **Charging Controller** | Quasi-static | `src/charging_controller.py` | Smart charging + V2G logic based on voltage and SOC |

It also extends the framework with `EVManager` (`src/ev_cosim_framework.py`), a subclass of the base `Manager` that implements a 7-step synchronization sequence.

## Running

From the project root:

```bash
# Base simulation (unchanged)
python -m src.run_co_simulation

# EV + V2G simulation (full year, normal load)
python -m src.run_ev_co_simulation

# V2G stress test (14 days, 3x passive load — demonstrates V2G activation)
python -m src.run_ev_stress_test
```

Output files are saved to the project root:
- `ev_results_base_overview_config{id}.png` — base-compatible 2x2 (for comparison)
- `ev_results_full_overview_config{id}.png` — full EV overview with all signals
- `ev_results_zoom_config{id}.png` — two-week zoom with home/away shading

## Synchronization sequence

The base uses a 4-step loop: Grid → Heat Pump → Room → Controller. The grid executes first so the controller always reacts to the **current** voltage (measure-then-act). FM-c preserves this causality and extends it to 7 steps:

```
Per timestep k:

  1. Driver Model           → is_home[k], temp_min[k], soc_depletion[k]
  2. Electric Grid           → V_grid[k]       (uses P_hp[k-1] + P_charge[k-1])
  3. Heat Pump               → Q_hp[k]         (uses P_hp[k-1])
  4. Room                    → T_room[k]       (uses Q_hp[k])
  5. EV Battery              → SOC[k]          (uses P_charge[k-1], soc_depletion[k])
  6. Thermostat Controller   → P_hp[k]         (uses V_grid[k], T_room[k], temp_min[k])
  7. Charging Controller     → P_charge[k]     (uses V_grid[k], SOC[k], is_home[k])
```

**Why this order?** Voltage has the highest physical priority — both controllers need the current grid state before deciding their setpoints. The Driver Model runs first because both controllers need `is_home` before making decisions. Steps 2–4 are unchanged from the base. Steps 5–7 extend it. Both controller outputs (`P_hp[k]`, `P_charge[k]`) are used by the grid at step `k+1`.

## New components — design and logic

### 1. Driver Model (`src/driver_model.py`)

A stateful callable class that models daily occupancy.

**Daily schedule** (re-sampled each day from uniform distributions):
- Departure: U(06:00, 07:00)
- Arrival: U(16:00, 17:00)
- SOC depletion at arrival: U(0%, 90%) — models unknown daily driving

**Outputs per timestep:**
- `is_home` (bool) — gates all EV charging/discharging
- `temp_min_dynamic` (float) — 20°C when home, 10°C when away (occupancy setback)
- `soc_depletion` (float) — non-zero only at the first timestep after arrival

The SOC depletion is applied as a step-change on arrival rather than continuous drain during driving, because the driver could be doing anything during the day (commuting, errands, charging at a destination). The uniform draw U(0, 0.90) covers everything from "worked from home" to "long road trip."

A configurable random seed (`random_seed: 42`) ensures reproducibility.

### 2. EV Battery Model (`src/ev_battery.py`)

A stateful callable class that integrates SOC using explicit Euler (matching the room model).

**ODE:**
```
Charging  (P ≥ 0):  dSOC/dt = η_ch × P / E_cap
Discharging (P < 0): dSOC/dt = P / (η_dis × E_cap)
```

The asymmetry is physically important: during discharge, dividing by `η_dis` (< 1) *increases* the rate of SOC depletion per unit of power delivered to the grid. This correctly models round-trip losses — you must deplete more stored energy than what arrives at the grid.

**Unit conversion:** `delta_t` in the config is in minutes, but `P [kW] / E_cap [kWh]` yields `1/hour`. The Euler step uses `delta_t / 60` to convert to hours.

**SOC is clamped to [0, 1]** after every step to prevent numerical overshoot.

**Parameters:**
| Parameter | Value | Source |
|-----------|-------|--------|
| E_cap | 50 kWh | Typical mid-range EV (e.g. VW ID.3, Nissan Leaf) |
| η_charge | 0.90 | AC Level 2 charging losses |
| η_discharge | 0.85 | V2G inverter losses (higher than charging) |
| Initial SOC | 1.0 | Assumes fully charged overnight |

### 3. Charging Controller (`src/charging_controller.py`)

A stateless function (like the base thermostat controller) with voltage-prioritised logic:

```
if not is_home           → 0 W (idle)
elif V < 0.98 (undervoltage):
    if SOC > 40%         → −3700 W (V2G discharge to raise voltage)
    else                 → 0 W (can't help, idle)
elif V > 1.02 (overvoltage):
    if SOC < 100%        → +7400 W (charge at max to lower voltage)
    else                 → 0 W (full, idle)
else (normal voltage):
    SOC < 20%            → +7400 W (full rate, deep recovery)
    20% ≤ SOC < 50%      → +5550 W (75% rate)
    SOC ≥ 50%            → +3700 W (50% rate, trickle)
    SOC = 100%           → 0 W
```

**V2G direction:** V2G discharges during **undervoltage** to inject power and raise voltage. During **overvoltage**, the EV charges at full rate to absorb power and lower voltage. This mirrors the heat pump controller (HP off during undervoltage, HP on during overvoltage) — both assets cooperate for voltage support.

**Parameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| P_charge_max | 7.4 kW | Level 2 AC, 32A @ 230V (SAE J1772) |
| P_discharge_max | 3.7 kW | Conservative V2G rate (half of charge rate) |
| SOC_min_V2G | 40% | Safety margin for next day's driving |

## Modifications to base files

All changes are backward-compatible — the base simulation runs identically.

### `src/controller.py`
Added optional keyword argument `temp_min_override: float = None`. When provided (by the Driver Model via the EVManager), it replaces the config's `minimum_temperature`. When `None` (default), the base config value is used. This enables **occupancy-based setback**: the thermostat dead-band narrows to [20°C, 30°C] when the driver is home (comfort) and widens to [10°C, 30°C] when away (energy saving / frost protection).

### `src/cosim_framework.py`
Added `**kwargs` to `Model.calculate()` so keyword arguments (like `temp_min_override`) can pass through the `Model` wrapper. Existing positional-only calls are unaffected.

### `src/load_configurations.py`
Added `ev_config.yaml` to the skip set so it is not mistakenly parsed as a simulation initialization config (it has no `config_id`). The EV run script loads it separately.

## Configuration

All EV parameters live in `configurations/ev_config.yaml`. The base configs (`config1.yaml`, `controller_config.yaml`) are unchanged.

```yaml
EVSettings:
  battery:
    capacity_kwh: 50
    efficiency_charge: 0.90
    efficiency_discharge: 0.85
    initial_soc: 1.0
  charging:
    p_charge_max_w: 7400
    p_discharge_max_w: 3700
    soc_min_v2g: 0.40
    voltage_min: 0.98
    voltage_max: 1.02
  driver:
    departure_earliest: 360     # 06:00
    departure_latest: 420       # 07:00
    arrival_earliest: 960       # 16:00
    arrival_latest: 1020        # 17:00
    soc_depletion_min: 0.0
    soc_depletion_max: 0.90
    temp_min_home: 20
    temp_min_away: 10
    random_seed: 42
```

## File overview

```
src/
├── run_co_simulation.py          # Base entry point (unchanged)
├── run_ev_co_simulation.py       # NEW — EV entry point
├── run_ev_stress_test.py         # NEW — V2G stress test (3x load, 14 days)
├── cosim_framework.py            # Base Model + Manager (minor: +**kwargs)
├── ev_cosim_framework.py         # NEW — EVManager subclass
├── controller.py                 # Base HP controller (minor: +temp_min_override)
├── charging_controller.py        # NEW — EV charging + V2G controller
├── driver_model.py               # NEW — occupancy + stochastic driving model
├── ev_battery.py                 # NEW — SOC dynamics (ODE)
├── grid.py                       # Unchanged
├── heat_pump.py                  # Unchanged
├── room.py                       # Unchanged
├── load_configurations.py        # Minor: skip ev_config.yaml
└── __init__.py

configurations/
├── config1.yaml                  # Base sim config (unchanged)
├── config_stress_test.yaml       # NEW — 14-day stress test config
├── controller_config.yaml        # HP controller config (unchanged)
└── ev_config.yaml                # NEW — all EV parameters
```

## V2G stress test

Under normal operating conditions (1x passive load), grid voltage at Customer_95 never drops below 0.98 p.u. — the network is well-designed and V2G is not needed. To demonstrate V2G activation, a stress test scales all passive consumer loads by 3x:

| Load scale | Min voltage | V2G events (14 days) |
|-----------|-------------|---------------------|
| 1.0x | 0.9840 p.u. | 0 |
| 2.0x | 0.9793 p.u. | 1 |
| 3.0x | 0.9684 p.u. | 16 |

The stress test (`python -m src.run_ev_stress_test`) runs 15 days at 3x load using `configurations/config_stress_test.yaml`. The `EVManager.run_simulation()` accepts an optional `load_scale` parameter for this purpose. Zoom plots from the stress test clearly show negative EV power (V2G discharge at −3700 W) during undervoltage events, with the HP simultaneously shutting off — both controllers cooperating for voltage support.

## Key design decisions

1. **Grid model is not modified.** The combined load `P_total = P_hp + P_charge` is computed in the EVManager and passed as a single value to the existing grid function. This avoids touching the most complex module.

2. **EVManager is a subclass, not a replacement.** The base `Manager` remains fully functional. `EVManager` overrides `__init__` and `run_simulation` to implement the 7-step sequence.

3. **Efficiency asymmetry matters.** Using the same `η` for charge and discharge would underestimate V2G losses. The battery model applies `η` in the numerator for charging and in the denominator for discharging.

4. **V2G activates on undervoltage, not overvoltage.** Discharging injects power into the grid, raising voltage — useful during undervoltage. Discharging during overvoltage would worsen the problem.

5. **Stochastic driving over deterministic.** Daily driving consumption is drawn from a uniform distribution rather than computed from a fixed commute distance. This is more honest about the uncertainty and avoids false precision.

6. **Occupancy setback is physically significant.** The 10°C/20°C dynamic dead-band creates a rebound heating peak at arrival that coincides with EV charging — a realistic demand spike the simulation should capture.

"""
Modular Smart Room Simulation - Main orchestration script.
SET3120 Practicum 2 B.3

Uses modules: heat_pump, room, controller, ambient.
Plotting and verification from P2.py (unchanged).

To verify against P2.py: run with same dt (e.g. dt=900). Results match
P2.simulate_smart_room(dt=900, method='explicit') within float precision.
"""

import numpy as np

import ambient
import controller as ctrl_module
import heat_pump
import room
from P2 import plot_results, verify_energy_balance

# ----- Simulation parameters (same defaults as P2.py) -----
dt = 900  # Time step [s]
t_end = 86400  # Duration [s]
C = 20e6  # Thermal capacitance [J/K]
R = 0.02  # Thermal resistance [K/W]
COP = 3.0
P_rated = 2000  # [W]
T_set = 20.0
dT_lower = 1.0
dT_upper = 1.0
V_nom = 230.0
V_tolerance = 0.02
T_room_init = 18.0
T_avg = 5.0
T_amplitude = 3.0
method = "explicit"

n_steps = int(t_end / dt)
tau = R * C

print("=== Modular Simulation Setup ===")
print(f"Time step: {dt} s ({dt/60:.1f} min)")
print(f"Total duration: {t_end} s ({t_end/3600:.1f} hours)")
print(f"Number of steps: {n_steps}")
print(f"Thermal time constant: {tau/3600:.1f} hours")
print(f"Discretization method: {method}")
print()

# ----- Initialize models -----
room_model = room.RoomModel(
    C=C, R=R, T_initial=T_room_init, dt=dt, method=method
)
thermostat = ctrl_module.ThermostatController(
    T_set=T_set,
    dT_lower=dT_lower,
    dT_upper=dT_upper,
    P_rated=P_rated,
    V_nom=V_nom,
    V_tolerance=V_tolerance,
)

# ----- Storage arrays (same layout as P2.py) -----
time_array = np.zeros(n_steps + 1)
T_room_array = np.zeros(n_steps + 1)
T_outside_array = np.zeros(n_steps + 1)
P_elec_array = np.zeros(n_steps + 1)
Q_hp_array = np.zeros(n_steps + 1)
state_array = np.zeros(n_steps + 1)
V_grid_array = np.zeros(n_steps + 1)

T_room_array[0] = T_room_init

# ----- Simulation loop -----
for k in range(n_steps):
    t = k * dt
    time_array[k] = t

    # 1. Outside temperature
    T_outside = ambient.get_outside_temperature(t, T_avg=T_avg, T_amplitude=T_amplitude)
    T_outside_array[k] = T_outside

    # 2. Grid voltage (constant for now)
    V_grid = V_nom
    V_grid_array[k] = V_grid

    # 3. Controller
    T_room = room_model.get_temperature()
    P_elec, state = thermostat.update(T_room, V_grid)
    P_elec_array[k] = P_elec
    state_array[k] = state

    # Debug (match P2.py style)
    if k < 5 or k % 100 == 0:
        st = "ON " if state else "OFF"
        print(f"Step {k:4d} | t={t/3600:5.2f}h | T_room={T_room:5.2f}°C | "
              f"State={st} | P={P_elec:4.0f}W")

    # 4. Heat pump
    Q_hp = heat_pump.heat_pump_function(P_elec, COP)
    Q_hp_array[k] = Q_hp

    # 5. Room update
    T_new = room_model.update(Q_hp, T_outside)
    T_room_array[k + 1] = T_new

# Final timestep (same as P2)
time_array[n_steps] = t_end
T_outside_array[n_steps] = T_outside_array[n_steps - 1]
V_grid_array[n_steps] = V_grid_array[n_steps - 1]
P_elec_array[n_steps] = P_elec_array[n_steps - 1]
Q_hp_array[n_steps] = Q_hp_array[n_steps - 1]
state_array[n_steps] = state_array[n_steps - 1]

results = {
    "time": time_array,
    "T_room": T_room_array,
    "T_outside": T_outside_array,
    "P_elec": P_elec_array,
    "Q_hp": Q_hp_array,
    "state": state_array,
    "V_grid": V_grid_array,
}

# ----- Verification -----
verify_energy_balance(results, C)

# ----- Plot (same as P2.py) -----
plot_results(results)

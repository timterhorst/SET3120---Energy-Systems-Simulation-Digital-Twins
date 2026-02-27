"""
Full simulation with voltage variations to demonstrate voltage control.

Runs the modular simulation with a time-varying grid voltage profile
(morning dip, evening spike) so the controller's voltage response is visible.
"""

import numpy as np
import matplotlib.pyplot as plt

import ambient
import controller as ctrl_module
import heat_pump
import room
from P2 import verify_energy_balance


def get_voltage_profile(t: float, V_nom: float = 230.0) -> float:
    """
    Time-varying voltage with dips and spikes.

    Parameters
    ----------
    t : float
        Time [s] from simulation start.
    V_nom : float
        Nominal voltage [V].

    Returns
    -------
    float
        Grid voltage [V].
    """
    # Voltage dip at 6am–8am, spike at 6pm–8pm
    if 6 * 3600 <= t < 8 * 3600:
        return 220.0  # Morning under-voltage
    elif 18 * 3600 <= t < 20 * 3600:
        return 236.0  # Evening over-voltage
    else:
        return V_nom


def simulate_with_voltage_profile():
    """Run 24h simulation with time-varying voltage; return results dict."""
    # Same parameters as run_modular_simulation
    dt = 900
    t_end = 86400
    C = 20e6
    R = 0.02
    COP = 3.0
    P_rated = 2000
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

    time_array = np.zeros(n_steps + 1)
    T_room_array = np.zeros(n_steps + 1)
    T_outside_array = np.zeros(n_steps + 1)
    P_elec_array = np.zeros(n_steps + 1)
    Q_hp_array = np.zeros(n_steps + 1)
    state_array = np.zeros(n_steps + 1)
    V_grid_array = np.zeros(n_steps + 1)

    T_room_array[0] = T_room_init

    for k in range(n_steps):
        t = k * dt
        time_array[k] = t

        T_outside = ambient.get_outside_temperature(t, T_avg=T_avg, T_amplitude=T_amplitude)
        T_outside_array[k] = T_outside

        # Time-varying voltage instead of constant V_nom
        V_grid = get_voltage_profile(t, V_nom)
        V_grid_array[k] = V_grid

        T_room = room_model.get_temperature()
        P_elec, state = thermostat.update(T_room, V_grid)
        P_elec_array[k] = P_elec
        state_array[k] = state

        if k < 8 or (6 * 3600 <= t < 8 * 3600) or (18 * 3600 <= t < 20 * 3600):
            st = "ON " if state else "OFF"
            print(f"Step {k:4d} | t={t/3600:5.2f}h | T_room={T_room:5.2f}°C | "
                  f"V={V_grid:5.1f}V | State={st} | P={P_elec:4.0f}W")

        Q_hp = heat_pump.heat_pump_function(P_elec, COP)
        Q_hp_array[k] = Q_hp
        T_new = room_model.update(Q_hp, T_outside)
        T_room_array[k + 1] = T_new

    time_array[n_steps] = t_end
    T_outside_array[n_steps] = T_outside_array[n_steps - 1]
    V_grid_array[n_steps] = V_grid_array[n_steps - 1]
    P_elec_array[n_steps] = P_elec_array[n_steps - 1]
    Q_hp_array[n_steps] = Q_hp_array[n_steps - 1]
    state_array[n_steps] = state_array[n_steps - 1]

    return {
        "time": time_array,
        "T_room": T_room_array,
        "T_outside": T_outside_array,
        "P_elec": P_elec_array,
        "Q_hp": Q_hp_array,
        "state": state_array,
        "V_grid": V_grid_array,
    }


def plot_results_with_voltage(results: dict) -> None:
    """Plot results including grid voltage to show controller response."""
    time_hours = results["time"] / 3600
    V_nom = 230.0
    V_tol = 0.02
    V_min, V_max = V_nom * (1 - V_tol), V_nom * (1 + V_tol)

    fig, axes = plt.subplots(4, 1, figsize=(12, 12))

    # 1. Temperatures
    ax1 = axes[0]
    ax1.plot(time_hours, results["T_room"], "b-", label="Room", lw=2)
    ax1.plot(time_hours, results["T_outside"], "c--", label="Outside", lw=1.5)
    ax1.axhline(y=20, color="r", linestyle=":", label="Setpoint 20°C")
    ax1.axhline(y=19, color="orange", linestyle=":", alpha=0.5)
    ax1.axhline(y=21, color="orange", linestyle=":", alpha=0.5)
    ax1.set_ylabel("Temperature [°C]")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Room and Outside Temperature")

    # 2. Grid voltage
    ax2 = axes[1]
    ax2.plot(time_hours, results["V_grid"], "m-", label="V_grid", lw=2)
    ax2.axhline(y=V_nom, color="k", linestyle="--", alpha=0.7, label=f"V_nom={V_nom}V")
    ax2.axhspan(V_min, V_max, alpha=0.15, color="green", label="±2% band")
    ax2.set_ylabel("Voltage [V]")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Grid Voltage (dip 6–8h, spike 18–20h)")

    # 3. Heat pump power and heat
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    ax3.plot(time_hours, results["P_elec"], "g-", label="Electric Power", lw=2)
    ax3_twin.plot(time_hours, results["Q_hp"], "orange", label="Heat Output", lw=2)
    ax3.set_ylabel("Electric Power [W]", color="g")
    ax3_twin.set_ylabel("Heat Output [W]", color="orange")
    ax3.set_title("Heat Pump Operation")
    ax3.legend(loc="upper left")
    ax3_twin.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    # 4. State
    ax4 = axes[3]
    ax4.fill_between(time_hours, 0, results["state"], alpha=0.3, color="green", label="Heat pump ON")
    ax4.set_xlabel("Time [hours]")
    ax4.set_ylabel("State (0=OFF, 1=ON)")
    ax4.set_ylim([-0.1, 1.2])
    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.3)
    ax4.set_title("Heat Pump State")

    plt.tight_layout()
    plt.savefig("smart_room_voltage_scenario.png", dpi=150)
    print("\nPlot saved as 'smart_room_voltage_scenario.png'")
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("VOLTAGE SCENARIO SIMULATION (24h with voltage dip/spike)")
    print("=" * 60)
    results = simulate_with_voltage_profile()
    verify_energy_balance(results, C=20e6)
    plot_results_with_voltage(results)

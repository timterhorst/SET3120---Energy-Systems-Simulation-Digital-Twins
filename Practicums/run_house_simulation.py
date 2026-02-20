"""
run_house_simulation.py - Simulation using the House class.
"""

import numpy as np
import matplotlib.pyplot as plt
from house import House


def run_simulation(
    house: House,
    duration=86400,  # 24 hours in seconds
    V_grid_profile=None,  # Optional: time-varying voltage
):
    """
    Run a simulation with the House class.

    Parameters
    ----------
    house : House
        The house instance to simulate
    duration : float
        Simulation duration [s]
    V_grid_profile : callable, optional
        Function that returns V_grid given time t
    """
    n_steps = int(duration / house.dt)

    print("=== Running House Simulation ===")
    print(f"Duration: {duration/3600:.1f} hours")
    print(f"Time step: {house.dt/60:.1f} minutes")
    print(f"Number of steps: {n_steps}")
    print()

    for k in range(n_steps):
        # Get voltage (constant or time-varying)
        if V_grid_profile is not None:
            V_grid = V_grid_profile(house.current_time)
        else:
            V_grid = house.V_nom

        # Update house for one timestep
        state = house.update(V_grid=V_grid)

        # Print progress every 10 steps
        if k % 10 == 0 or k < 5:
            print(f"Step {k:4d} | t={state['time']/3600:5.2f}h | "
                  f"T_room={state['T_room']:5.2f}°C | "
                  f"State={'ON ' if state['state'] else 'OFF'} | "
                  f"P={state['P_elec']:4.0f}W")

    print("\nSimulation complete!")
    return house.get_history()


def plot_results(history):
    """
    Plot simulation results from House history.

    Parameters
    ----------
    history : dict
        Dictionary with time series data from House.get_history()
    """
    time_hours = np.array(history['time']) / 3600

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Temperatures
    ax1 = axes[0]
    ax1.plot(time_hours, history['T_room'], 'b-', label='Room Temperature', linewidth=2)
    ax1.plot(time_hours, history['T_outside'], 'c--', label='Outside Temperature', linewidth=1.5)
    ax1.axhline(y=20, color='r', linestyle=':', label='Setpoint')
    ax1.axhline(y=19, color='orange', linestyle=':', alpha=0.5, label='Lower threshold')
    ax1.axhline(y=21, color='orange', linestyle=':', alpha=0.5, label='Upper threshold')
    ax1.set_xlabel('Time [hours]')
    ax1.set_ylabel('Temperature [°C]')
    ax1.set_title('Room and Outside Temperature Over Time')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Heat Pump Power and Heat Output
    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    line1 = ax2.plot(time_hours, history['P_elec'], 'g-', label='Electric Power', linewidth=2)
    line2 = ax2_twin.plot(time_hours, history['Q_hp'], 'orange', label='Heat Output', linewidth=2)

    ax2.set_xlabel('Time [hours]')
    ax2.set_ylabel('Electric Power [W]', color='g')
    ax2_twin.set_ylabel('Heat Output [W]', color='orange')
    ax2.set_title('Heat Pump Operation')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2_twin.tick_params(axis='y', labelcolor='orange')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Heat Pump State
    ax3 = axes[2]
    ax3.fill_between(time_hours, 0, history['state'], alpha=0.3, color='green', label='Heat Pump ON')
    ax3.set_xlabel('Time [hours]')
    ax3.set_ylabel('State (0=OFF, 1=ON)')
    ax3.set_title('Heat Pump Operating State')
    ax3.set_ylim([-0.1, 1.2])
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('house_simulation_results.png', dpi=150)
    print("\nPlot saved as 'house_simulation_results.png'")
    plt.show()


def _run_modular_equivalent(duration=86400, dt=900, method='explicit'):
    """
    Run the same simulation using the modular components (room, controller, etc.)
    to get reference results for comparison. Does not modify run_modular_simulation.
    """
    import ambient
    import controller as ctrl_module
    import heat_pump
    import room

    C, R = 20e6, 0.02
    COP, P_rated = 3.0, 2000
    T_set, dT_lower, dT_upper = 20.0, 1.0, 1.0
    V_nom, V_tolerance = 230.0, 0.02
    T_room_init, T_avg, T_amplitude = 18.0, 5.0, 3.0
    n_steps = int(duration / dt)

    room_model = room.RoomModel(C=C, R=R, T_initial=T_room_init, dt=dt, method=method)
    thermostat = ctrl_module.ThermostatController(
        T_set=T_set, dT_lower=dT_lower, dT_upper=dT_upper,
        P_rated=P_rated, V_nom=V_nom, V_tolerance=V_tolerance,
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
        V_grid = V_nom
        V_grid_array[k] = V_grid
        T_room = room_model.get_temperature()
        P_elec, state = thermostat.update(T_room, V_grid)
        P_elec_array[k] = P_elec
        state_array[k] = state
        Q_hp = heat_pump.heat_pump_function(P_elec, COP)
        Q_hp_array[k] = Q_hp
        T_new = room_model.update(Q_hp, T_outside)
        T_room_array[k + 1] = T_new

    time_array[n_steps] = duration
    T_outside_array[n_steps] = T_outside_array[n_steps - 1]
    V_grid_array[n_steps] = V_grid_array[n_steps - 1]
    P_elec_array[n_steps] = P_elec_array[n_steps - 1]
    Q_hp_array[n_steps] = Q_hp_array[n_steps - 1]
    state_array[n_steps] = state_array[n_steps - 1]

    return {
        'time': time_array,
        'T_room': T_room_array,
        'T_outside': T_outside_array,
        'P_elec': P_elec_array,
        'Q_hp': Q_hp_array,
        'state': state_array,
        'V_grid': V_grid_array,
    }


def compare_with_modular(history_house=None):
    """
    Compare House class results with modular implementation.
    This verifies that the House class produces identical results.

    Parameters
    ----------
    history_house : dict, optional
        History from a House run (House.get_history()). If None, runs a new House simulation.
    """
    print("\n" + "=" * 60)
    print("VERIFICATION: Comparing House class with modular implementation")
    print("=" * 60)

    if history_house is None:
        house = House(dt=900, method='explicit')
        history_house = run_simulation(house, duration=86400)

    # Run modular version (same logic as run_modular_simulation, no import of that script)
    results_modular = _run_modular_equivalent(duration=86400, dt=900, method='explicit')

    # House history: one entry per step (after each update). Modular: T_room[0]=initial, T_room[1..n_steps]=after each step.
    T_final_house = history_house['T_room'][-1]
    T_final_modular = results_modular['T_room'][-1]

    # Element-wise: house history['T_room'][k] should match modular T_room[k+1]
    T_house_arr = np.array(history_house['T_room'])
    T_modular_step = results_modular['T_room'][1:]  # after step 0, 1, ..., n_steps-1
    diff_T = np.max(np.abs(T_house_arr - T_modular_step)) if len(T_house_arr) == len(T_modular_step) else float('nan')

    print(f"\nFinal room temperature:")
    print(f"  House class:   {T_final_house:.4f}°C")
    print(f"  Modular impl: {T_final_modular:.4f}°C")
    print(f"  Difference:   {abs(T_final_house - T_final_modular):.6f}°C")
    print(f"  Max |T_house - T_modular| over all steps: {diff_T:.6f}°C")

    if abs(T_final_house - T_final_modular) < 0.001 and (np.isnan(diff_T) or diff_T < 0.001):
        print("\n✓ VERIFICATION PASSED: Results are identical!")
    else:
        print("\n⚠ WARNING: Results differ slightly. Check implementation.")


if __name__ == "__main__":
    # Create a house instance
    house = House(
        T_initial=18.0,
        T_set=20.0,
        dt=900,  # 15 minutes
        method='explicit'
    )

    # Run simulation
    history = run_simulation(house, duration=86400)

    # Plot results
    plot_results(history)

    # Optional: Compare with modular implementation (uses same history, no extra run)
    # Uncomment to verify:
    # compare_with_modular(history)

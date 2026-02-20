"""
Smart Room Simulation - Single Implementation
SET3120 Practicum 2

All models integrated in one script for initial testing.
"""

import numpy as np
import matplotlib.pyplot as plt


def simulate_smart_room(
    # Simulation parameters
    dt=900,  # Time step [s] - 15 minutes = 900 seconds
    t_end=86400,  # Simulation duration [s] - 24 hours
    
    # Thermal parameters
    C=20e6,  # Thermal capacitance [J/K]
    R=0.005,  # Thermal resistance [K/W]
    
    # Heat pump parameters
    COP=3.0,  # Coefficient of Performance [-]
    P_rated=2000,  # Rated electric power [W]
    
    # Controller parameters
    T_set=20.0,  # Temperature setpoint [°C]
    dT_lower=1.0,  # Lower hysteresis band [K]
    dT_upper=1.0,  # Upper hysteresis band [K]
    
    # Voltage parameters
    V_nom=230.0,  # Nominal voltage [V]
    V_tolerance=0.02,  # Voltage tolerance (2%)
    
    # Initial conditions
    T_room_init=18.0,  # Initial room temperature [°C]
    T_outside_init=5.0,  # Initial outside temperature [°C]
    
    # Discretization method
    method='explicit',  # 'explicit' or 'implicit'
):
    """
    Simulate a smart room with heat pump for one day.
    
    This monolithic function contains all models inline for testing purposes.
    Later we will refactor this into modular functions.
    """
    
    # Calculate derived parameters
    tau = R * C  # Thermal time constant [s]
    n_steps = int(t_end / dt)  # Number of simulation steps
    
    print(f"=== Simulation Setup ===")
    print(f"Time step: {dt} s ({dt/60:.1f} min)")
    print(f"Total duration: {t_end} s ({t_end/3600:.1f} hours)")
    print(f"Number of steps: {n_steps}")
    print(f"Thermal time constant: {tau/3600:.1f} hours")
    print(f"Discretization method: {method}")
    print()
    
    # Initialize storage arrays
    time_array = np.zeros(n_steps + 1)
    T_room_array = np.zeros(n_steps + 1)
    T_outside_array = np.zeros(n_steps + 1)
    P_elec_array = np.zeros(n_steps + 1)
    Q_hp_array = np.zeros(n_steps + 1)
    state_array = np.zeros(n_steps + 1)  # 0 = OFF, 1 = ON
    V_grid_array = np.zeros(n_steps + 1)
    
    # Set initial conditions
    T_room_array[0] = T_room_init
    T_outside_array[0] = T_outside_init
    state = 0  # Start with heat pump OFF
    state_array[0] = state
    
    # Simulation loop
    for k in range(n_steps):
        # Current time
        t = k * dt
        time_array[k] = t
        
        # ===== 1. AMBIENT TEMPERATURE MODEL (Optional) =====
        # Simple sinusoidal variation: colder at night, warmer during day
        # T_outside = T_avg + amplitude * sin(2π * t / 86400 - π/2)
        # This gives minimum at 6am, maximum at 6pm
        T_avg = 5.0  # Average outside temp [°C]
        T_amplitude = 3.0  # Daily temperature swing [K]
        T_outside = T_avg + T_amplitude * np.sin(2 * np.pi * t / 86400 - np.pi / 2)
        T_outside_array[k] = T_outside
        
        # ===== 2. GRID VOLTAGE (For now: constant, can be made variable) =====
        # In later practicums, this comes from PowerGridModel
        V_grid = V_nom  # Nominal voltage
        V_grid_array[k] = V_grid
        
        # ===== 3. CONTROLLER =====
        # Read current room temperature
        T_room = T_room_array[k]
        
        # Define temperature thresholds
        T_lower = T_set - dT_lower
        T_upper = T_set + dT_upper
        
        # Voltage limits
        V_min = V_nom * (1 - V_tolerance)
        V_max = V_nom * (1 + V_tolerance)
        
        # Controller logic (with voltage priority)
        if V_grid < V_min:
            # Voltage too low - FORCE OFF for safety
            state = 0
            reason = "Voltage too low"
        elif V_grid > V_max and T_room < T_upper:
            # Voltage too high and room not too warm - FORCE ON to help grid
            state = 1
            reason = "Voltage too high, help grid"
        else:
            # Normal thermostat operation (hysteresis logic)
            if state == 0:  # Currently OFF
                if T_room < T_lower:
                    state = 1
                    reason = "Temp below lower threshold"
                else:
                    reason = "Stay OFF (hysteresis)"
            else:  # Currently ON (state == 1)
                if T_room > T_upper:
                    state = 0
                    reason = "Temp above upper threshold"
                else:
                    reason = "Stay ON (hysteresis)"
        
        # Determine electric power based on state
        if state == 1:
            P_elec = P_rated
        else:
            P_elec = 0.0
        
        P_elec_array[k] = P_elec
        state_array[k] = state
        
        # Debug output (first few steps)
        if k < 5 or k % 100 == 0:
            print(f"Step {k:4d} | t={t/3600:5.2f}h | T_room={T_room:5.2f}°C | "
                  f"State={'ON ' if state else 'OFF'} | P={P_elec:4.0f}W | {reason}")
        
        # ===== 4. HEAT PUMP MODEL =====
        Q_hp = COP * P_elec
        Q_hp_array[k] = Q_hp
        
        # ===== 5. ROOM THERMAL DYNAMICS (ODE Integration) =====
        # Heat loss to outside
        Q_loss = (T_room - T_outside) / R
        
        # Discretization
        if method == 'explicit':
            # Explicit Euler: T[k+1] = T[k] + dt * f(T[k], ...)
            dT_dt = Q_hp / C - Q_loss / C
            T_room_new = T_room + dt * dT_dt
            
        elif method == 'implicit':
            # Implicit Euler: T[k+1] = T[k] + dt * f(T[k+1], ...)
            # For our linear ODE, we can solve analytically:
            # T[k+1] = [T[k] + (dt/C)*Q_hp + (dt/tau)*T_outside] / (1 + dt/tau)
            numerator = T_room + (dt / C) * Q_hp + (dt / tau) * T_outside
            denominator = 1 + dt / tau
            T_room_new = numerator / denominator
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'explicit' or 'implicit'")
        
        # Store new temperature
        T_room_array[k + 1] = T_room_new
    
    # Store final timestep values
    time_array[n_steps] = t_end
    T_outside_array[n_steps] = T_outside_array[n_steps - 1]  # Repeat last value
    V_grid_array[n_steps] = V_grid_array[n_steps - 1]
    P_elec_array[n_steps] = P_elec_array[n_steps - 1]
    Q_hp_array[n_steps] = Q_hp_array[n_steps - 1]
    state_array[n_steps] = state_array[n_steps - 1]
    
    # Return results as dictionary
    results = {
        'time': time_array,
        'T_room': T_room_array,
        'T_outside': T_outside_array,
        'P_elec': P_elec_array,
        'Q_hp': Q_hp_array,
        'state': state_array,
        'V_grid': V_grid_array,
    }
    
    return results


def plot_results(results):
    """Plot simulation results."""
    time_hours = results['time'] / 3600  # Convert to hours
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Temperatures
    ax1 = axes[0]
    ax1.plot(time_hours, results['T_room'], 'b-', label='Room Temperature', linewidth=2)
    ax1.plot(time_hours, results['T_outside'], 'c--', label='Outside Temperature', linewidth=1.5)
    ax1.axhline(y=20, color='r', linestyle=':', label='Setpoint (20°C)')
    ax1.axhline(y=19, color='orange', linestyle=':', alpha=0.5, label='Lower threshold (19°C)')
    ax1.axhline(y=21, color='orange', linestyle=':', alpha=0.5, label='Upper threshold (21°C)')
    ax1.set_xlabel('Time [hours]')
    ax1.set_ylabel('Temperature [°C]')
    ax1.set_title('Room and Outside Temperature Over Time')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Heat Pump Power and Heat Output
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(time_hours, results['P_elec'], 'g-', label='Electric Power', linewidth=2)
    line2 = ax2_twin.plot(time_hours, results['Q_hp'], 'orange', label='Heat Output', linewidth=2)
    
    ax2.set_xlabel('Time [hours]')
    ax2.set_ylabel('Electric Power [W]', color='g')
    ax2_twin.set_ylabel('Heat Output [W]', color='orange')
    ax2.set_title('Heat Pump Operation')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Heat Pump State
    ax3 = axes[2]
    ax3.fill_between(time_hours, 0, results['state'], alpha=0.3, color='green', label='Heat Pump ON')
    ax3.set_xlabel('Time [hours]')
    ax3.set_ylabel('State (0=OFF, 1=ON)')
    ax3.set_title('Heat Pump Operating State')
    ax3.set_ylim([-0.1, 1.2])
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('smart_room_results.png', dpi=150)
    print("\nPlot saved as 'smart_room_results.png'")
    plt.show()


def verify_energy_balance(results, C):
    """
    Simple verification: check if energy balance makes sense.
    
    Total energy added to room should approximately equal:
    (final temp - initial temp) * C + energy lost to environment
    """
    dt = results['time'][1] - results['time'][0]
    
    # Energy added by heat pump
    E_added = np.sum(results['Q_hp']) * dt  # [J]
    
    # Temperature change
    dT = results['T_room'][-1] - results['T_room'][0]
    E_stored = C * dT  # [J]
    
    print("\n=== Energy Balance Check ===")
    print(f"Energy added by heat pump: {E_added/1e6:.2f} MJ")
    print(f"Energy stored (temp increase): {E_stored/1e6:.2f} MJ")
    print(f"Energy lost to environment: {(E_added - E_stored)/1e6:.2f} MJ")
    print(f"Final room temperature: {results['T_room'][-1]:.2f}°C")


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("Starting Smart Room Simulation\n")
    
    # Run simulation with explicit method
    print("=" * 60)
    print("RUNNING EXPLICIT EULER METHOD")
    print("=" * 60)
    results_explicit = simulate_smart_room(
        method='explicit',
    )
    
    # Verify results
    verify_energy_balance(results_explicit, C=20e6)
    
    # Plot results
    plot_results(results_explicit)
    
    # Optional: Run with implicit method and compare
    print("\n" + "=" * 60)
    print("RUNNING IMPLICIT EULER METHOD")
    print("=" * 60)
    results_implicit = simulate_smart_room(
        method='implicit',
    )
    
    verify_energy_balance(results_implicit, C=20e6)
    
    # Compare methods
    print("\n=== Comparison ===")
    print(f"Explicit - Final temp: {results_explicit['T_room'][-1]:.4f}°C")
    print(f"Implicit - Final temp: {results_implicit['T_room'][-1]:.4f}°C")
    print(f"Difference: {abs(results_explicit['T_room'][-1] - results_implicit['T_room'][-1]):.4f}°C")
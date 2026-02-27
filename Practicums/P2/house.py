"""
house.py - House class that encapsulates all smart house components.
"""

from room import RoomModel
from controller import ThermostatController
from heat_pump import heat_pump_function
from ambient import get_outside_temperature


class House:
    """
    A smart house with heat pump, thermal model, and thermostat controller.

    This class encapsulates all sub-models and manages their interactions.
    It stores simulation history for later analysis and plotting.
    """

    def __init__(
        self,
        # Thermal parameters
        C=20e6,           # Thermal capacitance [J/K]
        R=0.02,           # Thermal resistance [K/W]
        T_initial=18.0,   # Initial room temperature [°C]

        # Controller parameters
        T_set=20.0,       # Temperature setpoint [°C]
        dT_lower=1.0,     # Lower hysteresis band [K]
        dT_upper=1.0,     # Upper hysteresis band [K]

        # Heat pump parameters
        COP=3.0,          # Coefficient of Performance [-]
        P_rated=2000,     # Rated electric power [W]

        # Voltage parameters
        V_nom=230.0,      # Nominal voltage [V]
        V_tolerance=0.02, # Voltage tolerance (±2%)

        # Simulation parameters
        dt=900,           # Time step [s]
        method='explicit', # Discretization method

        # Ambient temperature parameters
        T_avg=5.0,        # Average outside temperature [°C]
        T_amplitude=3.0,  # Daily temperature variation [K]
    ):
        """
        Initialize the House with all sub-models and parameters.
        """
        # Store parameters as instance attributes
        self.COP = COP
        self.P_rated = P_rated
        self.dt = dt
        self.T_avg = T_avg
        self.T_amplitude = T_amplitude
        self.V_nom = V_nom

        # Initialize sub-models
        self.room = RoomModel(
            C=C,
            R=R,
            T_initial=T_initial,
            dt=dt,
            method=method
        )

        self.controller = ThermostatController(
            T_set=T_set,
            dT_lower=dT_lower,
            dT_upper=dT_upper,
            P_rated=P_rated,
            V_nom=V_nom,
            V_tolerance=V_tolerance
        )

        # Initialize history storage for plotting
        self.history = {
            'time': [],
            'T_room': [],
            'T_outside': [],
            'P_elec': [],
            'Q_hp': [],
            'state': [],
            'V_grid': [],
        }

        # Initialize simulation clock
        self.current_time = 0.0

    def update(self, V_grid=None):
        """
        Update the house state for one timestep.

        This method orchestrates the sub-models in the correct sequence:
        1. Get outside temperature
        2. Controller decides heat pump power
        3. Heat pump produces heat
        4. Room temperature updates

        Parameters
        ----------
        V_grid : float, optional
            Grid voltage [V]. If None, uses nominal voltage.

        Returns
        -------
        dict
            Current state of all variables
        """
        # Use nominal voltage if not provided
        if V_grid is None:
            V_grid = self.V_nom

        # 1. Get outside temperature at current time
        T_outside = get_outside_temperature(
            t=self.current_time,
            T_avg=self.T_avg,
            T_amplitude=self.T_amplitude
        )

        # 2. Controller determines heat pump power
        T_room = self.room.get_temperature()
        P_elec, state = self.controller.update(T_room, V_grid)

        # 3. Heat pump converts electric power to heat
        Q_hp = heat_pump_function(P_elec, self.COP)

        # 4. Room temperature updates based on heat input
        T_room_new = self.room.update(Q_hp, T_outside)

        # Store results in history
        self.history['time'].append(self.current_time)
        self.history['T_room'].append(T_room_new)
        self.history['T_outside'].append(T_outside)
        self.history['P_elec'].append(P_elec)
        self.history['Q_hp'].append(Q_hp)
        self.history['state'].append(state)
        self.history['V_grid'].append(V_grid)

        # Advance simulation clock
        self.current_time += self.dt

        # Return current state as dictionary
        return {
            'time': self.current_time,
            'T_room': T_room_new,
            'T_outside': T_outside,
            'P_elec': P_elec,
            'Q_hp': Q_hp,
            'state': state,
            'V_grid': V_grid,
        }

    def get_history(self):
        """
        Get the complete simulation history.

        Returns
        -------
        dict
            Dictionary with time series of all variables
        """
        return self.history

    def reset(self, T_initial=None):
        """
        Reset the house to initial conditions.

        Parameters
        ----------
        T_initial : float, optional
            New initial room temperature. If None, keeps original.
        """
        # Reset sub-models
        if T_initial is not None:
            self.room.T_room = T_initial
        self.controller.state = 0

        # Clear history
        for key in self.history:
            self.history[key] = []

        # Reset clock
        self.current_time = 0.0

    def __repr__(self):
        """String representation of the House."""
        return (f"House(T_room={self.room.get_temperature():.2f}°C, "
                f"state={'ON' if self.controller.state else 'OFF'}, "
                f"time={self.current_time/3600:.2f}h)")

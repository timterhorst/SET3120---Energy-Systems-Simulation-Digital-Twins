"""
Thermostat controller with hysteresis and voltage protection.
SET3120 Practicum 2.
"""


class ThermostatController:
    """
    Hysteresis thermostat: turns heat pump ON/OFF based on room temperature
    and grid voltage. Voltage checks have priority over temperature.
    """

    def __init__(
        self,
        T_set: float,
        dT_lower: float,
        dT_upper: float,
        P_rated: float,
        V_nom: float,
        V_tolerance: float,
    ) -> None:
        """
        Initialize controller with setpoints and limits.

        Parameters
        ----------
        T_set : float
            Temperature setpoint [°C].
        dT_lower : float
            Lower hysteresis band [K] (turn ON when T_room < T_set - dT_lower).
        dT_upper : float
            Upper hysteresis band [K] (turn OFF when T_room > T_set + dT_upper).
        P_rated : float
            Rated electric power when ON [W].
        V_nom : float
            Nominal grid voltage [V].
        V_tolerance : float
            Voltage tolerance (e.g. 0.02 for ±2%).
        """
        self.T_set = T_set
        self.dT_lower = dT_lower
        self.dT_upper = dT_upper
        self.P_rated = P_rated
        self.V_nom = V_nom
        self.V_tolerance = V_tolerance
        self.V_min = V_nom * (1 - V_tolerance)
        self.V_max = V_nom * (1 + V_tolerance)
        self.T_lower = T_set - dT_lower
        self.T_upper = T_set + dT_upper
        self.state = 0  # 0 = OFF, 1 = ON

    def update(self, T_room: float, V_grid: float) -> tuple[float, int]:
        """
        Determine heat pump power based on temperature and voltage.

        Priority: (1) voltage safety, (2) hysteresis thermostat.

        Parameters
        ----------
        T_room : float
            Current room temperature [°C].
        V_grid : float
            Current grid voltage [V].

        Returns
        -------
        tuple[float, int]
            (P_elec [W], state) with state 0=OFF, 1=ON.
        """
        # 1. Voltage priority (safety and grid support)
        if V_grid < self.V_min:
            self.state = 0  # Voltage too low - FORCE OFF
        elif V_grid > self.V_max and T_room < self.T_upper:
            self.state = 1  # Voltage too high and room not too warm - FORCE ON
        else:
            # 2. Normal hysteresis thermostat
            if self.state == 0:  # Currently OFF
                if T_room < self.T_lower:
                    self.state = 1
            else:  # Currently ON
                if T_room > self.T_upper:
                    self.state = 0

        P_elec = self.P_rated if self.state == 1 else 0.0
        return (P_elec, self.state)

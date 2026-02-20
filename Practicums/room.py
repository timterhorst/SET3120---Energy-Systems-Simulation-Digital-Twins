"""
Room thermal dynamics for SET3120 Practicum 2.
ODE integration with explicit or implicit Euler.
"""


class RoomModel:
    """
    Lumped thermal model of a room: capacitance C, resistance R to outside.
    State: current room temperature T_room.
    """

    def __init__(
        self,
        C: float,
        R: float,
        T_initial: float,
        dt: float,
        method: str = "explicit",
    ) -> None:
        """
        Initialize room model with thermal parameters.

        Parameters
        ----------
        C : float
            Thermal capacitance [J/K].
        R : float
            Thermal resistance [K/W].
        T_initial : float
            Initial room temperature [°C].
        dt : float
            Time step [s].
        method : str, optional
            Discretization: 'explicit' or 'implicit' Euler.
        """
        self.C = C
        self.R = R
        self.T_room = T_initial
        self.dt = dt
        self.method = method
        self.tau = R * C  # Thermal time constant [s]

    def update(self, Q_hp: float, T_outside: float) -> float:
        """
        Update room temperature for one timestep.

        Parameters
        ----------
        Q_hp : float
            Heat input from heat pump [W].
        T_outside : float
            Outside temperature [°C].

        Returns
        -------
        float
            New room temperature [°C] after the timestep.
        """
        tau = self.tau
        dt = self.dt
        C = self.C
        T = self.T_room

        if self.method == "explicit":
            # Explicit Euler: T[k+1] = T[k] + dt * (Q_hp/C - (T[k]-T_out)/(R*C))
            dT_dt = Q_hp / C - (T - T_outside) / (self.R * C)
            T_new = T + dt * dT_dt
        elif self.method == "implicit":
            # Implicit Euler: T[k+1] = (T[k] + (dt/C)*Q_hp + (dt/tau)*T_out) / (1 + dt/tau)
            numerator = T + (dt / C) * Q_hp + (dt / tau) * T_outside
            denominator = 1 + dt / tau
            T_new = numerator / denominator
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'explicit' or 'implicit'")

        self.T_room = T_new
        return T_new

    def get_temperature(self) -> float:
        """Return current room temperature [°C]."""
        return self.T_room

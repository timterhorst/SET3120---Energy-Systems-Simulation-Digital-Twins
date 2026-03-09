"""EV battery model — dynamic model with explicit Euler integration for SOC."""


class EVBatteryModel:
    """Models the State of Charge (SOC) dynamics of an EV battery.

    ODE:
        Charging  (P >= 0): dSOC/dt = η_ch  * P / E_cap
        Discharging (P < 0): dSOC/dt = P / (η_dis * E_cap)

    The asymmetry ensures that discharging depletes the battery *faster* per unit
    of power delivered to the grid (round-trip losses).

    Integration uses explicit Euler, matching the room model convention.
    delta_t is converted from minutes (config units) to hours (kWh-consistent units).
    """

    def __init__(self, ev_settings: dict, delta_t: float):
        """
        Args:
            ev_settings: the 'EVSettings' dict from ev_config.yaml.
            delta_t: simulation timestep in minutes.
        """
        battery = ev_settings['battery']
        self.soc = battery['initial_soc']
        self.capacity_kwh = battery['capacity_kwh']
        self.eta_charge = battery['efficiency_charge']
        self.eta_discharge = battery['efficiency_discharge']
        self.delta_t_hours = delta_t / 60.0

    def __call__(self, p_charge_w: float, soc_depletion: float = 0.0) -> float:
        """Update SOC for one timestep.

        Args:
            p_charge_w: charging power [W]. Positive = grid-to-battery, negative = V2G.
            soc_depletion: SOC fraction to subtract for driving consumption
                           (applied before the Euler step, non-zero only at arrival).

        Returns:
            Updated SOC as a fraction in [0, 1].
        """
        if soc_depletion > 0.0:
            self.soc = max(0.0, self.soc - soc_depletion)

        p_charge_kw = p_charge_w / 1000.0

        if p_charge_kw >= 0:
            dsoc_dt = self.eta_charge * p_charge_kw / self.capacity_kwh
        else:
            dsoc_dt = p_charge_kw / (self.eta_discharge * self.capacity_kwh)

        self.soc += self.delta_t_hours * dsoc_dt
        self.soc = max(0.0, min(1.0, self.soc))

        return self.soc

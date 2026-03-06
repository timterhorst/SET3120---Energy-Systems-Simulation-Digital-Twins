"""Heat pump model."""


def heat_pump_function(power_setpoint: float, COP: float = 3.0) -> float:
    """Compute heat production from electric power using Coefficient of Performance.

    Parameters
    ----------
    power_setpoint : float
        Electric power consumed by the heat pump [W].
    COP : float, optional
        Coefficient of Performance [-], default 3.0 (typical air-source HP).

    Returns
    -------
    float
        Heat production [W]: COP * power_setpoint.
    """
    if power_setpoint < 0:
        raise ValueError("power_setpoint must be non-negative")
    return COP * power_setpoint
"""Heat pump model."""


def heat_pump_function(power_setpoint: float) -> float:
    """COP of the heat pump is 3, so 3W of electricity input produces 1W of heat output."""
    heat_production = 3 * power_setpoint  # Heat production as a function of power setpoint
    return heat_production

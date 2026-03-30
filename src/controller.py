"""Controller model."""


def controller_function(
    power_set_point_hp: float, voltage: float, temperature: float,
    controller_settings: dict, temp_min_override: float = None,
) -> float:
    """A simple controller for co-simulated coupled electric grid, heat pump, and building systems.
    
    The controller is used to adjust the power setpoint of the heat pump based boundary conditions.

    Uses bang-bang (hysteresis) control with a fixed-width dead band that
    tracks temp_min.  When temp_min_override lowers the setpoint (occupancy
    setback), the turn-off threshold drops by the same amount so the HP
    does not heat an empty house up to temp_max.

    Args:
        temp_min_override: if provided, replaces the minimum temperature from config.
            Used by the Driver Model to implement occupancy-based setback.
    """
    bc = controller_settings['ControllerSettings']['boundary_conditions']
    voltage_min = bc['minimum_voltage']
    voltage_max = bc['maximum_voltage']
    temp_min_default = bc['minimum_temperature']
    temp_max = bc['maximum_temperature']

    temp_min = temp_min_override if temp_min_override is not None else temp_min_default

    # The hysteresis band width is fixed at (temp_max − default_temp_min).
    # When setback lowers temp_min, the upper threshold tracks it so the
    # band stays the same width (e.g. [10, 14] instead of [10, 22]).
    deadband = temp_max - temp_min_default
    temp_off = temp_min + deadband

    p_adjust_step_size_voltage = controller_settings['ControllerSettings']['actions']['p_change_for_voltage']
    p_adjust_step_size_temp = controller_settings['ControllerSettings']['actions']['p_change_for_temperature']

    # Temperature control (voltage within limits)
    if voltage_min <= voltage <= voltage_max:
        if temperature > temp_off:
            power_set_point_hp = 0
        elif temperature < temp_min:
            power_set_point_hp = p_adjust_step_size_temp

    # Voltage control (takes priority over temperature)
    elif voltage < voltage_min:
        power_set_point_hp = 0
    elif voltage > voltage_max:
        if temperature < temp_max:
            power_set_point_hp = p_adjust_step_size_voltage
        else:
            power_set_point_hp = 0

    return power_set_point_hp

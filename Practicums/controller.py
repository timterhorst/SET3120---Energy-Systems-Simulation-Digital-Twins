"""Controller model."""


def controller_function(
    power_set_point_hp: float,
    voltage: float,
    temperature: float,
    controller_settings: dict,
    priority: str = "voltage",
) -> float:
    """A simple controller for co-simulated coupled electric grid, heat pump, and building systems.

    Parameters
    ----------
    priority : str
        "voltage"     – voltage control overrides temperature (original behaviour)
        "temperature" – temperature control overrides voltage
        "equal"       – both objectives weighted equally (average of desired setpoints)
    """
    bounds = controller_settings['ControllerSettings']['boundary_conditions']
    voltage_min = bounds['minimum_voltage']
    voltage_max = bounds['maximum_voltage']
    temp_min = bounds['minimum_temperature']
    temp_max = bounds['maximum_temperature']

    actions = controller_settings['ControllerSettings']['actions']
    p_voltage = actions['p_change_for_voltage']
    p_temp = actions['p_change_for_temperature']

    # Determine what each objective independently wants (None = no opinion)
    temp_desire = None
    if temperature > temp_max:
        temp_desire = 0
    elif temperature < temp_min:
        temp_desire = p_temp

    voltage_desire = None
    if voltage < voltage_min:
        voltage_desire = 0
    elif voltage > voltage_max:
        voltage_desire = p_voltage

    if priority == "voltage":
        # Voltage overrides: check voltage first, only do temperature if voltage OK
        if voltage_desire is not None:
            power_set_point_hp = voltage_desire
        elif temp_desire is not None:
            power_set_point_hp = temp_desire

    elif priority == "temperature":
        # Temperature overrides: check temperature first, only do voltage if temp OK
        if temp_desire is not None:
            power_set_point_hp = temp_desire
        elif voltage_desire is not None:
            power_set_point_hp = voltage_desire

    elif priority == "equal":
        # Both equally important: average the desired setpoints
        desires = [d for d in (temp_desire, voltage_desire) if d is not None]
        if desires:
            power_set_point_hp = sum(desires) / len(desires)

    return power_set_point_hp
"""Charging controller — quasi-static logic for EV smart charging and V2G."""


def charging_controller_function(
    voltage: float, soc: float, is_home: bool, ev_settings: dict,
) -> float:
    """Determine EV charging or discharging power.

    Priority logic (evaluated top-to-bottom):
        1. Not home          → idle (0 W)
        2. Undervoltage       → V2G discharge if SOC > reserve, else idle
        3. Overvoltage        → charge at max rate to act as additional load
        4. Normal voltage     → tiered charging based on SOC

    V2G discharges during undervoltage to inject power and raise voltage.
    During overvoltage the EV charges at full rate to increase load and lower voltage.
    This mirrors the heat pump controller's voltage-support strategy.

    Args:
        voltage: grid voltage at the connection point [p.u.].
        soc: current state of charge [-], in [0, 1].
        is_home: True if the EV is parked at home.
        ev_settings: the 'EVSettings' dict from ev_config.yaml.

    Returns:
        Charging power [W]. Positive = charging, negative = V2G discharge.
    """
    if not is_home:
        return 0.0

    charging = ev_settings['charging']
    p_max = charging['p_charge_max_w']
    p_dis = charging['p_discharge_max_w']
    soc_min_v2g = charging['soc_min_v2g']
    v_min = charging['voltage_min']
    v_max = charging['voltage_max']

    # --- Undervoltage: V2G to support grid ---
    if voltage < v_min:
        if soc > soc_min_v2g:
            return -p_dis
        return 0.0

    # --- Overvoltage: absorb power to reduce voltage ---
    if voltage > v_max:
        if soc < 1.0:
            return p_max
        return 0.0

    # --- Normal voltage: tiered smart charging ---
    if soc >= 1.0:
        return 0.0
    if soc < 0.20:
        return p_max
    if soc < 0.50:
        return 0.75 * p_max
    return 0.50 * p_max

"""
Heat pump model for SET3120 Practicum 2.
Converts electric power to heat output using COP.
"""


def heat_pump_function(P_elec: float, COP: float = 3.0) -> float:
    """
    Compute heat production from electric power.

    Parameters
    ----------
    P_elec : float
        Electric power consumed by the heat pump [W].
    COP : float, optional
        Coefficient of performance [-], default 3.0.

    Returns
    -------
    float
        Heat production Q_hp [W]: Q_hp = COP * P_elec.

    Raises
    ------
    ValueError
        If P_elec is negative.
    """
    if P_elec < 0:
        raise ValueError("P_elec must be non-negative")
    return COP * P_elec

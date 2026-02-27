"""
Ambient (outside) temperature profile for SET3120 Practicum 2.
Sinusoidal daily variation: minimum at 6am, maximum at 6pm.
"""

import math


def get_outside_temperature(
    t: float,
    T_avg: float = 5.0,
    T_amplitude: float = 3.0,
) -> float:
    """
    Compute outside temperature at a given time.

    Parameters
    ----------
    t : float
        Time [s] from simulation start.
    T_avg : float, optional
        Average outside temperature [°C], default 5.0.
    T_amplitude : float, optional
        Daily temperature swing [K], default 3.0.

    Returns
    -------
    float
        Outside temperature [°C].
        Formula: T_out = T_avg + T_amplitude * sin(2π * t / 86400 - π/2).
    """
    return T_avg + T_amplitude * math.sin(2 * math.pi * t / 86400 - math.pi / 2)

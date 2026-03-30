"""Load and interpolate KNMI hourly temperature data for the co-simulation."""

import numpy as np


def load_knmi_temperature(filepath: str, n_steps: int = 35040) -> np.ndarray:
    """Read KNMI hourly temperature and interpolate to quarter-hourly resolution.

    Args:
        filepath: path to KNMI_temp_data.txt (station 344, year 2019).
        n_steps: number of simulation timesteps (default 35040 for Δt=15 min).

    Returns:
        numpy array of shape (n_steps,) with outside temperature in °C.
    """
    rows = []
    with open(filepath, "r") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            stn = parts[0].strip()
            yyyymmdd = int(parts[1].strip())
            hh = int(parts[2].strip())
            t_raw = int(parts[3].strip())
            rows.append((stn, yyyymmdd, hh, t_raw))

    rows.sort(key=lambda r: (r[1], r[2]))

    assert len(rows) == 8760, (
        f"Expected 8760 hourly rows for a full year, got {len(rows)}"
    )

    hourly_temp = np.array([r[3] for r in rows], dtype=np.float64) / 10.0

    # KNMI HH=1 is the observation at 01:00, HH=24 at 24:00.
    # Hourly time points in minutes from Jan 1 00:00:
    #   60, 120, …, 525600
    hourly_minutes = np.arange(1, 8761) * 60.0

    # Simulation timesteps: k=0 → 00:15, k=1 → 00:30, …, k=35039 → 525600 min
    sim_minutes = np.arange(1, n_steps + 1) * 15.0

    # np.interp holds the boundary value for points outside xp range,
    # so the first few quarter-hours (15, 30, 45 min) get the HH=1 value.
    knmi_temp = np.interp(sim_minutes, hourly_minutes, hourly_temp)

    return knmi_temp

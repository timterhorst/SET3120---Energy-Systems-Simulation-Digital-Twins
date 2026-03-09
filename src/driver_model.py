"""Driver model — discrete event model for EV occupancy and driving behaviour."""
import numpy as np


class DriverModel:
    """Models daily departure/arrival schedule and stochastic driving energy consumption.

    At each timestep, produces:
        - is_home (bool): whether the EV is parked at home
        - temp_min_dynamic (float): thermostat lower bound (occupancy setback)
        - soc_depletion (float): SOC fraction consumed by driving (non-zero only at arrival)

    Departure and arrival times are re-sampled daily from uniform distributions.
    SOC depletion upon arrival is also sampled daily (uniform).
    """

    def __init__(self, ev_settings: dict):
        driver = ev_settings['driver']
        self.departure_earliest = driver['departure_earliest']
        self.departure_latest = driver['departure_latest']
        self.arrival_earliest = driver['arrival_earliest']
        self.arrival_latest = driver['arrival_latest']
        self.soc_depletion_min = driver['soc_depletion_min']
        self.soc_depletion_max = driver['soc_depletion_max']
        self.temp_min_home = driver['temp_min_home']
        self.temp_min_away = driver['temp_min_away']

        seed = driver.get('random_seed', None)
        self.rng = np.random.default_rng(seed)

        self._current_day = -1
        self._departure_time = 0.0
        self._arrival_time = 0.0
        self._soc_depletion_today = 0.0
        self._arrived_today = False

    def __call__(self, time_clock: float) -> tuple[bool, float, float]:
        """Determine driver status for the current simulation time.

        Args:
            time_clock: simulation time in minutes from the start.

        Returns:
            Tuple of (is_home, temp_min_dynamic, soc_depletion).
            soc_depletion is 0.0 on every timestep except the first timestep after arrival.
        """
        day = int(time_clock // 1440)
        time_in_day = time_clock % 1440

        if day != self._current_day:
            self._current_day = day
            self._departure_time = self.rng.uniform(self.departure_earliest, self.departure_latest)
            self._arrival_time = self.rng.uniform(self.arrival_earliest, self.arrival_latest)
            self._soc_depletion_today = self.rng.uniform(self.soc_depletion_min, self.soc_depletion_max)
            self._arrived_today = False

        is_home = time_in_day < self._departure_time or time_in_day >= self._arrival_time
        temp_min_dynamic = self.temp_min_home if is_home else self.temp_min_away

        # Apply driving SOC depletion exactly once, at the first timestep after arrival
        soc_depletion = 0.0
        if is_home and time_in_day >= self._arrival_time and not self._arrived_today:
            self._arrived_today = True
            soc_depletion = self._soc_depletion_today

        return is_home, temp_min_dynamic, soc_depletion

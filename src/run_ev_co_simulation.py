"""Run the EV + V2G co-simulation.

Usage (from the project root):
    python -m src.run_ev_co_simulation
"""
import os
from functools import partial

import yaml

from .charging_controller import charging_controller_function
from .controller import controller_function
from .cosim_framework import Model
from .driver_model import DriverModel
from .ev_battery import EVBatteryModel
from .ev_cosim_framework import EVManager
from .grid import electric_grid_function
from .heat_pump import heat_pump_function
from .load_configurations import load_configurations
from .room import RoomFunction


# 1. Load configurations
configurations_folder_path = './configurations'
controller_config, settings_configs = load_configurations(configurations_folder_path)

ev_config_path = os.path.join(configurations_folder_path, 'ev_config.yaml')
with open(ev_config_path, 'r') as f:
    ev_config = yaml.safe_load(f)

settings = settings_configs["config 1"]
ev_settings = ev_config['EVSettings']
delta_t = settings['InitializationSettings']['time']['delta_t']

# 2. Create model instances
electric_grid_model = Model(electric_grid_function)
heat_pump_model = Model(heat_pump_function)
room_model = Model(RoomFunction(settings))
controller_model = Model(partial(controller_function, controller_settings=controller_config))
driver_model = Model(DriverModel(ev_settings))
ev_battery_model = Model(EVBatteryModel(ev_settings, delta_t))
charging_controller_model = Model(partial(charging_controller_function, ev_settings=ev_settings))

# 3. Run EV + V2G co-simulation
models = [
    electric_grid_model,
    heat_pump_model,
    room_model,
    controller_model,
    driver_model,
    ev_battery_model,
    charging_controller_model,
]
manager = EVManager(models, settings)
manager.run_simulation()

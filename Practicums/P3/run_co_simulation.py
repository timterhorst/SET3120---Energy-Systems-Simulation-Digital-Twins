"""Run the co-simulation."""
import os
from functools import partial

from controller import controller_function
from cosim_framework import Manager, Model
from grid import electric_grid_function
from heat_pump import heat_pump_function
from load_configurations import load_configurations
from room import RoomFunction


# 1. Load configurations from the 'configurations' folder (next to this script)
_config_dir = os.path.dirname(os.path.abspath(__file__))
configurations_folder_path = os.path.join(_config_dir, 'configurations')
controller_config, settings_configs = load_configurations(configurations_folder_path)

print(settings_configs)

# Currently, 3 settings configurations are loaded
first_settings_config = settings_configs["config 1"]
second_settings_config = settings_configs["config 2"]
third_settings_config = settings_configs["config 3"]
fourth_settings_config = settings_configs["config 4"]
fifth_settings_config = settings_configs["config 5"] 
sixth_settings_config = settings_configs["config 6"]   

# Create model instances by wrapping the functions with the Model class 
electric_grid_model = Model(electric_grid_function)
heat_pump_model = Model(heat_pump_function)
controller_model = Model(partial(controller_function, controller_settings=controller_config))

# Run with the second settings_configuration
room_model = Model(RoomFunction(first_settings_config))

models = [electric_grid_model, heat_pump_model, room_model, controller_model]
manager = Manager(models, first_settings_config)
manager.run_simulation()

# Run with the second settings_configuration
room_model = Model(RoomFunction(second_settings_config))

models = [electric_grid_model, heat_pump_model, room_model, controller_model]
manager = Manager(models, second_settings_config)
manager.run_simulation()

# Run with the third settings_configuration
room_model = Model(RoomFunction(third_settings_config))

models = [electric_grid_model, heat_pump_model, room_model, controller_model]
manager = Manager(models, third_settings_config)
manager.run_simulation()

# Run with the fourth settings_configuration
room_model = Model(RoomFunction(fourth_settings_config))

models = [electric_grid_model, heat_pump_model, room_model, controller_model]
manager = Manager(models, fourth_settings_config)
manager.run_simulation()

# Run with the fifth settings_configuration
room_model = Model(RoomFunction(fifth_settings_config))

models = [electric_grid_model, heat_pump_model, room_model, controller_model]
manager = Manager(models, fifth_settings_config)
manager.run_simulation()

# Run with the sixth settings_configuration
room_model = Model(RoomFunction(sixth_settings_config))

models = [electric_grid_model, heat_pump_model, room_model, controller_model]
manager = Manager(models, sixth_settings_config)
manager.run_simulation()
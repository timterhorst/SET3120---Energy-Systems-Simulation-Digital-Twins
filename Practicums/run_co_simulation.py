"""Run the co-simulation."""
import os
from functools import partial

from controller import controller_function
from cosim_framework import Manager, Model
from grid import electric_grid_function
from heat_pump import heat_pump_function
from load_configurations import load_configurations
from room import RoomFunction

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Load configurations from the 'configurations' folder
configurations_folder_path = os.path.join(SCRIPT_DIR, 'configurations')
controller_config, settings_configs = load_configurations(configurations_folder_path)

# Resolve relative data paths in settings configs to absolute paths
for config in settings_configs.values():
    init = config['InitializationSettings']
    for key in ('passive_consumers_power_setpoints', 'grid_topology'):
        init[key] = os.path.join(SCRIPT_DIR, init[key])

# 2. Create model instances
electric_grid_model = Model(electric_grid_function)
heat_pump_model = Model(heat_pump_function)
# Change priority to "voltage", "temperature", or "equal"
CONTROLLER_PRIORITY = "voltage"
controller_model = Model(partial(
    controller_function, controller_settings=controller_config, priority=CONTROLLER_PRIORITY,
))

sim_config = settings_configs["config 1"]

smart_consumers = {
    "Customer_94": {"room": Model(RoomFunction(sim_config))},
    "Customer_95": {"room": Model(RoomFunction(sim_config))},
}

# 3. Run co-simulation
manager = Manager(
    electric_grid=electric_grid_model,
    heat_pump=heat_pump_model,
    controller=controller_model,
    smart_consumers=smart_consumers,
    settings_configuration=sim_config,
)
manager.run_simulation()


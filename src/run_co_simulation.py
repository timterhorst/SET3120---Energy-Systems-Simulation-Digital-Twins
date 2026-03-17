"""Run the co-simulation.

Usage (from the project root):
    python -m src.run_co_simulation          # defaults to config 1
    python -m src.run_co_simulation 99       # runs config 99 (3x stressed)
    python -m src.run_co_simulation --use-forecasted
    python -m src.run_co_simulation 99 --use-forecasted
"""
import argparse
import sys
from functools import partial

from .controller import controller_function
from .cosim_framework import Manager, Model
from .grid import electric_grid_function
from .heat_pump import heat_pump_function
from .load_configurations import load_configurations
from .room import RoomFunction


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the base co-simulation.")
    parser.add_argument(
        "config_id",
        nargs="?",
        default="1",
        help="Configuration id (e.g. 1 or 99). Defaults to 1.",
    )
    parser.add_argument(
        "--use-forecasted",
        action="store_true",
        help="Use the forecasted passive consumer dataset.",
    )
    return parser.parse_args(argv)


# 1. Load configurations from the 'configurations' folder (path relative to project root)
args = _parse_args(sys.argv[1:])
configurations_folder_path = './configurations'
controller_config, settings_configs = load_configurations(
    configurations_folder_path,
    use_forecasted=args.use_forecasted,
)

config_id = args.config_id
config_key = f"config {config_id}"
if config_key not in settings_configs:
    print(f"ERROR: '{config_key}' not found. Available: {list(settings_configs.keys())}")
    sys.exit(1)

# 2. Create model instances by wrapping the functions with the Model class
settings = settings_configs[config_key]
electric_grid_model = Model(electric_grid_function)
heat_pump_model = Model(heat_pump_function)
room_model = Model(RoomFunction(settings))
controller_model = Model(partial(controller_function, controller_settings=controller_config))

# 3. Run co-simulation with the given configurations
models = [electric_grid_model, heat_pump_model, room_model, controller_model]
manager = Manager(models, settings)
manager.run_simulation()

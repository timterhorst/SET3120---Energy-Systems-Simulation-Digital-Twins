"""Run the co-simulation."""
import argparse
import os
import pickle
from functools import partial

from controller import controller_function
from cosim_framework import Manager, Model
from grid import electric_grid_function
from heat_pump import heat_pump_function
from load_configurations import load_configurations
from room import RoomFunction

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIRST_RUN_RESULTS_PATH = os.path.join(SCRIPT_DIR, "first_run_results.pkl")


def main():
    parser = argparse.ArgumentParser(description="Run co-simulation with real or forecasted data.")
    parser.add_argument(
        "--use-forecasted",
        action="store_true",
        help="Use forecasted dataset (combined_active_power_forecasted.csv) and compare with first run.",
    )
    args = parser.parse_args()

    # 1. Load configurations from the 'configurations' folder
    configurations_folder_path = os.path.join(SCRIPT_DIR, "configurations")
    controller_config, settings_configs = load_configurations(
        configurations_folder_path, use_forecasted=args.use_forecasted
    )

    # Resolve relative data paths in settings configs to absolute paths
    for config in settings_configs.values():
        init = config["InitializationSettings"]
        for key in ("passive_consumers_power_setpoints", "grid_topology"):
            init[key] = os.path.join(SCRIPT_DIR, init[key])

    # 2. Create model instances
    electric_grid_model = Model(electric_grid_function)
    heat_pump_model = Model(heat_pump_function)
    CONTROLLER_PRIORITY = "voltage"
    controller_model = Model(partial(
        controller_function, controller_settings=controller_config, priority=CONTROLLER_PRIORITY,
    ))

    sim_config = settings_configs["config 1"]

    smart_consumers = {
        "Customer_94": {"room": Model(RoomFunction(sim_config))},
        "Customer_95": {"room": Model(RoomFunction(sim_config))},
    }

    manager = Manager(
        electric_grid=electric_grid_model,
        heat_pump=heat_pump_model,
        controller=controller_model,
        smart_consumers=smart_consumers,
        settings_configuration=sim_config,
    )

    if args.use_forecasted:
        if not os.path.isfile(FIRST_RUN_RESULTS_PATH):
            raise FileNotFoundError(
                f"First run results not found at {FIRST_RUN_RESULTS_PATH}. "
                "Run without --use-forecasted first: python run_co_simulation.py"
            )
        with open(FIRST_RUN_RESULTS_PATH, "rb") as f:
            previous_results = pickle.load(f)
        manager.run_simulation(previous_results=previous_results)
    else:
        times, plottable_state = manager.run_simulation(previous_results=None)
        with open(FIRST_RUN_RESULTS_PATH, "wb") as f:
            pickle.dump((times, plottable_state), f)
        print(f"First run results saved to {FIRST_RUN_RESULTS_PATH}")


if __name__ == "__main__":
    main()


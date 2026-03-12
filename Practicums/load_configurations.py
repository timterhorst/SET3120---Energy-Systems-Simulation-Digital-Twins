"""Functions for loading simulation configurations from YAML files."""
import os
import yaml


def load_configurations(
    configurations_folder_path: str,
    use_forecasted: bool = False,
) -> tuple[dict, list[dict[str, dict]]]:
    """Load configurations from YAML files in the specified folder path.

    Args:
        configurations_folder_path: Path to the folder containing YAML configs.
        use_forecasted: If True, use combined_active_power_forecasted.csv;
            otherwise use combined_active_power.csv for passive consumer setpoints.
    """
    power_setpoints_file = (
        "data/combined_active_power_forecasted.csv"
        if use_forecasted
        else "data/combined_active_power.csv"
    )

    config_files = [f for f in os.listdir(configurations_folder_path) if f.endswith('.yaml')]
    initialization_configurations = {}

    for config_file in config_files:
        if config_file == 'controller_config.yaml':
            with open(os.path.join(configurations_folder_path, config_file), 'r') as file:
                controller_configuration = yaml.safe_load(file)
            continue
        config_path = os.path.join(configurations_folder_path, config_file)
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
            try:
                config_id = config_data["InitializationSettings"]["config_id"]
            except KeyError as e:
                raise KeyError(f"Configuration file {config_file} is missing the 'config_id' key.") from e
            config_data["InitializationSettings"]["passive_consumers_power_setpoints"] = power_setpoints_file
            initialization_configurations[f"config {config_id}"] = config_data

    return controller_configuration, initialization_configurations
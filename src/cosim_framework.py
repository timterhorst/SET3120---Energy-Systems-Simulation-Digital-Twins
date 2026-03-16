"""Co-simulation framework module. Contains Model and Manager classes for running the co-simulation."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Model:
    """Wrapper class for modeling any physical process (e.g. power flow, heat production, etc.)."""
    
    def __init__(self, process_model):
        """Takes in the model of a physical process as a function or callable class."""
        if not callable(process_model):
            raise ValueError("The process must be a function or callable class.")
        self.process_model = process_model

    def calculate(self, *args, **kwargs) -> float:
        """Call the process function to perform calculations on an arbitrary number of inputs."""
        return self.process_model(*args, **kwargs)
    

class Manager:
    """The orchestrator manager for managing the data exchanged between the coupled models.
    
    NOTE: Currently, it is hardcoded to work with a particular sequence of execution of the coupled
    models as define in run_co_simulation.py.
    """
    
    def __init__(self, models: list[Model], settings_configuration: dict):
        self.models = models
        self.electric_grid = models[0]  # First model is the electric grid
        self.heat_pump = models[1]  # Second model is the heat pump
        self.room = models[2]  # Third model is the room
        self.controller = models[-1]  # Last model is the controller
        self.settings_configuration = settings_configuration

    def run_simulation(self):
        # Extract the relevant simulation parameters from the configuration
        config = self.settings_configuration
        config_id = config['InitializationSettings']['config_id']
        start_time = config['InitializationSettings']['time']['start_time']
        end_time = config['InitializationSettings']['time']['end_time']
        delta_t = config['InitializationSettings']['time']['delta_t']
        load_scale = config['InitializationSettings'].get('load_scale', 1.0)

        # Grid line data
        grid_topology = pd.read_csv(config['InitializationSettings']['grid_topology'])

        # Passive consumers (with optional load scaling for stressed grid scenarios)
        passive_consumer_power_setpoints = pd.read_csv(
            config['InitializationSettings']['passive_consumers_power_setpoints'], index_col="snapshots", parse_dates=True,
        )
        if load_scale != 1.0:
            passive_consumer_power_setpoints = passive_consumer_power_setpoints * load_scale

        # print(passive_consumer_power_setpoints.shape)
        # print(passive_consumer_power_setpoints.columns)

        # Smart consumer
        hp_power_setpoint = config['InitializationSettings']['initial_conditions']['heat_pump']['power_set_point']
        room_temperature = config['InitializationSettings']['initial_conditions']['room']['temperature']

        # Initialize lists to store data for plotting
        times = []
        smart_consumer_power_setpoint_over_time = []
        smart_consumer_voltage_over_time = []
        heat_pump_heat_output_over_time = []
        temperature_over_time = []
        p_at_grid_over_time = []

        print("=" * 65)
        print(f"Base Co-Simulation | config {config_id} | load_scale={load_scale}")
        print(f"t=[{start_time}, {end_time}] min | dt={delta_t} min")
        print(f"Initial HP power: {hp_power_setpoint} W | Initial T_room: {room_temperature} °C")
        print("=" * 65)

        time_steps = int((end_time - start_time) / delta_t)
        datetime_index = passive_consumer_power_setpoints.index[:time_steps]
    
        for time_step in range(time_steps):
            time_clock = start_time + time_step * delta_t
            corresponding_time_in_dataframe = datetime_index[time_step]

            # print(f"Time step {corresponding_time_in_dataframe} | Simulation time clock: {time_clock:.2f}")

            p_at_grid = hp_power_setpoint

            all_consumer_voltages = self.electric_grid.calculate(
                passive_consumer_power_setpoints, hp_power_setpoint, grid_topology, corresponding_time_in_dataframe,
            )
            smart_consumer_voltage = all_consumer_voltages["consumers"]["smart_consumer"]
            heat_production_from_hp = self.heat_pump.calculate(hp_power_setpoint)
            room_temperature = self.room.calculate(heat_production_from_hp)

            hp_power_setpoint = self.controller.calculate(hp_power_setpoint, smart_consumer_voltage, room_temperature)
            # print("-----------------------------------------------------------")
            # print(f"New power setpoint: {hp_power_setpoint}")
            # print("===========================================================")

            times.append(time_clock)
            p_at_grid_over_time.append(p_at_grid)
            smart_consumer_power_setpoint_over_time.append(hp_power_setpoint)
            smart_consumer_voltage_over_time.append(smart_consumer_voltage)
            heat_pump_heat_output_over_time.append(heat_production_from_hp)
            temperature_over_time.append(room_temperature)

            if time_step % 2000 == 0:
                print(f"[{time_step:>6}/{time_steps}] t={time_clock:>8.0f}min | "
                      f"V={smart_consumer_voltage:.4f} | T={room_temperature:.1f}°C | "
                      f"P_hp={hp_power_setpoint:.0f}W")

        print("=" * 65)
        print("Simulation complete. Generating plots & saving results...")

        self.plot_results(
            times,
            smart_consumer_voltage_over_time,
            temperature_over_time,
            smart_consumer_power_setpoint_over_time,
            heat_pump_heat_output_over_time,
            config_id,
        )

        self._save_results(
            config_id=config_id,
            datetime_index=datetime_index,
            times=times,
            voltages=smart_consumer_voltage_over_time,
            temperatures=temperature_over_time,
            hp_powers=smart_consumer_power_setpoint_over_time,
            heat_productions=heat_pump_heat_output_over_time,
            p_at_grid=p_at_grid_over_time,
        )

    def _save_results(self, config_id, datetime_index, **arrays):
        """Save simulation time-series to a compressed .npz file."""
        results_dir = os.path.join('results')
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, f"config{config_id}_base.npz")

        save_dict = {k: np.array(v) for k, v in arrays.items()}
        save_dict['datetime_index'] = np.array(datetime_index.astype(str))

        np.savez_compressed(path, **save_dict)
        print(f"  Results saved   → {path}")

    def plot_results(self, times, voltages, temperatures, power_setpoints, heat_productions, config_id):
        plt.style.use('ggplot')
        _, axs = plt.subplots(2, 2, figsize=(12, 8))

        plots = [
            (axs[0, 0], times, voltages, "Voltage Over Time", "Time [min]", "Voltage [V]", 'blue'),
            (axs[0, 1], times, temperatures, "Temperature Over Time", "Time [min]", "Temperature [°C]", 'red'),
            (axs[1, 0], times, power_setpoints, "Heat Pump Power Setpoint Over Time", "Time [min]", "Power Setpoint [W]", 'green'),
            (axs[1, 1], times, heat_productions, "Heat Production Over Time", "Time [min]", "Heat Production [W]", 'orange'),
        ]

        for ax, x, y, title, xlabel, ylabel, color in plots:
            ax.plot(x, y, color=color)
            ax.set_title(title, color='black')
            ax.set_xlabel(xlabel, color='black')
            ax.set_ylabel(ylabel, color='black')
            ax.tick_params(axis='x', colors='black')
            ax.tick_params(axis='y', colors='black')

        plt.tight_layout()
        plt.savefig(f"results_config{config_id}.png")

"""Co-simulation framework module. Contains Model and Manager classes for running the co-simulation."""
import os
import matplotlib.pyplot as plt
import pandas as pd


class Model:
    """Wrapper class for modeling any physical process (e.g. power flow, heat production, etc.)."""
    
    def __init__(self, process_model):
        """Takes in the model of a physical process as a function or callable class."""
        if not callable(process_model):
            raise ValueError("The process must be a function or callable class.")
        self.process_model = process_model

    def calculate(self, *args) -> float:
        """Call the process function to perform calculations on an arbitrary number of inputs."""
        return self.process_model(*args)
    

class Manager:
    """The orchestrator manager for managing the data exchanged between the coupled models.

    Supports one or more smart consumers, each with an independent room model
    but sharing the same heat-pump function and controller logic.
    """

    def __init__(
        self,
        electric_grid: Model,
        heat_pump: Model,
        controller: Model,
        smart_consumers: dict[str, dict],
        settings_configuration: dict,
    ):
        self.electric_grid = electric_grid
        self.heat_pump = heat_pump
        self.controller = controller
        self.smart_consumers = smart_consumers
        self.settings_configuration = settings_configuration

    def _plottable_state(self, state):
        """Extract only the time-series data from state (for saving/plotting)."""
        return {
            name: {
                k: list(v) for k, v in s.items()
                if k in ("voltages", "temperatures", "power_setpoints", "heat_productions")
            }
            for name, s in state.items()
        }

    def run_simulation(self, previous_results=None):
        config = self.settings_configuration
        config_id = config['InitializationSettings']['config_id']
        start_time = config['InitializationSettings']['time']['start_time']
        end_time = config['InitializationSettings']['time']['end_time']
        delta_t = config['InitializationSettings']['time']['delta_t']

        grid_topology = pd.read_csv(config['InitializationSettings']['grid_topology'])
        passive_consumer_power_setpoints = pd.read_csv(
            config['InitializationSettings']['passive_consumers_power_setpoints'],
            index_col="snapshots", parse_dates=True,
        )

        hp_initial = config['InitializationSettings']['initial_conditions']['heat_pump']['power_set_point']

        # Per-consumer state: each gets independent HP setpoint, room, and history
        state = {}
        for name, info in self.smart_consumers.items():
            state[name] = {
                "room": info["room"],
                "hp_power_setpoint": hp_initial,
                "voltages": [],
                "temperatures": [],
                "power_setpoints": [],
                "heat_productions": [],
            }

        times = []
        time_steps = int((end_time - start_time) / delta_t)

        print("===============================================================")
        print(f"Starting simulation | t={start_time}..{end_time}, delta_t={delta_t}")
        print(f"Smart consumers: {list(state.keys())}")
        print("===============================================================")

        for step in range(time_steps):
            time_clock = start_time + step * delta_t
            corresponding_time = passive_consumer_power_setpoints.index[step]

            # Collect current setpoints for all smart consumers
            setpoints = {n: s["hp_power_setpoint"] for n, s in state.items()}

            # Single power-flow run with all smart consumer injections
            all_voltages = self.electric_grid.calculate(
                passive_consumer_power_setpoints, setpoints, grid_topology, corresponding_time,
            )

            # Update each smart consumer independently
            for name, s in state.items():
                voltage = all_voltages["consumers"][name]
                heat_prod = self.heat_pump.calculate(s["hp_power_setpoint"])
                temperature = s["room"].calculate(heat_prod)
                s["hp_power_setpoint"] = self.controller.calculate(
                    s["hp_power_setpoint"], voltage, temperature,
                )
                s["voltages"].append(voltage)
                s["temperatures"].append(temperature)
                s["power_setpoints"].append(s["hp_power_setpoint"])
                s["heat_productions"].append(heat_prod)

            times.append(time_clock)

        plottable_state = self._plottable_state(state)
        self.plot_results(times, plottable_state, config_id, previous_results=previous_results)
        return times, plottable_state

    def plot_results(self, times, consumer_state, config_id, previous_results=None):
        """
        Plot simulation results. If previous_results is given, plot both original and
        forecasted runs on the same axes. previous_results is (times_prev, state_prev).
        """
        plt.style.use('ggplot')
        _, axs = plt.subplots(2, 2, figsize=(12, 8))

        metric_info = [
            (axs[0, 0], "voltages",         "Voltage Over Time",                "Voltage [p.u.]"),
            (axs[0, 1], "temperatures",      "Temperature Over Time",            "Temperature [°C]"),
            (axs[1, 0], "power_setpoints",   "Heat Pump Power Setpoint Over Time", "Power Setpoint [W]"),
            (axs[1, 1], "heat_productions",  "Heat Production Over Time",        "Heat Production [W]"),
        ]

        palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
        linestyles_single = ["-", "--", "-.", ":"]
        # For comparison: solid = original, dashed = forecasted
        linestyle_original, linestyle_forecasted = "-", "--"

        for ax, key, title, ylabel in metric_info:
            if previous_results is None:
                for i, (name, s) in enumerate(consumer_state.items()):
                    ax.plot(
                        times, s[key],
                        color=palette[i % len(palette)],
                        linestyle=linestyles_single[i % len(linestyles_single)],
                        linewidth=1.5,
                        label=name, alpha=0.85,
                    )
            else:
                times_prev, state_prev = previous_results
                for i, (name, s) in enumerate(state_prev.items()):
                    ax.plot(
                        times_prev, s[key],
                        color=palette[i % len(palette)],
                        linestyle=linestyle_original,
                        linewidth=1.5,
                        label=f"{name} (Original)", alpha=0.85,
                    )
                for i, (name, s) in enumerate(consumer_state.items()):
                    ax.plot(
                        times, s[key],
                        color=palette[i % len(palette)],
                        linestyle=linestyle_forecasted,
                        linewidth=1.5,
                        label=f"{name} (Forecasted)", alpha=0.85,
                    )
            ax.set_title(title, color='black')
            ax.set_xlabel("Time [min]", color='black')
            ax.set_ylabel(ylabel, color='black')
            ax.tick_params(axis='x', colors='black')
            ax.tick_params(axis='y', colors='black')
            ax.legend(fontsize='small')

        plt.tight_layout()
        save_dir = os.path.dirname(os.path.abspath(__file__))
        plt.savefig(os.path.join(save_dir, f"results_config{config_id}.png"))

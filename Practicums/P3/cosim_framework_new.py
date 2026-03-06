"""Co-simulation framework module. Contains Model and Manager classes for running the co-simulation."""
import matplotlib.pyplot as plt


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

        # Initialize previous timestep variables for Room-first synchronization
        self.heat_production_prev = 0.0  # Will be set properly before loop starts
        self.voltage_prev = 10.0  # Initialize with a default value (e.g., middle of range)

    def run_simulation(self):
        # Extract the relevant simulation parameters from the configuration
        config = self.settings_configuration
        config_id = config['InitializationSettings']['config_id']
        start_time = config['InitializationSettings']['time']['start_time']
        end_time = config['InitializationSettings']['time']['end_time']
        delta_t = config['InitializationSettings']['time']['delta_t']
        hp_power_setpoint = config['InitializationSettings']['initial_conditions']['heat_pump']['power_set_point']
        room_temperature = config['InitializationSettings']['initial_conditions']['room']['temperature']

        # Initialize lists to store data for plotting
        times = []
        voltages = []
        temperatures = []
        power_setpoints = []
        heat_productions = []

        print("===============================================================")
        print(f"Starting simulation at time {start_time}, ending at {end_time}, with time step delta_t: {delta_t}\n")
        print(f"Initial heat pump power_setpoint: {hp_power_setpoint}\n Initial heat pump temperature: {room_temperature}\n")
        print("===============================================================")

        time_steps = int((end_time - start_time) / delta_t)

        # Calculate initial heat production and voltage for the first timestep.
        # This ensures Room has a valid heat_production value when it executes first.
        self.heat_production_prev = self.heat_pump.calculate(hp_power_setpoint)
        self.voltage_prev = self.electric_grid.calculate(hp_power_setpoint)
        print(f"Initial heat production: {self.heat_production_prev}")
        print(f"Initial voltage: {self.voltage_prev}")

        for step in range(time_steps):
            current_time = start_time + step * delta_t
            print(f"Time step {step} | Current time: {current_time:.2f}")

            times.append(current_time)

            # Room-first (Jacobi-style) synchronization sequence

            # 1. ROOM updates first (uses heat production from PREVIOUS timestep)
            room_temperature = self.room.calculate(self.heat_production_prev)

            # 2. CONTROLLER decides power setpoint based on NEW room temperature, OLD voltage
            hp_power_setpoint = self.controller.calculate(
                hp_power_setpoint,
                self.voltage_prev,
                room_temperature,
            )

            # 3. GRID and HEAT PUMP execute (both use the NEW power setpoint from controller)
            voltage = self.electric_grid.calculate(hp_power_setpoint)
            heat_production_from_hp = self.heat_pump.calculate(hp_power_setpoint)

            # 4. Store current values for next timestep
            self.heat_production_prev = heat_production_from_hp
            self.voltage_prev = voltage

            print("-----------------------------------------------------------")
            print(f"New power setpoint: {hp_power_setpoint}")
            print("===========================================================")

            voltages.append(voltage)
            temperatures.append(room_temperature)
            power_setpoints.append(hp_power_setpoint)
            heat_productions.append(heat_production_from_hp)

        self.plot_results(times, voltages, temperatures, power_setpoints, heat_productions, config_id)

    def plot_results(self, times, voltages, temperatures, power_setpoints, heat_productions, config_id):
        plt.style.use('ggplot')
        _, axs = plt.subplots(2, 2, figsize=(12, 8))

        plots = [
            (axs[0, 0], times, voltages, "Voltage Over Time", "Time [min]", "Voltage [V]", 'blue'),
            (axs[0, 1], times, temperatures, "Temperature Over Time", "Time [min]", "Temperature [°C]", 'red'),
            (axs[1, 0], times, power_setpoints, "Heat Pump Power Setpoint Over Time", "Time [min]", "Power Setpoint [kW]", 'green'),
            (axs[1, 1], times, heat_productions, "Heat Production Over Time", "Time [min]", "Heat Production [kW]", 'orange'),
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

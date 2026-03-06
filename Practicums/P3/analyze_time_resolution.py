"""
Analyze the impact of timestep size on overshoot error.

This script runs the co-simulation with different timestep sizes and calculates
the error introduced by the first overshoot event.
"""

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import yaml
import os

from controller import controller_function
from cosim_framework import Manager, Model
from grid import electric_grid_function
from heat_pump import heat_pump_function
from room import RoomFunction


def calculate_overshoot_error(times, temperatures, upper_bound: float = 25.0):
    """
    Calculate the area of the first overshoot event.
    """
    if not times or not temperatures or len(times) != len(temperatures):
        return 0.0, None, None, 0.0

    overshoot_start = None
    overshoot_end = None

    for i, temp in enumerate(temperatures):
        if temp > upper_bound:
            if overshoot_start is None:
                overshoot_start = i
        elif overshoot_start is not None and overshoot_end is None:
            overshoot_end = i
            break

    if overshoot_start is None:
        return 0.0, None, None, 0.0

    if overshoot_end is None:
        overshoot_end = len(temperatures) - 1

    error = 0.0
    max_temp = temperatures[overshoot_start]

    for i in range(overshoot_start, overshoot_end):
        dt = times[i + 1] - times[i]
        excess_1 = max(0.0, temperatures[i] - upper_bound)
        excess_2 = max(0.0, temperatures[i + 1] - upper_bound)
        avg_excess = (excess_1 + excess_2) / 2.0
        error += avg_excess * dt
        max_temp = max(max_temp, temperatures[i], temperatures[i + 1])

    overshoot_start_time = times[overshoot_start]
    overshoot_end_time = times[overshoot_end]

    return error, overshoot_start_time, overshoot_end_time, max_temp


def run_simulation_with_timestep(delta_t: float, config_base_path: str = "./configurations/config6.yaml"):
    """
    Run simulation with a specific timestep and return time and temperature series.
    """
    # Load base configuration
    with open(config_base_path, "r") as f:
        config = yaml.safe_load(f)

    # Modify timestep
    config["InitializationSettings"]["time"]["delta_t"] = delta_t

    # Load controller config
    base_dir = os.path.dirname(os.path.abspath(__file__))
    controller_path = os.path.join(base_dir, "configurations", "controller_config.yaml")
    with open(controller_path, "r") as f:
        controller_config = yaml.safe_load(f)

    # Create models
    electric_grid_model = Model(electric_grid_function)
    heat_pump_model = Model(heat_pump_function)
    room_model = Model(RoomFunction(config))
    controller_model = Model(partial(controller_function, controller_settings=controller_config))

    models = [electric_grid_model, heat_pump_model, room_model, controller_model]

    # Run simulation, but suppress stdout spam
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        manager = Manager(models, config)
        manager.run_simulation()
        results = manager.get_results()
    finally:
        sys.stdout = old_stdout

    return results["times"], results["temperatures"]


def timestep_sensitivity_study():
    """
    Run simulations with different timestep sizes and analyze overshoot error.
    """
    timesteps = [0.5, 1, 2, 5, 10, 15, 30, 60]

    results = {
        "delta_t": [],
        "error": [],
        "overshoot_start": [],
        "overshoot_end": [],
        "max_overshoot": [],
    }

    print("=" * 60)
    print("TIME RESOLUTION SENSITIVITY STUDY")
    print("=" * 60)

    for dt in timesteps:
        print(f"\nRunning simulation with delta_t = {dt} min...")
        times, temperatures = run_simulation_with_timestep(dt)

        error, t_start, t_end, max_temp = calculate_overshoot_error(times, temperatures, upper_bound=25.0)

        results["delta_t"].append(dt)
        results["error"].append(error)
        results["overshoot_start"].append(t_start)
        results["overshoot_end"].append(t_end)
        results["max_overshoot"].append(max_temp)

        if error > 0.0:
            print("  ✓ Overshoot detected!")
            print(f"    Start time: {t_start:.1f} min")
            print(f"    End time:   {t_end:.1f} min")
            print(f"    Max temp:   {max_temp:.2f} °C")
            print(f"    Error:      {error:.2f} °C·min")
        else:
            print("  ✗ No overshoot occurred")

    plot_timestep_analysis(results)
    return results


def plot_timestep_analysis(results: dict):
    """
    Create visualization of timestep sensitivity analysis.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Error vs timestep
    ax1 = axes[0]
    ax1.plot(results["delta_t"], results["error"], "o-", linewidth=2, markersize=8)
    ax1.set_xlabel("Timestep Δt [minutes]", fontsize=12)
    ax1.set_ylabel("Overshoot Error [°C·min]", fontsize=12)
    ax1.set_title("Overshoot Error vs. Timestep Size", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")

    # Max overshoot vs timestep
    ax2 = axes[1]
    ax2.plot(results["delta_t"], results["max_overshoot"], "s-", linewidth=2, markersize=8, color="red")
    ax2.axhline(y=25, color="orange", linestyle="--", label="Upper boundary (25°C)")
    ax2.set_xlabel("Timestep Δt [minutes]", fontsize=12)
    ax2.set_ylabel("Maximum Temperature [°C]", fontsize=12)
    ax2.set_title("Maximum Overshoot vs. Timestep Size", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("timestep_sensitivity_analysis.png", dpi=150)
    print("\nPlot saved as 'timestep_sensitivity_analysis.png'")


if __name__ == "__main__":
    results = timestep_sensitivity_study()

    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Δt [min]':<12} {'Error [°C·min]':<18} {'Max Temp [°C]':<15}")
    print("-" * 60)
    for dt, err, max_t in zip(results["delta_t"], results["error"], results["max_overshoot"]):
        print(f"{dt:<12} {err:<18.2f} {max_t:<15.2f}")


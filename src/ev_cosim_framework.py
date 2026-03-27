"""Extended co-simulation framework for EV + V2G scenarios.

Defines EVManager, a subclass of Manager that extends the base 4-step synchronization
sequence to a 7-step sequence incorporating a Driver Model, EV Battery, and Charging
Controller alongside the existing grid, heat pump, room, and thermostat controller.

Synchronization sequence per timestep k:
    1. Driver Model        → is_home[k], temp_min[k], soc_depletion[k]
    2. Electric Grid       → V_grid[k]       (uses P_hp[k-1] + P_charge[k-1])
    3. Heat Pump           → Q_hp[k]         (uses P_hp[k-1])
    4. Room                → T_room[k]       (uses Q_hp[k])
    5. EV Battery          → SOC[k]          (uses P_charge[k-1], soc_depletion[k])
    6. Thermostat Controller → P_hp[k]       (uses V_grid[k], T_room[k], temp_min[k])
    7. Charging Controller → P_charge[k]     (uses V_grid[k], SOC[k], is_home[k])

Grid executes first so that both controllers see the current voltage (consistent with
the base implementation where the controller reacts to the current-step grid state).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Any

from .cosim_framework import Manager, Model
from .knmi_loader import load_knmi_temperature

MINUTES_PER_DAY = 1440
STEPS_PER_DAY = MINUTES_PER_DAY // 15  # 96 steps at dt=15 min


class EVManager(Manager):
    """Orchestrator for the extended EV + V2G co-simulation."""

    def __init__(self, models: list[Model], settings_configuration: dict):
        """
        Args:
            models: list of 7 Model-wrapped components in order:
                [electric_grid, heat_pump, room, thermostat_controller,
                 driver, ev_battery, charging_controller]
            settings_configuration: the base simulation config dict.
        """
        super().__init__(models[:4], settings_configuration)
        self.driver = models[4]
        self.ev_battery = models[5]
        self.charging_controller = models[6]

    def run_simulation(self):
        config = self.settings_configuration
        config_id = config['InitializationSettings']['config_id']
        start_time = config['InitializationSettings']['time']['start_time']
        end_time = config['InitializationSettings']['time']['end_time']
        delta_t = config['InitializationSettings']['time']['delta_t']
        load_scale = config['InitializationSettings'].get('load_scale', 1.0)

        grid_topology = pd.read_csv(config['InitializationSettings']['grid_topology'])
        passive_consumer_power_setpoints = pd.read_csv(
            config['InitializationSettings']['passive_consumers_power_setpoints'],
            index_col="snapshots", parse_dates=True,
        )
        if load_scale != 1.0:
            passive_consumer_power_setpoints = passive_consumer_power_setpoints * load_scale

        # Initial conditions
        hp_power_setpoint = config['InitializationSettings']['initial_conditions']['heat_pump']['power_set_point']
        p_charge = 0.0

        # Data logging lists
        times = []
        voltages = []
        temperatures = []
        hp_powers = []
        heat_productions = []
        ev_powers = []
        total_powers = []
        socs = []
        is_homes = []
        temp_min_dynamics = []

        time_steps = int((end_time - start_time) / delta_t)
        datetime_index = passive_consumer_power_setpoints.index[:time_steps]

        knmi_temp = load_knmi_temperature("data/KNMI_temp_data.txt", n_steps=time_steps)

        print("=" * 65)
        print(f"EV + V2G Co-Simulation | config {config_id} | load_scale={load_scale}")
        print(f"t=[{start_time}, {end_time}] min | dt={delta_t} min")
        print(f"Initial HP power: {hp_power_setpoint} W | "
              f"Initial SOC: {self.ev_battery.process_model.soc:.0%}")
        print("=" * 65)

        for step in range(time_steps):
            time_clock = start_time + step * delta_t
            ts = datetime_index[step]

            # --- Step 1: Driver Model ---
            is_home, temp_min_dynamic, soc_depletion = self.driver.calculate(time_clock)

            # --- Step 2: Electric Grid ---
            p_total = hp_power_setpoint + p_charge
            all_voltages = self.electric_grid.calculate(
                passive_consumer_power_setpoints, p_total, grid_topology, ts,
            )
            smart_consumer_voltage = all_voltages["consumers"]["smart_consumer"]

            # --- Step 3: Heat Pump ---
            heat_production = self.heat_pump.calculate(hp_power_setpoint)

            # --- Step 4: Room ---
            room_temperature = self.room.calculate(heat_production, knmi_temp[step])

            # --- Step 5: EV Battery ---
            soc = self.ev_battery.calculate(p_charge, soc_depletion)

            # --- Step 6: Thermostat Controller ---
            hp_power_setpoint = self.controller.calculate(
                hp_power_setpoint, smart_consumer_voltage, room_temperature,
                temp_min_override=temp_min_dynamic,
            )

            # --- Step 7: Charging Controller ---
            p_charge = self.charging_controller.calculate(
                smart_consumer_voltage, soc, is_home,
            )

            # Log all signals
            times.append(time_clock)
            voltages.append(smart_consumer_voltage)
            temperatures.append(room_temperature)
            hp_powers.append(hp_power_setpoint)
            heat_productions.append(heat_production)
            ev_powers.append(p_charge)
            total_powers.append(p_total)
            socs.append(soc)
            is_homes.append(is_home)
            temp_min_dynamics.append(temp_min_dynamic)

            if step % 2000 == 0:
                print(f"[{step:>6}/{time_steps}] t={time_clock:>8.0f}min | "
                      f"V={smart_consumer_voltage:.4f} | T={room_temperature:.1f}°C | "
                      f"SOC={soc:.1%} | P_hp={hp_power_setpoint:.0f}W | "
                      f"P_ev={p_charge:.0f}W | home={is_home}")

        print("=" * 65)
        print("Simulation complete. Generating plots & saving results...")

        data = dict(
            times=times, voltages=voltages, temperatures=temperatures,
            hp_powers=hp_powers, heat_productions=heat_productions,
            ev_powers=ev_powers, total_powers=total_powers,
            socs=socs, is_homes=is_homes, temp_min_dynamics=temp_min_dynamics,
        )

        self._plot_base_overview(data, config_id)
        self._plot_ev_overview(data, config_id)
        self._plot_weekly_zoom(data, config_id)
        self._save_results(data, config_id, datetime_index)

    def _save_results(self, data, config_id, datetime_index):
        """Save EV simulation time-series to a compressed .npz file."""
        import os
        results_dir = os.path.join('results')
        os.makedirs(results_dir, exist_ok=True)
        if self._is_forecasted_config1(config_id):
            path = os.path.join(results_dir, "config_ev_forecasted.npz")
        else:
            path = os.path.join(results_dir, f"config{config_id}_ev.npz")

        save_dict = {k: np.array(v) for k, v in data.items()}
        save_dict['datetime_index'] = np.array(datetime_index.astype(str))

        np.savez_compressed(path, **save_dict)
        print(f"  Results saved   → {path}")

    # ------------------------------------------------------------------
    # Figure 1: Base-compatible 2×2 (matches base Manager output format)
    # ------------------------------------------------------------------
    def _plot_base_overview(
        self,
        data: dict[str, Any],
        config_id: int | str,
        comparison: dict[str, Any] | None = None,
        base_label: str = "Original",
        comparison_label: str = "Forecasted",
    ):
        plt.style.use('ggplot')
        _, axs = plt.subplots(2, 2, figsize=(12, 8))

        plots = [
            (axs[0, 0], 'voltages', "Voltage Over Time", "Time [min]", "Voltage [p.u.]", 'blue'),
            (axs[0, 1], 'temperatures', "Temperature Over Time", "Time [min]", "Temperature [°C]", 'red'),
            (axs[1, 0], 'hp_powers', "Heat Pump Power Setpoint Over Time", "Time [min]", "Power Setpoint [W]", 'green'),
            (axs[1, 1], 'heat_productions', "Heat Production Over Time", "Time [min]", "Heat Production [W]", 'orange'),
        ]

        for ax, yk, title, xlabel, ylabel, color in plots:
            ax.plot(data['times'], data[yk], color=color, linewidth=0.6, label=base_label)
            if comparison is not None:
                ax.plot(
                    comparison['times'],
                    comparison[yk],
                    color=color,
                    linewidth=0.6,
                    linestyle="--",
                    alpha=0.85,
                    label=comparison_label,
                )
            ax.set_title(title, color='black')
            ax.set_xlabel(xlabel, color='black')
            ax.set_ylabel(ylabel, color='black')
            ax.tick_params(axis='x', colors='black')
            ax.tick_params(axis='y', colors='black')
            if comparison is not None:
                ax.legend(fontsize=7)

        plt.tight_layout()
        path = f"ev_results_base_overview_config{config_id}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Base overview  → {path}")

    # ------------------------------------------------------------------
    # Figure 2: Full EV overview — all signals, full simulation
    # ------------------------------------------------------------------
    def _plot_ev_overview(
        self,
        data: dict[str, Any],
        config_id: int | str,
        comparison: dict[str, Any] | None = None,
        base_label: str = "Original",
        comparison_label: str = "Forecasted",
    ):
        plt.style.use('ggplot')
        fig, axs = plt.subplots(3, 2, figsize=(14, 10))
        t = data['times']

        # (0,0) Voltage + limit lines
        axs[0, 0].plot(t, data['voltages'], color='blue', linewidth=0.3, label=base_label)
        if comparison is not None:
            axs[0, 0].plot(
                comparison['times'],
                comparison['voltages'],
                color='blue',
                linewidth=0.3,
                linestyle="--",
                alpha=0.85,
                label=comparison_label,
            )
        axs[0, 0].axhline(0.98, color='red', linestyle='--', linewidth=0.8, label='V_min')
        axs[0, 0].axhline(1.02, color='red', linestyle='--', linewidth=0.8, label='V_max')
        axs[0, 0].set_title("Grid Voltage at Smart Consumer", color='black')
        axs[0, 0].set_ylabel("Voltage [p.u.]", color='black')
        axs[0, 0].legend(fontsize=7)

        # (0,1) Temperature + dynamic temp_min
        axs[0, 1].plot(t, data['temperatures'], color='red', linewidth=0.3, label=f"T_room ({base_label})")
        if comparison is not None:
            axs[0, 1].plot(
                comparison['times'],
                comparison['temperatures'],
                color='red',
                linewidth=0.3,
                linestyle="--",
                alpha=0.85,
                label=f"T_room ({comparison_label})",
            )
        axs[0, 1].plot(t, data['temp_min_dynamics'], color='blue', linewidth=0.3,
                        alpha=0.6, label='temp_min (setback)')
        axs[0, 1].set_title("Room Temperature & Occupancy Setback", color='black')
        axs[0, 1].set_ylabel("Temperature [°C]", color='black')
        axs[0, 1].legend(fontsize=7)

        # (1,0) HP power
        axs[1, 0].plot(t, data['hp_powers'], color='green', linewidth=0.3, label=base_label)
        if comparison is not None:
            axs[1, 0].plot(
                comparison['times'],
                comparison['hp_powers'],
                color='green',
                linewidth=0.3,
                linestyle="--",
                alpha=0.85,
                label=comparison_label,
            )
        axs[1, 0].set_title("Heat Pump Power Setpoint", color='black')
        axs[1, 0].set_ylabel("Power [W]", color='black')
        if comparison is not None:
            axs[1, 0].legend(fontsize=7)

        # (1,1) EV charging power (positive = charging, negative = V2G)
        axs[1, 1].plot(t, data['ev_powers'], color='purple', linewidth=0.3, label=base_label)
        if comparison is not None:
            axs[1, 1].plot(
                comparison['times'],
                comparison['ev_powers'],
                color='purple',
                linewidth=0.3,
                linestyle="--",
                alpha=0.85,
                label=comparison_label,
            )
        axs[1, 1].axhline(0, color='black', linewidth=0.5)
        axs[1, 1].set_title("EV Charging Power (negative = V2G)", color='black')
        axs[1, 1].set_ylabel("Power [W]", color='black')
        if comparison is not None:
            axs[1, 1].legend(fontsize=7)

        # (2,0) SOC + V2G reserve line
        axs[2, 0].plot(t, data['socs'], color='orange', linewidth=0.3, label=base_label)
        if comparison is not None:
            axs[2, 0].plot(
                comparison['times'],
                comparison['socs'],
                color='orange',
                linewidth=0.3,
                linestyle="--",
                alpha=0.85,
                label=comparison_label,
            )
        axs[2, 0].axhline(0.40, color='red', linestyle='--', linewidth=0.8,
                           label='V2G reserve (40%)')
        axs[2, 0].set_title("EV State of Charge", color='black')
        axs[2, 0].set_ylabel("SOC [-]", color='black')
        axs[2, 0].set_ylim(-0.05, 1.05)
        axs[2, 0].legend(fontsize=7)

        # (2,1) Total power at connection point
        axs[2, 1].plot(t, data['total_powers'], color='teal', linewidth=0.3, label=base_label)
        if comparison is not None:
            axs[2, 1].plot(
                comparison['times'],
                comparison['total_powers'],
                color='teal',
                linewidth=0.3,
                linestyle="--",
                alpha=0.85,
                label=comparison_label,
            )
        axs[2, 1].axhline(0, color='black', linewidth=0.5)
        axs[2, 1].set_title("Total Power at Node (P_hp + P_charge)", color='black')
        axs[2, 1].set_ylabel("Power [W]", color='black')
        if comparison is not None:
            axs[2, 1].legend(fontsize=7)

        for row in axs:
            for ax in row:
                ax.set_xlabel("Time [min]", color='black')
                ax.tick_params(axis='x', colors='black')
                ax.tick_params(axis='y', colors='black')

        plt.tight_layout()
        path = f"ev_results_full_overview_config{config_id}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  EV overview    → {path}")

    # ------------------------------------------------------------------
    # Figure 3: One-week zoom with home/away shading (x-axis in hours)
    # ------------------------------------------------------------------
    def _plot_weekly_zoom(
        self,
        data: dict[str, Any],
        config_id: int | str,
        zoom_start_day: int = 7,
        zoom_days: int = 7,
        comparison: dict[str, Any] | None = None,
        base_label: str = "Original",
        comparison_label: str = "Forecasted",
    ):
        t = np.array(data['times'])

        zoom_start = zoom_start_day * MINUTES_PER_DAY
        zoom_end = (zoom_start_day + zoom_days) * MINUTES_PER_DAY
        mask = (t >= zoom_start) & (t < zoom_end)

        if mask.sum() == 0:
            print("  Zoom skipped (simulation shorter than zoom window)")
            return

        # Slice all signals; x-axis in hours relative to the start of the zoom window
        zt = t[mask]
        hours = (zt - zoom_start) / 60.0
        zv = np.array(data['voltages'])[mask]
        ztemp = np.array(data['temperatures'])[mask]
        ztmin = np.array(data['temp_min_dynamics'])[mask]
        zhp = np.array(data['hp_powers'])[mask]
        zev = np.array(data['ev_powers'])[mask]
        zsoc = np.array(data['socs'])[mask]
        zhome = np.array(data['is_homes'])[mask]
        ztot = np.array(data['total_powers'])[mask]

        # Optional comparison slice (same mask logic, assumes comparable time base)
        if comparison is not None:
            ct = np.array(comparison['times'])
            cmask = (ct >= zoom_start) & (ct < zoom_end)
            if cmask.sum() == 0:
                comparison = None
            else:
                cht = ct[cmask]
                chours = (cht - zoom_start) / 60.0
                czv = np.array(comparison['voltages'])[cmask]
                cztemp = np.array(comparison['temperatures'])[cmask]
                czhp = np.array(comparison['hp_powers'])[cmask]
                czev = np.array(comparison['ev_powers'])[cmask]
                czsoc = np.array(comparison['socs'])[cmask]
                cztot = np.array(comparison['total_powers'])[cmask]

        plt.style.use('ggplot')
        fig, axs = plt.subplots(3, 2, figsize=(16, 11))

        # Helper: shade "away" periods in light grey
        def shade_away(ax):
            away_start = None
            for i, home in enumerate(zhome):
                if not home and away_start is None:
                    away_start = hours[i]
                elif home and away_start is not None:
                    ax.axvspan(away_start, hours[i], alpha=0.12, color='grey')
                    away_start = None
            if away_start is not None:
                ax.axvspan(away_start, hours[-1], alpha=0.12, color='grey')

        # Helper: add vertical day separators and day labels
        def add_day_markers(ax):
            for d in range(1, zoom_days):
                ax.axvline(d * 24, color='black', ls=':', lw=0.4, alpha=0.4)

        # (0,0) Voltage + limits
        axs[0, 0].plot(hours, zv, color='blue', linewidth=0.8, label=base_label)
        if comparison is not None:
            axs[0, 0].plot(chours, czv, color='blue', linewidth=0.8, linestyle="--", alpha=0.85, label=comparison_label)
        axs[0, 0].axhline(0.98, color='red', ls='--', lw=0.8, label='V limits')
        axs[0, 0].axhline(1.02, color='red', ls='--', lw=0.8)
        shade_away(axs[0, 0])
        add_day_markers(axs[0, 0])
        axs[0, 0].set_title("Grid Voltage (grey = away)", color='black')
        axs[0, 0].set_ylabel("Voltage [p.u.]", color='black')
        axs[0, 0].legend(fontsize=7)

        # (0,1) Temperature + dynamic setback
        axs[0, 1].plot(hours, ztemp, color='red', linewidth=0.8, label=f"T_room ({base_label})")
        if comparison is not None:
            axs[0, 1].plot(chours, cztemp, color='red', linewidth=0.8, linestyle="--", alpha=0.85, label=f"T_room ({comparison_label})")
        axs[0, 1].step(hours, ztmin, color='blue', linewidth=1.0, alpha=0.7,
                        where='post', label='temp_min setback')
        shade_away(axs[0, 1])
        add_day_markers(axs[0, 1])
        axs[0, 1].set_title("Temperature & Setback (grey = away)", color='black')
        axs[0, 1].set_ylabel("Temperature [°C]", color='black')
        axs[0, 1].legend(fontsize=7)

        # (1,0) HP power
        axs[1, 0].plot(hours, zhp, color='green', linewidth=0.8, label=base_label)
        if comparison is not None:
            axs[1, 0].plot(chours, czhp, color='green', linewidth=0.8, linestyle="--", alpha=0.85, label=comparison_label)
        shade_away(axs[1, 0])
        add_day_markers(axs[1, 0])
        axs[1, 0].set_title("Heat Pump Power", color='black')
        axs[1, 0].set_ylabel("Power [W]", color='black')
        if comparison is not None:
            axs[1, 0].legend(fontsize=7)

        # (1,1) EV charging power
        axs[1, 1].plot(hours, zev, color='purple', linewidth=0.8, label=base_label)
        if comparison is not None:
            axs[1, 1].plot(chours, czev, color='purple', linewidth=0.8, linestyle="--", alpha=0.85, label=comparison_label)
        axs[1, 1].axhline(0, color='black', lw=0.5)
        shade_away(axs[1, 1])
        add_day_markers(axs[1, 1])
        axs[1, 1].set_title("EV Charging Power (grey = away)", color='black')
        axs[1, 1].set_ylabel("Power [W]", color='black')
        if comparison is not None:
            axs[1, 1].legend(fontsize=7)

        # (2,0) SOC + V2G reserve + home shading
        axs[2, 0].plot(hours, zsoc, color='orange', linewidth=0.8, label=base_label)
        if comparison is not None:
            axs[2, 0].plot(chours, czsoc, color='orange', linewidth=0.8, linestyle="--", alpha=0.85, label=comparison_label)
        axs[2, 0].axhline(0.40, color='red', ls='--', lw=0.8, label='V2G reserve')
        shade_away(axs[2, 0])
        add_day_markers(axs[2, 0])
        axs[2, 0].set_title("State of Charge (grey = away)", color='black')
        axs[2, 0].set_ylabel("SOC [-]", color='black')
        axs[2, 0].set_ylim(-0.05, 1.05)
        axs[2, 0].legend(fontsize=7)

        # (2,1) Total power at node with HP/EV breakdown
        axs[2, 1].plot(hours, ztot, color='teal', linewidth=0.8, label=f"P_total ({base_label})")
        axs[2, 1].plot(hours, zhp, color='green', linewidth=0.5, alpha=0.5, label=f"P_hp ({base_label})")
        axs[2, 1].plot(hours, zev, color='purple', linewidth=0.5, alpha=0.5, label=f"P_ev ({base_label})")
        if comparison is not None:
            axs[2, 1].plot(chours, cztot, color='teal', linewidth=0.8, linestyle="--", alpha=0.85, label=f"P_total ({comparison_label})")
            axs[2, 1].plot(chours, czhp, color='green', linewidth=0.5, linestyle="--", alpha=0.35, label=f"P_hp ({comparison_label})")
            axs[2, 1].plot(chours, czev, color='purple', linewidth=0.5, linestyle="--", alpha=0.35, label=f"P_ev ({comparison_label})")
        axs[2, 1].axhline(0, color='black', lw=0.5)
        shade_away(axs[2, 1])
        add_day_markers(axs[2, 1])
        axs[2, 1].set_title("Power Breakdown at Node (grey = away)", color='black')
        axs[2, 1].set_ylabel("Power [W]", color='black')
        axs[2, 1].legend(fontsize=7)

        for row in axs:
            for ax in row:
                ax.set_xlabel("Time [hours from start of week]", color='black')
                ax.set_xlim(0, zoom_days * 24)
                ax.set_xticks(np.arange(0, zoom_days * 24 + 1, 12))
                ax.tick_params(axis='x', colors='black')
                ax.tick_params(axis='y', colors='black')

        fig.suptitle(
            f"One-Week Detail — Days {zoom_start_day}–{zoom_start_day + zoom_days} "
            f"(dotted lines = midnight)",
            fontsize=13, color='black', y=1.01,
        )
        plt.tight_layout()
        path = f"ev_results_zoom_config{config_id}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Weekly zoom    → {path}")

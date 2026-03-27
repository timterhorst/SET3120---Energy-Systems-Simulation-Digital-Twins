"""Analyse saved simulation results and produce LaTeX-ready tables.

Usage (from the project root):
    python -m src.analyze_results
"""
import os
import sys

import numpy as np
import pandas as pd

RESULTS_DIR = "results"
TOTAL_STEPS = 35_040  # 365 days × 96 steps/day
KNMI_PATH = "data/KNMI_temp_data.txt"


def load(name: str) -> dict[str, np.ndarray] | None:
    path = os.path.join(RESULTS_DIR, f"{name}.npz")
    if not os.path.exists(path):
        return None
    return dict(np.load(path, allow_pickle=True))


def print_table(title: str, header: list[str], rows: list[list[str]]):
    """Print a pipe-separated table ready for LaTeX conversion."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")
    print()


def print_sectioned_table(title: str, header: list[str],
                          sections: list[tuple[str, list[list[str]]]]):
    """Print a pipe-separated table with named section groups."""
    n_cols = len(header)
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * n_cols) + "|")
    for section_name, rows in sections:
        print("|--- " + section_name + " " + "---|" * (n_cols - 1) + "---|")
        for row in rows:
            print("| " + " | ".join(str(c) for c in row) + " |")
    print()


def _parse_knmi(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse KNMI hourly file and return (months, temperatures_celsius).

    Both arrays have shape (8760,).  ``months`` holds the calendar month
    (1–12) for each hourly observation.
    """
    months, temps = [], []
    with open(filepath) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            yyyymmdd = parts[1].strip()
            month = int(yyyymmdd[4:6])
            t_raw = int(parts[3].strip())
            months.append(month)
            temps.append(t_raw / 10.0)
    return np.array(months), np.array(temps)


_SEASON_MAP = {
    "Winter (DJF)": {12, 1, 2},
    "Spring (MAM)": {3, 4, 5},
    "Summer (JJA)": {6, 7, 8},
    "Autumn (SON)": {9, 10, 11},
}


def knmi_summary_table(filepath: str = KNMI_PATH):
    """Print a report-ready summary table of the KNMI temperature data."""
    months, temp = _parse_knmi(filepath)

    # --- Section 1: annual statistics ---
    annual_rows = [
        ["Mean temperature",   f"{temp.mean():.2f}", "°C"],
        ["Std. deviation",     f"{temp.std():.2f}",  "°C"],
        ["Minimum temperature", f"{temp.min():.1f}", "°C"],
        ["Maximum temperature", f"{temp.max():.1f}", "°C"],
    ]

    # --- Section 2: seasonal statistics ---
    seasonal_rows = []
    for season, month_set in _SEASON_MAP.items():
        mask = np.isin(months, list(month_set))
        s = temp[mask]
        seasonal_rows.append([
            season,
            f"{s.mean():.2f}",
            f"{s.min():.1f}",
            f"{s.max():.1f}",
        ])

    # --- Section 3: monthly means ---
    monthly_rows = []
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    for m in range(1, 13):
        s = temp[months == m]
        monthly_rows.append([month_names[m - 1], f"{s.mean():.2f}", f"{s.min():.1f}", f"{s.max():.1f}"])

    # --- Section 4: derived climate indicators ---
    n_total = len(temp)
    frost_hours    = int((temp < 0).sum())
    ice_hours      = int((temp < -5).sum())
    summer_hours   = int((temp >= 25).sum())
    tropical_hours = int((temp >= 30).sum())
    hdd_base18     = float(np.maximum(18.0 - temp, 0).sum())

    indicator_rows = [
        ["Frost hours (T < 0 °C)",     f"{frost_hours}",    f"{frost_hours / n_total * 100:.2f}", "h / %"],
        ["Ice hours (T < −5 °C)",      f"{ice_hours}",      f"{ice_hours / n_total * 100:.2f}",   "h / %"],
        ["Summer hours (T ≥ 25 °C)",   f"{summer_hours}",   f"{summer_hours / n_total * 100:.2f}","h / %"],
        ["Tropical hours (T ≥ 30 °C)", f"{tropical_hours}", f"{tropical_hours / n_total * 100:.2f}", "h / %"],
        ["Heating degree-hours (base 18 °C)", f"{hdd_base18:.0f}", "—", "°C·h"],
    ]

    # --- Print annual + seasonal + indicators as a sectioned table ---
    annual_formatted = [[r[0], r[1], "—", r[2]] for r in annual_rows]
    seasonal_formatted = seasonal_rows
    indicator_formatted = indicator_rows

    print_sectioned_table(
        "Table 0a: KNMI Temperature Data — Annual & Seasonal Summary "
        "(Rotterdam stn 344, 2019)",
        ["Metric", "Value / Mean", "Min / Count", "Unit"],
        [
            ("Annual", annual_formatted),
            ("Seasonal (mean / min / max)", seasonal_formatted),
            ("Climate indicators", indicator_formatted),
        ],
    )

    # --- Print monthly breakdown as a flat table ---
    print_table(
        "Table 0b: KNMI Monthly Temperature Breakdown (Rotterdam stn 344, 2019)",
        ["Month", "Mean [°C]", "Min [°C]", "Max [°C]"],
        monthly_rows,
    )


def main():
    # ------------------------------------------------------------------
    # STEP 0 — KNMI weather data summary
    # ------------------------------------------------------------------
    if os.path.exists(KNMI_PATH):
        knmi_summary_table(KNMI_PATH)
    else:
        print(f"\n  WARNING: KNMI data not found at {KNMI_PATH} — skipping weather summary.\n")

    # ------------------------------------------------------------------
    # STEP 0 — Data availability check
    # ------------------------------------------------------------------
    datasets = {
        "config1_base": load("config1_base"),
        "config1_ev":   load("config1_ev"),
        "config99_base": load("config99_base"),
        "config99_ev":  load("config99_ev"),
    }

    print("\n" + "=" * 70)
    print("  STEP 0 — Data Availability")
    print("=" * 70)
    for name, data in datasets.items():
        status = "AVAILABLE" if data is not None else "MISSING"
        print(f"  {name:20s} : {status}")

    missing = [k for k, v in datasets.items() if v is None]
    if missing:
        print(f"\n  WARNING: missing datasets: {missing}")
        print("  Run the corresponding simulations first.")
        sys.exit(1)

    c1b = datasets["config1_base"]
    c1e = datasets["config1_ev"]
    c99b = datasets["config99_base"]
    c99e = datasets["config99_ev"]

    # ------------------------------------------------------------------
    # STEP 0 — HP sanity check (verify binary switching, not constant)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 0 — Heat Pump Sanity Check (config1_base, full year)")
    print("=" * 70)

    p_hp = c1b["hp_powers"].astype(float)
    q_hp = c1b["heat_productions"].astype(float)

    for label, arr in [("P_hp [W]", p_hp), ("Q_hp [W]", q_hp)]:
        print(f"  {label:12s}  min={arr.min():.1f}  max={arr.max():.1f}  "
              f"mean={arr.mean():.1f}  unique={np.unique(arr).tolist()}")

    if len(np.unique(p_hp)) <= 1:
        print("  ⚠  P_hp is CONSTANT — possible error!")
    else:
        print("  ✓  P_hp switches between multiple values — binary cycling confirmed.")

    # ------------------------------------------------------------------
    # Helper: build datetime index and summer mask
    # ------------------------------------------------------------------
    dt_c1 = pd.to_datetime(c1b["datetime_index"])
    summer_mask_c1 = dt_c1.month.isin([6, 7, 8])
    n_summer_c1 = summer_mask_c1.sum()

    dt_c99 = pd.to_datetime(c99b["datetime_index"])

    # ------------------------------------------------------------------
    # STEP 1a — Overvoltage suppression (config1, nominal)
    # ------------------------------------------------------------------
    v_base_c1 = c1b["voltages"].astype(float)
    v_ev_c1 = c1e["voltages"].astype(float)

    ov_base_full = int((v_base_c1 > 1.02).sum())
    ov_ev_full   = int((v_ev_c1 > 1.02).sum())
    ov_base_sum  = int((v_base_c1[summer_mask_c1] > 1.02).sum())
    ov_ev_sum    = int((v_ev_c1[summer_mask_c1] > 1.02).sum())

    # ------------------------------------------------------------------
    # STEP 1b — Power metrics (config1)
    # ------------------------------------------------------------------
    p_base_c1 = c1b["p_at_grid"].astype(float)
    p_ev_c1   = c1e["total_powers"].astype(float)
    p_charge_c1 = c1e["ev_powers"].astype(float)

    charge_mask_c1 = p_charge_c1 > 0
    v2g_mask_c1    = p_charge_c1 < 0

    # ORIGINAL TABLE CODE — PRESERVED
    # ------------------------------------------------------------------
    # Table 1: Nominal conditions summary
    # ------------------------------------------------------------------
    # rows_t1 = [
    #     ["Overvoltage steps (V>1.02), full year", f"{ov_base_full}", f"{ov_ev_full}", "count"],
    #     ["Overvoltage fraction, full year",
    #      f"{ov_base_full / TOTAL_STEPS * 100:.2f}", f"{ov_ev_full / TOTAL_STEPS * 100:.2f}", "%"],
    #     ["Overvoltage steps, summer (JJA)", f"{ov_base_sum}", f"{ov_ev_sum}", "count"],
    #     ["Overvoltage fraction, summer",
    #      f"{ov_base_sum / n_summer_c1 * 100:.2f}", f"{ov_ev_sum / n_summer_c1 * 100:.2f}", "%"],
    #     ["Peak P\\_hp (base)", f"{p_base_c1.max():.0f}", "—", "W"],
    #     ["Peak P\\_total (with EV)", "—", f"{p_ev_c1.max():.0f}", "W"],
    #     ["Mean P\\_hp (base)", f"{p_base_c1.mean():.1f}", "—", "W"],
    #     ["Mean P\\_total (with EV)", "—", f"{p_ev_c1.mean():.1f}", "W"],
    #     ["Mean P\\_charge (charging only, P>0)", "—",
    #      f"{p_charge_c1[charge_mask_c1].mean():.1f}" if charge_mask_c1.any() else "N/A", "W"],
    #     ["V2G discharge events (P\\_charge<0)", "—",
    #      f"{int(v2g_mask_c1.sum())}", "count"],
    #     ["Mean P\\_charge during V2G", "—",
    #      f"{p_charge_c1[v2g_mask_c1].mean():.1f}" if v2g_mask_c1.any() else "N/A", "W"],
    # ]
    # print_table("Table 1: Nominal Conditions Summary (config 1)",
    #             ["Metric", "Base (no EV)", "With EV", "Unit"], rows_t1)

    # NEW TABLE CODE
    # ------------------------------------------------------------------
    # Table 1: Nominal conditions summary (sectioned)
    # ------------------------------------------------------------------
    t1_voltage = [
        ["Overvoltage steps (V>1.02), full year", f"{ov_base_full}", f"{ov_ev_full}", "count"],
        ["Overvoltage fraction, full year",
         f"{ov_base_full / TOTAL_STEPS * 100:.2f}", f"{ov_ev_full / TOTAL_STEPS * 100:.2f}", "%"],
        ["Overvoltage steps, summer (JJA)", f"{ov_base_sum}", f"{ov_ev_sum}", "count"],
        ["Overvoltage fraction, summer",
         f"{ov_base_sum / n_summer_c1 * 100:.2f}", f"{ov_ev_sum / n_summer_c1 * 100:.2f}", "%"],
    ]
    t1_power = [
        ["Peak power at node", f"{p_base_c1.max():.0f}", f"{p_ev_c1.max():.0f}", "W"],
        ["Mean power at node", f"{p_base_c1.mean():.1f}", f"{p_ev_c1.mean():.1f}", "W"],
        ["Mean EV charging power (active, P>0)", "—",
         f"{p_charge_c1[charge_mask_c1].mean():.0f}" if charge_mask_c1.any() else "N/A", "W"],
        ["V2G discharge events (P<0)", "—", f"{int(v2g_mask_c1.sum())}", "count"],
    ]
    print_sectioned_table(
        "Table 1: Nominal Conditions Summary (config 1)",
        ["Metric", "Base (no EV)", "With EV", "Unit"],
        [("Grid voltage", t1_voltage), ("Node power", t1_power)],
    )

    # ------------------------------------------------------------------
    # Diagnostic: Overvoltage controller verification (config1, EV run)
    # ------------------------------------------------------------------
    is_home_c1 = c1e["is_homes"].astype(bool)
    ov_ev_mask = v_ev_c1 > 1.02

    absorb_correct = int((ov_ev_mask & (p_charge_c1 == 7400.0)).sum())
    away_correct   = int((ov_ev_mask & ~is_home_c1).sum())
    home_missed    = int((ov_ev_mask & is_home_c1 & (p_charge_c1 < 7400.0)).sum())

    print("=" * 70)
    print("  Diagnostic: Overvoltage Controller Verification (config 1, EV run)")
    print("=" * 70)
    print(f"  Total overvoltage timesteps (V>1.02):             {int(ov_ev_mask.sum()):>6}")
    print(f"  V>1.02 AND P_charge=7400 W (correct absorption):  {absorb_correct:>6}")
    print(f"  V>1.02 AND is_home=False (away, correct idle):     {away_correct:>6}")
    print(f"  V>1.02 AND is_home=True AND P_charge<7400 (missed):{home_missed:>6}")
    if home_missed > 0:
        soc_c1 = c1e["socs"].astype(float)
        missed_mask = ov_ev_mask & is_home_c1 & (p_charge_c1 < 7400.0)
        missed_socs = soc_c1[missed_mask]
        missed_charges = p_charge_c1[missed_mask]
        print(f"    → SOC at missed steps: min={missed_socs.min():.4f}  "
              f"max={missed_socs.max():.4f}  mean={missed_socs.mean():.4f}")
        print(f"    → P_charge at missed steps: unique={np.unique(missed_charges).tolist()}")
        full_soc_missed = int((missed_socs >= 1.0).sum())
        if full_soc_missed > 0:
            print(f"    → {full_soc_missed} of {home_missed} missed steps have SOC=1.0 "
                  "(battery full, cannot absorb more)")
    print()

    # ------------------------------------------------------------------
    # STEP 2a — Undervoltage mitigation (config99, stressed)
    # ------------------------------------------------------------------
    v_base_c99 = c99b["voltages"].astype(float)
    v_ev_c99   = c99e["voltages"].astype(float)

    uv_base = int((v_base_c99 < 0.98).sum())
    uv_ev   = int((v_ev_c99 < 0.98).sum())

    # ------------------------------------------------------------------
    # STEP 2b — Power metrics (config99)
    # ------------------------------------------------------------------
    p_base_c99  = c99b["p_at_grid"].astype(float)
    p_ev_c99    = c99e["total_powers"].astype(float)
    p_charge_c99 = c99e["ev_powers"].astype(float)

    charge_mask_c99 = p_charge_c99 > 0
    v2g_mask_c99    = p_charge_c99 < 0

    # ORIGINAL TABLE CODE — PRESERVED
    # ------------------------------------------------------------------
    # Table 2: Stressed grid summary
    # ------------------------------------------------------------------
    # rows_t2 = [
    #     ["Undervoltage steps (V<0.98)", f"{uv_base}", f"{uv_ev}", "count"],
    #     ["Undervoltage fraction",
    #      f"{uv_base / TOTAL_STEPS * 100:.2f}", f"{uv_ev / TOTAL_STEPS * 100:.2f}", "%"],
    #     ["Peak P\\_hp (base)", f"{p_base_c99.max():.0f}", "—", "W"],
    #     ["Peak P\\_total (with EV)", "—", f"{p_ev_c99.max():.0f}", "W"],
    #     ["Mean P\\_hp (base)", f"{p_base_c99.mean():.1f}", "—", "W"],
    #     ["Mean P\\_total (with EV)", "—", f"{p_ev_c99.mean():.1f}", "W"],
    #     ["Mean P\\_charge (charging only, P>0)", "—",
    #      f"{p_charge_c99[charge_mask_c99].mean():.1f}" if charge_mask_c99.any() else "N/A", "W"],
    #     ["V2G discharge events (P\\_charge<0)", "—",
    #      f"{int(v2g_mask_c99.sum())}", "count"],
    #     ["Mean P\\_charge during V2G", "—",
    #      f"{p_charge_c99[v2g_mask_c99].mean():.1f}" if v2g_mask_c99.any() else "N/A", "W"],
    # ]
    # print_table("Table 2: Stressed Grid Summary (config 99, 3× loading)",
    #             ["Metric", "Base (no EV)", "With EV", "Unit"], rows_t2)

    # NEW TABLE CODE
    # ------------------------------------------------------------------
    # Table 2: Stressed grid summary (sectioned)
    # ------------------------------------------------------------------
    t2_voltage = [
        ["Undervoltage steps (V<0.98), full year", f"{uv_base}", f"{uv_ev}", "count"],
        ["Undervoltage fraction, full year",
         f"{uv_base / TOTAL_STEPS * 100:.2f}", f"{uv_ev / TOTAL_STEPS * 100:.2f}", "%"],
    ]
    t2_power = [
        ["Peak power at node", f"{p_base_c99.max():.0f}", f"{p_ev_c99.max():.0f}", "W"],
        ["Mean power at node", f"{p_base_c99.mean():.1f}", f"{p_ev_c99.mean():.1f}", "W"],
        ["Mean EV charging power (active, P>0)", "—",
         f"{p_charge_c99[charge_mask_c99].mean():.0f}" if charge_mask_c99.any() else "N/A", "W"],
        ["V2G discharge events (P<0)", "—", f"{int(v2g_mask_c99.sum())}", "count"],
        ["Mean V2G discharge power", "—",
         f"{p_charge_c99[v2g_mask_c99].mean():.0f}" if v2g_mask_c99.any() else "N/A", "W"],
    ]
    print_sectioned_table(
        "Table 2: Stressed Grid Summary (config 99, 3× loading)",
        ["Metric", "Base (no EV)", "With EV", "Unit"],
        [("Grid voltage", t2_voltage), ("Node power", t2_power)],
    )

    # ------------------------------------------------------------------
    # STEP 2c — SOC floor violations (config99, EV run only)
    # ------------------------------------------------------------------
    soc_c99  = c99e["socs"].astype(float)
    home_c99 = c99e["is_homes"].astype(bool)

    n_home       = int(home_c99.sum())
    soc_at_home  = soc_c99[home_c99]
    violations   = int((soc_at_home < 0.40).sum())
    soc_min_year = float(soc_c99.min())

    # ORIGINAL TABLE CODE — PRESERVED
    # rows_t3 = [
    #     ["Total home timesteps", f"{n_home}", "count"],
    #     ["SOC < 0.40 while home", f"{violations}", "count"],
    #     ["SOC violation fraction (of home steps)",
    #      f"{violations / n_home * 100:.2f}" if n_home > 0 else "N/A", "%"],
    #     ["Minimum SOC (full year)", f"{soc_min_year:.4f}", "—"],
    # ]
    # print_table("Table 3: SOC Constraint Validation (config 99, 3× loading)",
    #             ["Metric", "Value", "Unit"], rows_t3)

    # NEW TABLE CODE
    # ------------------------------------------------------------------
    # Table 3: SOC constraint validation
    # ------------------------------------------------------------------
    rows_t3 = [
        ["Total home timesteps", f"{n_home}", "count"],
        ["SOC < 0.40 while home", f"{violations}", "count"],
        ["SOC violation fraction (of home steps)",
         f"{violations / n_home * 100:.2f}" if n_home > 0 else "N/A", "%"],
        ["Minimum SOC reached (full year)", f"{soc_min_year:.2f}", "—"],
    ]
    print_table("Table 3: SOC Constraint Validation (config 99, 3× loading)",
                ["Metric", "Value", "Unit"], rows_t3)

    # ------------------------------------------------------------------
    # DEBUG — HP-on while away and T_room well above setback
    # ------------------------------------------------------------------
    t_room_c1 = c1e["temperatures"].astype(float)
    t_min_dyn_c1 = c1e["temp_min_dynamics"].astype(float)
    p_hp_c1 = c1e["hp_powers"].astype(float)
    home_c1 = c1e["is_homes"].astype(bool)
    v_ev_c1_dbg = c1e["voltages"].astype(float)

    suspect = (p_hp_c1 > 0) & ~home_c1 & (t_room_c1 > t_min_dyn_c1 + 2)
    suspect_ov = suspect & (v_ev_c1_dbg > 1.02)
    suspect_real = suspect & (v_ev_c1_dbg <= 1.02)

    print("=" * 70)
    print("  DEBUG — HP on while away & T_room > T_min_dyn + 2 °C  (config 1 EV)")
    print("=" * 70)
    print(f"  Total suspect timesteps:     {int(suspect.sum()):>6}  "
          f"({suspect.sum() / TOTAL_STEPS * 100:.2f}%)")
    print(f"    Due to overvoltage (V>1.02, correct): {int(suspect_ov.sum()):>6}")
    print(f"    Remaining (temp-control bug):         {int(suspect_real.sum()):>6}")
    if suspect_real.any():
        print(f"  T_room  at bug steps: "
              f"{t_room_c1[suspect_real].min():.1f} – {t_room_c1[suspect_real].max():.1f} °C")
        print(f"  T_min_dyn at bug steps: "
              f"{t_min_dyn_c1[suspect_real].min():.1f} – {t_min_dyn_c1[suspect_real].max():.1f} °C")
    else:
        print("  ✓ Temperature-control setback is working correctly.")
    print()


if __name__ == "__main__":
    main()
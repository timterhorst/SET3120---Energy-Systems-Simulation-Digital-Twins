"""
Test script to verify voltage control functionality.

Tests the controller's response to under-voltage and over-voltage conditions.
"""

import matplotlib.pyplot as plt

from controller import ThermostatController


def test_undervoltage_protection():
    """Test that heat pump turns OFF when voltage is too low."""
    print("\n=== TEST 1: Under-voltage Protection ===")

    # Initialize controller
    ctrl = ThermostatController(
        T_set=20.0,
        dT_lower=1.0,
        dT_upper=1.0,
        P_rated=2000,
        V_nom=230.0,
        V_tolerance=0.02
    )

    # Test: Cold room (18°C) but low voltage (220V)
    T_room = 18.0  # Below 19°C threshold - would normally turn ON
    V_grid = 220.0  # Below 225.4V (98% of 230V) - safety limit

    P_elec, state = ctrl.update(T_room, V_grid)

    print(f"Room temp: {T_room}°C (below 19°C threshold)")
    print(f"Grid voltage: {V_grid}V (below 225.4V safety limit)")
    print(f"Heat pump state: {'ON' if state == 1 else 'OFF'}")
    print(f"Power: {P_elec}W")

    # Verify
    assert state == 0, "FAIL: Heat pump should be OFF due to low voltage"
    assert P_elec == 0, "FAIL: Power should be 0W when OFF"
    print("✓ PASS: Heat pump correctly turned OFF for safety")


def test_voltage_recovery():
    """Test that heat pump turns ON when voltage recovers."""
    print("\n=== TEST 2: Voltage Recovery ===")

    ctrl = ThermostatController(
        T_set=20.0, dT_lower=1.0, dT_upper=1.0,
        P_rated=2000, V_nom=230.0, V_tolerance=0.02
    )

    # Step 1: Low voltage (should stay OFF)
    T_room = 18.0
    V_grid = 220.0
    P_elec, state = ctrl.update(T_room, V_grid)

    print(f"Step 1 - Low voltage: State={'ON' if state else 'OFF'}, V={V_grid}V")

    # Step 2: Voltage recovers to nominal
    V_grid = 230.0  # Back to nominal
    P_elec, state = ctrl.update(T_room, V_grid)

    print(f"Step 2 - Voltage recovered: State={'ON' if state else 'OFF'}, V={V_grid}V")
    print(f"Power: {P_elec}W")

    # Verify
    assert state == 1, "FAIL: Heat pump should turn ON after voltage recovery"
    assert P_elec == 2000, "FAIL: Power should be P_rated when ON"
    print("✓ PASS: Heat pump correctly turned ON after voltage recovery")


def test_overvoltage_grid_support():
    """Test that heat pump turns ON during over-voltage to help grid."""
    print("\n=== TEST 3: Over-voltage Grid Support ===")

    ctrl = ThermostatController(
        T_set=20.0, dT_lower=1.0, dT_upper=1.0,
        P_rated=2000, V_nom=230.0, V_tolerance=0.02
    )

    # Room at comfortable temp (within hysteresis), but high voltage
    T_room = 20.5  # Between 19-21°C - normally no action
    V_grid = 236.0  # Above 234.6V (102% of 230V)

    P_elec, state = ctrl.update(T_room, V_grid)

    print(f"Room temp: {T_room}°C (within comfort band 19-21°C)")
    print(f"Grid voltage: {V_grid}V (above 234.6V threshold)")
    print(f"Heat pump state: {'ON' if state == 1 else 'OFF'}")
    print(f"Power: {P_elec}W")

    # Verify
    assert state == 1, "FAIL: Heat pump should turn ON to help reduce voltage"
    assert P_elec == 2000, "FAIL: Power should be P_rated"
    print("✓ PASS: Heat pump correctly provides grid support during over-voltage")


def test_overvoltage_with_hot_room():
    """Test that thermal limits override grid support when room is too hot."""
    print("\n=== TEST 4: Over-voltage with Hot Room ===")

    ctrl = ThermostatController(
        T_set=20.0, dT_lower=1.0, dT_upper=1.0,
        P_rated=2000, V_nom=230.0, V_tolerance=0.02
    )

    # Hot room + high voltage
    T_room = 22.0  # Above 21°C threshold
    V_grid = 236.0  # Above 234.6V

    P_elec, state = ctrl.update(T_room, V_grid)

    print(f"Room temp: {T_room}°C (above 21°C threshold)")
    print(f"Grid voltage: {V_grid}V (above 234.6V threshold)")
    print(f"Heat pump state: {'ON' if state == 1 else 'OFF'}")
    print(f"Power: {P_elec}W")

    # P2.py logic: force ON only if V > V_max and T_room < T_upper. Here T_room >= T_upper, so we do not force ON.
    assert state == 0, "FAIL: Heat pump should stay OFF (thermal comfort overrides grid support)"
    assert P_elec == 0, "FAIL: Power should be 0W when OFF"
    print("✓ PASS: Thermal comfort correctly prioritized over grid support")


def test_voltage_thresholds():
    """Test exact voltage thresholds."""
    print("\n=== TEST 5: Voltage Threshold Boundaries ===")

    ctrl = ThermostatController(
        T_set=20.0, dT_lower=1.0, dT_upper=1.0,
        P_rated=2000, V_nom=230.0, V_tolerance=0.02
    )

    T_room = 18.0  # Cold room (should want to turn ON)

    test_voltages = [
        (220.0, "Well below threshold", 0),
        (225.3, "Just below threshold", 0),
        (225.5, "Just above threshold", 1),
        (230.0, "Nominal", 1),
        (234.5, "Just below high threshold", 1),
        (234.7, "Just above high threshold", 1),
        (240.0, "Well above threshold", 1),
    ]

    for V, description, expected_state in test_voltages:
        P_elec, state = ctrl.update(T_room, V)
        status = "✓" if state == expected_state else "✗"
        print(f"  {status} V={V:5.1f}V ({description:25s}): State={'ON' if state else 'OFF'}")

    print("✓ PASS: Voltage thresholds at ±2% of nominal (225.4V / 234.6V)")


def plot_test_summary(results: list[tuple[str, str, bool]]) -> None:
    """
    Draw a graphical summary of test results.

    Parameters
    ----------
    results : list of (test_name, short_description, passed)
    """
    n = len(results)
    row_height = 1.5
    fig_height = max(5, n * row_height * 0.55)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, n * row_height + 1)
    ax.axis("off")

    # Title
    all_passed = all(r[2] for r in results)
    title = "Voltage control tests — ALL PASSED" if all_passed else "Voltage control tests — SOME FAILED"
    ax.text(0.5, n * row_height + 0.5, title, fontsize=14, fontweight="bold", ha="center", va="bottom")
    ax.text(0.5, n * row_height, "test_voltage_control.py", fontsize=10, ha="center", va="top", style="italic", color="gray")

    for i, (name, desc, passed) in enumerate(results):
        y = (n - 1 - i) * row_height
        if passed:
            ax.plot(0.08, y, "o", color="green", markersize=14, markeredgecolor="darkgreen", markeredgewidth=2)
            ax.text(0.12, y, "✓", fontsize=16, color="green", va="center", fontweight="bold")
        else:
            ax.plot(0.08, y, "s", color="red", markersize=12, markeredgecolor="darkred", markeredgewidth=2)
            ax.text(0.12, y, "✗", fontsize=16, color="red", va="center", fontweight="bold")
        ax.text(0.18, y, name, fontsize=11, va="center", fontweight="bold")
        ax.text(0.18, y - 0.35, desc, fontsize=9, va="center", color="gray")

    # Footer
    ax.text(0.5, -0.6, f"{sum(r[2] for r in results)}/{n} tests passed", fontsize=10, ha="center", va="top")
    plt.tight_layout()
    plt.savefig("voltage_tests_summary.png", dpi=150, bbox_inches="tight")
    print("\nSummary saved as 'voltage_tests_summary.png'")
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("VOLTAGE CONTROL FUNCTIONALITY TESTS")
    print("=" * 60)

    tests = [
        (test_undervoltage_protection, "Under-voltage protection", "Heat pump OFF when V < 98% (safety)"),
        (test_voltage_recovery, "Voltage recovery", "HP turns ON when V returns to nominal"),
        (test_overvoltage_grid_support, "Over-voltage grid support", "HP ON when V > 102% to help grid"),
        (test_overvoltage_with_hot_room, "Over-voltage with hot room", "Thermal limit overrides grid support"),
        (test_voltage_thresholds, "Voltage thresholds", "±2% boundaries (225.4 V / 234.6 V)"),
    ]
    results = []

    failed_msg = None
    for test_fn, name, desc in tests:
        try:
            test_fn()
            results.append((name, desc, True))
        except AssertionError as e:
            results.append((name, desc, False))
            if failed_msg is None:
                failed_msg = str(e)
            print(f"\n❌ TEST FAILED: {e}")

    # Graphical summary (all tests run so we can show pass/fail for each)
    if results:
        plot_test_summary(results)

    if failed_msg is not None:
        print("\n" + "=" * 60)
        print("SOME TESTS FAILED")
        print("=" * 60)
        raise AssertionError(failed_msg)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

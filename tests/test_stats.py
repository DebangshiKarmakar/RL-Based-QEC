"""Known-value tests for logical-error-rate statistics."""

import math

from qecrl.stats import per_round, shots_for_failures, wilson_interval


def test_wilson_interval_known_value() -> None:
    low, high = wilson_interval(50, 100)
    assert math.isclose(low, 0.40382982859014716, abs_tol=1e-12)
    assert math.isclose(high, 0.5961701714098528, abs_tol=1e-12)


def test_per_round_known_values() -> None:
    assert math.isclose(per_round(0.1, 2), 0.0527864045, abs_tol=1e-10)
    assert math.isclose(per_round(0.01, 1), 0.01, abs_tol=1e-12)


def test_shots_for_about_100_failures() -> None:
    assert shots_for_failures(0.017) == 5883

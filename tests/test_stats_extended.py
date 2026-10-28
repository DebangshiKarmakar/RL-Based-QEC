"""Extended tests for stats edge cases and bug fixes."""

import math
import pytest
from qecrl.stats import per_round, shots_for_failures, wilson_interval


def test_wilson_interval_zero_failures_non_negative() -> None:
    low, high = wilson_interval(0, 100)
    assert low >= 0.0
    assert not math.copysign(1.0, low) < 0  # ensure not -0.0
    assert low <= high <= 1.0


def test_wilson_interval_all_failures_capped_at_one() -> None:
    low, high = wilson_interval(100, 100)
    assert 0.0 <= low <= high
    assert high <= 1.0


def test_wilson_interval_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        wilson_interval(-1, 100)
    with pytest.raises(ValueError):
        wilson_interval(101, 100)
    with pytest.raises(ValueError):
        wilson_interval(10, 0)
    with pytest.raises(ValueError):
        wilson_interval(10, 100, z=-1.0)


def test_per_round_edge_cases() -> None:
    assert per_round(0.0, 3) == 0.0
    assert per_round(0.5, 3) == 0.5
    assert per_round(0.51, 3) == 0.5  # statistical saturation
    with pytest.raises(ValueError):
        per_round(-0.01, 3)
    with pytest.raises(ValueError):
        per_round(1.5, 3)
    with pytest.raises(ValueError):
        per_round(0.1, 0)


def test_shots_for_failures_zero_ler() -> None:
    assert math.isinf(shots_for_failures(0.0))
    with pytest.raises(ValueError):
        shots_for_failures(-0.01)
    with pytest.raises(ValueError):
        shots_for_failures(0.01, target_failures=0)


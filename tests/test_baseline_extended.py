"""Extended tests for baseline decoding and shape broadcasting bug fixes."""

import numpy as np
import pytest
from qecrl.baseline import count_failures, mwpm_predictions
from qecrl.noise import build_circuit


def test_count_failures_1d_vs_2d_no_false_broadcasting() -> None:
    # 1D predictions matching 2D observables
    preds = np.array([0, 1, 0])
    obs = np.array([[0], [1], [0]])
    assert count_failures(preds, obs) == 0

    # 1D predictions with one mismatch
    preds_mismatch = np.array([0, 0, 0])
    assert count_failures(preds_mismatch, obs) == 1


def test_count_failures_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        count_failures(np.array([0, 1]), np.array([[0], [1], [0]]))


def test_count_failures_empty() -> None:
    assert count_failures(np.empty((0, 1)), np.empty((0, 1))) == 0


def test_mwpm_predictions_single_event() -> None:
    circuit = build_circuit(3, 3, 0.005)
    dem = circuit.detector_error_model(decompose_errors=True)
    single_detection = np.zeros(circuit.num_detectors, dtype=bool)
    pred = mwpm_predictions(dem, single_detection)
    assert pred.shape == (1, 1)
    assert pred[0, 0] == 0


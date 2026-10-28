"""Extended tests for SyndromeDecodeEnv."""

import numpy as np
import pytest
from qecrl.env import SyndromeDecodeEnv


def test_step_before_reset_raises_runtime_error() -> None:
    env = SyndromeDecodeEnv()
    with pytest.raises(RuntimeError):
        env.step(0)


def test_step_after_terminated_raises_runtime_error() -> None:
    env = SyndromeDecodeEnv()
    env.reset()
    _, _, terminated, _, _ = env.step(0)
    assert terminated
    with pytest.raises(RuntimeError):
        env.step(0)


def test_env_basis_support() -> None:
    env_x = SyndromeDecodeEnv(distance=3, rounds=3, basis="x")
    obs, info = env_x.reset()
    assert obs.shape == (env_x.circuit.num_detectors,)
    _, _, term, _, _ = env_x.step(0)
    assert term


def test_env_invalid_batch_size() -> None:
    with pytest.raises(ValueError):
        SyndromeDecodeEnv(batch_size=0)


"""Extended tests for noise and circuit generation."""

import pytest
from qecrl.noise import build_circuit


def test_build_circuit_case_insensitive_basis() -> None:
    c_z = build_circuit(3, 3, 0.005, basis="Z")
    assert c_z.num_detectors > 0

    c_x = build_circuit(3, 3, 0.005, basis=" X ")
    assert c_x.num_detectors > 0


def test_build_circuit_invalid_params() -> None:
    with pytest.raises(ValueError):
        build_circuit(1, 3, 0.005)
    with pytest.raises(ValueError):
        build_circuit(3, 0, 0.005)
    with pytest.raises(ValueError):
        build_circuit(3, 3, -0.01)
    with pytest.raises(ValueError):
        build_circuit(3, 3, 1.5)
    with pytest.raises(ValueError):
        build_circuit(3, 3, 0.005, basis="y")


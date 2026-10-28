"""Stim circuit construction."""

import stim


def build_circuit(distance: int, rounds: int, p: float, basis: str = "z") -> stim.Circuit:
    """Build a rotated surface-code memory circuit with uniform noise."""
    if distance < 2:
        raise ValueError(f"Distance must be at least 2, got {distance}.")
    if rounds < 1:
        raise ValueError(f"Rounds must be at least 1, got {rounds}.")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Noise probability p must be in [0.0, 1.0], got {p}.")

    clean_basis = basis.strip().lower()
    if clean_basis not in ("x", "z"):
        raise ValueError(f"Basis must be 'x' or 'z', got {basis!r}.")

    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{clean_basis}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


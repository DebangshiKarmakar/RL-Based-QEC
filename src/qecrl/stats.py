"""Logical-error-rate statistics."""

import math


def wilson_interval(failures: int, shots: int, z: float = 1.96) -> tuple[float, float]:
    """Return a Wilson score interval for a failure probability."""
    if shots <= 0 or not 0 <= failures <= shots:
        raise ValueError("Require shots > 0 and 0 <= failures <= shots.")
    if z <= 0:
        raise ValueError("Require z > 0.")
    rate = failures / shots
    denominator = 1 + z * z / shots
    centre = (rate + z * z / (2 * shots)) / denominator
    half = z * math.sqrt(
        rate * (1 - rate) / shots + z * z / (4 * shots * shots)
    ) / denominator
    low = max(0.0, centre - half)
    high = min(1.0, centre + half)
    if low == 0.0:
        low = 0.0
    return low, high


def per_round(block_ler: float, rounds: int) -> float:
    """Convert a block logical error rate to a per-round rate."""
    if rounds <= 0:
        raise ValueError("Require rounds > 0.")
    if not 0 <= block_ler <= 1.0:
        raise ValueError("Require 0 <= block_ler <= 1.0.")
    if block_ler >= 0.5:
        return 0.5
    return (1 - (1 - 2 * block_ler) ** (1.0 / rounds)) / 2


def shots_for_failures(estimated_ler: float, target_failures: int = 100) -> int | float:
    """Estimate the shots needed to observe about target_failures failures."""
    if target_failures <= 0:
        raise ValueError("The target failure count must be positive.")
    if estimated_ler < 0:
        raise ValueError("The estimated LER cannot be negative.")
    if estimated_ler == 0:
        return math.inf
    return math.ceil(target_failures / estimated_ler)


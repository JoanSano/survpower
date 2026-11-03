"""Small shared helpers."""

from __future__ import annotations

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(iterable, **kwargs):
        return iterable


def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a Monte Carlo rejection rate."""
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(centre - half), float(centre + half)


def mc_se(p_hat: float, n: int) -> float:
    """Monte Carlo standard error of a simulated proportion."""
    return float(np.sqrt(max(p_hat * (1 - p_hat), 1e-12) / n))

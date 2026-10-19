"""Market-state feature interfaces."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def average_off_diagonal_correlation(returns: NDArray[np.float64]) -> float:
    """Compute the mean pairwise correlation for an asset return window."""
    correlation = np.corrcoef(returns, rowvar=False)
    upper = np.triu_indices_from(correlation, k=1)
    return float(correlation[upper].mean())


def relative_frobenius_distance(
    short_covariance: NDArray[np.float64],
    long_covariance: NDArray[np.float64],
    *,
    epsilon: float = 1e-12,
) -> float:
    """Measure short-versus-long covariance structure change."""
    numerator = np.linalg.norm(short_covariance - long_covariance, ord="fro")
    denominator = np.linalg.norm(long_covariance, ord="fro") + epsilon
    return float(numerator / denominator)


"""Base covariance estimation utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.covariance import LedoitWolf


def ledoit_wolf_covariance(
    returns: NDArray[np.float64], *, assume_zero_mean: bool = True
) -> NDArray[np.float64]:
    """Estimate a positive-definite covariance matrix with shrinkage."""
    estimator = LedoitWolf(assume_centered=assume_zero_mean)
    estimator.fit(returns)
    return np.asarray(estimator.covariance_, dtype=np.float64)


def convex_fusion(
    covariances: NDArray[np.float64], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Fuse K covariance matrices using non-negative weights that sum to one."""
    if covariances.ndim != 3:
        raise ValueError("covariances must have shape (K, N, N).")
    if weights.shape != (covariances.shape[0],):
        raise ValueError("weights must have shape (K,).")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must be non-negative and sum to one.")
    return np.tensordot(weights, covariances, axes=(0, 0))


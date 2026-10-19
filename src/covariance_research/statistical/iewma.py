"""Causal IEWMA: lagged volatility standardization and slower correlation updates."""

from __future__ import annotations

import numpy as np


def iterated_ewma(returns: np.ndarray, volatility_half_life: float, correlation_half_life: float,
                  clip: float = 4.2, diagonal_inflation: float = 0.0) -> np.ndarray:
    if returns.ndim != 2 or len(returns) < 20 or not np.isfinite(returns).all():
        raise ValueError("Need at least 20 finite observations.")
    if min(volatility_half_life, correlation_half_life) <= 0:
        raise ValueError("Half-lives must be positive.")
    count, dimension = returns.shape
    output = np.full((count, dimension, dimension), np.nan)
    variance = np.maximum(np.mean(returns[:20] ** 2, axis=0), 1e-16)
    correlation_state = np.eye(dimension)
    dv = np.exp2(-1 / volatility_half_life)
    dc = np.exp2(-1 / correlation_half_life)
    for t in range(19, count):
        if t > 19:
            # The denominator existed before observing r_t; no self-normalization.
            standardized = np.clip(returns[t] / np.sqrt(variance), -clip, clip)
            correlation_state = dc * correlation_state + (1 - dc) * np.outer(standardized, standardized)
            variance = dv * variance + (1 - dv) * returns[t] ** 2
        scale = np.sqrt(np.diag(correlation_state))
        correlation = correlation_state / np.outer(scale, scale)
        std = np.sqrt(variance)
        matrix = correlation * np.outer(std, std)
        matrix += np.diag(variance * diagonal_inflation)
        matrix += np.eye(dimension) * np.trace(matrix) / dimension * 1e-8
        output[t] = matrix
    return output


def precision_cholesky(covariances: np.ndarray) -> np.ndarray:
    """Lower L with C^{-1}=L L^T, not the inverse of chol(C)."""
    inverse = np.linalg.solve(covariances, np.broadcast_to(np.eye(covariances.shape[-1]), covariances.shape))
    inverse = (inverse + np.swapaxes(inverse, -1, -2)) / 2
    return np.linalg.cholesky(inverse)


def covariance_from_whitener(whitener: np.ndarray) -> np.ndarray:
    precision = whitener @ np.swapaxes(whitener, -1, -2)
    result = np.linalg.solve(precision, np.broadcast_to(np.eye(precision.shape[-1]), precision.shape))
    return (result + np.swapaxes(result, -1, -2)) / 2

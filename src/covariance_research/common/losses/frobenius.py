"""Explicit matrix-distance losses and offline future-risk target construction.

These helpers do not change the frozen NLL training or forecasting pipelines.
No normalization, epsilon, symmetry projection, or PSD repair is implicit.
"""

from __future__ import annotations

import numpy as np
import torch


def _validate_matrix_shapes(prediction_shape, target_shape):
    if (prediction_shape != target_shape or len(prediction_shape) < 2
            or prediction_shape[-1] == 0 or prediction_shape[-2] != prediction_shape[-1]):
        raise ValueError("Prediction and target need identical (..., N, N) shapes with N > 0.")


def _validate_reduction(reduction):
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'.")


def frobenius_covariance_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    squared: bool = False,
    reduction: str = "none",
) -> torch.Tensor:
    """Differentiable full-matrix Frobenius error, unsquared by default.

    Reductions act on per-matrix losses, never on individual matrix elements.
    `mean` computes mean(norm(E_t)), not norm(mean(E_t)) or sqrt(mean(E_t**2)).
    Equal prediction and target have exactly zero loss and zero chosen subgradient.
    """
    _validate_reduction(reduction)
    _validate_matrix_shapes(prediction.shape, target.shape)
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError("Use real floating-point tensors.")
    if prediction.dtype != target.dtype or prediction.device != target.device:
        raise ValueError("Prediction and target must share dtype and device.")
    if prediction.numel() == 0 or not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Prediction and target must be nonempty and finite.")
    error = prediction - target
    if squared:
        values = error.square().sum(dim=(-2, -1))
    else:
        # matrix_norm supplies a finite zero subgradient, unlike sqrt(sum(E**2)).
        values = torch.linalg.matrix_norm(error, ord="fro", dim=(-2, -1))
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    return values


def frobenius_covariance_error(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    squared: bool = False,
    reduction: str = "none",
):
    """NumPy counterpart with the same full-matrix and batch conventions."""
    _validate_reduction(reduction)
    prediction, target = np.asarray(prediction), np.asarray(target)
    _validate_matrix_shapes(prediction.shape, target.shape)
    if np.iscomplexobj(prediction) or np.iscomplexobj(target):
        raise ValueError("Use real arrays.")
    prediction, target = prediction.astype(np.float64), target.astype(np.float64)
    if prediction.size == 0 or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Prediction and target must be nonempty and finite.")
    error = prediction - target
    values = (np.square(error).sum(axis=(-2, -1)) if squared
              else np.linalg.norm(error, ord="fro", axis=(-2, -1)))
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    return values


def future_risk_targets(
    returns: np.ndarray,
    origins: np.ndarray,
    horizon: int = 20,
    *,
    center: bool = False,
    ddof: int = 0,
) -> np.ndarray:
    """Construct offline labels from t+1 through t+H, with explicit conventions.

    center=False, ddof=0 matches the project's frozen second-moment target.
    center=True, ddof=1 uses the conventional centered sample covariance.
    This helper never fills missing returns, shortens horizons, or repairs targets.
    Training/validation boundaries and label maturity remain the caller's job.
    """
    if (not isinstance(horizon, (int, np.integer)) or isinstance(horizon, (bool, np.bool_))
            or horizon <= 0):
        raise ValueError("horizon must be a positive integer.")
    if (not isinstance(ddof, (int, np.integer)) or isinstance(ddof, (bool, np.bool_))
            or not 0 <= ddof < horizon):
        raise ValueError("ddof must be an integer in [0, horizon).")
    if not isinstance(center, (bool, np.bool_)):
        raise ValueError("center must be an explicit boolean.")
    if not center and ddof != 0:
        raise ValueError("The uncentered second-moment convention requires ddof=0.")
    returns, origins = np.asarray(returns), np.asarray(origins)
    if (returns.ndim != 2 or returns.shape[1] == 0 or np.iscomplexobj(returns)
            or not np.isfinite(returns).all()):
        raise ValueError("Need a finite real (T, N) return matrix.")
    if (origins.ndim != 1 or origins.size == 0
            or not np.issubdtype(origins.dtype, np.integer)):
        raise ValueError("origins must be a nonempty one-dimensional integer array.")
    if np.any(origins < 0) or np.any(origins >= len(returns) - horizon):
        raise ValueError("Every origin must have a complete future horizon.")
    returns = returns.astype(np.float64)
    windows = returns[origins[:, None] + np.arange(1, horizon + 1)]
    if center:
        windows = windows - windows.mean(axis=1, keepdims=True)
    return np.einsum("bhi,bhj->bij", windows, windows) / (horizon - ddof)

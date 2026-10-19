"""Leakage-safe historical forecast reliability features."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def matured_forecast_end(current_index: int, horizon: int) -> int:
    """Return the exclusive upper bound for forecasts fully observed at current_index."""
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if current_index < 0:
        raise ValueError("current_index must be non-negative.")
    return max(0, current_index - horizon + 1)


def relative_recent_loss(
    losses: NDArray[np.float64],
    *,
    current_index: int,
    horizon: int,
    reliability_window: int,
) -> NDArray[np.float64]:
    """Average only matured historical losses and center across estimators."""
    if reliability_window <= 0:
        raise ValueError("reliability_window must be positive.")
    stop = matured_forecast_end(current_index, horizon)
    if stop > len(losses):
        raise ValueError("Loss history does not cover all matured forecasts.")
    start = max(0, stop - reliability_window)
    if start == stop:
        raise ValueError("No matured historical losses are available.")
    recent = losses[start:stop].mean(axis=0)
    if not np.isfinite(recent).all():
        raise ValueError("Historical losses must be finite.")
    return recent - recent.mean()


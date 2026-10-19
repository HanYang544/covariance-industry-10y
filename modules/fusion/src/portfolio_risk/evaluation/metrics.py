"""Portfolio performance metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def annualized_volatility(daily_returns: NDArray[np.float64]) -> float:
    return float(np.std(daily_returns, ddof=1) * np.sqrt(252.0))


def maximum_drawdown(daily_returns: NDArray[np.float64]) -> float:
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + daily_returns)))
    peaks = np.maximum.accumulate(wealth)
    return float(np.max(1.0 - wealth / peaks))


def empirical_cvar(daily_returns: NDArray[np.float64], confidence: float = 0.95) -> float:
    losses = -np.asarray(daily_returns)
    threshold = np.quantile(losses, confidence)
    return float(losses[losses >= threshold].mean())


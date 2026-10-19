"""Past-only market risk-state features; future labels are never accepted here."""

from __future__ import annotations

import numpy as np

from portfolio_risk.experiment import FEATURE_NAMES, market_features

VOL_WINDOWS = (5, 10, 20, 60, 120, 252)
EXPANDED_MARKET_NAMES = FEATURE_NAMES[:6] + [
    *[f"log_variance_{w}" for w in VOL_WINDOWS],
    "log_variance_ratio_5_20", "log_variance_ratio_10_60", "log_variance_ratio_60_252",
    "downside_share_20", "downside_share_60", "negative_market_frequency_20",
    "market_skewness_20", "market_excess_kurtosis_20", "market_tail_ratio_20",
    "market_tail_ratio_60", "mean_correlation_60", "mean_correlation_120",
    "correlation_dispersion_20", "correlation_dispersion_60", "common_risk_concentration_60",
    "market_momentum_5", "market_momentum_20", "market_momentum_60",
    "market_drawdown_20", "market_drawdown_60", "cross_sectional_dispersion_20",
    "cross_sectional_volatility_dispersion_20", "covariance_trace_ratio_20_252",
    "covariance_structure_change_20_60", "covariance_effective_rank_20",
    "covariance_condition_log_20", "short_volatility_of_volatility",
]


def expanded_market_features(history: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    """Return 39 risk-state descriptors using <=252 past trading days.

    Variance means uncentered second moment, consistent with the forecast target.
    Equal-weighted industry returns describe a market proxy, not an external index.
    Small denominator floors ensure numeric safety; actual returns are not clipped.
    """
    if len(history) < 252 or matrices.shape[0] != 4:
        raise ValueError("Expanded features require 252 past days and four covariance scales.")
    tiny = 1e-16
    short, medium = history[-20:], history[-60:]
    market = history.mean(axis=1)
    variance = {w: max(float(np.mean(history[-w:] ** 2)), tiny) for w in VOL_WINDOWS}
    correlation = {w: np.corrcoef(history[-w:], rowvar=False) for w in (20, 60, 120)}
    upper = np.triu_indices(history.shape[1], 1)
    m20 = market[-20:]
    centered = m20 - m20.mean()
    std = max(float(np.sqrt(np.mean(centered ** 2))), np.sqrt(tiny))

    def downside(block):
        return float(np.mean(np.minimum(block, 0) ** 2) / max(np.mean(block ** 2), tiny))

    def tail(w):
        block = market[-w:]
        return float(np.quantile(np.abs(block), 0.95) / max(np.sqrt(np.mean(block ** 2)), np.sqrt(tiny)))

    def drawdown(w):
        wealth = np.r_[1.0, np.cumprod(1 + market[-w:])]
        return float(np.min(wealth / np.maximum.accumulate(wealth) - 1))

    eig = np.linalg.eigvalsh(matrices[0])
    shares = eig / eig.sum()
    daily_variance = np.mean(short ** 2, axis=1)
    result = np.r_[market_features(history[-120:], matrices),
        [np.log(variance[w]) for w in VOL_WINDOWS],
        np.log(variance[5] / variance[20]), np.log(variance[10] / variance[60]),
        np.log(variance[60] / variance[252]), downside(short), downside(medium),
        np.mean(m20 < 0), np.mean(centered ** 3) / std ** 3,
        np.mean(centered ** 4) / std ** 4 - 3, tail(20), tail(60),
        correlation[60][upper].mean(), correlation[120][upper].mean(),
        correlation[20][upper].std(), correlation[60][upper].std(),
        np.linalg.eigvalsh(correlation[60])[-1] / history.shape[1],
        [market[-w:].sum() for w in (5, 20, 60)], drawdown(20), drawdown(60),
        np.sqrt(np.mean(np.var(short, axis=1))), np.std(np.sqrt(np.mean(short ** 2, axis=0))),
        np.log(np.trace(matrices[0]) / np.trace(matrices[3])),
        np.linalg.norm(matrices[0] - matrices[1]) / np.linalg.norm(matrices[1]),
        np.exp(-np.sum(shares * np.log(shares))) / history.shape[1],
        np.log(eig[-1] / eig[0]), np.std(np.log(np.maximum(daily_variance, tiny)))
    ]
    if len(result) != len(EXPANDED_MARKET_NAMES) or not np.isfinite(result).all():
        raise ValueError("Nonfinite or misaligned expanded features.")
    return result


def matured_reliability(losses: np.ndarray, t: int, horizon: int, window: int) -> np.ndarray:
    """Exactly window labels ending no later than t: origins t-H-M+1,...,t-H."""
    stop = t - horizon + 1
    start = stop - window
    if start < 0:
        raise ValueError("Insufficient matured labels.")
    recent = losses[start:stop].mean(axis=0)
    return recent - recent.mean()

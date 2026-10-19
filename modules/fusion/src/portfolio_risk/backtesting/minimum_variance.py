"""Long-only global minimum-variance portfolio optimization."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


def solve_long_only_gmv(
    covariance: NDArray[np.float64], initial: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Solve a fully invested, long-only minimum-variance portfolio."""
    asset_count = covariance.shape[0]
    scale = np.trace(covariance) / asset_count
    if not np.isfinite(covariance).all() or scale <= 0:
        raise ValueError("Covariance must be finite with positive average variance.")
    normalized = covariance / scale
    np.linalg.cholesky(normalized)
    equal = np.full(asset_count, 1 / asset_count)
    for starting in ([equal] if initial is None else [initial, equal]):
        result = minimize(
            lambda w: float(w @ normalized @ w), starting,
            jac=lambda w: 2 * normalized @ w,
            method="SLSQP", bounds=[(0, 1)] * asset_count,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                         "jac": lambda w: np.ones(asset_count)},
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if result.success and np.isclose(result.x.sum(), 1) and np.all(result.x >= -1e-10):
            weights = np.maximum(result.x, 0)
            return weights / weights.sum()
    raise RuntimeError(f"Portfolio optimization failed: {result.message}")


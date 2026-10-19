"""Six past-only dimensionless risk-state indicators; no neural dependencies."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

FEATURE_COLUMNS = ['volatility_ratio', 'average_correlation', 'absorption_ratio',
                   'effective_rank', 'subspace_rotation', 'correlation_drift']


class MarketRiskFeatures:
    def __init__(self, short_window: int = 20, long_window: int = 120,
                 top_k: int = 3, min_coverage: float = .95, min_assets: int = 30,
                 cov_method: str = 'ledoit_wolf', std_epsilon: float = 1e-8,
                 eigenvalue_epsilon: float = 1e-12):
        if not (2 <= short_window <= long_window and top_k >= 1 and min_assets >= 2
                and 0 < min_coverage <= 1 and std_epsilon > 0 and eigenvalue_epsilon > 0):
            raise ValueError('Invalid feature parameters')
        if cov_method != 'ledoit_wolf':
            raise ValueError('Only centered Ledoit-Wolf is supported')
        self.short_window, self.long_window, self.top_k = short_window, long_window, top_k
        self.min_coverage, self.min_assets = min_coverage, min_assets
        self.std_epsilon, self.eigenvalue_epsilon = std_epsilon, eigenvalue_epsilon

    def compute_covariance(self, returns: np.ndarray) -> tuple[np.ndarray, float]:
        model = LedoitWolf(assume_centered=False).fit(returns)
        return (model.covariance_ + model.covariance_.T) / 2, float(model.shrinkage_)

    def covariance_to_correlation(self, cov: np.ndarray) -> np.ndarray:
        sd = np.sqrt(np.maximum(np.diag(cov), self.std_epsilon ** 2))
        corr = np.clip(cov / np.outer(sd, sd), -1, 1)
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1)
        return corr

    def compute_volatility_ratio(self, short: np.ndarray, long: np.ndarray) -> float:
        return float(np.sqrt(np.trace(short) / max(np.trace(long), self.std_epsilon**2)))

    def compute_average_correlation(self, corr: np.ndarray) -> float:
        return float(corr[np.triu_indices(len(corr), 1)].mean())

    def compute_absorption_ratio(self, eigenvalues: np.ndarray) -> float:
        values = np.maximum(eigenvalues, 0)
        return float(np.sort(values)[-self.top_k:].sum() / values.sum()) if values.sum() > 0 else np.nan

    def compute_effective_rank(self, eigenvalues: np.ndarray) -> float:
        values = np.maximum(eigenvalues, 0)
        if values.sum() <= 0:
            return np.nan
        p = values / values.sum()
        return float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))) / len(p))

    def compute_subspace_rotation(self, short: np.ndarray, long: np.ndarray) -> float:
        return float(np.sqrt(np.clip(1 - np.linalg.norm(short.T @ long, 'fro')**2 / short.shape[1], 0, 1)))

    def compute_correlation_drift(self, short: np.ndarray, long: np.ndarray) -> float:
        return float(np.linalg.norm(short - long, 'fro') / len(short))

    def compute_at_date(self, returns: pd.DataFrame, date: object) -> tuple[np.ndarray, dict]:
        history = returns.loc[:date].tail(self.long_window)
        diag = dict(n_assets=0, asset_ids='[]', short_shrinkage=np.nan, long_shrinkage=np.nan,
                    short_total_variance=np.nan, long_total_variance=np.nan,
                    is_valid=False, invalid_reason='', subspace_boundary_degenerate=False)
        invalid = np.full(6, np.nan)
        if len(history) < self.long_window:
            diag['invalid_reason'] = 'insufficient_history'
            return invalid, diag
        x = history.to_numpy(dtype=np.float64)
        # Infinite observations indicate invalid assets; NaNs are allowed at low coverage loss.
        keep = (np.isfinite(x).mean(0) >= self.min_coverage) & ~np.isinf(x).any(0)
        x, names = x[:, keep], history.columns[keep]
        valid = []
        blocks = []
        for window in (self.short_window, self.long_window):
            block = x[-window:].copy()
            counts = np.isfinite(block).sum(0)
            means = np.divide(np.nansum(block, axis=0), counts, out=np.zeros(len(counts)), where=counts > 0)
            block = np.where(np.isnan(block), means, block)
            valid.append((counts > 0) & (block.std(0) > self.std_epsilon))
            blocks.append(block)
        keep = valid[0] & valid[1]
        short, long = (b[:, keep] for b in blocks)
        names = names[keep]
        n = len(names)
        diag.update(n_assets=n, asset_ids=json.dumps([str(v) for v in names]))
        if n < max(self.min_assets, self.top_k):
            diag['invalid_reason'] = 'insufficient_assets'
            return invalid, diag
        try:
            cs, ss = self.compute_covariance(short)
            cl, sl = self.compute_covariance(long)
            es, us = np.linalg.eigh(cs)
            el, ul = np.linalg.eigh(cl)
            if min(es.min(), el.min()) < -self.eigenvalue_epsilon:
                raise ValueError('material_negative_eigenvalue')
            es, el = np.maximum(es, 0), np.maximum(el, 0)
            if min(es.sum(), el.sum()) <= 0:
                raise ValueError('nonpositive_total_variance')
            rs, rl = self.covariance_to_correlation(cs), self.covariance_to_correlation(cl)
            values = np.array([self.compute_volatility_ratio(cs, cl), self.compute_average_correlation(rs),
                self.compute_absorption_ratio(el), self.compute_effective_rank(el),
                self.compute_subspace_rotation(us[:, -self.top_k:], ul[:, -self.top_k:]),
                self.compute_correlation_drift(rs, rl)])
            if not np.isfinite(values).all():
                raise ValueError('nonfinite_features')
            degenerate = n > self.top_k and any(e[-self.top_k] - e[-self.top_k-1] <= self.eigenvalue_epsilon * e.sum() for e in (es, el))
            diag.update(short_shrinkage=ss, long_shrinkage=sl, short_total_variance=float(es.sum()),
                        long_total_variance=float(el.sum()), is_valid=True, subspace_boundary_degenerate=bool(degenerate))
            return values, diag
        except (ValueError, np.linalg.LinAlgError) as error:
            diag['invalid_reason'] = str(error)
            return invalid, diag

    def transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        if not returns.index.is_unique or not returns.index.is_monotonic_increasing or not returns.columns.is_unique:
            raise ValueError('Unique sorted dates and unique asset IDs required')
        computed = [self.compute_at_date(returns, date) for date in returns.index]
        self.diagnostics_ = pd.DataFrame([d for _, d in computed], index=returns.index)
        return pd.DataFrame([v for v, _ in computed], index=returns.index, columns=FEATURE_COLUMNS)

    @staticmethod
    def standardize(features: pd.DataFrame, train_dates: pd.Index) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        train = features.loc[train_dates]
        if train.empty or not np.isfinite(train.to_numpy()).all():
            raise ValueError('Training features must be finite and nonempty')
        mean, scale = train.mean(), train.std(ddof=0).clip(lower=1e-12)
        return ((features - mean) / scale).clip(-3, 3), mean, scale

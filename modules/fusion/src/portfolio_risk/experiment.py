"""Reproducible first-round industry-10 experiment with explicit temporal boundaries."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import softmax

from portfolio_risk.config import load_config
from portfolio_risk.backtesting.minimum_variance import solve_long_only_gmv
from portfolio_risk.covariance.estimators import ledoit_wolf_covariance
from portfolio_risk.data.french import acquire_industry_10, parse_industry_zip, validate_return_matrix
from portfolio_risk.evaluation.metrics import annualized_volatility, empirical_cvar, maximum_drawdown
from portfolio_risk.losses.gaussian import gaussian_covariance_nll
from portfolio_risk.models.gating import LinearGate, MLPGate
from portfolio_risk.utils.random import set_random_seed

FEATURE_NAMES = [
    "log_short_long_volatility_ratio", "log_short_variance", "short_mean_correlation",
    "short_long_correlation_change", "covariance_structure_change", "common_risk_concentration",
    "relative_loss_20", "relative_loss_60", "relative_loss_120", "relative_loss_252",
]


def covariance_nll(covariance: np.ndarray, realized: np.ndarray) -> np.ndarray:
    """Normalized zero-mean NLL, omitting the same constant for every method."""
    cholesky = np.linalg.cholesky(covariance)
    logdet = 2 * np.log(np.diagonal(cholesky, axis1=-2, axis2=-1)).sum(axis=-1)
    solved = np.linalg.solve(covariance, realized)
    trace = np.trace(solved, axis1=-2, axis2=-1)
    return (logdet + trace) / (2 * covariance.shape[-1])


def split_prediction_indices(
    dates: pd.DatetimeIndex, start: str, end: str, horizon: int, stride: int,
    eligible: np.ndarray,
) -> np.ndarray:
    """Keep decision dates AND complete future labels within one split."""
    candidates = np.flatnonzero((dates >= start) & (dates <= end) & eligible)
    candidates = candidates[candidates + horizon < len(dates)]
    candidates = candidates[dates[candidates + horizon] <= pd.Timestamp(end)]
    return candidates[::stride]


def market_features(returns: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    short, long = returns[-20:], returns[-120:]
    short_variance = np.mean(short ** 2)
    long_variance = np.mean(long ** 2)
    short_correlation = np.corrcoef(short, rowvar=False)
    long_correlation = np.corrcoef(long, rowvar=False)
    upper = np.triu_indices(returns.shape[1], 1)
    return np.array([
        0.5 * np.log(short_variance / long_variance), np.log(short_variance),
        short_correlation[upper].mean(),
        short_correlation[upper].mean() - long_correlation[upper].mean(),
        np.linalg.norm(matrices[0] - matrices[2]) / np.linalg.norm(matrices[2]),
        np.linalg.eigvalsh(short_correlation)[-1] / returns.shape[1],
    ])


def build_arrays(frame: pd.DataFrame, config: dict) -> dict:
    returns = frame.to_numpy()
    count, dimension = returns.shape
    windows = config["forecast"]["lookback_windows"]
    horizon = config["forecast"]["horizon"]
    reliability_window = config["features"]["reliability_window"]
    first = max(windows) - 1
    covariance = np.full((count, len(windows), dimension, dimension), np.nan)
    realized = np.full((count, dimension, dimension), np.nan)
    losses = np.full((count, len(windows)), np.nan)
    features = np.full((count, 10), np.nan)
    ewma = {}
    for decay in config["evaluation"]["ewma_decays"]:
        series = np.empty((count, dimension, dimension))
        state = np.outer(returns[0], returns[0])
        for t, row in enumerate(returns):
            state = decay * state + (1 - decay) * np.outer(row, row)
            series[t] = state + np.eye(dimension) * (
                np.trace(state) / dimension * config["covariance"]["numerical_epsilon"]
            )
        ewma[decay] = series
    for t in range(first, count):
        for k, window in enumerate(windows):
            estimate = ledoit_wolf_covariance(returns[t - window + 1:t + 1])
            jitter = np.trace(estimate) / dimension * config["covariance"]["numerical_epsilon"]
            covariance[t, k] = estimate + np.eye(dimension) * jitter
        features[t, :6] = market_features(returns[max(0, t - 119):t + 1], covariance[t])
        if t + horizon < count:
            future = returns[t + 1:t + horizon + 1]
            realized[t] = future.T @ future / horizon
            losses[t] = covariance_nll(covariance[t], realized[t])
        # Include s=t-H; its last required return has just become observable at t.
        stop = t - horizon + 1
        start = stop - reliability_window
        if start >= first:
            recent = losses[start:stop].mean(axis=0)
            features[t, 6:] = recent - recent.mean()
        if (t - first) % 1000 == 0:
            print(f"Covariance and features: {t + 1}/{count} days", flush=True)
    eligible = np.isfinite(features).all(axis=1) & np.isfinite(realized).all(axis=(1, 2))
    splits = {
        name: split_prediction_indices(
            frame.index, bounds[0], bounds[1], horizon,
            config["forecast"][f"{name}_stride"], eligible,
        )
        for name, bounds in config["data"]["splits"].items()
    }
    for name, indices in splits.items():
        if not len(indices):
            raise ValueError(f"No valid samples for {name}.")
    np.linalg.cholesky(covariance[first:])
    return {
        "covariance": covariance, "realized": realized, "features": features,
        "losses": losses, "splits": splits, "ewma": ewma,
    }


def train_gate(
    arrays: dict, standardized: np.ndarray, columns: list[int], architecture: str,
    seed: int, config: dict, output: Path,
) -> tuple[np.ndarray, float, list[dict]]:
    set_random_seed(seed)
    model_config = config["model"]
    scale_count = arrays["covariance"].shape[1]
    model = (LinearGate(len(columns), output_dim=scale_count) if architecture == "linear" else
             MLPGate(len(columns), hidden_dim=model_config["hidden_dim"],
                     output_dim=scale_count)).double()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=model_config["learning_rate"],
        weight_decay=model_config["weight_decay"],
    )
    train, validation = arrays["splits"]["train"], arrays["splits"]["validation"]
    x = torch.from_numpy(standardized[:, columns])
    covariance = torch.from_numpy(arrays["covariance"])
    realized = torch.from_numpy(arrays["realized"])
    best_score, best_state, stale = float("inf"), None, 0
    history = []
    rng = np.random.default_rng(seed)
    for epoch in range(1, model_config["max_epochs"] + 1):
        model.train()
        shuffled = rng.permutation(train)
        for begin in range(0, len(train), model_config["batch_size"]):
            indices = shuffled[begin:begin + model_config["batch_size"]]
            weights = model(x[indices])
            fused = torch.einsum("bk,bkij->bij", weights, covariance[indices])
            loss = gaussian_covariance_nll(fused, realized[indices]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            weights = model(x[validation])
            fused = torch.einsum("bk,bkij->bij", weights, covariance[validation])
            score = gaussian_covariance_nll(fused, realized[validation]).mean().item()
            training_weights = model(x[train])
            training_fused = torch.einsum("bk,bkij->bij", training_weights, covariance[train])
            training_score = gaussian_covariance_nll(training_fused, realized[train]).mean().item()
        history.append({"epoch": epoch, "train_nll": training_score, "validation_nll": score})
        if score < best_score - 1e-7:
            best_score, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= model_config["early_stopping_patience"]:
            break
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "columns": columns, "architecture": architecture,
                "seed": seed}, output)
    with torch.no_grad():
        weights = model(x).numpy()
    return weights, best_score, history


def fit_fixed_fusion(covariances: np.ndarray, realized: np.ndarray) -> np.ndarray:
    """Fit a feature-free comparator using training labels only."""
    def objective(weights):
        fused = np.einsum("k,bkij->bij", weights, covariances)
        value = covariance_nll(fused, realized).mean()
        inverse = np.linalg.solve(fused, np.broadcast_to(np.eye(fused.shape[-1]), fused.shape))
        derivative = inverse - inverse @ realized @ inverse
        gradient = np.einsum("bij,bkji->bk", derivative, covariances).mean(0)
        gradient /= 2 * fused.shape[-1]
        return value, gradient
    candidates = []
    scale_count = covariances.shape[1]
    for initial in [np.full(scale_count, 1 / scale_count), *np.eye(scale_count)]:
        result = minimize(
            objective, initial, method="SLSQP", jac=True,
            bounds=[(0.0, 1.0)] * scale_count,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            options={"maxiter": 200, "ftol": 1e-10},
        )
        if result.success:
            candidates.append(result)
    if not candidates:
        raise RuntimeError("All fixed-fusion optimization starts failed.")
    return min(candidates, key=lambda item: item.fun).x


def paired_block_bootstrap(
    difference: np.ndarray, block_length: int, repetitions: int, seed: int,
) -> tuple[float, float, float]:
    """Circular moving blocks preserve paired observations and local time dependence."""
    rng = np.random.default_rng(seed)
    size = len(difference)
    starts = rng.integers(0, size, (repetitions, int(np.ceil(size / block_length))))
    indices = (starts[:, :, None] + np.arange(block_length)) % size
    indices = indices.reshape(repetitions, -1)[:, :size]
    means = difference[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    # Center under the null; one-sided alternative is positive improvement.
    pvalue = (1 + np.sum(means - difference.mean() >= difference.mean())) / (repetitions + 1)
    return float(lower), float(upper), float(pvalue)


def holm_adjust(pvalues: list[float]) -> np.ndarray:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(order))
    sorted_adjusted = np.maximum.accumulate(
        np.asarray(pvalues)[order] * (len(order) - np.arange(len(order)))
    )
    adjusted[order] = np.minimum(1, sorted_adjusted)
    return adjusted


def run_gmv_backtest(
    returns: np.ndarray, decisions: np.ndarray, covariances: np.ndarray,
    horizon: int, lag: int, cost_bps: float,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Non-overlapping buy-and-hold intervals; turnover uses drifted holdings."""
    asset_count = returns.shape[1]
    previous = np.full(asset_count, 1 / asset_count)
    observations, dates, turnovers = [], [], []
    for t, covariance in zip(decisions, covariances, strict=True):
        weights = solve_long_only_gmv(covariance, initial=previous)
        trading = np.abs(weights - previous).sum()
        turnovers.append(float(trading / 2))
        cost = cost_bps / 10000 * trading
        for offset, day in enumerate(range(t + 1 + lag, t + 1 + lag + horizon)):
            gross = float(weights @ returns[day])
            net = (1 - cost) * (1 + gross) - 1 if offset == 0 else gross
            observations.append(net)
            dates.append(day)
            weights = weights * (1 + returns[day]) / (1 + gross)
        previous = weights
    return np.asarray(observations), np.asarray(dates), turnovers


def write_figures(summary: pd.DataFrame, weights: dict, frame: pd.DataFrame,
                  test: np.ndarray, primary: str, yearly: pd.DataFrame, output: Path,
                  bootstrap: pd.DataFrame | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = summary.sort_values("test_nll")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#c15d35" if name == primary else "#437ab0" for name in ordered.method]
    ax.scatter(ordered.test_nll, ordered.method, c=colors, s=60)
    for row in ordered.itertuples():
        ax.annotate(f"{row.test_nll:.5f}", (row.test_nll, row.method),
                    xytext=(7, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_xlabel("Test Gaussian NLL per asset (lower is better; constant omitted)")
    ax.set_xlim(ordered.test_nll.min() - 0.01, ordered.test_nll.max() + 0.025)
    ax.set_title(f"Industry 10: validation-selected primary model = {primary}")
    ax.grid(axis="x", alpha=0.2)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output / "nll_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    rolling = frame.pow(2).mean(axis=1).rolling(20).mean().pow(0.5) * np.sqrt(252)
    axes[0].plot(frame.index[test], rolling.iloc[test], color="#333333")
    axes[0].set_ylabel("Market RMS risk (annualized)")
    axes[1].stackplot(frame.index[test], weights[primary].T, labels=["20", "60", "120", "252"])
    axes[1].set_ylabel("Mean gate weight across seeds")
    axes[1].legend(title="Window", loc="upper left", ncol=4)
    fig.tight_layout()
    fig.savefig(output / "dynamic_weights.png", dpi=180)
    plt.close(fig)
    yearly.pivot(index="year", columns="method", values="nll").plot(figsize=(11, 5))
    plt.ylabel("Mean test NLL")
    plt.tight_layout()
    plt.savefig(output / "yearly_nll.png", dpi=180)
    plt.close("all")
    if bootstrap is not None:
        selected = bootstrap[bootstrap.block_length == 3]
        fig, ax = plt.subplots(figsize=(10, 5))
        y = np.arange(len(selected))
        ax.errorbar(selected.improvement, y, xerr=[selected.improvement - selected.ci_lower,
                    selected.ci_upper - selected.improvement], fmt="o", capsize=4,
                    color="#437ab0")
        ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
        ax.set_yticks(y, selected.baseline)
        ax.invert_yaxis()
        ax.set_xlabel("Baseline NLL minus joint NLL (positive favors joint)")
        ax.set_title("Paired block bootstrap: pointwise 95% CI; 3-interval blocks")
        fig.tight_layout()
        fig.savefig(output / "nll_improvement.png", dpi=180)
        plt.close(fig)


def run_preliminary(config_path: Path) -> None:
    config = load_config(config_path)
    output = Path(config["project"]["output_dir"])
    for directory in [output, output / "checkpoints", output / "figures"]:
        directory.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    archive, manifest = acquire_industry_10(Path(config["data"]["raw_dir"]))
    full_frame, source_header = parse_industry_zip(archive)
    frame = full_frame.loc[config["data"]["history_start"]:config["data"]["splits"]["test"][1]]
    validate_return_matrix(frame)
    processed = Path(config["data"]["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    frame.to_csv(processed / "industry_10_returns.csv")
    quality = {
        "rows": len(frame), "assets": list(frame.columns),
        "first_date": str(frame.index[0].date()), "last_date": str(frame.index[-1].date()),
        "missing_values": int(frame.isna().sum().sum()), "duplicate_dates": 0,
        "source_header": source_header, "manifest": manifest,
    }
    (output / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(f"Validated {len(frame)} trading days across {frame.shape[1]} assets.", flush=True)
    arrays = build_arrays(frame, config)
    splits = arrays["splits"]
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    horizon = config["forecast"]["horizon"]
    for name, indices in splits.items():
        pd.DataFrame({
            "decision_date": frame.index[indices], "target_start": frame.index[indices + 1],
            "target_end": frame.index[indices + horizon],
            "latest_reliability_forecast": frame.index[indices - horizon],
            "latest_reliability_target_end": frame.index[indices],
        }).to_csv(output / f"{name}_dates.csv", index=False)
    mean = arrays["features"][train].mean(0)
    scale = arrays["features"][train].std(0)
    scale = np.maximum(scale, 1e-12)
    standardized = (arrays["features"] - mean) / scale
    np.savez(output / "feature_scaler.npz", mean=mean, scale=scale, names=FEATURE_NAMES)
    matrices, realized = arrays["covariance"], arrays["realized"]
    validation_losses, test_losses, test_covariances, all_weights = {}, {}, {}, {}
    for k, window in enumerate(config["forecast"]["lookback_windows"]):
        name = f"LW-{window}"
        validation_losses[name] = arrays["losses"][validation, k]
        test_losses[name] = arrays["losses"][test, k]
        test_covariances[name] = matrices[test, k]
    for decay, covariance in arrays["ewma"].items():
        name = f"EWMA-{decay}"
        validation_losses[name] = covariance_nll(covariance[validation], realized[validation])
        test_losses[name] = covariance_nll(covariance[test], realized[test])
        test_covariances[name] = covariance[test]
    baseline_weights = {
        "EqualFusion": np.full((len(frame), 4), 0.25),
        "FixedFusion": np.broadcast_to(fit_fixed_fusion(matrices[train], realized[train]),
                                      (len(frame), 4)),
    }
    scores = []
    for temperature in config["evaluation"]["reliability_temperatures"]:
        weight = softmax(-arrays["features"][validation, 6:] / temperature, axis=-1)
        score = covariance_nll(np.einsum("bk,bkij->bij", weight, matrices[validation]),
                               realized[validation]).mean()
        scores.append((score, temperature))
    temperature = min(scores)[1]
    baseline_weights["HistoricalLoss"] = softmax(-arrays["features"][:, 6:] / temperature, axis=-1)
    for name, weights in baseline_weights.items():
        validation_losses[name] = covariance_nll(
            np.einsum("bk,bkij->bij", weights[validation], matrices[validation]), realized[validation]
        )
        test_covariances[name] = np.einsum("bk,bkij->bij", weights[test], matrices[test])
        test_losses[name] = covariance_nll(test_covariances[name], realized[test])
        all_weights[name] = weights[test]
    training_records, seed_records = [], []
    columns_by_mode = {"Market": list(range(6)), "Reliability": list(range(6, 10)),
                       "Joint": list(range(10))}
    for architecture in config["model"]["architectures"]:
        for mode, columns in columns_by_mode.items():
            name = f"{architecture}-{mode}"
            seed_losses, seed_weights, seed_validation = [], [], []
            for seed in config["evaluation"]["random_seeds"]:
                weights, score, history = train_gate(
                    arrays, standardized, columns, architecture, seed, config,
                    output / "checkpoints" / f"{name}_seed{seed}.pt",
                )
                predicted = np.einsum("bk,bkij->bij", weights[test], matrices[test])
                loss = covariance_nll(predicted, realized[test])
                seed_losses.append(loss)
                seed_weights.append(weights[test])
                seed_validation.append(score)
                seed_records.append({"method": name, "seed": seed,
                                     "validation_nll": score, "test_nll": loss.mean()})
                for record in history:
                    training_records.append({"method": name, "seed": seed, **record})
                print(f"{name} seed={seed}: best validation={score:.6f}, epochs={len(history)}",
                      flush=True)
            # Mean per-origin loss across seeds is NOT the loss of an ensemble covariance.
            test_losses[name] = np.mean(seed_losses, axis=0)
            validation_losses[name] = np.asarray(seed_validation)
            all_weights[name] = np.mean(seed_weights, axis=0)
            test_covariances[name] = np.einsum("bk,bkij->bij", all_weights[name], matrices[test])
            pd.DataFrame(np.asarray(seed_losses).T, index=frame.index[test],
                         columns=[f"seed_{s}" for s in config["evaluation"]["random_seeds"]]
                         ).to_csv(output / f"{name}_seed_losses.csv")
    # Freeze the primary architecture using validation mean, before inference comparisons.
    primary = min([f"{a}-Joint" for a in config["model"]["architectures"]],
                  key=lambda name: validation_losses[name].mean())
    architecture = primary.split("-")[0]
    best_single = min([f"LW-{w}" for w in config["forecast"]["lookback_windows"]],
                      key=lambda name: validation_losses[name].mean())
    selected_ewma = min([name for name in test_losses if name.startswith("EWMA-")],
                        key=lambda name: validation_losses[name].mean())
    summary = pd.DataFrame([
        {"method": name, "validation_nll": validation_losses[name].mean(),
         "test_nll": loss.mean(), "test_seed_std": float(np.std(
             [r["test_nll"] for r in seed_records if r["method"] == name], ddof=1
         )) if "-Joint" in name or "-Market" in name or "-Reliability" in name else 0.0}
        for name, loss in test_losses.items()
    ]).sort_values("test_nll")
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(test_losses, index=frame.index[test]).rename_axis("decision_date").to_csv(
        output / "test_losses.csv"
    )
    pd.DataFrame(seed_records).to_csv(output / "seed_summary.csv", index=False)
    pd.DataFrame(training_records).to_csv(output / "training_history.csv", index=False)
    comparisons = [best_single, selected_ewma, "EqualFusion", "FixedFusion", "HistoricalLoss",
                   f"{architecture}-Market", f"{architecture}-Reliability"]
    bootstrap_records = []
    for block in config["evaluation"]["bootstrap_block_lengths"]:
        records = []
        for baseline in comparisons:
            difference = test_losses[baseline] - test_losses[primary]
            low, high, pvalue = paired_block_bootstrap(
                difference, block, config["evaluation"]["bootstrap_repetitions"],
                config["evaluation"]["bootstrap_seed"],
            )
            records.append({"baseline": baseline, "primary": primary, "block_length": block,
                            "improvement": difference.mean(), "ci_lower": low,
                            "ci_upper": high, "p_one_sided": pvalue})
        corrected = holm_adjust([r["p_one_sided"] for r in records])
        for record, pvalue in zip(records, corrected, strict=True):
            record["p_holm"] = pvalue
        bootstrap_records.extend(records)
    bootstrap = pd.DataFrame(bootstrap_records)
    bootstrap.to_csv(output / "paired_bootstrap.csv", index=False)
    yearly = pd.DataFrame([
        {"year": year, "method": name, "nll": loss[frame.index[test].year == year].mean(),
         "intervals": int(np.sum(frame.index[test].year == year))}
        for name, loss in test_losses.items() for year in range(2021, 2026)
    ])
    yearly.to_csv(output / "yearly_nll.csv", index=False)
    for name, weights in all_weights.items():
        pd.DataFrame(weights, index=frame.index[test], columns=["20", "60", "120", "252"]
                     ).to_csv(output / f"{name}_weights.csv")
    backtest_records = []
    # A lagged portfolio has shifted return dates. Compare methods only within the same lag.
    for lag in config["backtest"]["execution_lag_days"]:
        valid = test + horizon + lag < len(frame)
        for cost in config["backtest"]["transaction_cost_bps"]:
            for name in [best_single, selected_ewma, "EqualFusion", "FixedFusion",
                         "HistoricalLoss", primary]:
                daily, dates, turnover = run_gmv_backtest(
                    frame.to_numpy(), test[valid], test_covariances[name][valid], horizon, lag, cost
                )
                backtest_records.append({
                    "method": name, "lag_days": lag, "cost_bps": cost,
                    "annualized_volatility": annualized_volatility(daily),
                    "maximum_drawdown": maximum_drawdown(daily), "cvar_95": empirical_cvar(daily),
                    "mean_turnover": np.mean(turnover), "observations": len(daily),
                    "cumulative_return": float(np.prod(1 + daily) - 1),
                    "first_return_date": str(frame.index[dates[0]].date()),
                    "last_return_date": str(frame.index[dates[-1]].date()),
                    "gate_policy": "mean_seed_weights" if name == primary else "single_policy",
                })
                if cost == 0:
                    pd.DataFrame({"date": frame.index[dates], "return": daily}).to_csv(
                        output / f"backtest_{name}_lag{lag}.csv", index=False
                    )
    backtests = pd.DataFrame(backtest_records)
    backtests.to_csv(output / "backtest_summary.csv", index=False)
    # Regimes use training-only quantiles and dates observable at the forecast origin.
    low, high = np.quantile(arrays["features"][train, 1], [0.25, 0.75])
    regimes = np.where(arrays["features"][test, 1] < low, "low",
                       np.where(arrays["features"][test, 1] > high, "high", "middle"))
    regime_records = [
        {"regime": regime, "method": name, "nll": loss[regimes == regime].mean(),
         "intervals": int(np.sum(regimes == regime))}
        for name, loss in test_losses.items() for regime in ["low", "middle", "high"]
        if np.any(regimes == regime)
    ]
    pd.DataFrame(regime_records).to_csv(output / "regime_nll.csv", index=False)
    write_figures(summary, all_weights, frame, test, primary, yearly[
        yearly.method.isin([best_single, "FixedFusion", "HistoricalLoss", primary])
    ], output / "figures", bootstrap)
    metadata = {
        "primary_model": primary, "best_single_validation": best_single,
        "selected_ewma": selected_ewma, "historical_loss_temperature": temperature,
        "fixed_fusion_weights": baseline_weights["FixedFusion"][0].tolist(),
        "split_samples": {name: len(indices) for name, indices in splits.items()},
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "torch": torch.__version__, "source_sha256": manifest["sha256"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "regime_training_thresholds": [float(low), float(high)],
        "nll_policy": "mean per-origin NLL across 5 independently trained seeds",
        "backtest_policy": "mean softmax weights across 5 seeds; no test-based seed selection",
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_chinese_report(output, metadata, summary, bootstrap, yearly, backtests, quality)
    print(summary.to_string(index=False), flush=True)
    print(f"Experiment complete. Primary model chosen on validation: {primary}", flush=True)


def write_chinese_report(output: Path, metadata: dict, summary: pd.DataFrame,
                         bootstrap: pd.DataFrame, yearly: pd.DataFrame,
                         backtests: pd.DataFrame, quality: dict) -> None:
    def table(frame):
        header = "| " + " | ".join(frame.columns) + " |"
        separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
        rows = []
        for row in frame.itertuples(index=False, name=None):
            rows.append("| " + " | ".join(
                f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value)
                for value in row
            ) + " |")
        return "\n".join([header, separator, *rows])
    primary = metadata["primary_model"]
    architecture = primary.split("-")[0]
    core = bootstrap[bootstrap.block_length == 3]
    market = core[core.baseline == f"{architecture}-Market"].iloc[0]
    reliability = core[core.baseline == f"{architecture}-Reliability"].iloc[0]
    fixed = core[core.baseline == "FixedFusion"].iloc[0]
    complementary = (market.ci_lower > 0 and reliability.ci_lower > 0 and
                     market.p_holm < 0.05 and reliability.p_holm < 0.05)
    lines = [
        "# 10 行业数据初步实验报告", "",
        "## 本轮结论", "",
        f"验证集选择的联合模型为 `{primary}`。NLL 越低越好；省略高斯常数后负值是正常现象。",
        ("本轮两类信息互补性得到初步统计支持。" if complementary else
         "本轮尚不足以支持“市场状态与历史可靠性具有互补性”的核心假设。"),
        f"联合模型相对仅市场状态模型的 NLL 改善为 {market.improvement:.6f}，"
        f"95% 区间 [{market.ci_lower:.6f}, {market.ci_upper:.6f}]；"
        f"相对仅可靠性模型的改善为 {reliability.improvement:.6f}，"
        f"95% 区间 [{reliability.ci_lower:.6f}, {reliability.ci_upper:.6f}]。",
        f"相对固定融合的平均 NLL 改善为 {fixed.improvement:.6f}，"
        f"95% 时间块 Bootstrap 区间为 [{fixed.ci_lower:.6f}, {fixed.ci_upper:.6f}]。",
        "改善定义为基线 NLL 减去联合模型 NLL；正数表示联合模型较好。", "",
        "## 数据与固定实验设定", "",
        f"- 数据：Kenneth French 官方价值加权 10 行业日收益，{quality['source_header']}",
        f"- 清洗区间：{quality['first_date']}—{quality['last_date']}，共 {quality['rows']} 个交易日。",
        "- 训练：2005—2017；验证：2018—2020；测试：2021—2025。",
        "- 完整 20 日区间按固定步长取样，未覆盖的期末尾段不凑成短标签。"
        "本轮零滞后评测收益覆盖 2021-01-05—2025-12-10，滞后一日覆盖 2021-01-06—2025-12-11。",
        f"- 样本数：{metadata['split_samples']}；训练间隔 5 日，验证与测试间隔 20 日。",
        "- 目标：未来 20 日平均日频二阶矩，零均值口径。LW 使用 assume_centered=True。",
        "- 6 个市场状态特征、4 个成熟可靠性特征，可靠性回看 60 个预测起点。",
        "- 线性门控与 16 单元 MLP；每种输入配置各运行 5 个固定随机种子。",
        "- 固定融合权重仅用训练集拟合；单窗口、EWMA 参数、可靠性温度与主架构用验证集选择。",
        "- 特征标准化仅使用训练起点；所有标签完整落在所属时间段内。",
        f"- 验证集最优单窗口：{metadata['best_single_validation']}；"
        f"EWMA：{metadata['selected_ewma']}；历史损失温度：{metadata['historical_loss_temperature']}。",
        f"- 固定融合权重（20/60/120/252 日）：{metadata['fixed_fusion_weights']}。",
        f"- 原始文件 SHA256：`{metadata['source_sha256']}`。", "",
        "[官方数据说明](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_10_ind_port.html)",
        "", "## 预测结果", "",
        table(summary.rename(columns={"method": "方法", "validation_nll": "验证 NLL",
                                      "test_nll": "测试 NLL", "test_seed_std": "种子标准差"})), "",
        "LW-252 在本轮测试排序中较好，但这是完整单窗口结果的描述，不能把它事后作为"
        "验证集选定的单窗口；正式选定的单窗口仍是 LW-20。",
        "神经模型的主指标是各种子预测损失的平均值，test_seed_std 是各种子全时期平均损失的标准差。"
        "五个种子不是五份独立市场样本。", "",
        "## 成对统计检验", "",
        "下表采用长度 3 的连续 20 日区间块、5000 次循环时间块重采样。"
        "置信区间是逐项 95% 区间；p_holm 是 7 项单侧检验的 Holm 校正值。"
        "长度 1 与 6 的敏感性分析保存在 paired_bootstrap.csv。", "",
        table(core[["baseline", "improvement", "ci_lower", "ci_upper", "p_holm"]].rename(
            columns={"baseline": "对照方法", "improvement": "联合模型改善",
                     "ci_lower": "区间下界", "ci_upper": "区间上界", "p_holm": "校正后 p 值"}
        )), "",
        "## 分年份表现", "",
        table(yearly[yearly.method.isin([metadata['best_single_validation'], "FixedFusion",
                                       "HistoricalLoss", primary])].rename(columns={
            "year": "年份", "method": "方法", "nll": "测试 NLL", "intervals": "区间数量"
        })), "",
        "年度归属按预测起点年份划分，跨年持有区间保留在该起点年份。"
        "验证集选定的短窗口与测试期长窗口表现存在差异，提示模型选择对市场阶段敏感；"
        "这属于结果线索，尚未通过独立的滚动前向实验确认。",
        "## 下游组合检查", "",
        "使用相同的全投资、仅做多最小方差优化，持有 20 日并跟踪权重漂移。"
        "下表为零成本、零执行滞后的日频近似；费用 5/10 bps 和一日滞后结果保存在 backtest_summary.csv。"
        "联合模型回测使用五种子平均门控权重，因而属于预先规定的集成策略，"
        "与预测表中平均种子损失的口径不同。", "",
        table(backtests[(backtests.lag_days == 0) & (backtests.cost_bps == 0)][[
            "method", "annualized_volatility", "maximum_drawdown", "cvar_95", "mean_turnover"
        ]].rename(columns={"method": "方法", "annualized_volatility": "年化波动率",
                           "maximum_drawdown": "最大回撤", "cvar_95": "95% CVaR",
                           "mean_turnover": "平均单次换手率"})), "",
        "波动率、回撤和换手率在表中用小数表示，例如 0.1263 表示 12.63%。",
        "## 解释边界与下一步", "",
        "- 本轮为一个美国市场、10 个行业的首轮验证；49 行业、CM-IEWMA、滚动最优融合尚未实施。",
        "- 本轮结果不能替代对强外部方法的比较，也不能推广为真实可交易策略的盈利结论。",
        "- 历史收益来自当前下载版本，包含数据库事后修订，并非逐日历史发布版本。",
        "- 按训练期短期风险分位数划分时，测试区间只有中、高风险样本，没有低风险样本。"
        "因此不能依据本轮数据比较低风险状态下的方法表现。",
        "- 下一轮应冻结本轮结果，扩展 49 行业和外部基线；任何测试后调参必须作为新实验披露。",
        "", "## 图表", "",
        "![NLL 对比](figures/nll_comparison.png)", "",
        "![成对 NLL 改善及置信区间](figures/nll_improvement.png)", "",
        "![动态权重](figures/dynamic_weights.png)", "",
        "![分年份 NLL](figures/yearly_nll.png)", "",
    ]
    (output / "preliminary_report.md").write_text("\n".join(lines), encoding="utf-8")

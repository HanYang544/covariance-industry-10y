"""Matched mechanism controls, chronological selection audit, and paired risk inference.

This workflow preserves the existing recommendation. All available historical outcomes
are exploratory; a hashed prospective protocol is prepared, not retrospectively claimed.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

from portfolio_risk.config import load_config
from portfolio_risk.covariance.iewma import precision_cholesky
from portfolio_risk.data.french import parse_industry_zip, validate_return_matrix
from portfolio_risk.evaluation.metrics import (
    annualized_volatility, empirical_cvar, maximum_drawdown,
)
from portfolio_risk.experiment import (
    covariance_nll, fit_fixed_fusion, holm_adjust, paired_block_bootstrap,
    run_gmv_backtest, split_prediction_indices,
)
from portfolio_risk.models.state_whitener import (
    fit_convex_fusion, fit_precision_calibration, fit_state_model, whitener_statistics,
)
from portfolio_risk.state_research import prepare_research, training_indices
from portfolio_risk.window_ablation import markdown_table


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibration_design(train_features, features, clip):
    """Fit normalization on training rows only, including the intercept."""
    mean = train_features.mean(0)
    scale = np.maximum(train_features.std(0), 1e-12)
    return np.column_stack([
        np.ones(len(features)), np.clip((features - mean) / scale, -clip, clip),
    ])


def calibrate_covariance(train_cov, train_target, test_cov, train_x, test_x, config):
    """Give any comparator exactly the same four-feature precision calibration."""
    energy = np.trace(np.linalg.solve(train_cov, train_target), axis1=-2, axis2=-1)
    energy /= train_cov.shape[-1]
    train_design = calibration_design(train_x, train_x, config["feature_clip"])
    test_design = calibration_design(train_x, test_x, config["feature_clip"])
    theta, diagnostics = fit_precision_calibration(
        train_design, energy, config["calibration_ridge"],
    )
    result = test_cov * np.exp(-test_design @ theta)[:, None, None]
    return result, theta, diagnostics


def expanding_selection_audit(validation, penalty, minimum_years):
    """Replay outer years using only earlier stored validation scores for selection.

    This is an expanding, temporally cross-fitted audit of the existing registry,
    not an unseen independent confirmation or a new fully nested model search.
    """
    if minimum_years < 2:
        raise ValueError("At least two prior years required for standard-deviation scoring.")
    if validation.duplicated(["method", "year"]).any():
        raise ValueError("Duplicate method-year validation scores.")
    years = sorted(validation.year.unique())
    methods = sorted(validation.method.unique())
    rows = []
    for i in range(minimum_years, len(years)):
        year = years[i]
        past = validation[validation.year < year]
        scores = []
        for name in methods:
            values = past[past.method == name].excess_over_fixed.to_numpy()
            if len(values) != i or not np.isfinite(values).all():
                raise ValueError("Incomplete chronological validation registry.")
            scores.append((float(values.mean() + penalty * values.std(ddof=1)), name))
        selected = min(scores)[1]
        future = validation[(validation.year == year) & (validation.method == selected)]
        if len(future) != 1:
            raise ValueError("Missing outer-year evaluation.")
        row = future.iloc[0]
        rows.append(dict(
            outer_year=int(year), selection_year_last=int(years[i - 1]),
            selection_year_count=i, selected=selected, nll=float(row.nll),
            excess_over_fixed=float(row.excess_over_fixed), origins=int(row.origins),
        ))
    return pd.DataFrame(rows)


def paired_risk_bootstrap(primary, comparator, horizon, block_length, repetitions, seed):
    """Resample paired complete holding intervals, recomputing nonlinear metrics.

    p-values are centered-bootstrap approximations, not exact randomization tests.
    Drawdown intervals describe block-resampled paths, not chronology-invariant risk.
    """
    primary, comparator = np.asarray(primary), np.asarray(comparator)
    if min(horizon, block_length, repetitions) <= 0:
        raise ValueError("Positive sampling parameters required.")
    if (primary.ndim != 1 or primary.shape != comparator.shape
            or len(primary) % horizon or len(primary) < 2 * horizon
            or not np.isfinite(primary).all() or not np.isfinite(comparator).all()):
        raise ValueError("Need paired finite returns consisting of complete holding intervals.")
    n = len(primary) // horizon
    first, second = primary.reshape(n, horizon), comparator.reshape(n, horizon)
    metrics = dict(volatility=annualized_volatility, cvar_95=empirical_cvar,
                   maximum_drawdown=maximum_drawdown)
    observed = {name: function(comparator) - function(primary)
                for name, function in metrics.items()}
    samples = {name: [] for name in metrics}
    rng = np.random.default_rng(seed)
    # Batches avoid large peak allocations on long future records.
    for offset in range(0, repetitions, 100):
        batch = min(100, repetitions - offset)
        starts = rng.integers(0, n, (batch, int(np.ceil(n / block_length))))
        indices = ((starts[:, :, None] + np.arange(block_length)) % n).reshape(batch, -1)[:, :n]
        for index in indices:
            a, b = first[index].ravel(), second[index].ravel()
            for name, function in metrics.items():
                samples[name].append(function(b) - function(a))
    rows = []
    for name in metrics:
        draws = np.asarray(samples[name])
        lower, upper = np.quantile(draws, [.025, .975])
        p = (1 + np.sum(draws - observed[name] >= observed[name])) / (repetitions + 1)
        rows.append(dict(metric=name, improvement=observed[name], ci_lower=float(lower),
                         ci_upper=float(upper), p_one_sided_approx=float(p)))
    return rows


def matrix_in_original_order(covariance, permutation):
    inverse = np.argsort(permutation)
    return covariance[..., inverse, :][..., :, inverse]


def audit_registry(config, dimensions):
    """Fixed diagnostic controls: none are selected on the audit outcomes."""
    registry = [
        dict(name="Full-NoScale", family="whitener", experts=list(range(10)), calibrate=False),
        dict(name="LW-Only-Scale", family="whitener", experts=list(range(4)), calibrate=True),
        dict(name="IEWMA-Only-Scale", family="whitener", experts=list(range(4, 9)), calibrate=True),
        dict(name="NoDiagonal-Scale", family="whitener", experts=list(range(9)), calibrate=True),
        dict(name="Full-LabelPermutation", family="whitener", experts=list(range(10)),
             calibrate=True, permute_labels=True),
        dict(name="LW-252-Scale", family="covariance", experts=[3], calibrate=True),
        dict(name="FixedLW-Scale", family="covariance", experts=list(range(4)), calibrate=True),
        dict(name="FixedAll-NoScale", family="covariance", experts=list(range(10)), calibrate=False),
        dict(name="FixedAll-Scale", family="covariance", experts=list(range(10)), calibrate=True),
    ]
    permutations = [list(range(dimensions - 1, -1, -1))] + [
        np.random.default_rng(seed).permutation(dimensions).tolist()
        for seed in config["permutation_seeds"]
    ]
    for i, permutation in enumerate(permutations):
        registry.append(dict(name=f"Full-AssetOrder-{i}", family="whitener",
            experts=list(range(10)), calibrate=True, permutation=permutation))
    for mode in ("market", "reliability", "joint"):
        for seed in config["state_seeds"]:
            registry.append(dict(name=f"Matched-{mode}-Seed{seed}", family="state",
                mode=mode, states=config["matched_states"], ridge=config["matched_ridge"],
                calibrate=True, seed=seed))
    return registry


def fit_control(prepared, train, evaluate, spec, config):
    realized = prepared["arrays"]["realized"]
    x = prepared["features"]["global"]
    train_target = realized[train]
    fit_config = dict(config, seed=spec.get("seed", config["seed"]))
    if spec.get("permute_labels"):
        # Deterministic within-training-label permutation; evaluation labels remain intact.
        train_target = train_target[np.random.default_rng(config["seed"]).permutation(len(train))]
    if spec["family"] == "covariance":
        chosen = prepared["experts"][:, spec["experts"]]
        if chosen.shape[1] == 1:
            train_cov, test_cov = chosen[train, 0], chosen[evaluate, 0]
        else:
            weights = fit_fixed_fusion(chosen[train], train_target)
            train_cov = np.einsum("k,bkij->bij", weights, chosen[train])
            test_cov = np.einsum("k,bkij->bij", weights, chosen[evaluate])
        if spec["calibrate"]:
            test_cov, theta, _ = calibrate_covariance(
                train_cov, train_target, test_cov, x[train], x[evaluate], config,
            )
        else:
            theta = np.zeros(x.shape[1] + 1)
        return test_cov, dict(theta=theta)
    permutation = spec.get("permutation")
    if permutation is not None:
        selected = prepared["experts"][:, spec["experts"]]
        # Only requested finite rows are factored; the early warm-up contains NaNs.
        train_whiteners = precision_cholesky(selected[train][..., permutation, :][..., :, permutation])
        test_whiteners = precision_cholesky(selected[evaluate][..., permutation, :][..., :, permutation])
        train_target = train_target[..., permutation, :][..., :, permutation]
    else:
        chosen = spec.get("experts", list(range(10)))
        train_whiteners = prepared["whiteners"][train][:, chosen]
        test_whiteners = prepared["whiteners"][evaluate][:, chosen]
    state_x = prepared["features"][spec.get("mode", "global")]
    model_spec = dict(states=spec.get("states", 1), ridge=spec.get("ridge", 0.),
                      calibrate=spec["calibrate"])
    # The same four calibration inputs are used across all matched state controls.
    # This separates state-dependent fusion from the number of calibration features.
    statistics = whitener_statistics(train_whiteners, train_target)
    anchor, _ = fit_convex_fusion(np.ones((len(train), 1)), *statistics)
    model, _ = fit_state_model(train_whiteners, train_target, state_x[train], x[train],
                              model_spec, fit_config, statistics, anchor[0])
    prediction = model.predict(test_whiteners, state_x[evaluate], x[evaluate])[0]
    if permutation is not None:
        prediction = matrix_in_original_order(prediction, permutation)
    return prediction, dict(theta=model.theta, beta=model.beta)


def source_fingerprints(root):
    paths = [*root.glob("src/**/*.py"), *root.glob("tests/**/*.py"),
             *root.glob("configs/*.yaml"), *root.glob("scripts/*"),
             root / "pyproject.toml", root / "requirements-preliminary.txt"]
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(paths) if p.is_file()}


def make_confirmation_lock(config, research, source, primary, archive, as_of, root):
    """Lock one primary and one comparator for genuinely future data."""
    start = pd.Timestamp(config["confirmation_start"])
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    if start <= max(pd.Timestamp(as_of), today):
        raise ValueError("Confirmation must start after both the data snapshot and protocol day.")
    return dict(
        status="locked_waiting_for_future_data", independent_validation_available=False,
        created_at_utc=datetime.now(timezone.utc).isoformat(), decision_start=str(start.date()),
        minimum_complete_origins=config["confirmation_minimum_origins"],
        horizon_trading_days=20, origin_stride=20, origin_anchor="first available decision on/after decision_start",
        primary=primary, comparator="LW-252-Scale", model_configuration=research,
        primary_artifact_sha256=file_hash(source / "models" / "deployment_model.npz"),
        first_year_model_sha256=file_hash(source / "models" / f"evaluation_2026_{primary}.npz"),
        raw_snapshot_sha256=file_hash(archive), raw_snapshot_last=str(pd.Timestamp(as_of).date()),
        asset_order=json.loads((source / "deployment_manifest.json").read_text(encoding="utf-8"))["industry_columns"],
        refit_rule="annual; preceding ten calendar years; only labels matured before Jan 1; stride 5",
        first_refit_rule="use the saved evaluation_2026 primary for any 2026 origin; no midyear refit",
        comparator_rule="LW252 + identical four-feature precision calibration fitted with identical annual training rows",
        primary_endpoint="paired mean normalized NLL improvement; one-sided centered block-bootstrap p, alpha .05; block 3",
        secondary_endpoints=["GMV volatility", "CVaR95", "maximum drawdown", "turnover"],
        portfolio_rule="long-only fully invested; lag 1; 5 bps per unit two-sided notional; no new tuning",
        stopping_rule="first 30 fully matured nonoverlapping origins; no repeated significance peeking",
        data_rule="separate future archive; match assets, record hash; reject if pre-lock historical returns change",
        invalidation_rule="any architecture, hyperparameter, asset-order, outcome-rule change requires a new later lock",
        warning="A lock is not a completed independent validation. Revised historical data can invalidate confirmation.",
        source_sha256=source_fingerprints(root),
    )


def run_evidence_audit(config_path: Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path.cwd()
    config = load_config(config_path)
    research = load_config(config["research_config"])
    base = load_config(research["base_config"])
    source, output = Path(research["output_dir"]), Path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Audit artifacts already exist: choose a new output directory.")
    selection = json.loads((source / "selection.json").read_text(encoding="utf-8"))
    primary = selection["primary"]
    archive = Path(base["data"]["raw_dir"]) / "10_Industry_Portfolios_daily_CSV.zip"
    original = json.loads((source / "research_protocol.json").read_text(encoding="utf-8"))
    if file_hash(archive) != original["source_sha256"]:
        raise ValueError("Frozen data archive changed.")
    frame, _ = parse_industry_zip(archive)
    frame = frame.loc[base["data"]["history_start"]:]
    validate_return_matrix(frame)
    registry = audit_registry(config, frame.shape[1])
    preserved = {p.as_posix(): file_hash(p) for folder in (
        "preliminary", "window_ablation", "feature_search", "forward_audit",
        "state_whitener", "online_whitener",
    ) for p in (Path("outputs") / folder).rglob("*") if p.is_file()}
    output.mkdir(parents=True)
    protocol = dict(config=config, registry=registry, primary_unchanged=primary,
        created_at_utc=datetime.now(timezone.utc).isoformat(), source_sha256=file_hash(archive),
        evidence_status="exploratory; all historical evaluation periods already observed",
        selection_policy="no audit control replaces the previously selected primary",
        matched_controls="same annual training dates, mature labels, four calibration inputs, clip and ridge",
        comparisons="Holm within each period/block/lag/cost; NLL controls and volatility/CVaR families separately",
        limitations=["historical selection bias remains", "bootstrap is conditional on fitted models",
                     "no new universe/horizon study", "label permutation is a diagnostic, not a formal randomization test"],
        source_files_sha256=source_fingerprints(root), preserved_artifact_hashes=preserved)
    (output / "audit_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Protocol frozen: {len(registry)} diagnostic controls; recommendation unchanged", flush=True)
    prepared = prepare_research(frame, base, research)
    horizon = base["forecast"]["horizon"]
    with np.load(source / "selected_evaluation_covariances.npz") as saved:
        indices = saved["indices"].copy()
        predictions = {"Recommended": saved["primary"].copy(),
                       "CM-IEWMA": saved["cm_iewma"].copy(), "FixedLW": saved["fixed"].copy()}
    predictions["LW-252"] = prepared["experts"][indices, 3]
    predictions["EqualWeight-Covariance"] = prepared["experts"][indices, :4].mean(1)
    for spec in registry:
        predictions[spec["name"]] = np.full_like(predictions["Recommended"], np.nan)
    folds, fit_rows = [], []
    for year in research["evaluation_years"]:
        positions = np.flatnonzero(frame.index[indices].year == year)
        if not len(positions):
            continue
        evaluate = indices[positions]
        train = training_indices(frame, prepared, year, base, research)
        if frame.index[train[-1] + horizon] >= frame.index[evaluate[0]]:
            raise ValueError("Training labels cross the decision boundary.")
        folds.append(dict(year=year, train_origins=len(train), evaluation_origins=len(evaluate),
            training_label_last=str(frame.index[train[-1] + horizon].date()),
            decision_first=str(frame.index[evaluate[0]].date())))
        # Replicate, rather than silently trusting a stale saved primary.
        replication = dict(name="Replicated", family="whitener", experts=list(range(10)), calibrate=True)
        reproduced, _ = fit_control(prepared, train, evaluate, replication, research)
        np.testing.assert_allclose(reproduced, predictions["Recommended"][positions], rtol=1e-9, atol=1e-12)
        for spec in registry:
            prediction, parameters = fit_control(prepared, train, evaluate, spec, research)
            np.linalg.cholesky(prediction)
            predictions[spec["name"]][positions] = prediction
            folder = output / "models"
            folder.mkdir(exist_ok=True)
            np.savez_compressed(folder / f"{year}_{spec['name']}.npz", **parameters)
            fit_rows.append(dict(year=year, method=spec["name"], train_origins=len(train),
                                 calibration_parameters=len(parameters["theta"])))
        pd.DataFrame(folds).to_csv(output / "chronological_folds.csv", index=False)
        print(f"Matched audit year {year}: {len(registry)} controls + primary replication passed", flush=True)
    pd.DataFrame(fit_rows).to_csv(output / "fit_diagnostics.csv", index=False)
    realized = prepared["arrays"]["realized"][indices]
    losses = {name: covariance_nll(matrix, realized) for name, matrix in predictions.items()}
    pd.DataFrame(losses, index=frame.index[indices]).rename_axis("decision_date").to_csv(output / "evaluation_losses.csv")
    old_mask = frame.index[indices].year < 2026
    summary = pd.DataFrame([dict(method=name, test_2021_2025_nll=value[old_mask].mean(),
        diagnostic_2026_nll=value[~old_mask].mean()) for name, value in losses.items()])
    summary.to_csv(output / "summary.csv", index=False)
    yearly = pd.DataFrame([dict(method=name, year=year,
        nll=value[frame.index[indices].year == year].mean())
        for name, value in losses.items() for year in research["evaluation_years"]])
    yearly.to_csv(output / "yearly_nll.csv", index=False)
    np.savez_compressed(output / "evaluation_covariances.npz", indices=indices, **predictions)
    selection_audit = expanding_selection_audit(pd.read_csv(source / "validation_by_year.csv"),
        research["selection_std_penalty"], config["nested_minimum_years"])
    selection_audit.to_csv(output / "expanding_selection_audit.csv", index=False)
    nll_rows = []
    comparators = [name for name in predictions if name != "Recommended"]
    for segment, mask in (("2021_2025", old_mask), ("2026", ~old_mask)):
        for block in config["bootstrap_block_lengths"]:
            part = []
            for name in comparators:
                diff = losses[name][mask] - losses["Recommended"][mask]
                lo, hi, p = paired_block_bootstrap(diff, block, config["bootstrap_repetitions"], config["seed"])
                part.append(dict(segment=segment, comparator=name, block_length=block,
                    improvement=float(diff.mean()), ci_lower=lo, ci_upper=hi, p_one_sided=p))
            for row, p in zip(part, holm_adjust([r["p_one_sided"] for r in part]), strict=True):
                row["p_holm"] = float(p)
            nll_rows.extend(part)
    pd.DataFrame(nll_rows).to_csv(output / "nll_bootstrap.csv", index=False)
    calibration = []
    for name, covariance in predictions.items():
        energy = np.trace(np.linalg.solve(covariance, realized), axis1=-2, axis2=-1) / frame.shape[1]
        for year in research["evaluation_years"]:
            keep = frame.index[indices].year == year
            calibration.append(dict(method=name, year=year, mean_normalized_energy=energy[keep].mean(),
                median_normalized_energy=np.median(energy[keep]), origins=int(keep.sum())))
    pd.DataFrame(calibration).to_csv(output / "risk_calibration.csv", index=False)
    backtests, daily_records, risk_rows, stress = [], {}, [], []
    # Scalar calibration does not change GMV holdings; confirm numerically below.
    portfolio_names = ["Recommended", "LW-252", "LW-252-Scale", "CM-IEWMA", "FixedLW",
                       "FixedLW-Scale", "FixedAll-NoScale", "FixedAll-Scale",
                       "LW-Only-Scale", "IEWMA-Only-Scale", "NoDiagonal-Scale", "Full-NoScale"]
    for segment, mask in (("2021_2025", old_mask), ("2026", ~old_mask)):
        for lag in base["backtest"]["execution_lag_days"]:
            keep = mask & (indices + horizon + lag < len(frame))
            common_dates = None
            for name in portfolio_names + ["EqualWeight-Portfolio"]:
                if name == "EqualWeight-Portfolio":
                    # Identity covariance gives equal initial weights at each rebalance.
                    covariances = np.broadcast_to(np.eye(frame.shape[1]), (keep.sum(), frame.shape[1], frame.shape[1]))
                else:
                    covariances = predictions[name][keep]
                gross, days, turnover = run_gmv_backtest(frame.to_numpy(), indices[keep], covariances, horizon, lag, 0)
                if common_dates is None:
                    common_dates = days
                np.testing.assert_array_equal(days, common_dates)
                for cost in base["backtest"]["transaction_cost_bps"]:
                    net = gross.copy()
                    net[::horizon] = (1 - 2 * np.asarray(turnover) * cost / 10000) * (1 + net[::horizon]) - 1
                    daily_records[(segment, lag, cost, name)] = net
                    pd.DataFrame(dict(date=frame.index[days], net_return=net)).to_csv(
                        output / f"daily_{segment}_lag{lag}_cost{cost}_{name}.csv", index=False)
                    backtests.append(dict(segment=segment, lag_days=lag, cost_bps=cost, method=name,
                        annualized_volatility=annualized_volatility(net), cvar_95=empirical_cvar(net),
                        maximum_drawdown=maximum_drawdown(net), mean_turnover=float(np.mean(turnover)), days=len(net)))
                    for year in np.unique(frame.index[days].year):
                        values = net[frame.index[days].year == year]
                        stress.append(dict(segment=segment, lag_days=lag, cost_bps=cost, method=name,
                            year=int(year), annualized_volatility=annualized_volatility(values),
                            cvar_95=empirical_cvar(values), maximum_drawdown=maximum_drawdown(values), days=len(values)))
            for cost in base["backtest"]["transaction_cost_bps"]:
                for raw, scaled in (("Recommended", "Full-NoScale"), ("LW-252", "LW-252-Scale"),
                                    ("FixedLW", "FixedLW-Scale"), ("FixedAll-NoScale", "FixedAll-Scale")):
                    np.testing.assert_allclose(daily_records[(segment, lag, cost, raw)],
                        daily_records[(segment, lag, cost, scaled)], atol=1e-9, rtol=1e-7)
            print(f"Portfolio alignment and scalar invariance passed: {segment}, lag {lag}", flush=True)
    pd.DataFrame(backtests).to_csv(output / "backtest_summary.csv", index=False)
    pd.DataFrame(stress).to_csv(output / "yearly_portfolio_risk.csv", index=False)
    risk_comparators = ["LW-252-Scale", "CM-IEWMA", "FixedLW-Scale", "FixedAll-Scale", "EqualWeight-Portfolio"]
    for segment in ("2021_2025", "2026"):
        # Inference on the two predeclared representative policies; all six summaries above.
        for lag, cost in ((0, 0), (1, 5)):
            for block in config["bootstrap_block_lengths"]:
                part = []
                for name in risk_comparators:
                    rows = paired_risk_bootstrap(daily_records[(segment, lag, cost, "Recommended")],
                        daily_records[(segment, lag, cost, name)], horizon, block,
                        config["bootstrap_repetitions"], config["seed"])
                    part.extend(dict(segment=segment, lag_days=lag, cost_bps=cost,
                                     comparator=name, block_length=block, **row) for row in rows)
                inferential = [r for r in part if r["metric"] != "maximum_drawdown"]
                adjusted = holm_adjust([r["p_one_sided_approx"] for r in inferential])
                for row, p in zip(inferential, adjusted, strict=True):
                    row["p_holm_approx"] = float(p)
                risk_rows.extend(part)
            print(f"Paired portfolio bootstrap completed: {segment}, lag {lag}, cost {cost}", flush=True)
    pd.DataFrame(risk_rows).to_csv(output / "portfolio_bootstrap.csv", index=False)
    confirmation = make_confirmation_lock(config, research, source, primary, archive,
                                         frame.index[-1], root)
    lock = output / "confirmation_lock.json"
    lock.write_text(json.dumps(confirmation, indent=2), encoding="utf-8")
    (output / "confirmation_lock.sha256").write_text(file_hash(lock) + "\n", encoding="utf-8")
    metadata = dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=file_hash(archive), primary_replication_passed=True, scalar_gmv_invariance_passed=True,
        previous_artifacts_unchanged=True, independent_validation_available=False,
        config_sha256=file_hash(config_path), protocol_sha256=file_hash(output / "audit_protocol.json"),
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        scipy=scipy.__version__, sklearn=sklearn.__version__, torch=torch.__version__)
    for path, digest in preserved.items():
        if file_hash(Path(path)) != digest:
            raise RuntimeError(f"Preserved research artifact changed: {path}")
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_audit_report(output, summary, pd.DataFrame(risk_rows), selection_audit,
                       pd.DataFrame(backtests), config)
    print(summary.to_string(index=False), flush=True)
    print(f"Evidence audit complete: {output.resolve()}", flush=True)


def write_audit_report(output, summary, risks, selection_audit, backtest, config):
    selected = ["Recommended", "LW-252", "LW-252-Scale", "CM-IEWMA", "FixedLW",
                "FixedLW-Scale", "FixedAll-NoScale", "FixedAll-Scale", "Full-NoScale",
                "LW-Only-Scale", "IEWMA-Only-Scale", "NoDiagonal-Scale", "Full-LabelPermutation"]
    basic = summary[summary.method.isin(selected)]
    asset_order = summary[summary.method.str.startswith("Full-AssetOrder")]
    seeds = summary[summary.method.str.startswith("Matched-")]
    core_risk = risks[(risks.segment == "2021_2025") & (risks.lag_days == 0)
                     & (risks.block_length == 3) & (risks.metric != "maximum_drawdown")]
    economic = backtest[(backtest.segment == "2021_2025") & (backtest.lag_days == 1)
                       & (backtest.cost_bps == 5)]
    nll = pd.read_csv(output / "nll_bootstrap.csv")
    core_nll = nll[(nll.segment == "2021_2025") & (nll.block_length == 3)
                  & (nll.comparator.isin(["LW-252-Scale", "FixedLW-Scale", "FixedAll-Scale", "CM-IEWMA"]))]
    report = ["# 实验证据补强报告", "",
        "本轮不更换原推荐模型，不按新结果调参。所有现有时期均已被观察，以下结果都是探索性审计，不是新的独立验证。实验注册表、配置及源码校验值在评分前保存；旧六轮产物逐文件核验未改变。", "",
        "## 1. 公平对照与机制消融", "",
        "每年统一前十个自然年的已成熟标签、5 日训练步长、同一组评价起点。所有带 Scale 的新对照使用相同 5/20/60/252 日四个对数风险特征、训练期标准化、截断与正则强度。对照包括 LW252、固定 LW 协方差融合、十专家固定协方差融合；后者与推荐模型共享专家，区别是融合空间。", "",
        markdown_table(basic), "",
        "去校准、仅 LW、仅 IEWMA、去对角专家分别拆解风险尺度、多尺度专家家族和对角收缩贡献。只比较一个去掉模块的变体不代表完全分离交互作用；保留相同专家的协方差融合对照尤为重要。训练标签置乱会同时破坏风险水平与时间对应，属于故障诊断而非严格随机化显著性检验；评价标签从不置乱。", "",
        "### 同尺度校准强对照的配对 NLL 推断", "", markdown_table(core_nll), "",
        "improvement=对照损失−推荐损失，正数支持推荐。每个时期、块长内对全部预注册对照做 Holm；没有覆盖此前整个项目的选择偏差。2026 年只有 8 个区间，尤其块长 6 下推断极不稳定。", "",
        "## 2. 状态机制与随机种子", "",
        "市场、可靠性、联合均固定为 2 状态、ridge=0.3，使用同样四个校准特征和三个聚类种子。这样避免以不同状态数或校准维度替代对联合状态融合的检验。联合输入本身维度更多，PCA/聚类几何仍不同；种子不是独立市场样本。", "",
        markdown_table(seeds), "",
        "## 3. 资产排列稳健性", "",
        "在同一专家协方差上分别反转资产顺序、采用两个固定随机排列，重新计算精度 Cholesky 并重新拟合全局模型，预测后还原资产顺序再评分。市场总体风险特征及训练标签边界不变。", "",
        markdown_table(asset_order), "",
        "若结果变化明显，就不能称白化融合具有资产排列不变性。本次检查没有据此挑选有利排列，也没有引入新的排列集成主模型。", "",
        "## 4. 时间交叉拟合的选模审计", "",
        "读取既有 26 配置的逐年验证结果：最初三年作为内层选模历史，从第四年起只用更早年份的平均相对 NLL+0.25×年度标准差选配置，然后记录下一年的成绩。原逐年模型本身也只用当年之前成熟训练标签。", "",
        markdown_table(selection_audit), "",
        f"外层年度等权平均相对固定融合差：{selection_audit.excess_over_fixed.mean():.6f}；负数更好。", "",
        "这是对已存在候选注册表的时间前推交叉拟合审计，不是重新执行完整的未知超参数嵌套搜索，也不消除候选设计已经参考历史结果的偏差。", "",
        "## 5. 组合波动、尾部风险和成本", "",
        "所有比较方法在每个执行滞后内逐日收益日期完全一致；不同滞后不配对。先求解零成本持仓，再按每次双边交易额在持有首日计费，等价于原回测。等权组合为每期重新等权、期内买入持有。", "",
        "### 1 日延迟、5 bps 的经济效果", "", markdown_table(economic), "",
        "### 零延迟/零成本：重算非线性风险指标的配对区块区间", "", markdown_table(core_risk), "",
        "将每日收益按完整 20 日持有区间组织，在预测区间层以长度 1/3/6 的循环移动块同时重采样两策略；每次重算年化波动与经验 CVaR95，不把逐日风险或种子当作独立样本。5000 次，输出逐项百分位 95% 区间。单侧 p 为中心化 bootstrap 近似，不是精确检验；同一时期/政策/块长内对五个对照×两个指标做 Holm。", "",
        "最大回撤也输出重采样路径区间，但不做显著性声明：换序会改变路径，循环跨边界和较少尾部样本带来局限。所有区间条件于已拟合模型，没有重采样训练过程。年度风险表提供不同市场时期的直接诊断，而非事后挑选有利危机段。", "",
        "整体乘以正数不会改变全投资 GMV 的最优权重。因此去掉尺度校准能改变 NLL，却不应改变 GMV 结果。四组对应回测逐日一致性已自动核验。risk_calibration.csv 记录 tr(Ĉ⁻¹S)/N，理想均值约 1；这不是分布覆盖率检验。", "",
        "## 6. 未来独立确认：已锁定，尚未完成", "",
        f"confirmation_lock.json 指定决策日从 {config['confirmation_start']} 起，固定原推荐对 LW252+同尺度校准；第一个完整 {config['confirmation_minimum_origins']} 个不重叠区间完成后统一评价，不反复显著性窥视。主终点是配对 NLL，组合指标为次终点；一年只按固定规则重拟合。", "",
        "SHA256 锁文件记录现有数据、源码、模型、资产顺序及规则。新数据必须另存并核对历史重叠部分；历史修订须报告并可能使确认无效。当前尚无完整的新未来样本，所以 independent_validation_available=false。锁定协议不是已经完成独立验证，也不保证未来胜出。", "",
        "## 7. 尚未覆盖的项目", "",
        "其他资产市场/49 行业、5/60 日预测期限、训练历史长度、真正点时数据修订管理和重新拟合全过程的不确定性尚未实验。应另立协议；49 行业与 10 行业来自相同基础股票市场，不等于完全独立市场。详细公式与最新实验总协议分别见 docs/evidence_audit_method.md 和 docs/experimental_protocol.md。",
    ]
    (output / "evidence_report.md").write_text("\n".join(report), encoding="utf-8")

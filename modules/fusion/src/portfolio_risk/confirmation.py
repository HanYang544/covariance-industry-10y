"""Evaluate the hashed prospective protocol only when sufficient unseen data exists."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from portfolio_risk.config import load_config
from portfolio_risk.data.french import parse_industry_zip, validate_return_matrix
from portfolio_risk.evidence_audit import file_hash, fit_control, source_fingerprints
from portfolio_risk.evaluation.metrics import annualized_volatility, empirical_cvar, maximum_drawdown
from portfolio_risk.experiment import covariance_nll, paired_block_bootstrap, run_gmv_backtest
from portfolio_risk.models.state_whitener import StateWhitenerModel
from portfolio_risk.state_research import prepare_research, training_indices, predict_research_model
from portfolio_risk.window_ablation import markdown_table


def validate_confirmation_frame(lock, frozen, future):
    """Require unchanged historical observations and the exact locked asset order."""
    validate_return_matrix(future)
    if list(future.columns) != lock["asset_order"] or list(frozen.columns) != lock["asset_order"]:
        raise ValueError("Asset names/order differ from the confirmation lock.")
    if not frozen.index.isin(future.index).all():
        raise ValueError("Future archive omits frozen historical observations.")
    historical = future.loc[frozen.index]
    if not np.array_equal(historical.to_numpy(), frozen.to_numpy()):
        raise ValueError("Historical returns were revised; confirmation is invalid under this lock.")
    # An inserted historical date also changes temporal lags and is a revision.
    if not future.index[future.index <= frozen.index[-1]].equals(frozen.index):
        raise ValueError("Historical calendar changed under this lock.")
    start = pd.Timestamp(lock["decision_start"])
    candidates = np.flatnonzero(future.index >= start)
    # Anchor before maturity filtering, so subsequent runs cannot shift the origin grid.
    origins = candidates[::lock["origin_stride"]]
    origins = origins[origins + lock["horizon_trading_days"] < len(future)]
    required = lock["minimum_complete_origins"]
    if len(origins) < required:
        raise ValueError(f"Only {len(origins)} mature future origins; lock requires {required}.")
    return origins[:required]


def verify_lock_artifacts(lock_path, lock, root):
    checksum = lock_path.with_suffix(".sha256")
    if file_hash(lock_path) != checksum.read_text(encoding="utf-8").strip():
        raise ValueError("Confirmation lock checksum changed.")
    current = source_fingerprints(root)
    for name, digest in lock["source_sha256"].items():
        if current.get(name) != digest:
            raise ValueError(f"Locked source/configuration changed: {name}; establish a new future lock.")
    research = lock["model_configuration"]
    source = Path(research["output_dir"])
    if file_hash(source / "models" / "deployment_model.npz") != lock["primary_artifact_sha256"]:
        raise ValueError("Locked primary artifact changed.")
    if file_hash(source / "models" / f"evaluation_2026_{lock['primary']}.npz") != lock["first_year_model_sha256"]:
        raise ValueError("Locked first-year prediction model changed.")
    return source


def run_confirmation(config_path):
    config = load_config(config_path)
    lock_path = Path(config["lock_path"])
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    source = verify_lock_artifacts(lock_path, lock, Path.cwd())
    archive = Path(config["future_archive"])
    if not archive.is_file():
        raise RuntimeError("Separate future data archive not available; independent confirmation remains pending.")
    research = lock["model_configuration"]
    base = load_config(research["base_config"])
    frozen_archive = Path(base["data"]["raw_dir"]) / "10_Industry_Portfolios_daily_CSV.zip"
    if file_hash(frozen_archive) != lock["raw_snapshot_sha256"]:
        raise ValueError("Frozen raw snapshot changed.")
    frozen, _ = parse_industry_zip(frozen_archive)
    future, _ = parse_industry_zip(archive)
    frozen = frozen.loc[base["data"]["history_start"]:]
    future = future.loc[base["data"]["history_start"]:]
    indices = validate_confirmation_frame(lock, frozen, future)
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Confirmation already evaluated; do not repeat significance peeking.")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    prepared = prepare_research(future, base, research)
    if not prepared["eligible"][indices].all():
        raise ValueError("Locked future origins have invalid features or targets.")
    # All predictions are saved before any endpoint comparison is made.
    predictions = {name: np.full((len(indices), future.shape[1], future.shape[1]), np.nan)
                   for name in ("Recommended", "LW-252-Scale")}
    for year in sorted(future.index[indices].year.unique()):
        positions = np.flatnonzero(future.index[indices].year == year)
        evaluate = indices[positions]
        train = training_indices(future, prepared, year, base, research)
        if future.index[train[-1] + lock["horizon_trading_days"]] >= future.index[evaluate[0]]:
            raise ValueError("Unmatured labels in annual refit.")
        if year == 2026:
            saved = StateWhitenerModel.load(source / "models" / f"evaluation_{year}_{lock['primary']}.npz")
            primary = predict_research_model(saved, prepared, evaluate, dict(mode="global"))[0]
        else:
            primary, _ = fit_control(prepared, train, evaluate,
                dict(family="whitener", experts=list(range(10)), calibrate=True), research)
        baseline, _ = fit_control(prepared, train, evaluate,
            dict(family="covariance", experts=[3], calibrate=True), research)
        predictions["Recommended"][positions] = primary
        predictions["LW-252-Scale"][positions] = baseline
    output.mkdir(parents=True)
    np.savez_compressed(output / "covariances.npz", indices=indices, **predictions)
    target = prepared["arrays"]["realized"][indices]
    losses = {name: covariance_nll(covariance, target) for name, covariance in predictions.items()}
    pd.DataFrame(losses, index=future.index[indices]).rename_axis("decision_date").to_csv(output / "losses.csv")
    difference = losses["LW-252-Scale"] - losses["Recommended"]
    lo, hi, p = paired_block_bootstrap(difference, 3, 5000, research["seed"])
    rows = []
    # One-day execution requires one additional return after the final label.
    horizon = lock["horizon_trading_days"]
    enough_for_portfolio = indices[-1] + horizon + 1 < len(future)
    if enough_for_portfolio:
        for name, covariance in predictions.items():
            daily, dates, turnover = run_gmv_backtest(future.to_numpy(), indices, covariance, horizon, 1, 5)
            pd.DataFrame(dict(date=future.index[dates], net_return=daily)).to_csv(output / f"daily_{name}.csv", index=False)
            rows.append(dict(method=name, annualized_volatility=annualized_volatility(daily),
                             cvar_95=empirical_cvar(daily), maximum_drawdown=maximum_drawdown(daily),
                             mean_turnover=float(np.mean(turnover))))
    result = dict(status="completed", evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
        independent_temporal_validation=True, historical_revision_check_passed=True,
        lock_sha256=file_hash(lock_path), future_archive_sha256=file_hash(archive),
        origins=len(indices), first_decision=str(future.index[indices[0]].date()),
        final_label=str(future.index[indices[-1] + horizon].date()), improvement=float(difference.mean()),
        ci_lower=lo, ci_upper=hi, p_one_sided_approx=p,
        primary_endpoint_supported=bool(difference.mean() > 0 and p < .05),
        secondary_portfolio_complete=bool(enough_for_portfolio),
        caution="temporal confirmation conditional on unchanged data and locked code; not cross-market proof")
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output / "backtest_summary.csv", index=False)
    report = ["# 未来独立确认结果", "",
        f"首个锁定的 {len(indices)} 个成熟非重叠区间，原推荐对 LW252+同尺度校准的平均 NLL 改善为 {difference.mean():.6f}，95% 区间 [{lo:.6f}, {hi:.6f}]，单侧中心化 bootstrap 近似 p={p:.6f}。", "",
        f"主终点判定：{'支持正向改善' if result['primary_endpoint_supported'] else '未确认正向改善'}。保留不利结果，不据此更改锁定模型。", "",
        markdown_table(pd.DataFrame(rows)) if rows else "1 日执行滞后的最后一日收益尚缺，次终点暂不完整。",
    ]
    (output / "confirmation_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)

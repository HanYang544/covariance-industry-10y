"""Validation-only selection of a matured-error online correction layer."""

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from portfolio_risk.config import load_config
from portfolio_risk.data.french import parse_industry_zip
from portfolio_risk.evaluation.metrics import annualized_volatility, empirical_cvar, maximum_drawdown
from portfolio_risk.experiment import covariance_nll, paired_block_bootstrap, holm_adjust, run_gmv_backtest, split_prediction_indices
from portfolio_risk.models.online_whitener import online_prediction
from portfolio_risk.models.state_whitener import StateWhitenerModel, fit_convex_fusion
from portfolio_risk.state_research import prepare_research, fit_research_model, research_candidates
from portfolio_risk.window_ablation import markdown_table


def online_candidates(config, selection, research_specs):
    result = []
    for parent_kind in config["parents"]:
        name = selection["primary"] if parent_kind == "primary" else selection["selected_modes"][parent_kind]
        parent_spec = next(s for s in research_specs if s["name"] == name)
        result.append(dict(name=f"{parent_kind}-static", parent=name, parent_mode=parent_spec["mode"],
                           window=0, anchor_penalty=0., correct=False))
        for window, penalty, correct in itertools.product(config["reliability_windows"], config["anchor_penalties"], config["scalar_correction"]):
            result.append(dict(name=f"{parent_kind}-m{window}-p{penalty:g}-c{int(correct)}",
                               parent=name, parent_mode=parent_spec["mode"], window=window,
                               anchor_penalty=penalty, correct=correct))
    return result


def run_online_research(config_path: Path):
    torch.set_num_threads(1)
    config = load_config(config_path)
    research = load_config(Path(config["research_config"]))
    base = load_config(Path(research["base_config"]))
    source = Path(research["output_dir"])
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "selection.json").exists():
        raise RuntimeError("Online selection already frozen; choose a new output directory.")
    source_selection = json.loads((source / "selection.json").read_text(encoding="utf-8"))
    source_protocol = json.loads((source / "research_protocol.json").read_text(encoding="utf-8"))
    specs = online_candidates(config, source_selection, source_protocol["candidates"])
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()}
    protocol = dict(config=config, candidates=specs, research_config=research,
        created_at_utc=datetime.now(timezone.utc).isoformat(), preserved_source_hashes=hashes,
        selection_years=research["validation_years"], later_outcomes_used_for_selection=False,
        evidence_status="hierarchical validation reuse and previously seen later periods; exploratory")
    (output / "online_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    archive = Path(base["data"]["raw_dir"]) / "10_Industry_Portfolios_daily_CSV.zip"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != source_protocol["source_sha256"]:
        raise ValueError("Frozen source data changed.")
    frame, _ = parse_industry_zip(archive)
    frame = frame.loc[base["data"]["history_start"]:]
    prepared = prepare_research(frame, base, research)
    horizon = base["forecast"]["horizon"]
    validation = split_prediction_indices(frame.index, "2012-01-01", "2020-12-31", horizon, 20, prepared["eligible"])
    original_validation = pd.read_csv(source / "validation_by_year.csv")
    rows, per_origin = [], []
    print(f"Predeclared online search: {len(specs)} candidates, training-free local convex updates", flush=True)
    for year in research["validation_years"]:
        indices = validation[frame.index[validation].year == year]
        parents = {n: StateWhitenerModel.load(source / "models" / f"validation_{year}_{n}.npz") for n in {s["parent"] for s in specs}}
        reference = float(original_validation[(original_validation.method == source_selection["primary"]) & (original_validation.year == year)].iloc[0].nll)
        for spec in specs:
            matrices = np.stack([online_prediction(parents[spec["parent"]], prepared, t, horizon, spec, config)[0] for t in indices])
            losses = covariance_nll(matrices, prepared["arrays"]["realized"][indices])
            rows.append(dict(method=spec["name"], year=year, nll=losses.mean(), excess_over_static=losses.mean() - reference))
            per_origin.extend(dict(method=spec["name"], decision_date=str(frame.index[t].date()), nll=float(loss)) for t, loss in zip(indices, losses, strict=True))
        pd.DataFrame(rows).to_csv(output / "validation_by_year.csv", index=False)
        print(f"Online validation {year} done: {len(indices)} origins × {len(specs)} candidates", flush=True)
    validation_frame = pd.DataFrame(rows)
    search_rows = []
    for spec in specs:
        data = validation_frame[validation_frame.method == spec["name"]]
        excess = data.excess_over_static.to_numpy()
        search_rows.append(dict(**spec, mean_validation_nll=data.nll.mean(), mean_excess=excess.mean(),
            std_excess=excess.std(ddof=1), selection_score=excess.mean() + research["selection_std_penalty"] * excess.std(ddof=1)))
    search = pd.DataFrame(search_rows).sort_values("selection_score")
    search.to_csv(output / "validation_search.csv", index=False)
    pd.DataFrame(per_origin).to_csv(output / "validation_losses.csv", index=False)
    primary = str(search.iloc[0]["name"])
    spec = next(s for s in specs if s["name"] == primary)
    selected_joint = str(search[search.parent_mode == "joint"].iloc[0]["name"])
    selected = dict(primary=primary, spec=spec, selected_joint=selected_joint,
        selected_at_utc=datetime.now(timezone.utc).isoformat(), later_outcomes_used_for_selection=False,
        protocol_sha256=hashlib.sha256((output / "online_protocol.json").read_bytes()).hexdigest())
    (output / "selection.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(f"Online configuration locked before later scores: {primary}", flush=True)
    evaluation_indices = np.load(source / "selected_evaluation_covariances.npz")["indices"]
    matrices = {s["name"]: np.full((len(evaluation_indices), frame.shape[1], frame.shape[1]), np.nan) for s in specs}
    weights_rows, scale_rows = [], []
    for year in research["evaluation_years"]:
        positions = np.flatnonzero(frame.index[evaluation_indices].year == year)
        indices = evaluation_indices[positions]
        parents = {n: StateWhitenerModel.load(source / "models" / f"evaluation_{year}_{n}.npz") for n in {s["parent"] for s in specs}}
        for item in specs:
            for position, t in zip(positions, indices, strict=True):
                matrix, weights, scale, correction = online_prediction(parents[item["parent"]], prepared, t, horizon, item, config)
                matrices[item["name"]][position] = matrix
                if item["name"] == primary:
                    weights_rows.append(dict(date=str(frame.index[t].date()), **dict(zip(prepared["names"], weights, strict=True))))
                    scale_rows.append(dict(date=str(frame.index[t].date()), variance_scale=scale, online_correction=correction))
        print(f"Online evaluation {year} done with frozen configuration", flush=True)
    pd.DataFrame(weights_rows).to_csv(output / "selected_weights.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(output / "selected_variance_scale.csv", index=False)
    losses = {name: covariance_nll(matrix, prepared["arrays"]["realized"][evaluation_indices]) for name, matrix in matrices.items()}
    dates = frame.index[evaluation_indices]
    old_losses = pd.read_csv(source / "evaluation_losses.csv", index_col=0, parse_dates=True)
    if not np.array_equal(old_losses.index.values, dates.values):
        raise ValueError("Different evaluation dates.")
    reference_names = list(dict.fromkeys([source_selection["primary"], source_selection["selected_modes"]["joint"],
        "CM-IEWMA", "LW-20", "LW-60", "LW-120", "LW-252", "FixedFusion-LW", "EqualFusion-LW", "HistoricalLoss-LW",
        "EWMA-0.94", "EWMA-0.97", *prepared["names"][4:9]]))
    for name in reference_names:
        losses[f"Reference-{name}"] = old_losses[name].to_numpy()
    test_mask = dates.year < 2026
    summary = pd.DataFrame([dict(method=name, test_2021_2025_nll=loss[test_mask].mean(), diagnostic_2026_nll=loss[~test_mask].mean(), primary=name == primary)
                           for name, loss in losses.items()]).sort_values("test_2021_2025_nll")
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(losses, index=dates).rename_axis("decision_date").to_csv(output / "evaluation_losses.csv")
    yearly = pd.DataFrame([dict(method=name, year=year, nll=loss[dates.year == year].mean()) for name, loss in losses.items() for year in research["evaluation_years"]])
    yearly.to_csv(output / "yearly_nll.csv", index=False)
    rows = []
    for block in base["evaluation"]["bootstrap_block_lengths"]:
        part = []
        for name in [f"Reference-{n}" for n in reference_names]:
            diff = losses[name][test_mask] - losses[primary][test_mask]
            low, high, p = paired_block_bootstrap(diff, block, base["evaluation"]["bootstrap_repetitions"], base["evaluation"]["bootstrap_seed"])
            part.append(dict(primary=primary, baseline=name, block_length=block, improvement=diff.mean(), ci_lower=low, ci_upper=high, p_one_sided=p))
        for row, p in zip(part, holm_adjust([r["p_one_sided"] for r in part]), strict=True):
            row["p_holm"] = p
        rows += part
    pd.DataFrame(rows).to_csv(output / "paired_bootstrap.csv", index=False)
    chosen = list(dict.fromkeys([primary, selected_joint, "primary-static", "joint-static"]))
    backtest_rows = []
    for segment, mask in [("2021_2025", test_mask), ("2026", ~test_mask)]:
        for name, lag, cost in itertools.product(chosen, base["backtest"]["execution_lag_days"], base["backtest"]["transaction_cost_bps"]):
            keep = mask & (evaluation_indices + horizon + lag < len(frame))
            daily, dt, turnover = run_gmv_backtest(frame.to_numpy(), evaluation_indices[keep], matrices[name][keep], horizon, lag, cost)
            backtest_rows.append(dict(segment=segment, method=name, lag_days=lag, cost_bps=cost,
                annualized_volatility=annualized_volatility(daily), maximum_drawdown=maximum_drawdown(daily),
                cvar_95=empirical_cvar(daily), mean_turnover=np.mean(turnover), days=len(daily)))
            if name == primary:
                pd.DataFrame(dict(date=frame.index[dt], net_return=daily)).to_csv(output / f"selected_returns_{segment}_lag{lag}_cost{cost}.csv", index=False)
    backtest = pd.DataFrame(backtest_rows)
    backtest.to_csv(output / "backtest_summary.csv", index=False)
    parent_spec = next(s for s in research_candidates(research) if s["name"] == spec["parent"])
    deploy_train = split_prediction_indices(frame.index, f"{frame.index[-1].year - research['training_history_years']}-01-01", str(frame.index[-1].date()), horizon, 5, prepared["eligible"])
    anchor, _ = fit_convex_fusion(np.ones((len(deploy_train), 1)), prepared["diagonal"][deploy_train], prepared["gram"][deploy_train])
    deployment, _ = fit_research_model(prepared, deploy_train, parent_spec, research, anchor[0])
    deployment.save(output / "deployment_parent_model.npz")
    forecast, weights, scale, correction = online_prediction(deployment, prepared, len(frame) - 1, horizon, spec, config)
    pd.DataFrame(forecast, index=frame.columns, columns=frame.columns).to_csv(output / "latest_covariance_forecast.csv")
    manifest = dict(as_of=str(frame.index[-1].date()), model=primary, online_spec=spec,
        online_config=config, parent=spec["parent"], expert_names=prepared["names"], industries=frame.columns.tolist(),
        variance_multiplier=scale, online_correction=correction, weights=weights.tolist(),
        horizon_trading_days=horizon, training_label_last=str(frame.index[deploy_train[-1] + horizon].date()),
        data_status="frozen archive, not live data")
    (output / "deployment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    np.savez_compressed(output / "selected_covariances.npz", indices=evaluation_indices, primary=matrices[primary], selected_joint=matrices[selected_joint])
    display_names = [primary, selected_joint, "primary-static", "joint-static", *[f"Reference-{n}" for n in reference_names]]
    display = summary[summary.method.isin(display_names)]
    report = ["# 成熟误差驱动的在线凸更新", "",
        "本轮属于同一研究过程内的分层探索；上一层模型已经由验证筛选，再次使用同一验证期会增加选择偏差。不能将本轮 2026 年成绩称为独立验证。", "",
        f"预定搜索 {len(specs)} 个配置；仅按九个 2012—2020 年度的前推成绩选择 `{primary}`，联合父模型的选定版本为 `{selected_joint}`。保留静态候选，验证没有改善时允许不采用在线层。", "",
        "## 验证选择", "", markdown_table(search.head(8)[["name", "mean_excess", "std_excess", "selection_score"]]), "",
        "## 后续时期诊断", "", markdown_table(display[["method", "test_2021_2025_nll", "diagnostic_2026_nll"]]), "",
        "在线层使用 origin s=t-H-M+1,...,t-H 的完整 20 日标签，不读取未成熟收益。每个时点以当年父模型权重为锚点，在 simplex 上最小化近期白化 NLL + lambda/2×权重偏差平方；随后可选拟合一个有界、带二次惩罚的精度标量偏移。", "",
        "近期窗口 M=20/60/120 个连续交易日起点，标签高度重叠，不能当成 M 个独立样本。父模型在当前时点可利用的历史上拟合，近期层是截至当前日的在线重新估计；标量纠偏使用的是当前父模型在历史上的反事实预测，不声称这些是当时真实发出的样本外预测。", "",
        "## 组合结果", "", markdown_table(backtest[(backtest.segment == "2021_2025") & (backtest.lag_days == 0) & (backtest.cost_bps == 0)][["method", "annualized_volatility", "maximum_drawdown", "cvar_95"]]), "",
        "deployment_parent_model.npz 与 deployment_manifest.json 共同定义可加载模型；latest_covariance_forecast.csv 是截至 2026-08-31 的研究预测。完整公式、资料来源、时间边界和局限见 docs/state_whitener_method.md。所有配置成绩均保留，不按后来成绩改选模型。"]
    (output / "online_report.md").write_text("\n".join(report), encoding="utf-8")
    for filename, digest in hashes.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Parent research artifact changed: {filename}")
    (output / "run_metadata.json").write_text(json.dumps(dict(primary=primary,
        completed_at_utc=datetime.now(timezone.utc).isoformat(), parent_artifacts_unchanged=True,
        later_outcomes_used_for_selection=False), indent=2), encoding="utf-8")
    print(display.to_string(index=False), flush=True)
    print(f"Online research completed: {output.resolve()}", flush=True)

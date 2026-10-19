"""Rolling-origin research pipeline; no post-2020 outcomes are used for selection."""

from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from portfolio_risk.config import load_config
from portfolio_risk.covariance.iewma import iterated_ewma, precision_cholesky, covariance_from_whitener
from portfolio_risk.data.french import parse_industry_zip, validate_return_matrix
from portfolio_risk.evaluation.metrics import annualized_volatility, empirical_cvar, maximum_drawdown
from portfolio_risk.experiment import (build_arrays, covariance_nll, fit_fixed_fusion, paired_block_bootstrap,
                                       holm_adjust, run_gmv_backtest, split_prediction_indices)
from portfolio_risk.models.state_whitener import (fit_convex_fusion, fit_state_model, whitener_statistics,
                                                  StateWhitenerModel)
from portfolio_risk.risk_features import EXPANDED_MARKET_NAMES, expanded_market_features, matured_reliability
from portfolio_risk.window_ablation import markdown_table

MARKET_NAMES = ["log_short_long_volatility_ratio", "log_variance_5", "log_variance_20",
                "log_variance_60", "log_variance_252", "short_mean_correlation",
                "short_long_correlation_change", "common_risk_concentration", "downside_share_20",
                "downside_share_60", "market_momentum_20", "market_drawdown_60",
                "market_tail_ratio_20", "short_volatility_of_volatility"]


def research_candidates(config):
    specs = []
    for mode in config["candidate_modes"]:
        counts = [1] if mode == "global" else config["state_counts"]
        strengths = [0.] if mode == "global" else config["ridge_strengths"]
        for count, ridge, calibrate in itertools.product(counts, strengths, config["calibration"]):
            name = f"{mode}-s{count}-r{ridge:g}-c{int(calibrate)}"
            specs.append(dict(name=name, mode=mode, states=count, ridge=ridge, calibrate=calibrate))
    return specs


def prepare_research(frame, base, config):
    arrays = build_arrays(frame, base)
    returns, count = frame.to_numpy(), len(frame)
    names = [f"LW-{w}" for w in base["forecast"]["lookback_windows"]]
    iewma = []
    for index, (vol, cor) in enumerate(config["iewma_half_lives"]):
        iewma.append(iterated_ewma(returns, vol, cor, config["standardized_return_clip"],
                                 config["fast_diagonal_inflation"] if index == 0 else 0))
        names.append(f"IEWMA-{vol}-{cor}")
    diagonal = np.zeros_like(iewma[0])
    positions = np.arange(frame.shape[1])
    diagonal[:, positions, positions] = np.diagonal(iewma[0], axis1=-2, axis2=-1)
    experts = np.concatenate([arrays["covariance"], np.stack(iewma, axis=1), diagonal[:, None]], axis=1)
    names.append("Diagonal-IEWMA-21")
    first = 251
    whiteners = np.full_like(experts, np.nan)
    whiteners[first:] = precision_cholesky(experts[first:])
    target_valid = np.arange(first, count - base["forecast"]["horizon"])
    losses = np.full((count, len(names)), np.nan)
    losses[target_valid] = covariance_nll(experts[target_valid], arrays["realized"][target_valid, None])
    start = first + base["forecast"]["horizon"] + config["reliability_window"] - 1
    market = np.full((count, len(MARKET_NAMES)), np.nan)
    reliability = np.full((count, len(names)), np.nan)
    columns = [EXPANDED_MARKET_NAMES.index(n) for n in MARKET_NAMES]
    for t in range(start, count):
        market[t] = expanded_market_features(returns[t - 251:t + 1], arrays["covariance"][t])[columns]
        reliability[t] = matured_reliability(losses, t, base["forecast"]["horizon"], config["reliability_window"])
    eligible = np.isfinite(market).all(1) & np.isfinite(reliability).all(1) & np.isfinite(arrays["realized"]).all((1, 2))
    diagonal_stat = np.full((count, len(names), frame.shape[1]), np.nan)
    gram = np.full((count, len(names), len(names)), np.nan)
    diagonal_stat[target_valid], gram[target_valid] = whitener_statistics(whiteners[target_valid], arrays["realized"][target_valid])
    features = {"global": market[:, 1:5], "market": market, "reliability": reliability,
                "joint": np.column_stack([market, reliability])}
    return dict(arrays=arrays, experts=experts, whiteners=whiteners, names=names, features=features,
                eligible=eligible, diagonal=diagonal_stat, gram=gram, expert_losses=losses)


def training_indices(frame, prepared, year, base, config):
    start = f"{max(2005, year - config['training_history_years'])}-01-01"
    end = f"{year - 1}-12-31"
    return split_prediction_indices(frame.index, start, end, base["forecast"]["horizon"],
                                    base["forecast"]["train_stride"], prepared["eligible"])


def fit_research_model(prepared, indices, spec, config, anchor):
    features = prepared["features"][spec["mode"]]
    return fit_state_model(prepared["whiteners"][indices], prepared["arrays"]["realized"][indices],
        features[indices], features[indices], spec, config,
        (prepared["diagonal"][indices], prepared["gram"][indices]), anchor)


def predict_research_model(model, prepared, indices, spec):
    x = prepared["features"][spec["mode"]][indices]
    return model.predict(prepared["whiteners"][indices], x, x)


def cm_iewma_predictions(prepared, returns, indices, lookback):
    selected = prepared["whiteners"][:, 4:9]
    forecasts, rows = [], []
    for t in indices:
        # At close t, r_s is known. The forecast used to score it was made at s-1.
        observed = np.arange(t - lookback + 1, t + 1)
        target = returns[observed, :, None] * returns[observed, None, :]
        diagonal, gram = whitener_statistics(selected[observed - 1], target)
        weights, _ = fit_convex_fusion(np.ones((lookback, 1)), diagonal, gram)
        forecasts.append(covariance_from_whitener(np.einsum("k,kij->ij", weights[0], selected[t])))
        rows.append(weights[0])
    return np.stack(forecasts), np.stack(rows)


def run_state_research(config_path: Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = load_config(config_path)
    base = load_config(Path(config["base_config"]))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "selection.json").exists():
        raise RuntimeError("Frozen selection already exists: use a separately named experiment.")
    (output / "models").mkdir(exist_ok=True)
    archive = Path(base["data"]["raw_dir"]) / "10_Industry_Portfolios_daily_CSV.zip"
    hash_value = hashlib.sha256(archive.read_bytes()).hexdigest()
    old_meta = json.loads((Path(base["project"]["output_dir"]) / "run_metadata.json").read_text(encoding="utf-8"))
    if hash_value != old_meta["source_sha256"]:
        raise ValueError("Frozen data changed.")
    old_outputs = [Path("outputs") / p for p in ("preliminary", "window_ablation", "feature_search", "forward_audit")]
    original_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for directory in old_outputs for p in directory.rglob("*") if p.is_file()}
    specs = research_candidates(config)
    protocol = dict(config=config, base_config=base, candidates=specs,
        created_at_utc=datetime.now(timezone.utc).isoformat(), source_sha256=hash_value,
        selection_rule="mean annual validation excess NLL over annual fixed covariance fusion + 0.25 std annual excess",
        selection_years=config["validation_years"], later_outcomes_used_for_selection=False,
        evidence_status="all later evaluation periods already seen in previous experiments; exploratory",
        references=["https://stanford.edu/~boyd/papers/cov_pred_finance.html",
                    "https://stanford.edu/~boyd/papers/forecasting_covariances.html",
                    "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1365738",
                    "https://ora.ox.ac.uk/objects/uuid:64e98570-6bdf-4a6a-9067-c8cd5b24d8d5"],
        preserved_artifact_hashes=original_hashes)
    (output / "research_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    frame, _ = parse_industry_zip(archive)
    frame = frame.loc[base["data"]["history_start"]:]
    validate_return_matrix(frame)
    print(f"Frozen research search: {len(specs)} candidates, {len(config['validation_years'])} chronological validation years", flush=True)
    prepared = prepare_research(frame, base, config)
    horizon = base["forecast"]["horizon"]
    validation_all = split_prediction_indices(frame.index, f"{min(config['validation_years'])}-01-01",
        f"{max(config['validation_years'])}-12-31", horizon, 20, prepared["eligible"])
    validation_rows, solver_rows, fold_rows = [], [], []
    for year in config["validation_years"]:
        train = training_indices(frame, prepared, year, base, config)
        val = validation_all[frame.index[validation_all].year == year]
        fixed_cov_weight = fit_fixed_fusion(prepared["arrays"]["covariance"][train], prepared["arrays"]["realized"][train])
        fixed_cov = np.einsum("k,bkij->bij", fixed_cov_weight, prepared["arrays"]["covariance"][val])
        baseline_score = covariance_nll(fixed_cov, prepared["arrays"]["realized"][val]).mean()
        anchor, _ = fit_convex_fusion(np.ones((len(train), 1)), prepared["diagonal"][train], prepared["gram"][train])
        fold_rows.append(dict(year=year, train_origins=len(train), validation_origins=len(val),
            train_first=str(frame.index[train[0]].date()), train_last=str(frame.index[train[-1]].date()),
            train_label_last=str(frame.index[train[-1] + horizon].date()),
            decision_first=str(frame.index[val[0]].date()), decision_last=str(frame.index[val[-1]].date())))
        for spec in specs:
            model, info = fit_research_model(prepared, train, spec, config, anchor[0])
            prediction = predict_research_model(model, prepared, val, spec)[0]
            score = covariance_nll(prediction, prepared["arrays"]["realized"][val]).mean()
            validation_rows.append(dict(method=spec["name"], year=year, nll=score,
                                        excess_over_fixed=score - baseline_score, origins=len(val)))
            solver_rows.append(dict(stage="validation", method=spec["name"], year=year,
                                    fusion_iterations=info["fusion"]["iterations"],
                                    scale_iterations=info["calibration"].get("iterations", 0)))
            model.save(output / "models" / f"validation_{year}_{spec['name']}.npz")
        pd.DataFrame(validation_rows).to_csv(output / "validation_by_year.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(output / "chronological_folds.csv", index=False)
        print(f"Validation year {year}: {len(train)} past-only training origins; {len(val)} forecasts; all {len(specs)} fits completed", flush=True)
    validation_frame = pd.DataFrame(validation_rows)
    search_rows = []
    for spec in specs:
        scores = validation_frame[validation_frame.method == spec["name"]]
        excess = scores.excess_over_fixed.to_numpy()
        search_rows.append(dict(**spec, validation_nll=scores.nll.mean(),
            mean_excess_over_fixed=excess.mean(), std_annual_excess=excess.std(ddof=1),
            selection_score=excess.mean() + config["selection_std_penalty"] * excess.std(ddof=1)))
    search = pd.DataFrame(search_rows).sort_values("selection_score")
    search.to_csv(output / "validation_search.csv", index=False)
    primary = str(search.iloc[0]["name"])
    selected_modes = {mode: str(search[search["mode"] == mode].iloc[0]["name"]) for mode in config["candidate_modes"]}
    pure_joint = str(search[(search["mode"] == "joint") & (~search.calibrate)].iloc[0]["name"])
    selection = dict(primary=primary, selected_modes=selected_modes, selected_pure_joint=pure_joint,
        selected_at_utc=datetime.now(timezone.utc).isoformat(), later_outcomes_used_for_selection=False,
        protocol_sha256=hashlib.sha256((output / "research_protocol.json").read_bytes()).hexdigest())
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"Selection locked BEFORE later scores: {primary}; best joint={selected_modes['joint']}", flush=True)
    old_test = prepared["arrays"]["splits"]["test"]
    forward = split_prediction_indices(frame.index, "2026-01-01", str(frame.index[-1].date()),
                                        horizon, 20, prepared["eligible"])
    evaluation_indices = np.r_[old_test, forward]
    all_predictions = {s["name"]: np.full((len(evaluation_indices), frame.shape[1], frame.shape[1]), np.nan) for s in specs}
    baseline_names = [*prepared["names"][:-1], "EWMA-0.94", "EWMA-0.97", "EqualFusion-LW", "FixedFusion-LW", "HistoricalLoss-LW", "CM-IEWMA"]
    all_predictions.update({n: np.full((len(evaluation_indices), frame.shape[1], frame.shape[1]), np.nan) for n in baseline_names})
    weight_rows, scale_rows, cm_rows, evaluation_folds = [], [], [], []
    for year in config["evaluation_years"]:
        positions = np.flatnonzero(frame.index[evaluation_indices].year == year)
        indices = evaluation_indices[positions]
        if not len(indices):
            continue
        train = training_indices(frame, prepared, year, base, config)
        anchor, _ = fit_convex_fusion(np.ones((len(train), 1)), prepared["diagonal"][train], prepared["gram"][train])
        evaluation_folds.append(dict(year=year, train_origins=len(train), evaluation_origins=len(indices),
            train_label_last=str(frame.index[train[-1] + horizon].date()), decision_first=str(frame.index[indices[0]].date())))
        for spec in specs:
            model, info = fit_research_model(prepared, train, spec, config, anchor[0])
            prediction, weights, scales, states = predict_research_model(model, prepared, indices, spec)
            all_predictions[spec["name"]][positions] = prediction
            model.save(output / "models" / f"evaluation_{year}_{spec['name']}.npz")
            solver_rows.append(dict(stage="evaluation", method=spec["name"], year=year,
                fusion_iterations=info["fusion"]["iterations"], scale_iterations=info["calibration"].get("iterations", 0)))
            if spec["name"] == primary:
                weight_rows.extend(dict(date=str(frame.index[t].date()), **dict(zip(prepared["names"], w, strict=True))) for t, w in zip(indices, weights, strict=True))
                scale_rows.extend(dict(date=str(frame.index[t].date()), variance_multiplier=float(a)) for t, a in zip(indices, scales, strict=True))
        for k, name in enumerate(prepared["names"][:-1]):
            all_predictions[name][positions] = prepared["experts"][indices, k]
        for decay, covariance in prepared["arrays"]["ewma"].items():
            all_predictions[f"EWMA-{decay}"][positions] = covariance[indices]
        fixed_weight = fit_fixed_fusion(prepared["arrays"]["covariance"][train], prepared["arrays"]["realized"][train])
        all_predictions["FixedFusion-LW"][positions] = np.einsum("k,bkij->bij", fixed_weight, prepared["arrays"]["covariance"][indices])
        all_predictions["EqualFusion-LW"][positions] = prepared["arrays"]["covariance"][indices].mean(1)
        historical_weights = softmax(-prepared["arrays"]["features"][indices, 6:] /
                                     old_meta["historical_loss_temperature"], axis=1)
        all_predictions["HistoricalLoss-LW"][positions] = np.einsum("bk,bkij->bij", historical_weights,
                                                                  prepared["arrays"]["covariance"][indices])
        all_predictions["CM-IEWMA"][positions], cm_weights = cm_iewma_predictions(prepared, frame.to_numpy(), indices, config["cm_lookback"])
        cm_rows.extend(dict(date=str(frame.index[t].date()), **{n: float(w) for n, w in zip(prepared["names"][4:9], row, strict=True)}) for t, row in zip(indices, cm_weights, strict=True))
        print(f"Evaluation {year} completed with annual refit; {len(indices)} origins", flush=True)
    pd.DataFrame(solver_rows).to_csv(output / "solver_diagnostics.csv", index=False)
    pd.DataFrame(evaluation_folds).to_csv(output / "evaluation_folds.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(output / "selected_weights.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(output / "selected_variance_scale.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(output / "cm_iewma_weights.csv", index=False)
    realized = prepared["arrays"]["realized"][evaluation_indices]
    losses = {name: covariance_nll(matrix, realized) for name, matrix in all_predictions.items()}
    losses_frame = pd.DataFrame(losses, index=frame.index[evaluation_indices]).rename_axis("decision_date")
    losses_frame.to_csv(output / "evaluation_losses.csv")
    test_mask = frame.index[evaluation_indices].year < 2026
    summary = pd.DataFrame([dict(method=name, test_2021_2025_nll=loss[test_mask].mean(),
        diagnostic_2026_nll=loss[~test_mask].mean(), primary=name == primary,
        method_type="candidate" if name in [s["name"] for s in specs] else "baseline") for name, loss in losses.items()]).sort_values("test_2021_2025_nll")
    summary.to_csv(output / "summary.csv", index=False)
    yearly = pd.DataFrame([dict(method=name, year=year, nll=loss[frame.index[evaluation_indices].year == year].mean())
        for name, loss in losses.items() for year in config["evaluation_years"]])
    yearly.to_csv(output / "yearly_nll.csv", index=False)
    selected_names = list(dict.fromkeys([primary, *selected_modes.values(), pure_joint]))
    comparators = baseline_names + [n for n in selected_names if n != primary]
    boot_rows = []
    for block in base["evaluation"]["bootstrap_block_lengths"]:
        part = []
        for name in comparators:
            diff = losses[name][test_mask] - losses[primary][test_mask]
            low, high, p = paired_block_bootstrap(diff, block,
                base["evaluation"]["bootstrap_repetitions"], base["evaluation"]["bootstrap_seed"])
            part.append(dict(primary=primary, baseline=name, block_length=block,
                improvement=diff.mean(), ci_lower=low, ci_upper=high, p_one_sided=p))
        for row, p in zip(part, holm_adjust([r["p_one_sided"] for r in part]), strict=True):
            row["p_holm"] = p
        boot_rows += part
    pd.DataFrame(boot_rows).to_csv(output / "paired_bootstrap.csv", index=False)
    backtest_rows = []
    for segment, mask in [("2021_2025", test_mask), ("2026", ~test_mask)]:
        for name, lag, cost in itertools.product(list(dict.fromkeys([*selected_names, *baseline_names])),
            base["backtest"]["execution_lag_days"], base["backtest"]["transaction_cost_bps"]):
            keep = mask & (evaluation_indices + horizon + lag < len(frame))
            daily, dates, turnover = run_gmv_backtest(frame.to_numpy(), evaluation_indices[keep], all_predictions[name][keep], horizon, lag, cost)
            backtest_rows.append(dict(segment=segment, method=name, lag_days=lag, cost_bps=cost,
                annualized_volatility=annualized_volatility(daily), maximum_drawdown=maximum_drawdown(daily),
                cvar_95=empirical_cvar(daily), mean_turnover=np.mean(turnover), days=len(daily)))
            if name == primary:
                pd.DataFrame(dict(date=frame.index[dates], net_return=daily)).to_csv(
                    output / f"selected_daily_returns_{segment}_lag{lag}_cost{cost}.csv", index=False)
    backtest = pd.DataFrame(backtest_rows)
    backtest.to_csv(output / "backtest_summary.csv", index=False)
    # Deployment refit uses every fully matured label as of the archive's final day.
    spec = next(s for s in specs if s["name"] == primary)
    last = len(frame) - 1
    deploy_start = f"{max(2005, frame.index[-1].year - config['training_history_years'])}-01-01"
    deploy_train = split_prediction_indices(frame.index, deploy_start, str(frame.index[-1].date()), horizon, 5, prepared["eligible"])
    anchor, _ = fit_convex_fusion(np.ones((len(deploy_train), 1)), prepared["diagonal"][deploy_train], prepared["gram"][deploy_train])
    deployment, _ = fit_research_model(prepared, deploy_train, spec, config, anchor[0])
    deployment.save(output / "models" / "deployment_model.npz")
    forecast, weights, scale, probabilities = predict_research_model(deployment, prepared, np.array([last]), spec)
    pd.DataFrame(forecast[0], index=frame.columns, columns=frame.columns).to_csv(output / "latest_covariance_forecast.csv")
    deployed = dict(as_of=str(frame.index[-1].date()), model=primary, expert_names=prepared["names"],
        industry_columns=frame.columns.tolist(), market_features=MARKET_NAMES,
        reliability_features=[f"relative_nll_m60_{n}" for n in prepared["names"]],
        mode=spec["mode"], variance_multiplier=float(scale[0]), weights=weights[0].tolist(),
        state_probabilities=probabilities[0].tolist(), train_origins=len(deploy_train),
        training_label_last=str(frame.index[deploy_train[-1] + horizon].date()),
        horizon_trading_days=horizon, status="research model; frozen archive forecast, not current live market data")
    (output / "deployment_manifest.json").write_text(json.dumps(deployed, indent=2), encoding="utf-8")
    np.savez_compressed(output / "selected_evaluation_covariances.npz", indices=evaluation_indices,
        primary=all_predictions[primary], cm_iewma=all_predictions["CM-IEWMA"], fixed=all_predictions["FixedFusion-LW"])
    display = summary[summary.method.isin([*selected_names, *baseline_names])]
    report = ["# 市场状态引导的凸白化融合：研究实验", "",
        "本轮基于原始研究资料重新设计统计方法。所有候选与选择规则先保存；2012—2020 年九个年度前推验证决定配置，随后才评价 2021—2026 年。后者此前已被查看，所以本轮仍属探索，2026 年也不再声称独立留出。", "",
        f"最终按预定稳健验证规则选定 `{primary}`。最优联合分支为 `{selected_modes['joint']}`；如果两者不同，不为保持互补性设想而强行选联合模型。", "",
        "## 候选与选择", "", markdown_table(search.head(8)[["name", "mean_excess_over_fixed", "std_annual_excess", "selection_score"]]), "",
        "每年只使用当年之前已经完整成熟的标签，至多回看 10 个自然年；不使用未来年参数。验证按年度等权，评分为相对当年固定协方差融合的平均 NLL 差 + 0.25×年度差的标准差。主模型、全球/市场/可靠性/联合及纯联合分支均在评价之前锁定。", "",
        "## 后续时期成绩", "", markdown_table(display[["method", "test_2021_2025_nll", "diagnostic_2026_nll"]]), "",
        "所有模型共用 62 个旧测试起点和 8 个 2026 起点。每年按预定规则重拟合参数，固定融合基线同样重拟合；不能与旧的全程冻结神经模型分数直接当成同设定比较。全部 26 个候选结果均保留，不按评价分数更换推荐模型。", "",
        "## 组合风险", "", markdown_table(backtest[(backtest.segment == "2021_2025") & (backtest.lag_days == 0) & (backtest.cost_bps == 0)][["method", "annualized_volatility", "maximum_drawdown", "cvar_95"]]), "",
        "单纯整体方差校准不直接改变 GMV 权重。NLL、投资组合波动、回撤与尾部风险须分别评价。区块 bootstrap 和 Holm 校正保存在 paired_bootstrap.csv，未消除整个研究过程的多轮选择偏差。", "",
        "## 可用模型", "",
        "models/deployment_model.npz 是按已锁定配置、只用截至原始快照末日已成熟标签重拟合的研究模型；deployment_manifest.json 记录资产顺序、特征、专家、训练标签末日与预测日期。latest_covariance_forecast.csv 是未来 20 日平均日频风险矩阵预测，不是 20 日累计收益协方差。当前快照截至 2026-08-31，不代表实时市场。", "",
        "## 研究出处与完整方法", "",
        "状态分组、PCA、正则化强度、20 日标签与凸风险水平校准是本项目的改造，而不是声称复现了论文的全部实验。IEWMA 与 CM-IEWMA 对照采用其基本公式与行业半衰期设置，并明确记录本项目的初始化和时间边界。完整推导见 docs/state_whitener_method.md。", "",
        "- [Johansson 等：多尺度 IEWMA 与白化融合](https://stanford.edu/~boyd/papers/cov_pred_finance.html)",
        "- [Barratt 与 Boyd：凸优化协方差预测](https://stanford.edu/~boyd/papers/forecasting_covariances.html)",
        "- [Corsi：多时间尺度波动的 HAR 思路](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1365738)",
        "- [Barndorff-Nielsen 等：下行半方差](https://ora.ox.ac.uk/objects/uuid:64e98570-6bdf-4a6a-9067-c8cd5b24d8d5)"]
    (output / "research_report.md").write_text("\n".join(report), encoding="utf-8")
    for filename, digest in original_hashes.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Preserved artifact changed: {filename}")
    (output / "run_metadata.json").write_text(json.dumps(dict(primary=primary,
        completed_at_utc=datetime.now(timezone.utc).isoformat(), previous_artifacts_unchanged=True,
        validation_origins=len(validation_all), evaluation_origins=len(evaluation_indices),
        later_outcomes_used_for_selection=False, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest()), indent=2), encoding="utf-8")
    print(display.to_string(index=False), flush=True)
    print(f"Research completed: {output.resolve()}", flush=True)

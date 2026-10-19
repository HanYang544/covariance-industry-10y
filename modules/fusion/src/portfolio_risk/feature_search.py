"""Predeclared feature/tuning exploration; validation selection precedes test scoring."""

from __future__ import annotations

import copy
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
from portfolio_risk.data.french import parse_industry_zip, validate_return_matrix
from portfolio_risk.evaluation.metrics import annualized_volatility, empirical_cvar, maximum_drawdown
from portfolio_risk.experiment import (build_arrays, covariance_nll, fit_fixed_fusion, holm_adjust,
                                       paired_block_bootstrap, run_gmv_backtest)
from portfolio_risk.losses.gaussian import gaussian_covariance_nll
from portfolio_risk.models.anchored import AnchoredRiskGate
from portfolio_risk.risk_features import EXPANDED_MARKET_NAMES, expanded_market_features, matured_reliability
from portfolio_risk.utils.random import set_random_seed
from portfolio_risk.window_ablation import markdown_table


def candidates(search: dict) -> list[dict]:
    result = []
    for mode in search["modes"]:
        windows = search["reliability_windows"] if mode == "expanded_joint" else [60]
        strengths = search["gate_strengths"] if mode == "expanded_joint" else [1.0]
        for architecture, window, strength, calibrate in itertools.product(
                search["architectures"], windows, strengths, search["risk_calibration"]):
            name = f"{mode}-{architecture}-r{window}-g{strength:g}-c{int(calibrate)}"
            result.append(dict(name=name, mode=mode, architecture=architecture,
                               reliability_window=window, strength=strength, calibrate=calibrate))
    return result


def train_candidate(arrays, standardized, anchor, spec, search, seed, path):
    set_random_seed(seed)
    model = AnchoredRiskGate(standardized.shape[1], anchor, spec["architecture"],
                            search["hidden_dim"], spec["strength"], spec["calibrate"],
                            search["max_abs_log_scale"]).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=search["learning_rate"],
                                 weight_decay=search["weight_decay"])
    train, val = arrays["splits"]["train"], arrays["splits"]["validation"]
    x = torch.from_numpy(standardized)
    covariance = torch.from_numpy(arrays["covariance"])
    realized = torch.from_numpy(arrays["realized"])

    def predict(indices):
        w, log_scale = model(x[indices])
        matrix = torch.einsum("bk,bkij->bij", w, covariance[indices]) * log_scale.exp()[:, None, None]
        return w, log_scale, matrix

    def score(indices):
        return gaussian_covariance_nll(predict(indices)[2], realized[indices]).mean().item()

    # Epoch 0 is a genuine candidate, preventing forced degradation of the anchor.
    with torch.no_grad():
        best = score(val)
        history = [dict(epoch=0, train_nll=score(train), validation_nll=best)]
    best_state, best_epoch, stale = copy.deepcopy(model.state_dict()), 0, 0
    rng = np.random.default_rng(seed)
    for epoch in range(1, search["max_epochs"] + 1):
        model.train()
        order = rng.permutation(train)
        for begin in range(0, len(order), 32):
            index = order[begin:begin + 32]
            weights, log_scale, matrix = predict(index)
            loss = (gaussian_covariance_nll(matrix, realized[index]).mean() +
                    search["regularization"] * model.penalty(weights, log_scale))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_score, train_score = score(val), score(train)
        history.append(dict(epoch=epoch, train_nll=train_score, validation_nll=val_score))
        if val_score < best - 1e-7:
            best, best_state, best_epoch, stale = val_score, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= search["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    torch.save(dict(state_dict=best_state, spec=spec, seed=seed,
                    best_epoch=best_epoch, input_dim=standardized.shape[1]), path)
    # Exclude test labels and test predictions from fitting/selection completely.
    with torch.no_grad():
        validation_matrix = predict(val)[2].numpy()
    return model, validation_matrix, best_epoch, history


def run_feature_search(config_path: Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    search_config = load_config(config_path)
    config = load_config(Path(search_config["base_config"]))
    search = search_config["search"]
    output = Path(search_config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "selection.json").exists():
        raise RuntimeError("A completed selection exists. Use a new output directory for a new experiment.")
    (output / "checkpoints").mkdir(exist_ok=True)
    raw = Path(config["data"]["raw_dir"]) / "10_Industry_Portfolios_daily_CSV.zip"
    raw_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
    reference = Path(config["project"]["output_dir"])
    expected_hash = json.loads((reference / "run_metadata.json").read_text(encoding="utf-8"))["source_sha256"]
    if raw_hash != expected_hash:
        raise ValueError("Data archive differs from the frozen preliminary run.")
    original_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in reference.rglob("*") if p.is_file()}
    frame, raw_header = parse_industry_zip(raw)
    frame = frame.loc[config["data"]["history_start"]:config["data"]["splits"]["test"][1]]
    validate_return_matrix(frame)
    specs = candidates(search)
    protocol = dict(search_config=search_config, base_config=config, candidates=specs,
                    raw_sha256=raw_hash, status="exploratory_reused_test_period",
                    selection_rule="minimum validation NLL of 5-seed covariance ensemble; expanded_joint only",
                    test_selection_used=False, market_feature_count=len(EXPANDED_MARKET_NAMES),
                    created_at_utc=datetime.now(timezone.utc).isoformat(), raw_header=raw_header,
                    original_artifact_hashes=original_hashes)
    (output / "search_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Protocol saved before training: {len(specs)} configurations, {len(search['random_seeds'])} seeds", flush=True)
    arrays = build_arrays(frame, config)
    train, validation, test = [arrays["splits"][s] for s in ("train", "validation", "test")]
    all_indices = np.r_[train, validation, test]
    expanded = np.full((len(frame), len(EXPANDED_MARKET_NAMES)), np.nan)
    for t in all_indices:
        expanded[t] = expanded_market_features(frame.to_numpy()[t - 251:t + 1], arrays["covariance"][t])
    anchor = fit_fixed_fusion(arrays["covariance"][train], arrays["realized"][train])
    features_by_case = {}
    names_by_case = {}
    for mode in search["modes"]:
        for window in search["reliability_windows"] if mode == "expanded_joint" else [60]:
            q = np.full((len(frame), 4), np.nan)
            for t in all_indices:
                q[t] = matured_reliability(arrays["losses"], t, config["forecast"]["horizon"], window)
            market = expanded if mode.startswith("expanded") else arrays["features"][:, :6]
            features = market if mode == "expanded_market" else np.column_stack([market, q])
            names = list(EXPANDED_MARKET_NAMES if mode.startswith("expanded") else EXPANDED_MARKET_NAMES[:6])
            if mode != "expanded_market":
                names += [f"relative_loss_{w}_m{window}" for w in config["forecast"]["lookback_windows"]]
            mean, scale = features[train].mean(0), np.maximum(features[train].std(0), 1e-12)
            standardized = (features - mean) / scale
            if not np.isfinite(standardized[all_indices]).all():
                raise ValueError("Nonfinite standardized features.")
            features_by_case[mode, window] = standardized
            names_by_case[mode, window] = names
            np.savez(output / f"scaler_{mode}_r{window}.npz", mean=mean, scale=scale, names=names)
            pd.DataFrame(features[all_indices], index=frame.index[all_indices], columns=names).to_csv(
                output / f"features_{mode}_r{window}.csv")
    trained, validation_matrices, history_rows, seed_rows, candidate_rows = {}, {}, [], [], []
    for number, spec in enumerate(specs, 1):
        name = spec["name"]
        x = features_by_case[spec["mode"], spec["reliability_window"]]
        models, matrices = [], []
        for seed in search["random_seeds"]:
            model, val_matrix, epoch, history = train_candidate(
                arrays, x, anchor, spec, search, seed, output / "checkpoints" / f"{name}_seed{seed}.pt")
            models.append(model)
            matrices.append(val_matrix)
            seed_rows.append(dict(method=name, seed=seed, best_epoch=epoch,
                                  validation_nll=covariance_nll(val_matrix, arrays["realized"][validation]).mean()))
            history_rows.extend(dict(method=name, seed=seed, **row) for row in history)
        ensemble = np.mean(matrices, axis=0)
        score = covariance_nll(ensemble, arrays["realized"][validation]).mean()
        trained[name], validation_matrices[name] = models, ensemble
        candidate_rows.append(dict(**spec, feature_count=x.shape[1], validation_nll=score,
                                   mean_best_epoch=np.mean([r["best_epoch"] for r in seed_rows if r["method"] == name])))
        pd.DataFrame(candidate_rows).to_csv(output / "validation_search.csv", index=False)
        pd.DataFrame(history_rows).to_csv(output / "training_history.csv", index=False)
        print(f"[{number}/{len(specs)}] {name}: validation={score:.6f}", flush=True)
    eligible = [r for r in candidate_rows if r["mode"] == "expanded_joint"]
    primary = min(eligible, key=lambda r: r["validation_nll"])["name"]
    pure_fusion = min([r for r in eligible if not r["calibrate"]], key=lambda r: r["validation_nll"])["name"]
    controls = {mode: min([r for r in candidate_rows if r["mode"] == mode],
                          key=lambda r: r["validation_nll"])["name"]
                for mode in ("expanded_market", "original_joint")}
    selection = dict(primary=primary, selected_pure_fusion=pure_fusion, controls=controls,
                     anchor=anchor.tolist(), test_selection_used=False,
                     selected_at_utc=datetime.now(timezone.utc).isoformat(),
                     protocol_sha256=hashlib.sha256((output / "search_protocol.json").read_bytes()).hexdigest())
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"Validation selection frozen before test scoring: {primary}", flush=True)
    test_covariances, validation_losses, test_losses = {}, {}, {}
    for spec in specs:
        name = spec["name"]
        x = torch.from_numpy(features_by_case[spec["mode"], spec["reliability_window"]][test])
        covariance = torch.from_numpy(arrays["covariance"][test])
        predicted, weights, scales = [], [], []
        for seed, model in zip(search["random_seeds"], trained[name], strict=True):
            with torch.no_grad():
                w, log_scale = model(x)
                matrix = torch.einsum("bk,bkij->bij", w, covariance) * log_scale.exp()[:, None, None]
            predicted.append(matrix.numpy())
            weights.append(w.numpy())
            scales.append(log_scale.exp().numpy())
            for row in seed_rows:
                if row["method"] == name and row["seed"] == seed:
                    row["test_nll"] = covariance_nll(predicted[-1], arrays["realized"][test]).mean()
        test_covariances[name] = np.mean(predicted, axis=0)
        test_losses[name] = covariance_nll(test_covariances[name], arrays["realized"][test])
        validation_losses[name] = covariance_nll(validation_matrices[name], arrays["realized"][validation])
        # Scale-weight interaction: ensemble uses mean(a*w), not mean(a)*mean(w).
        coefficient = np.mean(np.asarray(weights) * np.asarray(scales)[:, :, None], axis=0)
        pd.DataFrame(coefficient, index=frame.index[test], columns=["20", "60", "120", "252"]).to_csv(
            output / f"{name}_covariance_coefficients.csv")
    pd.DataFrame(seed_rows).to_csv(output / "seed_summary.csv", index=False)
    baseline_covariances = {}
    for k, window in enumerate(config["forecast"]["lookback_windows"]):
        baseline_covariances[f"LW-{window}"] = arrays["covariance"][:, k]
    for decay, matrix in arrays["ewma"].items():
        baseline_covariances[f"EWMA-{decay}"] = matrix
    for name, weight in {"FixedFusion": anchor, "EqualFusion": np.full(4, .25)}.items():
        baseline_covariances[name] = np.einsum("k,bkij->bij", weight, arrays["covariance"])
    best_temperature = min(config["evaluation"]["reliability_temperatures"], key=lambda temp:
        covariance_nll(np.einsum("bk,bkij->bij", softmax(-arrays["features"][validation, 6:] / temp, axis=-1),
                                 arrays["covariance"][validation]), arrays["realized"][validation]).mean())
    baseline_covariances["HistoricalLoss"] = np.einsum("bk,bkij->bij",
        softmax(-arrays["features"][:, 6:] / best_temperature, axis=-1), arrays["covariance"])
    for name, matrix in baseline_covariances.items():
        test_covariances[name] = matrix[test]
        test_losses[name] = covariance_nll(matrix[test], arrays["realized"][test])
        validation_losses[name] = covariance_nll(matrix[validation], arrays["realized"][validation])
    # Compare original learned gates using their actual covariance ensembles, too.
    for name in (f"{a}-{m}" for a in ("linear", "mlp") for m in ("Market", "Reliability", "Joint")):
        table = pd.read_csv(reference / f"{name}_weights.csv", index_col=0)
        if not np.array_equal(pd.to_datetime(table.index).values, frame.index[test].values):
            raise ValueError("Old and new test origins do not match.")
        matrix = np.einsum("bk,bkij->bij", table.to_numpy(), arrays["covariance"][test])
        method = f"Original-{name}-ensemble"
        test_covariances[method] = matrix
        test_losses[method] = covariance_nll(matrix, arrays["realized"][test])
        # Old validation is average individual losses, explicitly kept separate.
        validation_losses[method] = np.array([np.nan])
    summary = pd.DataFrame([dict(method=name, validation_nll=validation_losses[name].mean(),
                                test_nll=loss.mean(), primary=name == primary,
                                method_type="candidate" if name in trained else "baseline")
                            for name, loss in test_losses.items()]).sort_values("test_nll")
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(test_losses, index=frame.index[test]).rename_axis("decision_date").to_csv(output / "test_losses.csv")
    np.savez_compressed(output / "selected_covariances.npz", dates=frame.index[test].astype(str).to_numpy(dtype=str),
                        primary=test_covariances[primary], pure_fusion=test_covariances[pure_fusion],
                        fixed=test_covariances["FixedFusion"], lw252=test_covariances["LW-252"])
    comparators = list(baseline_covariances) + [n for n in test_losses if n.startswith("Original-")]
    comparators += list(controls.values())
    bootstrap_rows = []
    for block in config["evaluation"]["bootstrap_block_lengths"]:
        block_rows = []
        for baseline in comparators:
            difference = test_losses[baseline] - test_losses[primary]
            low, high, p = paired_block_bootstrap(difference, block,
                config["evaluation"]["bootstrap_repetitions"], config["evaluation"]["bootstrap_seed"])
            block_rows.append(dict(primary=primary, baseline=baseline, block_length=block,
                                   improvement=difference.mean(), ci_lower=low, ci_upper=high, p_one_sided=p))
        for row, corrected in zip(block_rows, holm_adjust([r["p_one_sided"] for r in block_rows]), strict=True):
            row["p_holm"] = corrected
        bootstrap_rows += block_rows
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(output / "paired_bootstrap.csv", index=False)
    yearly = pd.DataFrame([dict(year=year, method=name, nll=loss[frame.index[test].year == year].mean())
                           for name, loss in test_losses.items() for year in range(2021, 2026)])
    yearly.to_csv(output / "yearly_nll.csv", index=False)
    chosen = list(dict.fromkeys([primary, pure_fusion, *controls.values(), *baseline_covariances]))
    backtests = []
    for lag, cost, name in itertools.product(config["backtest"]["execution_lag_days"],
                                           config["backtest"]["transaction_cost_bps"], chosen):
        valid = test + config["forecast"]["horizon"] + lag < len(frame)
        daily, dates, turnover = run_gmv_backtest(frame.to_numpy(), test[valid],
            test_covariances[name][valid], config["forecast"]["horizon"], lag, cost)
        backtests.append(dict(method=name, lag_days=lag, cost_bps=cost,
                             annualized_volatility=annualized_volatility(daily),
                             maximum_drawdown=maximum_drawdown(daily), cvar_95=empirical_cvar(daily),
                             mean_turnover=np.mean(turnover), days=len(daily)))
        pd.DataFrame(dict(date=frame.index[dates], daily_return=daily)).to_csv(
            output / f"backtest_{name}_lag{lag}_cost{cost}.csv", index=False)
    backtest = pd.DataFrame(backtests)
    backtest.to_csv(output / "backtest_summary.csv", index=False)
    report_methods = [primary, pure_fusion, *controls.values(), *baseline_covariances]
    comparison = summary[summary.method.isin(report_methods)].copy()
    primary_nll = summary.set_index("method").loc[primary, "test_nll"]
    wins = [n for n in comparators if test_losses[n].mean() > primary_nll]
    report = ["# 扩展市场风险特征与稳健调参：探索实验", "",
              "本轮是在已经查看过 2021—2025 年结果后设计的探索实验，不能视作全新独立验证。所有配置在训练前写入 search_protocol.json，先按验证集选定模型并保存 selection.json，再计算测试成绩；没有根据测试成绩更换主模型。", "",
              f"市场状态特征从 6 个增至 {len(EXPANDED_MARKET_NAMES)} 个；联合模型另加 4 个历史可靠性特征。共 {len(specs)} 个配置，每个 5 个随机种子，输出为协方差矩阵的算术集成，评分并非各随机种子损失的平均。", "",
              f"验证集选定主模型：`{primary}`。纯融合分支：`{pure_fusion}`。", "",
              markdown_table(comparison[["method", "validation_nll", "test_nll"]]), "",
              f"主模型测试 NLL 为 {primary_nll:.9f}，低于 {len(wins)}/{len(comparators)} 个预定对照（含旧模型矩阵集成和新控制组）；这只是均值比较，不等于所有差异统计显著。", "",
              "## 训练与建模", "",
              "数据、预测期、协方差估计、训练/验证/测试日期与第一轮完全相同。所有输入只由决策日及此前收益构造。可靠性只使用目标区间已经完整结束的样本，回看 20 或 60 个起点。标准化均值与标准差仅用训练起点计算。", "",
              "固定融合权重 beta 仅在训练集优化 NLL。本轮门控以 beta 为锚点，残差输出层权重、偏置初始化为 0，epoch 0 也参与验证早停。线性层直接产生残差；MLP 为 D→8→(4 或 5)，隐藏激活为 tanh。", "",
              "动态权重 u=softmax(log(beta)+z)，w=(1-g)beta+g*u，g∈{0.25,1}。纯融合预测 C=sum(w_k*C_k)。校准版本加一个标量头 a=exp(log(2)*tanh(z_scale))∈(0.5,2)，预测 C=a*sum(w_k*C_k)。该头校正整体风险水平，不能把它的优势归因于多尺度权重机制。", "",
              "训练目标为归一化高斯 NLL + 0.03×[KL(w||beta)+(log a)^2]；Adam 学习率 0.0003、权重衰减 0.001、批量 32、最多 100 轮、耐心 15 轮、梯度范数上限 1。早停及配置选择使用未经正则项修饰的验证 NLL。每个种子分别早停，再集成协方差。", "",
              "## 统计与投资组合评估", "",
              "paired_bootstrap.csv 保留全部对照的配对循环区块 bootstrap，区块 1/3/6，5000 次重采样，并在每种区块长度下进行 Holm 校正。它没有修正整个多轮探索的选择偏差，不是确认性检验。", "",
              markdown_table(bootstrap[(bootstrap.block_length == 3) & bootstrap.baseline.isin(["LW-252", "FixedFusion", *controls.values()])][["baseline", "improvement", "ci_lower", "ci_upper", "p_holm"]]), "",
              "长仓 GMV、权重漂移、换手、0/5/10 bps 成本及 0/1 天延迟计算口径沿用 README。校准系数为正的整体标量，因此对给定权重产生的 GMV 持仓没有直接影响；协方差 NLL 的改进不自动带来投资组合风险指标的改进。", "",
              markdown_table(backtest[(backtest.lag_days == 0) & (backtest.cost_bps == 0)][["method", "annualized_volatility", "maximum_drawdown", "cvar_95"]]), "",
              "## 输出与复核", "",
              "validation_search.csv：训练前预定配置的完整验证结果；summary.csv：全部测试结果，不只保留赢家；seed_summary.csv 与 training_history.csv：种子差异和逐轮轨迹；scaler_*.npz 与 features_*.csv：训练标准化参数和实际输入；checkpoints/：全部模型；selected_covariances.npz：选定预测矩阵；yearly_nll.csv：逐年结果。", "",
              "后续需要预先锁定模型，在未使用的新时间区间或独立资产样本上确认。不能承诺所有指标、所有年份及所有数据都超过每一种基线。"]
    (output / "feature_search_report.md").write_text("\n".join(report), encoding="utf-8")
    for filename, digest in original_hashes.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Original artifact was changed: {filename}")
    (output / "run_metadata.json").write_text(json.dumps(dict(
        completed_at_utc=datetime.now(timezone.utc).isoformat(), original_artifacts_unchanged=True,
        split_samples={k: len(v) for k, v in arrays["splits"].items()},
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        primary=primary, primary_test_nll=primary_nll, mean_nll_wins=len(wins),
        comparison_count=len(comparators), test_selection_used=False), indent=2), encoding="utf-8")
    print(comparison[["method", "validation_nll", "test_nll"]].to_string(index=False), flush=True)
    print(f"Completed: {output.resolve()}", flush=True)

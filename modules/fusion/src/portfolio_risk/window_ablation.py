"""Exploratory shorter-window comparisons and evidence-based first-round diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from scipy.stats import spearmanr

from portfolio_risk.config import load_config
from portfolio_risk.data.french import acquire_industry_10, parse_industry_zip, validate_return_matrix
from portfolio_risk.evaluation.metrics import annualized_volatility, empirical_cvar, maximum_drawdown
from portfolio_risk.experiment import (
    FEATURE_NAMES, build_arrays, covariance_nll, fit_fixed_fusion, holm_adjust,
    paired_block_bootstrap, run_gmv_backtest, train_gate,
)
from portfolio_risk.models.gating import LinearGate, MLPGate
from portfolio_risk.utils.random import set_random_seed


def subset_scales(arrays: dict, indices: list[int]) -> dict:
    """Keep identical dates and market features; recenter reliability for selected scales."""
    selected_reliability = arrays["features"][:, np.array(indices) + 6].copy()
    selected_reliability -= selected_reliability.mean(axis=1, keepdims=True)
    return {
        "covariance": arrays["covariance"][:, indices],
        "realized": arrays["realized"],
        "features": np.column_stack([arrays["features"][:, :6], selected_reliability]),
        "losses": arrays["losses"][:, indices],
        "splits": arrays["splits"], "ewma": arrays["ewma"],
    }


def markdown_table(frame: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(frame.columns) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(
            f"{v:.6f}" if isinstance(v, (float, np.floating)) else str(v) for v in row
        ) + " |")
    return "\n".join(lines)


def original_diagnostics(arrays: dict, frame: pd.DataFrame, config: dict,
                         reference: Path, output: Path) -> dict:
    history = pd.read_csv(reference / "training_history.csv")
    best = history.loc[history.groupby(["method", "seed"]).validation_nll.idxmin()]
    best.to_csv(output / "original_best_epochs.csv", index=False)
    slopes = []
    for (method, seed), curve in history.groupby(["method", "seed"]):
        curve = curve.sort_values("epoch")
        slopes.append({"method": method, "seed": seed,
                       "train_change": curve.iloc[-1].train_nll - curve.iloc[0].train_nll,
                       "validation_change": curve.iloc[-1].validation_nll - curve.iloc[0].validation_nll})
    pd.DataFrame(slopes).to_csv(output / "original_training_changes.csv", index=False)
    records = []
    for split, indices in arrays["splits"].items():
        for k, window in enumerate(config["forecast"]["lookback_windows"]):
            for year in np.unique(frame.index[indices].year):
                chosen = indices[frame.index[indices].year == year]
                records.append({"split": split, "year": year, "window": window,
                                "nll": arrays["losses"][chosen, k].mean(), "origins": len(chosen)})
    pd.DataFrame(records).to_csv(output / "single_window_by_year.csv", index=False)
    # A retrospective oracle is diagnostic only; it is never fed to a predictor.
    oracle_rows, correlation_rows, reliability_rows = [], [], []
    for split, indices in arrays["splits"].items():
        past = arrays["features"][indices, 6:]
        actual = arrays["losses"][indices]
        actual_centered = actual - actual.mean(axis=1, keepdims=True)
        rank_correct = np.mean(past.argmin(axis=1) == actual.argmin(axis=1))
        reliability_rows.append({"split": split, "best_window_match_rate": rank_correct,
                                 "nominal_chance_reference": 0.25,
                                 "reliability_future_spearman": spearmanr(
                                     past.ravel(), actual_centered.ravel()).statistic})
        for k, window in enumerate(config["forecast"]["lookback_windows"]):
            oracle_rows.append({"split": split, "window": window,
                                "future_best_frequency": np.mean(actual.argmin(axis=1) == k),
                                "mean_nll": actual[:, k].mean(),
                                "relative_loss_std": past[:, k].std()})
        feature_frame = pd.DataFrame(arrays["features"][indices], columns=FEATURE_NAMES)
        corr = feature_frame.corr()
        corr.to_csv(output / f"feature_correlation_{split}.csv")
        for market in FEATURE_NAMES[:6]:
            for reliability in FEATURE_NAMES[6:]:
                correlation_rows.append({"split": split, "market_feature": market,
                                         "reliability_feature": reliability,
                                         "correlation": corr.loc[market, reliability]})
    pd.DataFrame(oracle_rows).to_csv(output / "future_best_window_diagnostic.csv", index=False)
    pd.DataFrame(reliability_rows).to_csv(output / "reliability_diagnostic.csv", index=False)
    pd.DataFrame(correlation_rows).to_csv(output / "cross_feature_correlation.csv", index=False)
    validation = arrays["splits"]["validation"]
    in_2020 = frame.index[validation].year == 2020
    attribution = []
    for k, window in enumerate(config["forecast"]["lookback_windows"]):
        values = arrays["losses"][validation, k]
        attribution.append({"window": window, "validation_nll": values.mean(),
                            "validation_before_2020_nll": values[~in_2020].mean(),
                            "validation_2020_nll": values[in_2020].mean()})
    pd.DataFrame(attribution).to_csv(output / "validation_selection_attribution.csv", index=False)
    train = arrays["splits"]["train"]
    mean, scale = arrays["features"][train].mean(0), arrays["features"][train].std(0)
    standardized = (arrays["features"] - mean) / np.maximum(scale, 1e-12)
    initial_scores = []
    for architecture in config["model"]["architectures"]:
        for mode, columns in {"Market": list(range(6)), "Reliability": list(range(6, 10)),
                              "Joint": list(range(10))}.items():
            name = f"{architecture}-{mode}"
            for seed in config["evaluation"]["random_seeds"]:
                set_random_seed(seed)
                model = (LinearGate(len(columns)) if architecture == "linear" else
                         MLPGate(len(columns), config["model"]["hidden_dim"])).double()
                with torch.no_grad():
                    weights = model(torch.from_numpy(standardized[validation][:, columns])).numpy()
                fused = np.einsum("bk,bkij->bij", weights, arrays["covariance"][validation])
                score = covariance_nll(fused, arrays["realized"][validation]).mean()
                trained = best[(best.method == name) & (best.seed == seed)].iloc[0].validation_nll
                initial_scores.append({"method": name, "seed": seed,
                                       "untrained_validation_nll": score,
                                       "best_trained_validation_nll": trained,
                                       "training_improvement": score - trained})
    pd.DataFrame(initial_scores).to_csv(output / "initialization_diagnostic.csv", index=False)
    weight_rows = []
    for path in reference.glob("*_weights.csv"):
        weights = pd.read_csv(path, index_col=0)
        for window in weights:
            weight_rows.append({"method": path.name.removesuffix("_weights.csv"), "window": window,
                                "mean_weight": weights[window].mean(),
                                "weight_std": weights[window].std()})
    pd.DataFrame(weight_rows).to_csv(output / "original_weight_diagnostic.csv", index=False)
    result = {
        "best_epoch_one_count": int((best.epoch == 1).sum()), "trained_model_count": len(best),
        "linear_joint_best_epochs": best[best.method == "linear-Joint"].epoch.tolist(),
        "mlp_joint_best_epochs": best[best.method == "mlp-Joint"].epoch.tolist(),
        "training_improved_validation_worsened": int(sum(
            row["train_change"] < 0 and row["validation_change"] > 0 for row in slopes
        )),
        "untrained_better_than_selected_trained_count": int(sum(
            row["training_improvement"] < 0 for row in initial_scores
        )),
    }
    (output / "diagnostic_metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_case(case: str, windows: list[int], arrays: dict, config: dict,
             output: Path) -> dict:
    directory = output / case
    (directory / "checkpoints").mkdir(parents=True, exist_ok=True)
    k_count = len(windows)
    train, validation, test = [arrays["splits"][s] for s in ["train", "validation", "test"]]
    covariance, realized, features = arrays["covariance"], arrays["realized"], arrays["features"]
    mean, scale = features[train].mean(0), np.maximum(features[train].std(0), 1e-12)
    standardized = (features - mean) / scale
    np.savez(directory / "feature_scaler.npz", mean=mean, scale=scale,
             names=FEATURE_NAMES[:6] + [f"relative_loss_{w}" for w in windows])
    val_losses, test_losses, test_covariances, gate_weights = {}, {}, {}, {}
    for k, window in enumerate(windows):
        name = f"LW-{window}"
        val_losses[name] = arrays["losses"][validation, k]
        test_losses[name] = arrays["losses"][test, k]
        test_covariances[name] = covariance[test, k]
    fixed = fit_fixed_fusion(covariance[train], realized[train])
    weights_by_method = {
        "EqualFusion": np.full((len(features), k_count), 1 / k_count),
        "FixedFusion": np.broadcast_to(fixed, (len(features), k_count)),
    }
    temperatures = []
    for temperature in config["evaluation"]["reliability_temperatures"]:
        weights = softmax(-features[validation, 6:] / temperature, axis=-1)
        score = covariance_nll(np.einsum("bk,bkij->bij", weights, covariance[validation]),
                               realized[validation]).mean()
        temperatures.append((score, temperature))
    temperature = min(temperatures)[1]
    weights_by_method["HistoricalLoss"] = softmax(-features[:, 6:] / temperature, axis=-1)
    for name, weights in weights_by_method.items():
        val_losses[name] = covariance_nll(
            np.einsum("bk,bkij->bij", weights[validation], covariance[validation]), realized[validation]
        )
        test_covariances[name] = np.einsum("bk,bkij->bij", weights[test], covariance[test])
        test_losses[name] = covariance_nll(test_covariances[name], realized[test])
        gate_weights[name] = weights[test]
    seed_records, curves = [], []
    for architecture in config["model"]["architectures"]:
        for mode, columns in {"Market": list(range(6)), "Reliability": list(range(6, 6 + k_count)),
                              "Joint": list(range(6 + k_count))}.items():
            name = f"{architecture}-{mode}"
            losses, weights, validation_scores = [], [], []
            for seed in config["evaluation"]["random_seeds"]:
                prediction, score, history = train_gate(
                    arrays, standardized, columns, architecture, seed, config,
                    directory / "checkpoints" / f"{name}_seed{seed}.pt",
                )
                predicted_covariance = np.einsum("bk,bkij->bij", prediction[test], covariance[test])
                loss = covariance_nll(predicted_covariance, realized[test])
                losses.append(loss)
                weights.append(prediction[test])
                validation_scores.append(score)
                seed_records.append({"method": name, "seed": seed,
                                     "validation_nll": score, "test_nll": loss.mean()})
                curves.extend({"method": name, "seed": seed, **row} for row in history)
                print(f"{case} {name} seed={seed}: validation={score:.6f}, epochs={len(history)}",
                      flush=True)
            test_losses[name] = np.mean(losses, axis=0)
            val_losses[name] = np.asarray(validation_scores)
            gate_weights[name] = np.mean(weights, axis=0)
            test_covariances[name] = np.einsum("bk,bkij->bij", gate_weights[name], covariance[test])
            pd.DataFrame(np.asarray(losses).T, columns=[f"seed_{s}" for s in
                         config["evaluation"]["random_seeds"]]).to_csv(
                directory / f"{name}_seed_losses.csv", index=False
            )
    summary = pd.DataFrame([
        {"case": case, "windows": "/".join(map(str, windows)), "method": name,
         "validation_nll": val_losses[name].mean(), "test_nll": loss.mean(),
         "test_seed_std": np.std([r["test_nll"] for r in seed_records if r["method"] == name],
                                 ddof=1) if name.startswith(("linear-", "mlp-")) else 0.0}
        for name, loss in test_losses.items()
    ])
    primary = min([f"{a}-Joint" for a in config["model"]["architectures"]],
                  key=lambda name: val_losses[name].mean())
    best_single = min([f"LW-{w}" for w in windows], key=lambda name: val_losses[name].mean())
    summary.to_csv(directory / "summary.csv", index=False)
    pd.DataFrame(seed_records).to_csv(directory / "seed_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(directory / "training_history.csv", index=False)
    pd.DataFrame(test_losses).to_csv(directory / "test_losses.csv", index=False)
    for name, weights in gate_weights.items():
        pd.DataFrame(weights, columns=[str(w) for w in windows]).to_csv(
            directory / f"{name}_weights.csv", index=False
        )
    metadata = {"case": case, "windows": windows, "primary_model": primary,
                "best_single_validation": best_single, "fixed_weights": fixed.tolist(),
                "reliability_temperature": temperature,
                "market_feature_windows_unchanged": [20, 120],
                "test_origins": len(test), "feature_count": 6 + k_count,
                "status": "exploratory_after_first_round_test_observation"}
    (directory / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"metadata": metadata, "summary": summary, "losses": test_losses,
            "covariances": test_covariances, "weights": gate_weights}


def run_window_ablation(config_path: Path) -> None:
    specification = load_config(config_path)
    config = load_config(specification["base_config"])
    output, reference = Path(specification["output_dir"]), Path(specification["reference_output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "ablation_report.md").exists():
        raise FileExistsError("This exploratory run already exists; use a new output directory.")
    reference_hashes = {str(path.relative_to(reference)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in reference.rglob("*") if path.is_file()}
    original_meta = json.loads((reference / "run_metadata.json").read_text(encoding="utf-8"))
    if hashlib.sha256(Path(specification["base_config"]).read_bytes()).hexdigest() != original_meta[
        "config_sha256"
    ]:
        raise ValueError("First-round configuration changed; comparison would not be controlled.")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    (output / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (output / "base_config.yaml").write_text(Path(specification["base_config"]).read_text(
        encoding="utf-8"), encoding="utf-8")
    archive, manifest = acquire_industry_10(Path(config["data"]["raw_dir"]))
    if manifest["sha256"] != original_meta["source_sha256"]:
        raise ValueError("Ablation must use the exact first-round data snapshot.")
    full_frame, _ = parse_industry_zip(archive)
    frame = full_frame.loc[config["data"]["history_start"]:config["data"]["splits"]["test"][1]]
    validate_return_matrix(frame)
    arrays = build_arrays(frame, config)
    for split, indices in arrays["splits"].items():
        saved = pd.read_csv(reference / f"{split}_dates.csv", parse_dates=["decision_date"])
        np.testing.assert_array_equal(saved.decision_date.to_numpy(), frame.index[indices].to_numpy())
        saved.to_csv(output / f"{split}_dates.csv", index=False)
    diagnostics = original_diagnostics(arrays, frame, config, reference, output)
    cases = {}
    for case, windows in specification["window_groups"].items():
        indices = [config["forecast"]["lookback_windows"].index(window) for window in windows]
        cases[case] = run_case(case, windows, subset_scales(arrays, indices), config, output)
    original_summary = pd.read_csv(reference / "summary.csv")
    original_summary.insert(0, "windows", "20/60/120/252")
    original_summary.insert(0, "case", "original_4")
    summary = pd.concat([original_summary, *[c["summary"] for c in cases.values()]], ignore_index=True)
    summary.to_csv(output / "all_summary.csv", index=False)
    original_losses = pd.read_csv(reference / "test_losses.csv", index_col=0)
    records = []
    for block in config["evaluation"]["bootstrap_block_lengths"]:
        block_records = []
        for case, result in cases.items():
            primary = result["metadata"]["primary_model"]
            architecture = primary.split("-")[0]
            contrasts = {
                "same_case_FixedFusion": result["losses"]["FixedFusion"],
                "same_case_EqualFusion": result["losses"]["EqualFusion"],
                "same_case_HistoricalLoss": result["losses"]["HistoricalLoss"],
                "same_case_Market": result["losses"][f"{architecture}-Market"],
                "same_case_Reliability": result["losses"][f"{architecture}-Reliability"],
                "original_4_Joint": original_losses[original_meta["primary_model"]].to_numpy(),
                "original_4_FixedFusion": original_losses.FixedFusion.to_numpy(),
            }
            for comparator, baseline in contrasts.items():
                difference = baseline - result["losses"][primary]
                low, high, pvalue = paired_block_bootstrap(
                    difference, block, config["evaluation"]["bootstrap_repetitions"],
                    config["evaluation"]["bootstrap_seed"],
                )
                block_records.append({"case": case, "primary": primary, "comparator": comparator,
                                      "block_length": block, "improvement": difference.mean(),
                                      "ci_lower": low, "ci_upper": high, "p_one_sided": pvalue})
        corrected = holm_adjust([r["p_one_sided"] for r in block_records])
        for record, pvalue in zip(block_records, corrected, strict=True):
            record["p_holm_14"] = pvalue
        records.extend(block_records)
    bootstrap = pd.DataFrame(records)
    bootstrap.to_csv(output / "paired_bootstrap.csv", index=False)
    test = arrays["splits"]["test"]
    yearly = pd.DataFrame([
        {"case": case, "method": method, "year": int(year),
         "nll": loss[frame.index[test].year == year].mean()}
        for case, result in cases.items() for method, loss in result["losses"].items()
        for year in np.unique(frame.index[test].year)
    ])
    yearly.to_csv(output / "yearly_nll.csv", index=False)
    backtests = []
    for case, result in cases.items():
        primary = result["metadata"]["primary_model"]
        for lag in config["backtest"]["execution_lag_days"]:
            valid = test + config["forecast"]["horizon"] + lag < len(frame)
            for cost in config["backtest"]["transaction_cost_bps"]:
                for method in ["EqualFusion", "FixedFusion", "HistoricalLoss", primary]:
                    daily, days, turnover = run_gmv_backtest(
                        frame.to_numpy(), test[valid], result["covariances"][method][valid],
                        config["forecast"]["horizon"], lag, cost,
                    )
                    backtests.append({"case": case, "method": method, "lag_days": lag,
                                      "cost_bps": cost, "annualized_volatility": annualized_volatility(daily),
                                      "maximum_drawdown": maximum_drawdown(daily),
                                      "cvar_95": empirical_cvar(daily), "mean_turnover": np.mean(turnover),
                                      "observations": len(daily)})
    backtest_table = pd.DataFrame(backtests)
    backtest_table.to_csv(output / "backtest_summary.csv", index=False)
    write_ablation_report(output, reference, original_meta, cases, summary, bootstrap,
                          backtest_table, diagnostics)
    write_ablation_figures(output, reference, cases, summary, bootstrap)
    for relative, digest in reference_hashes.items():
        assert hashlib.sha256((reference / relative).read_bytes()).hexdigest() == digest
    (output / "reference_artifact_hashes.json").write_text(
        json.dumps(reference_hashes, indent=2), encoding="utf-8"
    )
    print("Window ablation complete; original first-round artifacts are unchanged.", flush=True)
    print(summary[summary.method.isin(["FixedFusion", "linear-Joint", "mlp-Joint"])]
          .to_string(index=False), flush=True)


def write_ablation_report(output: Path, reference: Path, original_meta: dict, cases: dict,
                          summary: pd.DataFrame, bootstrap: pd.DataFrame,
                          backtests: pd.DataFrame, diagnostics: dict) -> None:
    primary_rows = []
    for case, primary in {"original_4": original_meta["primary_model"], **{
        name: result["metadata"]["primary_model"] for name, result in cases.items()
    }}.items():
        chosen = summary[(summary.case == case) & (summary.method == primary)].iloc[0]
        fixed = summary[(summary.case == case) & (summary.method == "FixedFusion")].iloc[0]
        primary_rows.append({"窗口": chosen.windows, "验证所选模型": primary,
                             "验证 NLL": chosen.validation_nll, "联合测试 NLL": chosen.test_nll,
                             "固定融合测试 NLL": fixed.test_nll})
    core = bootstrap[bootstrap.block_length == 3]
    table = pd.DataFrame(primary_rows)
    table.to_csv(output / "selected_comparison.csv", index=False)
    reliable = pd.read_csv(output / "reliability_diagnostic.csv")
    test_reliability = reliable[reliable.split == "test"].iloc[0]
    validation_reliability = reliable[reliable.split == "validation"].iloc[0]
    cross_correlation = pd.read_csv(output / "cross_feature_correlation.csv")
    maximum_cross_correlation = cross_correlation[
        cross_correlation.split == "train"
    ].correlation.abs().max()
    year_window = pd.read_csv(output / "single_window_by_year.csv")
    validation_years = year_window[year_window.split == "validation"].pivot(
        index="year", columns="window", values="nll"
    ).reset_index()
    validation_years.columns = ["年份", *[f"LW-{w}" for w in validation_years.columns[1:]]]
    original_backtest = pd.read_csv(reference / "backtest_summary.csv")
    original_backtest.insert(0, "case", "original_4")
    backtest_all = pd.concat([original_backtest, backtests], ignore_index=True)
    main_backtest = backtest_all[(backtest_all.lag_days == 0) & (backtest_all.cost_bps == 0) & (
        (backtest_all.method == "FixedFusion") | backtest_all.apply(
            lambda r: r.method == (original_meta["primary_model"] if r.case == "original_4" else
                                   cases[r.case]["metadata"]["primary_model"]), axis=1
        )
    )][["case", "method", "annualized_volatility", "maximum_drawdown", "mean_turnover"]]
    lines = [
        "# 原实验原因分析与短窗口融合探索", "",
        "## 实验性质与控制条件", "",
        "本轮在观察首轮测试结果之后提出，只能作为探索性分析。仍使用 2021—2025 年的同一"
        "62 个测试区间，不能当作新增独立样本或确认性检验。窗口取 20/60 日与 20/60/120 日。",
        "保持历史数据版本、预测日期、20 日标签、6 个市场状态特征、训练参数和 5 个种子相同。"
        "可靠性仍回看 60 个已经成熟的历史预测，但按当前可融合的窗口重新中心化。"
        "两个窗口时仍用 20/120 日描述市场状态，以隔离融合候选数量的变化。",
        "各窗口组合内的主架构仍按平均验证 NLL 选择。两个新组合共训练 60 个模型，首轮结果完整保留。",
        "", "## 为什么首轮没有超过固定融合", "",
        f"**证据一：验证最优训练轮次很早。** 首轮 {diagnostics['trained_model_count']} 个模型中，"
        f"{diagnostics['best_epoch_one_count']} 个最佳模型来自第 1 轮；线性联合模型的最佳轮次为 "
        f"{diagnostics['linear_joint_best_epochs']}。"
        f"{diagnostics['training_improved_validation_worsened']} 个模型出现训练 NLL 改善、"
        "末轮验证 NLL 反而变差。说明当前训练方向难以迁移到验证环境；"
        "不能仅据此断言优化器没有收敛或特征完全无效。",
        f"补查同种子的未训练随机初始化后，有 "
        f"{diagnostics.get('untrained_better_than_selected_trained_count', '待计算')}/30 个模型的"
        "未训练验证 NLL 甚至优于所选训练 checkpoint。"
        "这说明当前结果也受初始化和早停影响，不能把全部优势都归因于已学到的动态机制。"
        "这个检查没有参与模型选择，未替换首轮训练结果。",
        "", "**证据二：验证与测试对窗口的偏好不同。** 验证集选择 LW-20，但在测试期，"
        "LW-120/252 总体较好。下面列出验证期分年份单窗口 NLL：", "",
        markdown_table(validation_years), "",
        "按同一批验证起点计算，去掉 2020 年后验证单窗口排序偏好长窗口；加入 2020 年后改选"
        "20 日。这是历史分段的选择敏感性诊断，没有用这个删年结果重新调参。"
        "其跨时期稳定性尚需滚动前向实验确认。"
        "不能把训练期和验证期 NLL 的绝对差直接解释为过拟合，因为它们对应不同市场环境。",
        "", "**证据三：联合门控给短窗口的权重偏高。** 首轮线性联合模型测试期平均给予 20 日"
        "窗口约 20.6%，固定融合仅约 5.2%。短窗口响应快，但样本更少，协方差误差和收缩偏差"
        "可能更明显。在该测试期，用更多短期估计替代长窗口并不一定提高预测质量。",
        "", "**证据四：成熟可靠性具有滞后，而且尺度偏好未必稳定。** 当前可靠性是预测后 20 日"
        "才成熟、再平均最近 60 个预测起点，概括的信息覆盖约 80 个交易日。"
        f"测试起点上，历史可靠性选出的最低损失窗口与未来实际最佳窗口一致率为 "
        f"{test_reliability.best_window_match_rate:.1%}，相对损失的展平 Spearman 相关为 "
        f"{test_reliability.reliability_future_spearman:.3f}。"
        "这只是诊断统计，没有控制类别频率或时间依赖；名义 25% 均匀猜测率不能当作正式显著性基线。",
        f"同一相关统计在验证集仅为 {validation_reliability.reliability_future_spearman:.3f}。"
        "这提示历史可靠性信号并非处处无效，而是跨市场阶段的稳定性有限；"
        "展平相关也包含不同尺度长期优劣的影响，不能直接当作动态选择能力。",
        "", "**尚未证实的解释：特征冗余和有效样本量不足。** 市场状态与可靠性都来自相同收益"
        "和估计器，可能携带重复信息；训练标签重叠也使 651 个起点的有效独立样本更少。"
        f"训练集两组特征间最大绝对 Pearson 相关为 {maximum_cross_correlation:.3f}，"
        "没有显示非常强的简单线性重复。特征相关矩阵已保存；"
        "因此当前不能把高度冗余当作已经证实的主要原因。",
        "", "## 两个或三个短窗口的实际结果", "", markdown_table(table), "",
        "NLL 越低越好。跨窗口比较采用完全一致的测试起点；每个组合的固定融合均只用训练集拟合。",
    ]
    original = table.iloc[0]["联合测试 NLL"]
    for case, result in cases.items():
        selected = result["metadata"]["primary_model"]
        row = summary[(summary.case == case) & (summary.method == selected)].iloc[0]
        delta = original - row.test_nll
        direction = "改善" if delta > 0 else "变差"
        lines.extend(["", f"- `{case}` 的验证所选联合模型为 `{selected}`，相对首轮联合模型"
                      f"测试 NLL {direction} {abs(delta):.6f}。"])
    lines.extend([
        "", "## 成对比较与不确定性", "",
        "改善 = 对照 NLL − 当前联合模型 NLL，正值表示当前联合模型较好。主表使用 3 个区间"
        "的循环时间块，5000 次重采样；p 值在两个组合共 14 项比较内作 Holm 校正。"
        "这些区间不修正此前观察测试数据的研究者选择，因此仍属探索性证据。", "",
        markdown_table(core[["case", "comparator", "improvement", "ci_lower", "ci_upper",
                             "p_holm_14"]].rename(columns={"case": "窗口组合", "comparator": "对照",
                             "improvement": "改善", "ci_lower": "区间下界", "ci_upper": "区间上界",
                             "p_holm_14": "校正后 p 值"})), "",
        "## 下游风险检查", "",
        "下表为零成本、零执行滞后。联合模型回测使用平均种子门控权重，预测表使用平均种子损失。"
        "年化波动率与回撤使用小数，例如 0.126 表示 12.6%。", "",
        markdown_table(main_backtest.rename(columns={"case": "窗口组合", "method": "方法",
            "annualized_volatility": "年化波动率", "maximum_drawdown": "最大回撤",
            "mean_turnover": "平均单次换手率"})), "",
        "## 后续建议", "",
        "先针对训练与验证的窗口偏好差异进行滚动前向评估，并检查可靠性回看期。"
        "如果尝试 10/20/60 日或删减输入特征，应作为新的单因素探索另行记录。"
        "对同一测试期继续筛选得到的结果，需要未来新数据或预先固定的历史前向方案再验证。",
        "", "## 图表", "",
        "![不同窗口组合的 NLL](figures/window_comparison.png)", "",
        "![相对首轮联合模型的改善](figures/relative_to_original.png)", "",
    ])
    (output / "ablation_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_ablation_figures(output: Path, reference: Path, cases: dict,
                           summary: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for method, marker in [("FixedFusion", "s"), ("linear-Joint", "o"), ("mlp-Joint", "^")]:
        rows = summary[summary.method == method]
        ax.plot(rows.windows, rows.test_nll, marker=marker, label=method)
    ax.set_ylabel("Test NLL per asset (lower is better)")
    ax.set_xlabel("Covariance lookback windows in trading days")
    ax.set_title("Exploratory ablation; identical test origins; five seeds per gate")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(directory / "window_comparison.png", dpi=180)
    plt.close(fig)
    chosen = bootstrap[(bootstrap.block_length == 3) & (
        bootstrap.comparator == "original_4_Joint")]
    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(chosen))
    ax.errorbar(chosen.improvement, y,
                xerr=[chosen.improvement - chosen.ci_lower, chosen.ci_upper - chosen.improvement],
                fmt="o", capsize=5)
    ax.axvline(0, color="#555555", linestyle="--")
    ax.set_yticks(y, chosen.case)
    ax.set_xlabel("Original joint NLL minus reduced-window joint NLL (positive favors reduced)")
    ax.set_title("Reused test data: paired pointwise 95% intervals, exploratory only")
    fig.tight_layout()
    fig.savefig(directory / "relative_to_original.png", dpi=180)
    plt.close(fig)

"""Once-only 2026 audit of frozen models: no fitting, no model/seed selection."""

from __future__ import annotations

import hashlib
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
from portfolio_risk.experiment import build_arrays, covariance_nll, paired_block_bootstrap, holm_adjust, run_gmv_backtest
from portfolio_risk.models.anchored import AnchoredRiskGate
from portfolio_risk.models.gating import LinearGate, MLPGate
from portfolio_risk.risk_features import expanded_market_features, matured_reliability
from portfolio_risk.window_ablation import markdown_table


def run_forward_audit(config_path: Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    search_config = load_config(config_path)
    config = load_config(Path(search_config["base_config"]))
    search = search_config["search"]
    source = Path(search_config["output_dir"])
    output = source.parent / "forward_audit"
    output.mkdir(exist_ok=True)
    if (output / "summary.csv").exists():
        raise RuntimeError("Forward audit already exists; never repeatedly tune against this holdout.")
    selection_path = source / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    protocol = json.loads((source / "search_protocol.json").read_text(encoding="utf-8"))
    reference = Path(config["project"]["output_dir"])
    old_metadata = json.loads((reference / "run_metadata.json").read_text(encoding="utf-8"))
    raw = Path(config["data"]["raw_dir"]) / "10_Industry_Portfolios_daily_CSV.zip"
    if hashlib.sha256(raw.read_bytes()).hexdigest() != protocol["raw_sha256"]:
        raise ValueError("Frozen archive changed.")
    selected_names = list(dict.fromkeys([selection["primary"], selection["selected_pure_fusion"],
                                         *selection["controls"].values()]))
    frozen_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in source.rglob("*") if p.is_file()}
    # Write the holdout rule before parsing/scoring 2026 returns.
    audit_protocol = dict(start="2026-01-01", end="last complete target in frozen archive",
                          selected_names=selected_names, selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                          source_artifact_hashes=frozen_hashes, no_training=True,
                          no_reselection=True, created_at_utc=datetime.now(timezone.utc).isoformat(),
                          evidence_status="new temporal holdout, short revised-data snapshot; exploratory audit")
    (output / "audit_protocol.json").write_text(json.dumps(audit_protocol, indent=2), encoding="utf-8")
    frame, _ = parse_industry_zip(raw)
    frame = frame.loc[config["data"]["history_start"]:]
    validate_return_matrix(frame)
    if frame.index[-1] < pd.Timestamp("2026-02-01"):
        raise ValueError("Frozen archive has insufficient 2026 holdout data.")
    config["data"]["splits"]["test"] = ["2026-01-01", str(frame.index[-1].date())]
    arrays = build_arrays(frame, config)
    indices = arrays["splits"]["test"]
    expanded = np.stack([expanded_market_features(frame.to_numpy()[t - 251:t + 1], arrays["covariance"][t])
                         for t in indices])
    predictions = {}
    for name in selected_names:
        spec = next(s for s in protocol["candidates"] if s["name"] == name)
        mode, window = spec["mode"], spec["reliability_window"]
        features = expanded if mode.startswith("expanded") else arrays["features"][indices, :6]
        if mode != "expanded_market":
            q = np.stack([matured_reliability(arrays["losses"], t, config["forecast"]["horizon"], window) for t in indices])
            features = np.column_stack([features, q])
        scaler = np.load(source / f"scaler_{mode}_r{window}.npz")
        x = torch.from_numpy((features - scaler["mean"]) / scaler["scale"])
        seed_matrices = []
        for seed in search["random_seeds"]:
            state = torch.load(source / "checkpoints" / f"{name}_seed{seed}.pt", weights_only=True)
            model = AnchoredRiskGate(features.shape[1], np.array(selection["anchor"]), spec["architecture"],
                                    search["hidden_dim"], spec["strength"], spec["calibrate"], search["max_abs_log_scale"]).double()
            model.load_state_dict(state["state_dict"])
            model.eval()
            with torch.no_grad():
                weights, log_scale = model(x)
            matrix = np.einsum("bk,bkij->bij", weights.numpy(), arrays["covariance"][indices])
            seed_matrices.append(matrix * log_scale.exp().numpy()[:, None, None])
        predictions[name] = np.mean(seed_matrices, axis=0)
    for k, window in enumerate(config["forecast"]["lookback_windows"]):
        predictions[f"LW-{window}"] = arrays["covariance"][indices, k]
    for decay, matrix in arrays["ewma"].items():
        predictions[f"EWMA-{decay}"] = matrix[indices]
    predictions["FixedFusion"] = np.einsum("k,bkij->bij", selection["anchor"], arrays["covariance"][indices])
    predictions["EqualFusion"] = arrays["covariance"][indices].mean(1)
    predictions["HistoricalLoss"] = np.einsum("bk,bkij->bij", softmax(
        -arrays["features"][indices, 6:] / old_metadata["historical_loss_temperature"], axis=-1), arrays["covariance"][indices])
    old_scaler = np.load(reference / "feature_scaler.npz")
    original_x = (arrays["features"][indices] - old_scaler["mean"]) / old_scaler["scale"]
    for architecture in ("linear", "mlp"):
        for mode, cols in {"Market": list(range(6)), "Reliability": list(range(6, 10)), "Joint": list(range(10))}.items():
            name = f"{architecture}-{mode}"
            weights = []
            for seed in config["evaluation"]["random_seeds"]:
                checkpoint = torch.load(reference / "checkpoints" / f"{name}_seed{seed}.pt", weights_only=True)
                model = (LinearGate(len(cols)) if architecture == "linear" else
                         MLPGate(len(cols), config["model"]["hidden_dim"])).double()
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()
                with torch.no_grad():
                    weights.append(model(torch.from_numpy(original_x[:, cols])).numpy())
            predictions[f"Original-{name}-ensemble"] = np.einsum("bk,bkij->bij", np.mean(weights, axis=0), arrays["covariance"][indices])
    losses = {n: covariance_nll(m, arrays["realized"][indices]) for n, m in predictions.items()}
    summary = pd.DataFrame([dict(method=n, holdout_nll=l.mean(), origins=len(l), primary=n == selection["primary"])
                            for n, l in losses.items()]).sort_values("holdout_nll")
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(losses, index=frame.index[indices]).rename_axis("decision_date").to_csv(output / "holdout_losses.csv")
    pd.DataFrame(dict(decision_date=frame.index[indices], label_start=frame.index[indices + 1],
                      label_end=frame.index[indices + config["forecast"]["horizon"]])).to_csv(output / "holdout_dates.csv", index=False)
    comparison_names = [n for n in losses if n not in selected_names]
    rows = []
    for name in comparison_names:
        diff = losses[name] - losses[selection["primary"]]
        low, high, p = paired_block_bootstrap(diff, 3, config["evaluation"]["bootstrap_repetitions"],
                                             config["evaluation"]["bootstrap_seed"])
        rows.append(dict(baseline=name, improvement=diff.mean(), ci_lower=low, ci_upper=high, p_one_sided=p))
    for row, p in zip(rows, holm_adjust([r["p_one_sided"] for r in rows]), strict=True):
        row["p_holm"] = p
    pd.DataFrame(rows).to_csv(output / "paired_bootstrap.csv", index=False)
    backtests = []
    for name, matrix in predictions.items():
        daily, dates, turnover = run_gmv_backtest(frame.to_numpy(), indices, matrix, config["forecast"]["horizon"], 0, 0)
        backtests.append(dict(method=name, annualized_volatility=annualized_volatility(daily),
                             maximum_drawdown=maximum_drawdown(daily), cvar_95=empirical_cvar(daily), days=len(daily)))
    backtest = pd.DataFrame(backtests)
    backtest.to_csv(output / "backtest_summary.csv", index=False)
    primary = selection["primary"]
    report = ["# 2026 年冻结模型留出审计", "",
              "这是主模型和全部调参已锁定后，对此前未用于实验评分的 2026 年时间段的一次性检查；没有重新训练、挑种子或换模型。审计后该时间段也已经被使用，不能继续反复调参再称独立留出。", "",
              f"原始快照截至 {frame.index[-1].date()}。共 {len(indices)} 个完整、非重叠 20 日评价区间；实际收益覆盖 {frame.index[indices[0] + 1].date()}—{frame.index[indices[-1] + config['forecast']['horizon']].date()}。主模型仍为 `{primary}`。", "",
              markdown_table(summary[["method", "holdout_nll", "origins"]]), "",
              "NLL 越低越好；本轮所有神经方法统一评分协方差矩阵集成。小样本无法代表整年或所有市场状态，原始历史也不是点时发布版本。", "",
              "## 零成本、零延迟组合结果", "", markdown_table(backtest), "",
              "## 解释边界", "",
              "与 2021—2025 年比较分开呈现。区块 bootstrap 仅作为探索描述，极少的区间数意味着统计不确定性很高。留出期某些指标胜出，不意味着多指标、所有年份和所有基线都稳定胜出。", "",
              "audit_protocol.json 固定规则及输入模型校验值；holdout_dates.csv 明确标签边界；holdout_losses.csv 保留逐区间损失。"]
    (output / "forward_audit_report.md").write_text("\n".join(report), encoding="utf-8")
    for filename, digest in frozen_hashes.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen model artifacts changed: {filename}")
    (output / "run_metadata.json").write_text(json.dumps(dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        origins=len(indices), raw_end=str(frame.index[-1].date()), primary=primary,
        primary_holdout_nll=losses[primary].mean(), source_artifacts_unchanged=True), indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"Completed one-time holdout audit: {output.resolve()}", flush=True)

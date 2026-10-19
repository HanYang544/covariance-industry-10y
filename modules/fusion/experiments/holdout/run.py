"""One frozen train/validation/test comparison; no test-time parameter refits."""
import hashlib
import json
import copy
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


from portfolio_risk.config import load_config
from portfolio_risk.data.french import parse_industry_zip, validate_return_matrix
from portfolio_risk.state_research import prepare_research, MARKET_NAMES

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/dynamic_holdout"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lw-only', action='store_true')
    parser.add_argument('--four-features', action='store_true')
    parser.add_argument('--six-features', action='store_true')
    parser.add_argument('--loss', choices=['frobenius'], default='frobenius')
    args = parser.parse_args()
    if args.four_features and args.six_features:
        parser.error('Choose four or six features, not both')
    global OUT
    OUT = ROOT / ('outputs/dynamic_lw_holdout' if args.lw_only else 'outputs/dynamic_holdout')
    if args.four_features:
        OUT = OUT.with_name(OUT.name + '_four_features')
    input_dim = 4 if args.four_features else 14
    if args.six_features:
        OUT = OUT.with_name(OUT.name + '_six_features_k3')
        input_dim = 6
    expert_count = 4 if args.lw_only else 10
    method = 'DynamicLW-EarlyStop' if args.lw_only else 'DynamicGate-EarlyStop'
    if args.four_features:
        method += '-4F'
    if args.six_features:
        method += '-6F-K3'
    torch.set_num_threads(1)
    torch.manual_seed(20261012)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "test_results.csv").exists():
        raise RuntimeError("Completed holdout exists; preserve it and choose a new output directory.")
    base = load_config(ROOT / "configs/preliminary.yaml")
    config = load_config(ROOT / "configs/state_whitener.yaml")
    archive = ROOT / "data/raw/10_Industry_Portfolios_daily_CSV.zip"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    expected = json.loads((ROOT / "data/raw/industry_10_manifest.json").read_text("utf-8"))["sha256"]
    assert digest == expected, "Frozen data changed"
    protocol = dict(splits=base["data"]["splits"], horizon=20,
        target="future 20-day uncentered mean outer product", loss="mean unsquared full-matrix Frobenius",
        architecture="dynamic market-state gate + precision-factor fusion + scale calibration",
        optimization="joint full-batch Adam; softmax simplex weights; bounded scale coefficients",
        learning_rate=0.01, max_epochs=500, patience=30, seed=20261012,
        refit_on_validation=False, test_parameter_refit=False, source_sha256=digest,
        caveat="Historical test period has previously been inspected; this is not new independent confirmation.")
    protocol.update(architecture=f"{input_dim} market features -> Linear({input_dim},16) -> ReLU -> Linear(16,{expert_count}) -> Softmax; precision-factor fusion + 4-feature scale calibration",
                    gate_features=MARKET_NAMES[1:5] if args.four_features else MARKET_NAMES,
                    trainable_parameters=16 * input_dim + 21 + 17 * expert_count,
                    expert_count=expert_count, lw_only=args.lw_only)
    if args.six_features:
        from portfolio_risk.market_risk_features import MarketRiskFeatures, FEATURE_COLUMNS
        protocol.update(gate_features=FEATURE_COLUMNS, feature_covariance='centered LedoitWolf',
                        short_window=20, long_window=120, top_k=3, min_assets=10,
                        min_assets_adaptation='Dataset has 10 industries; extractor default remains 30')
    protocol.update(training_loss=args.loss, early_stopping_metric=args.loss)
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    frame, _ = parse_industry_zip(archive)
    frame = frame.loc[base["data"]["history_start"]:"2025-12-31"]
    validate_return_matrix(frame)
    prepared = prepare_research(frame, base, config)
    expert_names = prepared['names'][:expert_count]
    whiteners = prepared['whiteners'][:, :expert_count]
    arrays = prepared["arrays"]
    train, val, test = (arrays["splits"][key] for key in ("train", "validation", "test"))
    assert train[-1] + 20 < val[0] and val[-1] + 20 < test[0]
    features = prepared["features"]["global"]
    mean, scale = features[train].mean(0), np.maximum(features[train].std(0), 1e-12)
    design = np.column_stack([np.ones(len(frame)), np.clip((features - mean) / scale, -3, 3)])
    gate_features = prepared["features"]["global" if args.four_features else "market"]
    if args.six_features:
        extractor = MarketRiskFeatures(min_assets=10, top_k=3)
        raw_features = extractor.transform(frame)
        raw_features.to_csv(OUT / 'market_risk_features.csv')
        extractor.diagnostics_.to_csv(OUT / 'feature_diagnostics.csv')
        standardized, _, _ = extractor.standardize(raw_features, frame.index[train])
        standardized.to_csv(OUT / 'standardized_features.csv')
        gate_features = raw_features.to_numpy()
        assert np.isfinite(gate_features[np.r_[train, val, test]]).all(), 'Invalid evaluation features'
    gate_mean = gate_features[train].mean(0)
    gate_scale = np.maximum(gate_features[train].std(0), 1e-12)
    gate_input = np.clip((gate_features - gate_mean) / gate_scale, -3, 3)
    # Only train/validation tensors are passed to the optimization loop.
    tensors = {name: tuple(torch.tensor(a[idx], dtype=torch.float64) for a in
               (whiteners, design, arrays["realized"], gate_input))
               for name, idx in (("train", train), ("validation", val))}
    theta = torch.nn.Parameter(torch.zeros(5, dtype=torch.float64))
    gate = torch.nn.Sequential(torch.nn.Linear(input_dim, 16), torch.nn.ReLU(), torch.nn.Linear(16, expert_count)).double()
    optimizer = torch.optim.Adam([*gate.parameters(), theta], lr=protocol["learning_rate"])

    def predict(w, x, g):
        weights = gate(g).softmax(-1)
        factor = torch.einsum("bk,bkij->bij", weights, w)
        return torch.linalg.inv(factor @ factor.transpose(-1, -2)) * torch.exp(-x @ theta)[:, None, None]

    def loss(w, x, y, g):
        prediction = predict(w, x, g)
        return torch.linalg.matrix_norm(prediction - y, ord="fro").mean()

    best, stale, history = float("inf"), 0, []
    # Include the initial model as a valid checkpoint.
    for epoch in range(protocol["max_epochs"] + 1):
        if epoch:
            optimizer.zero_grad()
            loss(*tensors["train"]).backward()
            optimizer.step()
            with torch.no_grad():
                theta[0].clamp_(-1, 1)
                theta[1:].clamp_(-0.25, 0.25)
        with torch.no_grad():
            tr, va = float(loss(*tensors["train"])), float(loss(*tensors["validation"]))
        history.append(dict(epoch=epoch, **{f'train_{args.loss}':tr, f'validation_{args.loss}':va}))
        if va < best - 1e-12:
            best, best_epoch, stale = va, epoch, 0
            checkpoint = theta.detach().clone()
            gate_checkpoint = copy.deepcopy(gate.state_dict())
        else:
            stale += 1
        if epoch % 25 == 0:
            print(f"Epoch {epoch}: validation={va:.8g}; best={best_epoch}", flush=True)
        if stale >= protocol["patience"]:
            break
    with torch.no_grad():
        theta.copy_(checkpoint)
        gate.load_state_dict(gate_checkpoint)
    pd.DataFrame(history).to_csv(OUT / "training_history.csv", index=False)
    torch.save(gate.state_dict(), OUT / "gate.pt")
    np.savez(OUT / "model.npz", theta=theta.detach().numpy(), feature_mean=mean, feature_scale=scale,
             gate_mean=gate_mean, gate_scale=gate_scale, expert_names=np.array(expert_names))
    # Evaluate the test set once after the checkpoint has been restored.
    predictions = {name: prepared["experts"][test, k] for k, name in enumerate(prepared["names"][:-1])}
    predictions.update({f"EWMA-{decay}": matrix[test] for decay, matrix in arrays["ewma"].items()})
    with torch.no_grad():
        predictions[method] = predict(torch.tensor(whiteners[test]),
                                                   torch.tensor(design[test]), torch.tensor(gate_input[test])).numpy()
        weights = gate(torch.tensor(gate_input[test])).softmax(-1).numpy()
        np.testing.assert_allclose(weights.sum(1), 1)
        assert np.std(weights, axis=0).max() > 1e-8, "Gate did not produce changing weights"
        pd.DataFrame(weights, index=frame.index[test], columns=expert_names).rename_axis("date").to_csv(OUT / "test_weights.csv")
    if args.lw_only:
        with np.load(ROOT / 'outputs/dynamic_holdout/test_predictions.npz') as previous:
            np.testing.assert_array_equal(previous['dates'], frame.index[test].astype(str).to_numpy(dtype=str))
            np.testing.assert_array_equal(previous['target'], arrays['realized'][test])
            for name in predictions:
                if name != method:
                    np.testing.assert_allclose(previous[name], predictions[name], rtol=1e-12, atol=1e-15)
            predictions['DynamicGate-EarlyStop'] = previous['DynamicGate-EarlyStop'].copy()
    truth = arrays["realized"][test]
    rows, origin_rows = [], []
    for name, matrix in predictions.items():
        assert np.isfinite(matrix).all()
        np.linalg.cholesky(matrix)
        frob = np.linalg.norm(matrix - truth, axis=(1, 2))
        rows.append(dict(method=name, test_frobenius=frob.mean(), samples=len(test)))
        origin_rows.extend(dict(method=name, date=str(frame.index[t].date()), frobenius=float(f))
                           for t, f in zip(test, frob))
    results = pd.DataFrame(rows).sort_values(f"test_{args.loss}")
    results.to_csv(OUT / "test_results.csv", index=False)
    pd.DataFrame(origin_rows).to_csv(OUT / "test_by_origin.csv", index=False)
    np.savez_compressed(OUT / "test_predictions.npz", dates=frame.index[test].astype(str).to_numpy(dtype=str),
                        target=truth, **predictions)
    splits = {name: dict(samples=len(idx), first=str(frame.index[idx[0]].date()),
              last_origin=str(frame.index[idx[-1]].date()), last_label=str(frame.index[idx[-1] + 20].date()))
              for name, idx in (("train", train), ("validation", val), ("test", test))}
    (OUT / "run_metadata.json").write_text(json.dumps(dict(splits=splits, best_epoch=best_epoch,
        stopped_epoch=epoch, **{f'best_validation_{args.loss}':best}, training_loss=args.loss,
        test_methods=len(predictions)), indent=2), encoding="utf-8")
    table = ["| 方法 | 测试 Frobenius ↓ |", "| --- | ---: |"]
    table += [f"| {r.method} | {r.test_frobenius:.8g} |" for r in results.itertuples()]
    report = ["# 基础方法与设计方法：固定测试集结果", "",
        "训练：2005—2017；验证：2018—2020；测试：2021—2025。每个样本的未来 20 日标签完整落在所属区间。",
        f"样本数：训练 {len(train)}，验证 {len(val)}，测试 {len(test)}。训练起点间隔 5 日，验证与测试间隔 20 日。", "",
        "## 测试结果", "", *table, "",
        "## 本次训练方式", "",
        "设计方法保留十专家精度 Cholesky 融合与四周期风险校准结构；本次改用联合 Adam 训练、Softmax 权重和有界校准系数，以支持验证早停，区别于旧版两阶段优化。",
        "损失是逐样本完整矩阵 Frobenius 范数的均值（不平方）；标签保持未来 20 日零均值二阶矩。全批量学习率 0.01，最多 500 轮，验证连续 30 轮无改善早停；无额外正则项。",
        f"最佳轮次 {best_epoch}，停止轮次 {epoch}；恢复最佳验证参数后评估测试集。测试期间不重训，不用测试成绩选择轮次或改参数。",
        "特征标准化只拟合训练集。LW、EWMA、IEWMA 的历史状态随可观察收益更新；这属于预测输入更新。",
        "本次比较 11 个统计基线设置与设计方法；Frobenius为训练、早停与测试指标，越小越好。", "",
        "这里的测试区间曾在历史实验中被查看，因此是固定划分下的测试比较，不是全新独立验证。", "",
        "[逐样本结果](test_by_origin.csv) · [训练曲线数据](training_history.csv) · [完整协议](protocol.json)"]
    report[0] = "# 风险状态动态门控：固定测试集结果"
    report = [line.replace("设计方法保留十专家精度 Cholesky 融合与四周期风险校准结构；本次改用联合 Adam 训练、Softmax 权重和有界校准系数，以支持验证早停，区别于旧版两阶段优化。",
                "设计方法：14 个市场风险特征 → Linear(14,16) → ReLU → Linear(16,10) → Softmax，得到逐日变化的专家权重；随后精度因子融合，再经过四特征风险尺度校准。门控 410 个参数，校准 5 个，共 415 个。所有标准化仅拟合训练集。")
                for line in report]
    report += ["", "本次只运行一个预先设定的结构与随机种子，没有用测试成绩选配置。", "", "[逐日专家权重](test_weights.csv) · [模型说明](../../docs/model.md)"]
    if args.lw_only:
        report = [line.replace('Linear(16,10)', 'Linear(16,4)').replace('门控 410 个参数，校准 5 个，共 415 个', '门控 308 个参数，校准 5 个，共 313 个')
                  .replace('本次只比较 11 个统计基线设置与一个设计方法', '本次比较 11 个统计基线、四 LW 动态门控以及已有十专家动态门控') for line in report]
        report[0] = '# 四个 LW 专家的动态门控：测试结果'
        report += ['', '四专家为 LW-20、LW-60、LW-120、LW-252。只改变专家集合与门控输出维度；输入仍为相同 14 个风险特征，保留精度因子融合和尺度校准，并非直接对协方差加权。已有十专家预测直接读取，测试日期、标签及基线矩阵已核对一致。']
    if args.four_features:
        count = protocol['trainable_parameters']
        report = [line.replace('14 个市场风险特征', '4 个对数风险特征').replace('Linear(14,16)', 'Linear(4,16)')
                  .replace('相同 14 个风险特征', '5/20/60/252 日四个对数风险特征')
                  .replace('门控 410 个参数，校准 5 个，共 415 个', f'门控 {count-5} 个参数，校准 5 个，共 {count} 个')
                  .replace('门控 308 个参数，校准 5 个，共 313 个', f'门控 {count-5} 个参数，校准 5 个，共 {count} 个') for line in report]
        report[0] += '（仅四个输入特征）'
        report += ['', '本次门控仅输入过去 5、20、60、252 日所有行业平方收益均值的对数；不再输入相关、动量、回撤等额外状态。校准层仍用相同四个风险特征。']
    if args.six_features:
        count = protocol['trainable_parameters']
        report = [line.replace('14 个市场风险特征', '6 个新风险状态指标').replace('Linear(14,16)', 'Linear(6,16)')
                  .replace('相同 14 个风险特征', '六个新风险状态指标')
                  .replace('门控 410 个参数，校准 5 个，共 415 个', f'门控 {count-5} 个参数，校准 5 个，共 {count} 个')
                  .replace('门控 308 个参数，校准 5 个，共 313 个', f'门控 {count-5} 个参数，校准 5 个，共 {count} 个') for line in report]
        report[0] += '（六指标输入）'
        report += ['', '门控仅输入 VR、AC、AR、ER、SR、CD，非拼接旧特征。校准仍使用四周期风险特征。20/120 日去均值 LW，AR 与 SR 均使用 k=3；当前仅十行业，因此 min_assets=10，模块默认仍为30。AC 使用45对行业，ER 除以10，CD 比较10×10相关矩阵。特征协方差与预测标签口径不同：后者仍是未去均值二阶矩。', '', '[六指标说明](../../docs/market_risk_features.md) · [原始指标](market_risk_features.csv) · [标准化指标](standardized_features.csv) · [诊断](feature_diagnostics.csv)']
    (OUT / "README.md").write_text("\n\n".join(report[:1]) + "\n" + "\n".join(report[1:]), encoding="utf-8")
    print(results.to_string(index=False), flush=True)
    print(f"Saved: {OUT}", flush=True)


if __name__ == "__main__":
    main()

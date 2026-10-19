"""Validate and summarize the four prespecified dynamic-gate experiments."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
CASES = [('dynamic_holdout', 'DynamicGate-EarlyStop', 14, 10),
         ('dynamic_lw_holdout', 'DynamicLW-EarlyStop', 14, 4),
         ('dynamic_holdout_four_features', 'DynamicGate-EarlyStop-4F', 4, 10),
         ('dynamic_lw_holdout_four_features', 'DynamicLW-EarlyStop-4F', 4, 4)]
CASES += [('dynamic_holdout_six_features_k3', 'DynamicGate-EarlyStop-6F-K3', 6, 10),
          ('dynamic_lw_holdout_six_features_k3', 'DynamicLW-EarlyStop-6F-K3', 6, 4)]


def main():
    rows = []
    with np.load(ROOT / 'outputs/dynamic_holdout/test_predictions.npz') as reference:
        dates, truth = reference['dates'].copy(), reference['target'].copy()
        baseline = {k: reference[k].copy() for k in reference.files if k.startswith(('LW-', 'EWMA-', 'IEWMA-'))}
    for directory, method, inputs, experts in CASES:
        path = ROOT / 'outputs' / directory
        meta = json.loads((path / 'run_metadata.json').read_text(encoding='utf-8-sig'))
        history = pd.read_csv(path / 'training_history.csv')
        assert history.loc[history.validation_frobenius.idxmin(), 'epoch'] == meta['best_epoch']
        weights = pd.read_csv(path / 'test_weights.csv', index_col=0).to_numpy()
        assert weights.shape == (len(dates), experts) and np.all(weights >= 0)
        np.testing.assert_allclose(weights.sum(1), 1)
        assert weights.std(0).max() > 1e-8
        gate = torch.nn.Sequential(torch.nn.Linear(inputs, 16), torch.nn.ReLU(), torch.nn.Linear(16, experts)).double()
        gate.load_state_dict(torch.load(path / 'gate.pt', weights_only=True))
        with np.load(path / 'model.npz') as model:
            assert model['gate_mean'].shape == (inputs,)
        with np.load(path / 'test_predictions.npz') as saved:
            np.testing.assert_array_equal(saved['dates'], dates)
            np.testing.assert_array_equal(saved['target'], truth)
            for k, matrix in baseline.items():
                np.testing.assert_allclose(saved[k], matrix, rtol=1e-12, atol=1e-15)
            matrix = saved[method]
            np.linalg.cholesky(matrix)
            frobenius = np.linalg.norm(matrix - truth, axis=(1, 2)).mean()
        score = pd.read_csv(path / 'test_results.csv').set_index('method').loc[method]
        np.testing.assert_allclose(frobenius, score.test_frobenius)
        rows.append(dict(method=method, features=inputs, experts=experts, test_frobenius=frobenius,
                         best_epoch=meta['best_epoch'], stopped_epoch=meta['stopped_epoch'],
                         validation_frobenius=meta['best_validation_frobenius']))
    out = ROOT / 'outputs/feature_count_comparison'
    out.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'summary.csv', index=False)
    lines = ['# 动态门控：四、六、十四特征比较', '',
             '四特征为 5/20/60/252 日对数平方收益均值；六特征为 VR、AC、AR、ER、SR、CD；十四特征为原市场状态输入。各自分别融合十专家和四个 LW，保留精度因子融合与四特征风险校准。六指标按用户十行业设定使用 min_assets=10、top_k=3；不混入早先k=5结果。', '',
             '训练 2005—2017，验证 2018—2020，测试 2021—2025，共 62 个测试起点。全批量 Adam，学习率 0.01，隐藏层 16，种子 20261012，最多 500 轮，验证耐心 30 轮；恢复最佳验证模型。', '',
             '| 输入特征 | 专家数 | 测试 Frobenius ↓ | 最佳轮次 | 停止轮次 |',
             '| ---: | ---: | ---: | ---: | ---: |']
    lines += [f'| {r.features} | {r.experts} | {r.test_frobenius:.9f} | {r.best_epoch} | {r.stopped_epoch} |' for r in frame.itertuples()]
    lines += ['', '已核对各组测试日期、标签及 11 个基线矩阵一致（浮点容差内），预测正定，权重非负且和为 1，并随日期变化。所有成绩从保存预测重新计算核验。', '',
              '这是单个种子下的追加实验，没有依据测试结果继续调参。输入维度改变也会改变参数数量与随机初始化，差异不能全归因于特征信息；历史测试已被查看，不是独立确认。', '']
    lines += [f'- [{inputs} 特征 / {experts} 专家](../{directory}/README.md)' for directory, _, inputs, experts in CASES]
    (out / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(frame.to_string(index=False))
    print('PASS: checkpoints, early stopping, dynamic weights, common test data, predictions and metrics')


if __name__ == '__main__':
    main()

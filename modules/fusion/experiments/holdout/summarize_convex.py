"""Reconstruct saved convex models, independently score and publish comparisons."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from portfolio_risk.models.convex_gate import ConvexGate

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/convex_fusion'

def main():
    torch.set_num_threads(1)
    table=pd.read_csv(OUT/'comparison.csv');data=np.load(OUT/'verification_inputs.npz')
    rows=[]
    for r in table.itertuples():
        folder=OUT/r.directory;z=np.load(folder/'test_predictions.npz')
        model=ConvexGate(r.experts,bool(r.calibrated))
        model.load_state_dict(torch.load(folder/'model.pt',weights_only=True))
        with torch.no_grad(): pred,w=model(torch.tensor(data['experts'][:,:r.experts]),torch.tensor(data['state']),torch.tensor(data['design']),True)
        np.testing.assert_array_equal(pred.numpy(),z['prediction'])
        np.testing.assert_array_equal(z['target'],data['target']);np.testing.assert_array_equal(z['dates'],data['dates'])
        weights=pd.read_csv(folder/'test_weights.csv',index_col=0).to_numpy()
        np.testing.assert_allclose(w.numpy(),weights,rtol=1e-12,atol=1e-15)
        manual=np.einsum('bk,bkij->bij',weights,data['experts'][:,:r.experts])
        if r.calibrated: manual*=np.exp(-data['design']@model.theta.detach().numpy())[:,None,None]
        np.testing.assert_allclose(manual,pred.numpy(),rtol=1e-12,atol=1e-15)
        np.linalg.cholesky(manual)
        scores=np.linalg.norm(manual-data['target'],axis=(1,2))
        np.testing.assert_allclose(scores.mean(),r.test_score,rtol=1e-10,atol=1e-12)
        history=pd.read_csv(folder/'training_history.csv')
        assert int(history.loc[history.validation_loss.idxmin(),'epoch'])==r.best_epoch
        pd.DataFrame(dict(date=data['dates'],loss=scores)).to_csv(folder/'test_by_origin.csv',index=False)
        rows.append(dict(experts=r.experts,objective=r.objective,method='Convex+Calibration' if r.calibrated else 'Convex-Pure',test_score=r.test_score,best_epoch=r.best_epoch))
    for k in [4,10]:
        for obj in ['frobenius']:
            folder=ROOT/'outputs'/(('dynamic_lw_holdout' if k==4 else 'dynamic_holdout')+'_six_features_k3'+(''))
            z=np.load(folder/'test_predictions.npz');key=next(x for x in z.files if x.startswith('Dynamic') and '6F-K3' in x)
            np.testing.assert_array_equal(z['target'],data['target']);np.testing.assert_array_equal(z['dates'],data['dates'])
            p=z[key];y=z['target']
            score=np.linalg.norm(p-y,axis=(1,2)).mean()
            rows.append(dict(experts=k,objective=obj,method='Precision+Calibration',test_score=score,best_epoch=json.loads((folder/'run_metadata.json').read_text())['best_epoch']))
    pd.DataFrame(rows).to_csv(OUT/'fusion_comparison.csv',index=False)
    lines=['# 动态协方差凸组合与精度融合','',
        '六指标k=3，4个LW或10专家，未来20日未中心化二阶矩标签。训练651、验证37、测试62，预测日期与原实验一致。4个预设Frobenius模型独立训练和对应验证早停，所有检查点锁定后才计算测试分数。','',
        '## 三种公式','',
        r'权重均为 $w_t=\operatorname{Softmax}(\operatorname{MLP}(x_t))$，MLP为6→16→K，随状态动态变化，不是固定权重。','',
        r'- Convex-Pure：$\widehat C_t=\sum_kw_{t,k}C_{t,k}$。严格凸组合，不增加校准。',
        r'- Convex+Calibration：$\widehat C_t=(\sum_kw_{t,k}C_{t,k})\exp(-z_t^\top\theta)$。凸组合后再乘正标量；最终矩阵不一定仍在原专家的凸包内。',
        r'- Precision+Calibration：原方法，$\widehat C_t=(A_tA_t^\top)^{-1}\exp(-z_t^\top\theta)$，$A_t=\sum_kw_{t,k}L_{t,k}$。','',
        '两种凸组合版本都保持正定：专家为正定且Softmax权重为正；校准因子也为正。测试矩阵均通过数值Cholesky检查，没有通过事后特征值修正改变预测。','',
        '校准版与原版仅更换融合方式；纯凸组合版还去掉校准。共有训练参数保持Adam 0.01、全批量、最多500轮、patience=30、seed=20261012、min_delta=1e-12，含第0轮候选，门控使用相同种子初始化。纯模型比校准模型少5个参数。','']
    results=pd.DataFrame(rows)
    for obj in ['frobenius']:
        lines += [f'## {obj.upper()}：对应训练与测试','', '| 专家数 | 纯凸组合 | 凸组合＋校准 | 原精度融合＋校准 |','| --- | ---: | ---: | ---: |']
        for k in [4,10]:
            sub=results[(results.experts==k)&(results.objective==obj)].set_index('method').test_score
            lines.append(f'| {k} | {sub["Convex-Pure"]:.9f} | {sub["Convex+Calibration"]:.9f} | {sub["Precision+Calibration"]:.9f} |')
        lines.append('')
    lines += ['以上分数均越小越好。仅为单种子、已查看历史测试集上的探索，不代表统计显著优劣；不得按测试最低分回选轮次或更改实验配置。', '',
        '十专家纯凸组合Frobenius的最佳检查点为第0轮，不能解释为训练带来的收益。', '',
        '已核验：保存模型恢复预测、numpy直接组合、权重与日期一致、损失独立重算、最佳验证轮次、正定性。原动态实验文件哈希未变化。','',
        '[全部分数和轮次](comparison.csv) · [三方法比较](fusion_comparison.csv) · [协议](protocol.json) · [锁定检查点](selection.json)']
    text='\n'.join(lines)+'\n';(OUT/'README.md').write_text(text,encoding='utf-8')
    report=ROOT/'outputs/experiment_results.md';existing=report.read_text(encoding='utf-8')
    if '## 6 动态协方差凸组合' in existing: raise RuntimeError('Already merged')
    summary=text[text.index('## FROBENIUS'):text.index('以上分数')].replace('## ','### ')
    report.write_text(existing+'\n## 6 动态协方差凸组合\n\n六指标，4/10专家；纯凸组合与保留校准两版。训练协议与原方法一致，本表仅保留Frobenius训练和评价。\n\n'+summary+'\n纯凸组合不加校准；校准版最终不一定在专家矩阵凸包内，但仍保持正定。单种子历史测试，不做显著性结论。\n\n[完整方法和记录](convex_fusion/README.md)\n',encoding='utf-8')
    print(results.to_string(index=False))

if __name__=='__main__': main()

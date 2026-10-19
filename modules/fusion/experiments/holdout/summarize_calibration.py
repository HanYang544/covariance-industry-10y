"""Audit new precision fits and complete the fusion x calibration comparison."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from portfolio_risk.models.precision_gate import PrecisionGate
from portfolio_risk.covariance.iewma import precision_cholesky

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'outputs/precision_no_calibration'
OUT=ROOT/'outputs/calibration_comparison'

def main():
    if OUT.exists(): raise RuntimeError('Refusing overwrite')
    torch.set_num_threads(1)
    source=pd.read_csv(SOURCE/'comparison.csv')
    data=np.load(SOURCE/'verification_inputs.npz')
    reference=np.load(ROOT/'outputs/convex_fusion/verification_inputs.npz')
    for key in ['state','design','target','dates']:
        np.testing.assert_array_equal(data[key],reference[key])
    np.testing.assert_allclose(data['whiteners'],precision_cholesky(reference['experts']),rtol=1e-12,atol=1e-12)
    rows=[]
    for r in source.itertuples():
        folder=SOURCE/r.directory;z=np.load(folder/'test_predictions.npz')
        model=PrecisionGate(r.experts,False)
        model.load_state_dict(torch.load(folder/'model.pt',weights_only=True))
        assert 'theta' not in model.state_dict()
        with torch.no_grad(): pred,w=model(torch.tensor(data['whiteners'][:,:r.experts]),torch.tensor(data['state']),return_weights=True)
        np.testing.assert_array_equal(pred.numpy(),z['prediction'])
        np.testing.assert_array_equal(z['target'],data['target']);np.testing.assert_array_equal(z['dates'],data['dates'])
        weights=pd.read_csv(folder/'test_weights.csv',index_col=0).to_numpy()
        np.testing.assert_allclose(w.numpy(),weights,atol=1e-15,rtol=1e-12)
        factor=np.einsum('bk,bkij->bij',weights,data['whiteners'][:,:r.experts])
        manual=np.linalg.inv(factor@factor.swapaxes(-1,-2))
        np.testing.assert_allclose(manual,z['prediction'],rtol=1e-12,atol=1e-14)
        np.linalg.cholesky(manual)
        score=np.linalg.norm(manual-data['target'],axis=(1,2))
        np.testing.assert_allclose(score.mean(),r.test_score,rtol=1e-10,atol=1e-12)
        h=pd.read_csv(folder/'training_history.csv')
        assert int(h.loc[h.validation_loss.idxmin(),'epoch'])==r.best_epoch
        pd.DataFrame(dict(date=data['dates'],loss=score)).to_csv(folder/'test_by_origin.csv',index=False)
        rows.append(dict(experts=r.experts,objective=r.objective,fusion='precision',calibrated=False,test_score=r.test_score,best_epoch=r.best_epoch))
    previous=pd.read_csv(ROOT/'outputs/convex_fusion/fusion_comparison.csv')
    for r in previous.itertuples():
        rows.append(dict(experts=r.experts,objective=r.objective,fusion='precision' if r.method.startswith('Precision') else 'covariance',calibrated=r.method.endswith('Calibration'),test_score=r.test_score,best_epoch=r.best_epoch))
    table=pd.DataFrame(rows)
    assert len(table)==8 and not table.duplicated(['experts','objective','fusion','calibrated']).any()
    OUT.mkdir();table.to_csv(OUT/'comparison.csv',index=False)
    effects=[];lines=['# 融合方式 × 有无风险校准','',
        '本表保留8项Frobenius记录：4/10专家、两种融合、有无校准。六指标k=3、6→16→K门控、种子20261012、Adam 0.01、最多500轮、验证patience=30均不变。','',
        '无校准不是把已有模型的校准系数改为1，而是完全不创建5个校准参数，重新训练门控并按对应验证损失选检查点。检验的是保留/移除校准模块的整体效果。', '',
        r'协方差融合：$C_t=\sum_kw_{t,k}C_{t,k}$。精度融合：$C_t=(A_tA_t^\top)^{-1}$，$A_t=\sum_kw_{t,k}L_{t,k}$。有校准时再乘$\exp(-z_t^\top\theta)$，无校准直接输出$C_t$。','']
    for obj in ['frobenius']:
        lines += [f'## {obj.upper()}：对应训练与测试','', '| 专家数 | 协方差无校准 | 协方差有校准 | 精度无校准 | 精度有校准 |','| --- | ---: | ---: | ---: | ---: |']
        for k in [4,10]:
            vals=[]
            for fusion in ['covariance','precision']:
                sub=table[(table.experts==k)&(table.objective==obj)&(table.fusion==fusion)].set_index('calibrated').test_score
                vals.extend([sub[False],sub[True]])
                effects.append(dict(experts=k,objective=obj,fusion=fusion,without=sub[False],with_calibration=sub[True],change_with_minus_without=sub[True]-sub[False]))
            lines.append('| '+str(k)+' | '+' | '.join(f'{v:.9f}' for v in vals)+' |')
        lines.append('')
    pd.DataFrame(effects).to_csv(OUT/'calibration_effects.csv',index=False)
    lines += ['均越小越好。各分支分别验证早停，不通过测试选择轮次。10专家纯凸组合选中第0轮初始化模型，不能解释为训练增益。','',
        '新分支已核验：保存模型恢复、从协方差独立重建精度因子、numpy融合、预测/日期/标签/权重一致、损失独立重算、验证最佳轮次和正定性。当前仍是单种子且历史测试已查看，不据此宣称统计显著性。','',
        '[完整8项结果及最佳轮次](comparison.csv) · [校准前后差值](calibration_effects.csv) · [新训练协议](../precision_no_calibration/protocol.json) · [凸组合记录](../convex_fusion/README.md)']
    text='\n'.join(lines)+'\n';(OUT/'README.md').write_text(text,encoding='utf-8')
    (SOURCE/'README.md').write_text('# 精度融合无校准\n\n两组Frobenius训练记录。[完整有无校准对比](../calibration_comparison/README.md)。\n',encoding='utf-8')
    report=ROOT/'outputs/experiment_results.md';content=report.read_text(encoding='utf-8')
    if '## 7 融合方式与校准' in content: raise RuntimeError('Already merged')
    tables=text[text.index('## FROBENIUS'):text.index('均越小越好')].replace('## ','### ')
    report.write_text(content+'\n## 7 融合方式与校准：完整比较\n\n补跑精度融合无校准，门控从头训练；其他设置不变。新旧结果严格对应同一六指标、标签和预测日期。\n\n'+tables+'\n[校准消融完整记录](calibration_comparison/README.md)。所有分支均为动态权重，单种子历史测试。\n',encoding='utf-8')
    print(pd.DataFrame(effects).to_string(index=False))

if __name__=='__main__': main()

"""Train-only ridge expert combination; select alpha on validation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from portfolio_risk.config import load_config
from portfolio_risk.data.french import parse_industry_zip
from portfolio_risk.state_research import prepare_research

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/ridge_ablation'

def scores(pred,y,obj):
    return np.linalg.norm(pred-y,axis=(1,2))

def project(pred,floor):
    e,u=np.linalg.eigh((pred+pred.swapaxes(-1,-2))/2)
    return (u*np.maximum(e,floor)[:,None,:])@u.swapaxes(-1,-2),int((e.min(-1)<floor).sum())

def main():
    if OUT.exists(): raise RuntimeError('Refusing overwrite')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    OUT.mkdir()
    alphas=[0.,1e-6,1e-5,1e-4,1e-3,.01,.1,1.,10.,100.]
    protocol=dict(expert_counts=[4,10],ridge_alphas=alphas,ridge='pooled full-matrix entries, K shared coefficients, no intercept, normalized mean squared error plus alpha*||beta||^2',
        ridge_selection='validation unsquared Frobenius, coefficients fitted only on train',
        spd_floor='1e-6 times mean training target diagonal; applied before validation and test scoring',
        target='existing future20 uncentered second moment; NOT centered CTT task',
        test_rule='ridge alpha selected on validation, lock before test scoring',
        limitations='4 vs10 changes expert types as well as count; ridge differs from dynamic fusion architecture; single historical holdout')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    base=load_config(ROOT/'configs/preliminary.yaml');config=load_config(ROOT/'configs/state_whitener.yaml')
    frame,_=parse_industry_zip(ROOT/'data/raw/10_Industry_Portfolios_daily_CSV.zip')
    frame=frame.loc[base['data']['history_start']:'2025-12-31']
    p=prepare_research(frame,base,config);a=p['arrays']
    tr,va,te=(a['splits'][k] for k in ['train','validation','test'])
    y=a['realized'];experts=p['experts']
    unit=float(np.diagonal(y[tr],axis1=-2,axis2=-1).mean());floor=unit*1e-6
    selections=[];ridge_search=[];models={}
    for k in [4,10]:
        folder=OUT/f'experts{k}';folder.mkdir()
        # A row represents one (date, matrix entry), a column one expert.
        x=experts[tr,:k].transpose(0,2,3,1).reshape(-1,k)/unit
        target=y[tr].reshape(-1)/unit
        gram=x.T@x/len(x);rhs=x.T@target/len(x)
        candidates=[]
        for alpha in alphas:
            beta=np.linalg.lstsq(gram+alpha*np.eye(k),rhs,rcond=None)[0]
            v,repairs=project(np.einsum('k,tkij->tij',beta,experts[va,:k]),floor)
            row=dict(experts=k,alpha=alpha,validation_frobenius=float(scores(v,y[va],'frobenius').mean()),validation_spd_repairs=repairs)
            candidates.append((row,beta));ridge_search.append(row)
        selected,beta=min(candidates,key=lambda v:v[0]['validation_frobenius'])
        np.savez(folder/'ridge.npz',beta=beta,unit=unit,floor=floor,alpha=selected['alpha'],expert_names=np.array(p['names'][:k]))
        models[(k,'ridge')]=beta;selections.append(dict(method='Ridge',**selected))
    pd.DataFrame(ridge_search).to_csv(OUT/'ridge_validation_search.csv',index=False)
    (OUT/'selection.json').write_text(json.dumps(selections,indent=2))
    print('Selections locked; scoring test.',flush=True)
    rows=[];weights=[];origin=[]
    for k in [4,10]:
        folder=OUT/f'experts{k}'
        pred,repairs=project(np.einsum('k,tkij->tij',models[(k,'ridge')],experts[te,:k]),floor)
        np.savez_compressed(folder/'ridge_test.npz',prediction=pred,target=y[te],dates=frame.index[te].astype(str).to_numpy(dtype=str))
        score=scores(pred,y[te],'frobenius')
        rows.append(dict(experts=k,objective='frobenius',method='Ridge',test_score=score.mean(),spd_repairs=repairs))
        origin.extend(dict(experts=k,objective='frobenius',method='Ridge',date=str(frame.index[t].date()),loss=float(s)) for t,s in zip(te,score))
        weights.extend(dict(experts=k,method='Ridge',expert=n,coefficient=float(v)) for n,v in zip(p['names'][:k],models[(k,'ridge')]))
        directory=('dynamic_lw_holdout' if k==4 else 'dynamic_holdout')+'_six_features_k3'
        prev=np.load(ROOT/'outputs'/directory/'test_predictions.npz')
        name=next(name for name in prev.files if name.startswith('Dynamic') and '6F-K3' in name)
        np.testing.assert_array_equal(prev['target'],y[te])
        np.testing.assert_array_equal(prev['dates'],frame.index[te].astype(str).to_numpy(dtype=str))
        score=scores(prev[name],y[te],'frobenius')
        rows.append(dict(experts=k,objective='frobenius',method='DynamicGate-6F',test_score=score.mean(),spd_repairs=0))
        origin.extend(dict(experts=k,objective='frobenius',method='DynamicGate-6F',date=str(frame.index[t].date()),loss=float(v)) for t,v in zip(te,score))
    pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(weights).to_csv(OUT/'coefficients.csv',index=False)
    pd.DataFrame(origin).to_csv(OUT/'test_by_origin.csv',index=False)
    lines=['# Ridge 固定线性融合对照','',
        '4/10专家；系数仅训练集拟合，验证Frobenius选择alpha，测试不重拟合。目标为未来20日未中心化二阶矩。','',
        '每个专家一个跨日期、跨矩阵元素共享的系数；无截距，系数允许负数，不要求和为1。完整矩阵展平，输入和标签同时除以训练标签平均方差。目标是平均平方误差加alpha乘系数平方和。预测若非正定，特征值截断至训练平均方差的1e-6倍；验证和测试一致处理。','',
        'Ridge是协方差线性组合，动态模型为精度因子融合加风险校准；二者损失和结构不同，不能用此对照单独证明动态性有效。Ridge采用平方误差加L2训练。','',
        '| 专家数 | 方法 | 测试Frobenius ↓ |','| --- | --- | ---: |']
    table=pd.DataFrame(rows)
    for r in table.itertuples(): lines.append(f'| {r.experts} | {r.method} | {r.test_score:.9f} |')
    lines += ['', '[系数](coefficients.csv) · [验证选参](selection.json) · [逐期损失](test_by_origin.csv)']
    (OUT/'README.md').write_text('\\n'.join(lines),encoding='utf-8')
    print(table.to_string(index=False),flush=True)

if __name__=='__main__': main()

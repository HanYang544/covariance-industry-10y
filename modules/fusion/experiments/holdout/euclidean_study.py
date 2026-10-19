"""Ten fresh Euclidean fits; unchanged six-feature model/training settings."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from portfolio_risk.config import load_config
from portfolio_risk.data.french import parse_industry_zip
from portfolio_risk.state_research import prepare_research
from portfolio_risk.models.convex_gate import ConvexGate
from portfolio_risk.models.precision_gate import PrecisionGate
from portfolio_risk.losses.euclidean import euclidean_loss

ROOT=Path(__file__).resolve().parents[2]
SUITE=ROOT.parent.parent
OUT=ROOT/'outputs/euclidean_study'
IDEA_PATH=ROOT.parent/'forecasting/src/portfolio_risk/models/structured_ideas.py'
spec=importlib.util.spec_from_file_location('suite_structured_ideas',IDEA_PATH)
ideas=importlib.util.module_from_spec(spec);spec.loader.exec_module(ideas)

def distance(pred,target):
    i,j=np.triu_indices(target.shape[-1])
    return np.linalg.norm((pred-target)[:,i,j],axis=1)

def main():
    if OUT.exists():raise RuntimeError('Refuse to overwrite experiment')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    configurations=[dict(name=f'{fusion}_{k}_{"calibrated" if cal else "pure"}',fusion=fusion,experts=k,calibrated=cal)
                    for fusion in ['covariance','precision'] for k in [4,10] for cal in [False,True]]
    configurations += [dict(name=f'idea{i}',idea=i) for i in [1,2]]
    OUT.mkdir()
    archive=ROOT/'data/raw/10_Industry_Portfolios_daily_CSV.zip'
    protocol=dict(objective='euclidean',definition='mean unsquared L2 of upper triangle including diagonal; no off-diagonal sqrt(2) weight',
       configurations=configurations,features=6,top_k=3,assets=10,seed=20261012,optimizer='Adam',learning_rate=.01,
       max_epochs=500,patience=30,min_delta=1e-12,batch='full train',target='future20 uncentered mean outer product',
       source_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),test_rule='all checkpoints locked before any test scoring',
       limitations='single seed; reused historical test; original Idea1 q3/Idea2 q10 LW120; no parameter search')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    base=load_config(ROOT/'configs/preliminary.yaml');config=load_config(ROOT/'configs/state_whitener.yaml')
    frame,_=parse_industry_zip(archive);frame=frame.loc[base['data']['history_start']:'2025-12-31']
    print('Preparing unchanged data and expert matrices...',flush=True)
    prepared=prepare_research(frame,base,config);a=prepared['arrays']
    tr,va,te=(a['splits'][k] for k in ['train','validation','test'])
    assert (len(tr),len(va),len(te))==(651,37,62)
    assert tr[-1]+20<va[0] and va[-1]+20<te[0]
    ref=ROOT/'outputs/dynamic_holdout_six_features_k3'
    raw=pd.read_csv(ref/'market_risk_features.csv',index_col=0,parse_dates=True)
    saved=np.load(ref/'model.npz');np.testing.assert_array_equal(raw.index,frame.index)
    np.testing.assert_allclose(raw.iloc[tr].mean(0),saved['gate_mean'],rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(raw.iloc[tr].std(0,ddof=0),saved['gate_scale'],rtol=1e-12,atol=1e-14)
    state=np.clip((raw.to_numpy()-saved['gate_mean'])/saved['gate_scale'],-3,3)
    f=prepared['features']['global'];mean=f[tr].mean(0);scale=np.maximum(f[tr].std(0),1e-12)
    np.testing.assert_allclose(mean,saved['feature_mean']);np.testing.assert_allclose(scale,saved['feature_scale'])
    design=np.column_stack([np.ones(len(frame)),np.clip((f-mean)/scale,-3,3)])
    dates=frame.index[te].astype(str).to_numpy(dtype=str)
    old=np.load(ref/'test_predictions.npz')
    np.testing.assert_array_equal(old['dates'],dates);np.testing.assert_array_equal(old['target'],a['realized'][te])
    np.testing.assert_array_equal(old['LW-120'],a['covariance'][te,2])
    # Build only eligible splits: early covariance rows contain unavailable histories.
    batches={}
    for name,idx in [('train',tr),('validation',va),('test',te)]:
        eig,u=ideas.spectral_basis(a['covariance'][idx,2])
        batches[name]=dict(zip(['experts','whiteners','state','design','target','values','basis'],
            [torch.tensor(v,dtype=torch.float64) for v in [prepared['experts'][idx],prepared['whiteners'][idx],state[idx],design[idx],a['realized'][idx],eig,u]]))
    np.savez_compressed(OUT/'verification_inputs.npz',**{k:v.numpy() for k,v in batches['test'].items()},dates=dates)
    np.savez(OUT/'scaler.npz',gate_mean=saved['gate_mean'],gate_scale=saved['gate_scale'],feature_mean=mean,feature_scale=scale)
    pd.DataFrame([dict(split=n,samples=len(ix),first=str(frame.index[ix[0]]),last_origin=str(frame.index[ix[-1]]),last_label=str(frame.index[ix[-1]+20])) for n,ix in [('train',tr),('validation',va),('test',te)]]).to_csv(OUT/'splits.csv',index=False)
    def create(c):
        return ideas.StructuredRiskModel(c['idea']).double() if 'idea' in c else (ConvexGate if c['fusion']=='covariance' else PrecisionGate)(c['experts'],c['calibrated'])
    def predict(model,c,split):
        b=batches[split]
        if 'idea' in c:return model(b['state'],b['values'],b['basis'])
        matrices=b['experts' if c['fusion']=='covariance' else 'whiteners'][:,:c['experts']]
        return model(matrices,b['state'],b['design'])
    results=[]
    for c in configurations:
        folder=OUT/c['name'];folder.mkdir();torch.manual_seed(20261012)
        model=create(c);optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        best=float('inf');best_epoch=0;history=[]
        for epoch in range(501):
            if epoch:
                optimizer.zero_grad();loss=euclidean_loss(predict(model,c,'train'),batches['train']['target'])
                loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters())
                optimizer.step()
                if hasattr(model,'constrain'):model.constrain()
            with torch.no_grad():
                scores={n:float(euclidean_loss(predict(model,c,n),batches[n]['target'])) for n in ['train','validation']}
            assert np.isfinite(list(scores.values())).all()
            history.append(dict(epoch=epoch,train_loss=scores['train'],validation_loss=scores['validation']))
            if scores['validation']<best-1e-12:
                best=scores['validation'];best_epoch=epoch;checkpoint=copy.deepcopy(model.state_dict())
            if epoch-best_epoch>=30:break
        torch.save(checkpoint,folder/'model.pt');pd.DataFrame(history).to_csv(folder/'training_history.csv',index=False)
        results.append(dict(**c,objective='euclidean',validation_score=best,best_epoch=best_epoch,stopped_epoch=epoch))
        print(f'{c["name"]}: validation {best:.9f}, best/stop {best_epoch}/{epoch}',flush=True)
    (OUT/'selection.json').write_text(json.dumps(results,indent=2))
    for c,r in zip(configurations,results):
        folder=OUT/c['name'];model=create(c);model.load_state_dict(torch.load(folder/'model.pt',weights_only=True))
        with torch.no_grad():p=predict(model,c,'test').numpy()
        y=batches['test']['target'].numpy();np.linalg.cholesky(p)
        losses=distance(p,y);r['test_score']=float(losses.mean())
        np.testing.assert_allclose(euclidean_loss(torch.tensor(p),torch.tensor(y)).item(),r['test_score'],rtol=1e-12)
        np.savez_compressed(folder/'test_predictions.npz',prediction=p,target=y,dates=dates)
        pd.DataFrame(dict(date=dates,euclidean=losses)).to_csv(folder/'test_by_origin.csv',index=False)
        (folder/'result.json').write_text(json.dumps(r,indent=2))
    pd.DataFrame(results).to_csv(OUT/'comparison.csv',index=False)
    baseline=[]
    for key in old.files:
        if key.startswith(('LW-','IEWMA-','EWMA-')):
            baseline.append(dict(method=key,euclidean=float(distance(old[key],old['target']).mean())))
    pd.DataFrame(baseline).to_csv(OUT/'baselines.csv',index=False)
    lines=['# Euclidean独立训练实验','','六特征、10资产，未来20日未中心化二阶矩；训练/验证/测试651/37/62。', '',
      '损失为含对角线上三角55元素的未平方欧氏距离，非对角只计一次。每组从头训练、验证Euclidean早停、恢复最佳权重再测试。', '',
      'Adam 0.01、全批量、最多500轮、耐心30、seed 20261012；模型结构不变。Idea1为LW120/q3，Idea2为LW120/q10。历史调参和4/14特征实验不在本轮范围。', '',
      '| 方法 | 测试Euclidean ↓ | 验证Euclidean | 最佳/停止轮次 |','| --- | ---: | ---: | --- |']
    for r in results:lines.append(f'| {r["name"]} | {r["test_score"]:.9f} | {r["validation_score"]:.9f} | {r["best_epoch"]}/{r["stopped_epoch"]} |')
    lines+=['','## 统计基线','','同一批测试日期和标签，原生统计估计器不进行损失训练。','','| 方法 | 测试Euclidean ↓ |','| --- | ---: |']
    for r in baseline:lines.append(f'| {r["method"]} | {r["euclidean"]:.9f} |')
    lines+=['','Frobenius历史记录保留。不同损失数值不能直接比较大小；单种子、重复查看的历史测试，仅作探索。不包含Ridge重训，因为其原目标为平方误差加L2。', '',
      '[协议](protocol.json) · [模型分数](comparison.csv) · [基线分数](baselines.csv)']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(pd.DataFrame(results)[['name','test_score','best_epoch']].to_string(index=False),flush=True)

if __name__=='__main__':main()

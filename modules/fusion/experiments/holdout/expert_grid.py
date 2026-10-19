"""Explicit test-ranked exploratory grid; validation still selects epochs."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from portfolio_risk.config import load_config
from portfolio_risk.data.french import parse_industry_zip
from portfolio_risk.state_research import prepare_research
from portfolio_risk.models.precision_gate import PrecisionGate
from portfolio_risk.losses.euclidean import euclidean_loss

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/expert_grid'

def candidates():
    return [dict(name=f'l{l}_i{i}_d{d}',lw=l,iewma=i,diagonal=d,experts=l+i+d,
                 indices=list(range(l))+list(range(4,4+i))+([9] if d else []))
            for l,i,d in itertools.product(range(5),range(6),range(2)) if l+i+d>=2]

def objective(p,y,o):
    return torch.linalg.matrix_norm(p-y,ord='fro').mean() if o=='frobenius' else euclidean_loss(p,y)

def main():
    if OUT.exists():raise RuntimeError('Refusing overwrite')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    grid=candidates();assert len(grid)==56
    archive=ROOT/'data/raw/10_Industry_Portfolios_daily_CSV.zip'
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest==json.loads((ROOT/'outputs/euclidean_study/protocol.json').read_text())['source_sha256']
    OUT.mkdir()
    protocol=dict(grid=grid,grid_definition='prefix LW20/60/120/252 x prefix five IEWMA x optional diagonal; K>=2',
        selection='TEST-ranked exploratory search explicitly requested; validation selects epoch only',
        source_sha256=digest,features=6,top_k=3,assets=10,seed=20261012,learning_rate=.01,
        optimizer='Adam',max_epochs=500,patience=30,min_delta=1e-12,batch='full train',
        model='PrecisionGate calibrated=True',target='future20 uncentered mean outer product',
        objectives=['frobenius','euclidean'],limitations='single seed; selected test score is optimistic, not independent generalization evidence; family-prefix grid not exhaustive subsets',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    base=load_config(ROOT/'configs/preliminary.yaml');config=load_config(ROOT/'configs/state_whitener.yaml')
    frame,_=parse_industry_zip(archive);frame=frame.loc[base['data']['history_start']:'2025-12-31']
    prepared=prepare_research(frame,base,config);a=prepared['arrays']
    tr,va,te=[a['splits'][s] for s in ['train','validation','test']]
    assert (len(tr),len(va),len(te))==(651,37,62)
    assert tr[-1]+20<va[0] and va[-1]+20<te[0]
    ref=ROOT/'outputs/dynamic_holdout_six_features_k3'
    raw=pd.read_csv(ref/'market_risk_features.csv',index_col=0,parse_dates=True);saved=np.load(ref/'model.npz')
    np.testing.assert_array_equal(raw.index,frame.index)
    np.testing.assert_allclose(raw.iloc[tr].mean(axis=0),saved['gate_mean'],rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(raw.iloc[tr].std(axis=0,ddof=0),saved['gate_scale'],rtol=1e-12,atol=1e-14)
    state=np.clip((raw.to_numpy()-saved['gate_mean'])/saved['gate_scale'],-3,3)
    f=prepared['features']['global'];mean=f[tr].mean(0);scale=np.maximum(f[tr].std(0),1e-12)
    np.testing.assert_allclose(mean,saved['feature_mean']);np.testing.assert_allclose(scale,saved['feature_scale'])
    design=np.column_stack([np.ones(len(frame)),np.clip((f-mean)/scale,-3,3)])
    dates=frame.index[te].astype(str).to_numpy(dtype=str)
    old=np.load(ref/'test_predictions.npz')
    np.testing.assert_array_equal(dates,old['dates']);np.testing.assert_array_equal(a['realized'][te],old['target'])
    b={n:tuple(torch.tensor(v[idx],dtype=torch.float64) for v in [prepared['whiteners'],state,design,a['realized']])
       for n,idx in a['splits'].items()}
    np.savez_compressed(OUT/'verification_inputs.npz',whiteners=b['test'][0].numpy(),state=b['test'][1].numpy(),design=b['test'][2].numpy(),target=b['test'][3].numpy(),dates=dates,names=np.array(prepared['names']))
    np.savez(OUT/'scaler.npz',gate_mean=saved['gate_mean'],gate_scale=saved['gate_scale'],feature_mean=mean,feature_scale=scale)
    rows=[]
    for c in grid:
        for o in ['frobenius','euclidean']:
            name=c['name']+'_'+o;folder=OUT/name;folder.mkdir();torch.manual_seed(20261012)
            model=PrecisionGate(c['experts'],True);optimizer=torch.optim.Adam(model.parameters(),lr=.01)
            def predict(split):
                w,x,d,y=b[split]
                return model(w[:,c['indices']],x,d)
            best=float('inf');best_epoch=0;history=[]
            for epoch in range(501):
                if epoch:
                    optimizer.zero_grad();objective(predict('train'),b['train'][3],o).backward()
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                    optimizer.step();model.constrain()
                with torch.no_grad():t,v=[float(objective(predict(n),b[n][3],o)) for n in ['train','validation']]
                assert np.isfinite([t,v]).all()
                history.append(dict(epoch=epoch,train_loss=t,validation_loss=v))
                if v<best-1e-12:best=v;best_epoch=epoch;checkpoint=copy.deepcopy(model.state_dict())
                if epoch-best_epoch>=30:break
            torch.save(checkpoint,folder/'model.pt');pd.DataFrame(history).to_csv(folder/'training_history.csv',index=False)
            model.load_state_dict(torch.load(folder/'model.pt',weights_only=True))
            w,x,d,y=b['test']
            with torch.no_grad():p,weights=model(w[:,c['indices']],x,d,True)
            p=p.numpy();weights=weights.numpy();np.linalg.cholesky(p);np.testing.assert_allclose(weights.sum(1),1)
            err=p-y.numpy();ii,jj=np.triu_indices(10)
            scores=np.linalg.norm(err,axis=(1,2)) if o=='frobenius' else np.linalg.norm(err[:,ii,jj],axis=1)
            np.testing.assert_allclose(scores.mean(),objective(torch.tensor(p),y,o).item(),rtol=1e-12)
            r=dict(**c,run=name,objective=o,expert_names='; '.join(prepared['names'][k] for k in c['indices']),
                   best_epoch=best_epoch,stopped_epoch=epoch,validation_score=best,test_score=float(scores.mean()),parameters=sum(p.numel() for p in model.parameters()))
            np.savez_compressed(folder/'test_predictions.npz',prediction=p,target=y.numpy(),dates=dates,weights=weights)
            pd.DataFrame(dict(date=dates,loss=scores)).to_csv(folder/'test_by_origin.csv',index=False)
            (folder/'result.json').write_text(json.dumps(r,indent=2),encoding='utf-8');rows.append(r)
            print(f'{len(rows)}/112 {name}: test={r["test_score"]:.9f} best={best_epoch}',flush=True)
    table=pd.DataFrame(rows).sort_values(['objective','test_score','name'],kind='stable')
    table.to_csv(OUT/'comparison.csv',index=False)
    winners=table.groupby(['objective','experts'],sort=True).head(1).sort_values(['objective','experts'])
    winners.to_csv(OUT/'best_by_count.csv',index=False)
    selected=table.groupby('objective',sort=True).head(1)
    selected.to_csv(OUT/'test_selected.csv',index=False)
    lines=['# 动态门控专家网格：按测试集排名','','固定六特征、精度因子融合＋校准；56套集合×两种独立损失=112次训练。每组验证早停，恢复权重后评分测试。', '',
        '**测试集参与组合筛选：最低分存在选择偏差，不能当成独立泛化证明。单种子；不是全部专家子集穷举。**', '',
        '网格：LW数0–4、IEWMA数0–5、对角专家开/关，总专家数至少2。LW依次20/60/120/252；IEWMA依次21-63/63-125/125-250/250-500/500-1000。每类取前几个，对角专家为Diagonal-IEWMA-21。', '',
        '训练/验证/测试651/37/62，Adam 0.01，全批量，seed 20261012，最多500轮、耐心30，含第0轮。改变专家数也改变输出层参数量，不能把差异只归因于数量。', '',
        '## 各数量的测试最优集合','','| 损失 | 专家数 | LW数 | IEWMA数 | 对角 | 测试误差 ↓ | 最佳轮次 |','| --- | ---: | ---: | ---: | --- | ---: | ---: |']
    for r in winners.itertuples():lines.append(f'| {r.objective} | {r.experts} | {r.lw} | {r.iewma} | {bool(r.diagonal)} | {r.test_score:.9f} | {r.best_epoch} |')
    lines+=['','## 各损失前五名','','| 损失 | 专家数 | 专家组成 | 测试误差 ↓ |','| --- | ---: | --- | ---: |']
    for r in table.groupby('objective').head(5).itertuples():lines.append(f'| {r.objective} | {r.experts} | {r.expert_names} | {r.test_score:.9f} |')
    lines+=['','[全部112组](comparison.csv) · [按数量最优](best_by_count.csv) · [测试筛选结果](test_selected.csv) · [协议](protocol.json)']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(selected[['objective','experts','expert_names','test_score']].to_string(index=False),flush=True)

if __name__=='__main__':main()

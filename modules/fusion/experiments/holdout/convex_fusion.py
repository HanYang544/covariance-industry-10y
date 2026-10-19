"""Eight predeclared arithmetic fusion fits, preserving existing experiments."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from portfolio_risk.config import load_config
from portfolio_risk.data.french import parse_industry_zip
from portfolio_risk.state_research import prepare_research
from portfolio_risk.models.convex_gate import ConvexGate,covariance_loss

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/convex_fusion'

def main():
    if OUT.exists(): raise RuntimeError('Preserve completed experiment; refusing overwrite')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    archive=ROOT/'data/raw/10_Industry_Portfolios_daily_CSV.zip'
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest==json.loads((ROOT/'data/raw/industry_10_manifest.json').read_text())['sha256']
    originals={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for d in (ROOT/'outputs').glob('dynamic_*') for f in d.rglob('*') if f.is_file()}
    OUT.mkdir()
    protocol=dict(experts=[4,10],calibrated=[False,True],losses=['frobenius'],features=6,top_k=3,
        fusion='sum softmax(state network) * expert covariance; calibrated branch multiplies by exp(-design@theta)',
        target='future20 mean uncentered return outer product; same original fusion task',
        optimizer='Adam',learning_rate=.01,max_epochs=500,patience=30,min_delta=1e-12,seed=20261012,
        source_sha256=digest,originals_sha256=originals,
        test_rule='freeze eight configs before train; validation early stopping only; all checkpoints locked before test',
        limitations='single seed and previously inspected historical test; pure model removes calibration, calibrated model changes fusion only')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    base=load_config(ROOT/'configs/preliminary.yaml');config=load_config(ROOT/'configs/state_whitener.yaml')
    frame,_=parse_industry_zip(archive);frame=frame.loc[base['data']['history_start']:'2025-12-31']
    prepared=prepare_research(frame,base,config);a=prepared['arrays']
    tr,va,te=(a['splits'][k] for k in ['train','validation','test'])
    assert tr[-1]+20<va[0] and va[-1]+20<te[0]
    # Reuse only train-fitted six-feature transforms, verified against raw features.
    reference=ROOT/'outputs/dynamic_holdout_six_features_k3'
    raw=pd.read_csv(reference/'market_risk_features.csv',index_col=0,parse_dates=True)
    np.testing.assert_array_equal(raw.index,frame.index)
    saved=np.load(reference/'model.npz')
    np.testing.assert_allclose(raw.iloc[tr].mean(axis=0),saved['gate_mean'],rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(raw.iloc[tr].std(axis=0,ddof=0),saved['gate_scale'],rtol=1e-12,atol=1e-14)
    state=np.clip((raw.to_numpy()-saved['gate_mean'])/saved['gate_scale'],-3,3)
    f=prepared['features']['global'];mean=f[tr].mean(0);scale=np.maximum(f[tr].std(0),1e-12)
    np.testing.assert_allclose(mean,saved['feature_mean']);np.testing.assert_allclose(scale,saved['feature_scale'])
    design=np.column_stack([np.ones(len(frame)),np.clip((f-mean)/scale,-3,3)])
    np.savez(OUT/'scaler.npz',gate_mean=saved['gate_mean'],gate_scale=saved['gate_scale'],feature_mean=mean,feature_scale=scale)
    with np.load(reference/'test_predictions.npz') as old:
        np.testing.assert_array_equal(old['target'],a['realized'][te])
        np.testing.assert_array_equal(old['dates'],frame.index[te].astype(str).to_numpy(dtype=str))
    records=[]
    for k in [4,10]:
        batches={split:tuple(torch.tensor(v[ix],dtype=torch.float64) for v in [prepared['experts'][:,:k],state,design,a['realized']]) for split,ix in [('train',tr),('validation',va)]}
        for calibrated in [False,True]:
            for obj in ['frobenius']:
                label=f'experts{k}_{"calibrated" if calibrated else "pure"}_{obj}'
                folder=OUT/label;folder.mkdir()
                torch.manual_seed(20261012);model=ConvexGate(k,calibrated)
                opt=torch.optim.Adam(model.parameters(),lr=.01)
                best=float('inf');history=[]
                for epoch in range(501):
                    if epoch:
                        c,x,d,y=batches['train'];opt.zero_grad()
                        loss=covariance_loss(model(c,x,d),y,obj)
                        loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters())
                        opt.step();model.constrain()
                    with torch.no_grad():
                        values={}
                        for split,(c,x,d,y) in batches.items(): values[split]=float(covariance_loss(model(c,x,d),y,obj))
                    assert np.isfinite(list(values.values())).all()
                    history.append(dict(epoch=epoch,train_loss=values['train'],validation_loss=values['validation']))
                    if values['validation']<best-1e-12:
                        best=values['validation'];best_epoch=epoch;checkpoint=copy.deepcopy(model.state_dict())
                    if epoch-best_epoch>=30: break
                torch.save(checkpoint,folder/'model.pt')
                pd.DataFrame(history).to_csv(folder/'training_history.csv',index=False)
                row=dict(experts=k,calibrated=calibrated,objective=obj,directory=label,validation_score=best,best_epoch=best_epoch,stopped_epoch=epoch,parameters=sum(p.numel() for p in model.parameters()))
                records.append(row)
                print(f'{label}: validation={best:.9g}, epoch={best_epoch}/{epoch}',flush=True)
    (OUT/'selection.json').write_text(json.dumps(records,indent=2))
    np.savez_compressed(OUT/'verification_inputs.npz',experts=prepared['experts'][te],state=state[te],design=design[te],target=a['realized'][te],dates=frame.index[te].astype(str).to_numpy(dtype=str),names=np.array(prepared['names']))
    print('All eight checkpoints locked; testing.',flush=True)
    comparison=[]
    for row in records:
        k,obj=row['experts'],row['objective'];folder=OUT/row['directory']
        model=ConvexGate(k,row['calibrated']);model.load_state_dict(torch.load(folder/'model.pt',weights_only=True))
        with torch.no_grad():
            pred,w=model(torch.tensor(prepared['experts'][te,:k]),torch.tensor(state[te]),torch.tensor(design[te]),True)
            score=float(covariance_loss(pred,torch.tensor(a['realized'][te]),obj));pred=pred.numpy();w=w.numpy()
        np.linalg.cholesky(pred);np.testing.assert_allclose(w.sum(1),1)
        assert w.std(0).max()>1e-8
        np.savez_compressed(folder/'test_predictions.npz',prediction=pred,target=a['realized'][te],dates=frame.index[te].astype(str).to_numpy(dtype=str))
        pd.DataFrame(w,index=frame.index[te],columns=prepared['names'][:k]).to_csv(folder/'test_weights.csv')
        comparison.append(dict(**row,test_score=score,min_eigenvalue=float(np.linalg.eigvalsh(pred).min())))
    pd.DataFrame(comparison).to_csv(OUT/'comparison.csv',index=False)
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in originals.items())
    print(pd.DataFrame(comparison).to_string(index=False),flush=True)

if __name__=='__main__': main()

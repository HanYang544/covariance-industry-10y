"""Repeat every retained main-report configuration without seed/test selection."""
import copy
import hashlib
import importlib.util
import itertools
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from portfolio_risk.config import load_config
from portfolio_risk.data.french import parse_industry_zip
from portfolio_risk.state_research import prepare_research
from portfolio_risk.models.convex_gate import ConvexGate
from portfolio_risk.models.precision_gate import PrecisionGate
from portfolio_risk.models.joint_refinement import JointRefinement
from portfolio_risk.losses.euclidean import euclidean_loss

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT.parent.parent
OUT = ROOT / 'outputs/multiseed'
IDEA_PATH = ROOT.parent / 'forecasting/src/portfolio_risk/models/structured_ideas.py'
spec = importlib.util.spec_from_file_location('multiseed_ideas', IDEA_PATH)
ideas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ideas)
SEEDS = [20261012, 20261013, 20261014]
OBJECTIVES = ['frobenius', 'euclidean']


def configurations():
    result = []
    for fusion, k, cal in itertools.product(['covariance', 'precision'], [4, 10], [False, True]):
        result.append(dict(id=f'{fusion}_{k}_{"calibrated" if cal else "pure"}', kind='gate',
                           fusion=fusion, experts=k, calibrated=cal, indices=list(range(k)), group='fusion'))
    for l, i, d in itertools.product(range(5), range(6), range(2)):
        if l+i+d < 2:
            continue
        indices = list(range(l))+list(range(4, 4+i))+([9] if d else [])
        alias = next((c for c in result if c['kind']=='gate' and c['fusion']=='precision'
                      and c['calibrated'] and c['indices']==indices), None)
        if alias is None:
            alias = dict(id=f'grid_l{l}_i{i}_d{d}', kind='gate', fusion='precision',
                         experts=len(indices), calibrated=True, indices=indices, group='grid')
            result.append(alias)
        alias['grid'] = dict(name=f'l{l}_i{i}_d{d}', lw=l, iewma=i, diagonal=d)
    for idea, window in itertools.product([1, 2], [20, 60, 120]):
        result.append(dict(id=f'idea{idea}_lw{window}', kind='idea', group='idea',
                           idea=idea, window=window, q=3 if idea==1 else 10))
    for fusion in ['precision', 'covariance']:
        for variant, q in [(v,q) for v in ['spectral','learned'] for q in [1,3,5]]+[('idea2',10)]:
            result.append(dict(id=f'joint_{fusion}_{variant}_q{q}', kind='joint', group='joint',
                               fusion=fusion, variant=variant, q=q, experts=10, calibrated=False))
    assert len(result)==82 and len({c['id'] for c in result})==82
    assert sum('grid' in c for c in result)==56
    return result


def objective(p, y, name):
    if name=='frobenius':
        return torch.linalg.matrix_norm(p-y, ord='fro').mean()
    if name=='euclidean':
        return euclidean_loss(p,y)
    raise ValueError(name)


def numpy_scores(p,y,name):
    error=p-y
    if name=='frobenius':
        return np.linalg.norm(error,axis=(1,2))
    if name=='euclidean':
        i,j=np.triu_indices(y.shape[-1])
        return np.linalg.norm(error[:,i,j],axis=1)
    raise ValueError(name)


def create(c):
    if c['kind']=='joint':
        return JointRefinement(c['variant'],c['q'],c['fusion'])
    if c['kind']=='idea':
        return ideas.StructuredRiskModel(c['idea']).double()
    return (ConvexGate if c['fusion']=='covariance' else PrecisionGate)(c['experts'],c['calibrated'])


def predict(model,c,b,details=False):
    if c['kind']=='idea':
        return model(b['state'],b[f'values_{c["window"]}'],b[f'basis_{c["window"]}'])
    key='experts' if c['fusion']=='covariance' else 'whiteners'
    if c['kind']=='joint':
        return model(b[key],b['state'],diagnostics=details)
    return model(b[key][:,c['indices']],b['state'],b['design'],return_weights=details)


def prepare():
    base=load_config(ROOT/'configs/preliminary.yaml')
    config=load_config(ROOT/'configs/state_whitener.yaml')
    frame,_=parse_industry_zip(ROOT/'data/raw/10_Industry_Portfolios_daily_CSV.zip')
    frame=frame.loc[base['data']['history_start']:'2025-12-31']
    p=prepare_research(frame,base,config);a=p['arrays']
    tr,va,te=[a['splits'][s] for s in ['train','validation','test']]
    assert (len(tr),len(va),len(te))==(651,37,62)
    assert tr[-1]+20<va[0] and va[-1]+20<te[0]
    ref=ROOT/'outputs/dynamic_holdout_six_features_k3'
    raw=pd.read_csv(ref/'market_risk_features.csv',index_col=0,parse_dates=True)
    saved=np.load(ref/'model.npz')
    np.testing.assert_array_equal(raw.index,frame.index)
    mean=raw.iloc[tr].mean(0).to_numpy();scale=raw.iloc[tr].std(0,ddof=0).to_numpy()
    np.testing.assert_allclose(mean,saved['gate_mean'],rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(scale,saved['gate_scale'],rtol=1e-12,atol=1e-14)
    state=np.clip((raw.to_numpy()-mean)/scale,-3,3)
    f=p['features']['global'];fm=f[tr].mean(0);fs=np.maximum(f[tr].std(0),1e-12)
    np.testing.assert_allclose(fm,saved['feature_mean'])
    np.testing.assert_allclose(fs,saved['feature_scale'])
    design=np.column_stack([np.ones(len(frame)),np.clip((f-fm)/fs,-3,3)])
    dates=frame.index[te].astype(str).to_numpy(dtype=str)
    old=np.load(ref/'test_predictions.npz')
    np.testing.assert_array_equal(dates,old['dates'])
    np.testing.assert_array_equal(a['realized'][te],old['target'])
    # The Idea branch must use exactly the same six features, not just the same dimension.
    idea_raw=pd.read_csv(ROOT.parent/'forecasting/outputs/structured_ideas/raw_features.csv',index_col=0,parse_dates=True)
    np.testing.assert_array_equal(idea_raw.index,raw.index)
    np.testing.assert_allclose(idea_raw.to_numpy(),raw.to_numpy(),equal_nan=True)
    batches={}
    for name,idx in a['splits'].items():
        b=dict(experts=p['experts'][idx],whiteners=p['whiteners'][idx],state=state[idx],
               design=design[idx],target=a['realized'][idx])
        for w in [20,60,120]:
            k=base['forecast']['lookback_windows'].index(w)
            b[f'values_{w}'],b[f'basis_{w}']=ideas.spectral_basis(a['covariance'][idx,k])
        assert all(np.isfinite(v).all() for v in b.values())
        np.savez_compressed(OUT/f'inputs_{name}.npz',**b,dates=frame.index[idx].astype(str).to_numpy(dtype=str))
        batches[name]={k:torch.tensor(v,dtype=torch.float64) for k,v in b.items()}
    np.savez(OUT/'scaler.npz',gate_mean=mean,gate_scale=scale,feature_mean=fm,feature_scale=fs)
    pd.DataFrame([dict(split=n,samples=len(ix),first=str(frame.index[ix[0]]),last_origin=str(frame.index[ix[-1]]),
                       last_label=str(frame.index[ix[-1]+20])) for n,ix in a['splits'].items()]).to_csv(OUT/'splits.csv',index=False)
    return batches,dates


def train(c,o,seed,batches):
    folder=OUT/'runs'/c['id']/o/str(seed);folder.mkdir(parents=True,exist_ok=True)
    selected=folder/'selection.json'
    if selected.exists():
        r=json.loads(selected.read_text());assert r['configuration']==c and r['seed']==seed and r['objective']==o
        return r
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=create(c);initial=copy.deepcopy(model.state_dict())
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    best=float('inf');best_epoch=0;history=[]
    for epoch in range(501):
        grad=0.;gate_grad=0.
        if epoch:
            optimizer.zero_grad()
            objective(predict(model,c,batches['train']),batches['train']['target'],o).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            grad=float(sum(p.grad.square().sum() for p in model.parameters()).sqrt())
            if c['kind']=='joint':
                gate_grad=float(sum(p.grad.square().sum() for p in model.fusion.parameters()).sqrt())
            optimizer.step()
            if hasattr(model,'constrain'):model.constrain()
        with torch.no_grad():
            t,v=[float(objective(predict(model,c,batches[s]),batches[s]['target'],o)) for s in ['train','validation']]
        assert np.isfinite([t,v,grad,gate_grad]).all()
        history.append(dict(epoch=epoch,train_loss=t,validation_loss=v,total_grad_norm=grad,gate_grad_norm=gate_grad))
        if v<best-1e-12:
            best=v;best_epoch=epoch;checkpoint=copy.deepcopy(model.state_dict())
        if epoch-best_epoch>=30:break
    torch.save(checkpoint,folder/'model.pt')
    pd.DataFrame(history).to_csv(folder/'training_history.csv',index=False)
    r=dict(id=c['id'],configuration=c,objective=o,seed=seed,best_epoch=best_epoch,stopped_epoch=epoch,
           validation_score=best,parameters=sum(p.numel() for p in model.parameters()),
           parameter_change_l2=float(sum((checkpoint[k]-initial[k]).double().square().sum() for k in initial).sqrt()))
    if c['kind']=='joint':
        assert max(h['gate_grad_norm'] for h in history)>0
        r['gate_parameter_change_l2']=float(sum((checkpoint[k]-initial[k]).square().sum() for k in initial if k.startswith('fusion.')).sqrt())
    selected.write_text(json.dumps(r,indent=2),encoding='utf-8')
    return r


def project(p,floor):
    values,u=np.linalg.eigh((p+p.swapaxes(-1,-2))/2)
    return (u*np.maximum(values,floor)[:,None,:])@u.swapaxes(-1,-2)


def train_ridge(k,seed,b):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    folder=OUT/'runs'/f'ridge_{k}'/'frobenius'/str(seed);folder.mkdir(parents=True,exist_ok=True)
    y=b['train']['target'].numpy();experts=b['train']['experts'].numpy()
    unit=float(np.diagonal(y,axis1=-2,axis2=-1).mean());floor=unit*1e-6
    x=experts[:,:k].transpose(0,2,3,1).reshape(-1,k)/unit;target=y.reshape(-1)/unit
    gram=x.T@x/len(x);rhs=x.T@target/len(x);candidates=[]
    for alpha in [0.,1e-6,1e-5,1e-4,1e-3,.01,.1,1.,10.,100.]:
        beta=np.linalg.lstsq(gram+alpha*np.eye(k),rhs,rcond=None)[0]
        pred=project(np.einsum('k,tkij->tij',beta,b['validation']['experts'].numpy()[:,:k]),floor)
        candidates.append((float(numpy_scores(pred,b['validation']['target'].numpy(),'frobenius').mean()),alpha,beta))
    val,alpha,beta=min(candidates,key=lambda x:x[0])
    np.savez(folder/'model.npz',beta=beta,unit=unit,floor=floor,alpha=alpha)
    pd.DataFrame([dict(alpha=a,validation_loss=v) for v,a,_ in candidates]).to_csv(folder/'validation_search.csv',index=False)
    r=dict(id=f'ridge_{k}',configuration=dict(id=f'ridge_{k}',kind='ridge',experts=k,group='ridge'),
           objective='frobenius',seed=seed,validation_score=val,alpha=alpha,parameters=k)
    (folder/'selection.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    return r


def evaluate(r,b,dates):
    c=r['configuration'];o=r['objective'];folder=OUT/'runs'/r['id']/o/str(r['seed'])
    extra={}
    if c['kind']=='ridge':
        z=np.load(folder/'model.npz')
        prediction=project(np.einsum('k,tkij->tij',z['beta'],b['experts'].numpy()[:,:c['experts']]),float(z['floor']))
    else:
        model=create(c);model.load_state_dict(torch.load(folder/'model.pt',weights_only=True));model.eval()
        with torch.no_grad():
            if c['kind']=='idea':p=predict(model,c,b)
            elif c['kind']=='gate':
                p,w=predict(model,c,b,True);extra['weights']=w.numpy()
            else:
                p,d=predict(model,c,b,True);extra=dict(weights=d['weights'].numpy(),base=d['base'].numpy())
            prediction=p.numpy()
    np.linalg.cholesky(prediction)
    target=b['target'].numpy();scores=numpy_scores(prediction,target,o)
    r['test_score']=float(scores.mean())
    np.testing.assert_allclose(r['test_score'],objective(torch.tensor(prediction),b['target'],o).item(),rtol=1e-12)
    np.savez_compressed(folder/'test_predictions.npz',prediction=prediction,target=target,dates=dates,**extra)
    pd.DataFrame(dict(date=dates,loss=scores)).to_csv(folder/'test_by_origin.csv',index=False)
    (folder/'result.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    return {k:v for k,v in r.items() if k!='configuration'} | dict(kind=c['kind'],group=c['group'])


def legacy_score(c,o):
    forecast=ROOT.parent/'forecasting/outputs'
    if c['kind']=='ridge':
        table=pd.read_csv(ROOT/'outputs/ridge_ablation/comparison.csv')
        return float(table[(table.method=='Ridge') & (table.experts==c['experts'])].iloc[0].test_score)
    if c['kind']=='joint':
        if c['variant']=='idea2':directory=f'joint_idea2_{c["fusion"]}'
        elif c['fusion']=='covariance':directory=f'joint_convex_q{c["q"]}'
        else:directory='joint_refinement' if c['q']==3 else f'joint_refinement_q{c["q"]}'
        table=pd.read_csv(ROOT/'outputs'/directory/'comparison.csv')
        return float(table[(table.variant==c['variant']) & (table.objective==o)].iloc[0].test_score)
    if c['kind']=='idea':
        if c['window']!=120:
            table=pd.read_csv(forecast/'idea_base_windows/comparison.csv')
            return float(table[(table.idea==c['idea']) & (table.window==c['window']) & (table.objective==o)].iloc[0].test_score)
        if o=='frobenius':
            table=pd.read_csv(forecast/'structured_ideas/comparison.csv')
            return float(table[(table.method==f'Idea{c["idea"]}') & (table.objective==o)].iloc[0].test_score)
        table=pd.read_csv(ROOT/'outputs/euclidean_study/comparison.csv')
        return float(table[table.name==f'idea{c["idea"]}'].iloc[0].test_score)
    if c['group']=='grid':
        table=pd.read_csv(ROOT/'outputs/expert_grid/comparison.csv')
        return float(table[(table.name==c['grid']['name']) & (table.objective==o)].iloc[0].test_score)
    table=pd.read_csv(ROOT/('outputs/calibration_comparison/comparison.csv' if o=='frobenius' else 'outputs/euclidean_study/comparison.csv'))
    return float(table[(table.fusion==c['fusion']) & (table.experts==c['experts']) & (table.calibrated==c['calibrated'])].iloc[0].test_score)


def summarize(table):
    rows=[]
    for (name,o),g in table.groupby(['id','objective'],sort=False):
        assert len(g)==3 and sorted(g.seed)==SEEDS
        rows.append(dict(id=name,objective=o,n=3,mean=g.test_score.mean(),std=g.test_score.std(ddof=1),
                         validation_mean=g.validation_score.mean(),validation_std=g.validation_score.std(ddof=1)))
    return pd.DataFrame(rows)


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    configs=configurations();OUT.mkdir(exist_ok=True)
    code_files=[Path(__file__),IDEA_PATH,ROOT/'src/portfolio_risk/models/joint_refinement.py',
                ROOT/'src/portfolio_risk/models/convex_gate.py',ROOT/'src/portfolio_risk/models/precision_gate.py',
                ROOT/'src/portfolio_risk/state_research.py',ROOT/'src/portfolio_risk/losses/euclidean.py']
    protocol=dict(seeds=SEEDS,configurations=configs,neural_fits=492,ridge_fits=6,
        scope='All retained six-feature main-report configurations including all 56 expert-grid candidates; duplicate configurations share runs.',
        excluded='Previously removed 4/14-feature experiments, LW100/252/504 and historical learning-rate/q searches, NLL and archived methods are not revived.',
        optimizer='Adam',learning_rate=.01,max_epochs=500,patience=30,min_delta=1e-12,include_epoch0=True,
        batch='full train',dtype='float64',threads=1,device='cpu',features=6,assets=10,
        target='future20 uncentered mean outer product',objective=OBJECTIVES,
        selection='Each seed independently validation-early-stopped. All checkpoints locked before any test scoring. No seed selection.',
        aggregation='Arithmetic mean and sample standard deviation (ddof=1) over three per-seed mean test losses, NOT pooled dates or ensemble predictions.',
        grid_selection='Rank fixed candidate sets by their three-seed mean TEST loss, exploratory, never average per-seed winners.',
        ridge='Deterministic pooled squared-error+L2 with validation Frobenius alpha selection; actually refitted three times, not a Frobenius-trained network.',
        source_sha256=hashlib.sha256((ROOT/'data/raw/10_Industry_Portfolios_daily_CSV.zip').read_bytes()).hexdigest(),
        code_sha256={str(f.relative_to(SUITE)):hashlib.sha256(f.read_bytes()).hexdigest() for f in code_files})
    assert protocol['source_sha256']==json.loads((ROOT/'outputs/euclidean_study/protocol.json').read_text())['source_sha256']
    protocol_path=OUT/'protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==protocol,'Cannot resume a changed protocol'
    else:protocol_path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    batches,dates=prepare();records=[]
    for c,o,seed in itertools.product(configs,OBJECTIVES,SEEDS):
        r=train(c,o,seed,batches);records.append(r)
        print(f'{len(records)}/492 {c["id"]} {o} seed={seed} val={r["validation_score"]:.9f} epoch={r["best_epoch"]}/{r["stopped_epoch"]}',flush=True)
    for k,seed in itertools.product([4,10],SEEDS):records.append(train_ridge(k,seed,batches))
    (OUT/'selection.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('All 498 selections locked. Evaluating saved checkpoints on test.',flush=True)
    table=pd.DataFrame([evaluate(r,batches['test'],dates) for r in records])
    table.to_csv(OUT/'per_seed.csv',index=False)
    summarize(table).to_csv(OUT/'summary.csv',index=False)
    audit=[]
    for r in records:
        if r['seed']!=SEEDS[0]:continue
        previous=legacy_score(r['configuration'],r['objective'])
        np.testing.assert_allclose(r['test_score'],previous,rtol=1e-8,atol=1e-13)
        audit.append(dict(id=r['id'],objective=r['objective'],previous=previous,reproduced=r['test_score'],absolute_difference=abs(previous-r['test_score'])))
    pd.DataFrame(audit).to_csv(OUT/'legacy_seed_audit.csv',index=False)
    (OUT/'completion.json').write_text(json.dumps(dict(neural_fits=492,ridge_fits=6,summary_rows=166,legacy_matches=len(audit),seeds=SEEDS),indent=2),encoding='utf-8')
    print('Complete: 492 neural fits + 6 deterministic Ridge refits; 166 mean/std rows; all historical-seed scores reproduced.',flush=True)


if __name__=='__main__':main()

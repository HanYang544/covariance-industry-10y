"""Run the retained 82 configurations against frozen 14-ETF targets."""
import argparse
import concurrent.futures
import hashlib
import itertools
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import multiseed as legacy
from portfolio_risk.models.joint_refinement_nd import JointRefinementND

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/paper_etf_migration'
BATCHES=None

def configurations():
    configs=legacy.configurations()
    for c in configs:
        if c['kind']=='idea' and c['idea']==2:c['q']=14
        if c['kind']=='joint' and c['variant']=='idea2':
            c['q']=14;c['id']=c['id'].replace('_q10','_q14')
    return configs

def create(c):
    if c['kind']=='joint':return JointRefinementND(c['variant'],c['q'],c['fusion'],dimension=14)
    if c['kind']=='idea':return legacy.ideas.StructuredRiskModel(c['idea'],dimension=14,components=3).double()
    return (legacy.ConvexGate if c['fusion']=='covariance' else legacy.PrecisionGate)(c['experts'],c['calibrated'])

def initialize():
    global BATCHES
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    legacy.OUT=OUT;legacy.create=create
    BATCHES={}
    for s in ['train','validation']:
        a=dict(np.load(OUT/f'inputs_{s}.npz'))
        BATCHES[s]={k:torch.tensor(v,dtype=torch.float64) for k,v in a.items() if k not in ['dates','origins']}

def fit(task):
    c,o,seed=task;t=time.time()
    result=legacy.train(c,o,seed,BATCHES)
    return result,time.time()-t

def ridge(k,objective,seed):
    b=BATCHES;folder=OUT/'runs'/f'ridge_{k}'/objective/str(seed);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'selection.json').exists():return json.loads((folder/'selection.json').read_text())
    y=b['train']['target'].numpy();experts=b['train']['experts'].numpy()
    unit=float(np.diagonal(y,axis1=-2,axis2=-1).mean());floor=unit*1e-6
    x=experts[:,:k].transpose(0,2,3,1).reshape(-1,k)/unit;target=y.reshape(-1)/unit
    gram=x.T@x/len(x);rhs=x.T@target/len(x);candidates=[]
    for alpha in [0.,1e-6,1e-5,1e-4,1e-3,.01,.1,1.,10.,100.]:
        beta=np.linalg.lstsq(gram+alpha*np.eye(k),rhs,rcond=None)[0]
        p=legacy.project(np.einsum('k,tkij->tij',beta,b['validation']['experts'].numpy()[:,:k]),floor)
        candidates.append((float(legacy.numpy_scores(p,b['validation']['target'].numpy(),objective).mean()),alpha,beta))
    val,alpha,beta=min(candidates,key=lambda x:x[0]);np.savez(folder/'model.npz',beta=beta,unit=unit,floor=floor,alpha=alpha)
    pd.DataFrame([dict(alpha=a,validation_loss=v) for v,a,_ in candidates]).to_csv(folder/'validation_search.csv',index=False)
    r=dict(id=f'ridge_{k}',configuration=dict(id=f'ridge_{k}',kind='ridge',experts=k,group='ridge'),objective=objective,seed=seed,validation_score=val,alpha=alpha,parameters=k)
    (folder/'selection.json').write_text(json.dumps(r,indent=2),encoding='utf-8');return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--benchmark',action='store_true');ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    initialize();configs=configurations()
    if args.benchmark:
        for c in [configs[0],next(c for c in configs if c['id']=='joint_precision_spectral_q3'),next(c for c in configs if c['id']=='joint_precision_idea2_q14')]:
            torch.manual_seed(legacy.SEEDS[0]);m=create(c);opt=torch.optim.Adam(m.parameters(),lr=.01);t=time.time()
            for _ in range(5):
                opt.zero_grad();p=legacy.predict(m,c,BATCHES['train']);loss=legacy.objective(p,BATCHES['train']['target'],'frobenius');loss.backward()
                assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in m.parameters());opt.step()
            print(c['id'],dict(seconds_per_epoch=(time.time()-t)/5,loss=float(loss.detach())),flush=True)
        return
    files=[Path(__file__),Path(__file__).with_name('prepare_etf_migration.py'),Path(legacy.__file__),legacy.IDEA_PATH,
           ROOT/'src/portfolio_risk/models/joint_refinement_nd.py',ROOT/'src/portfolio_risk/models/convex_gate.py',
           ROOT/'src/portfolio_risk/models/precision_gate.py',ROOT/'src/portfolio_risk/market_risk_features.py',ROOT/'src/portfolio_risk/covariance/iewma.py']
    protocol=dict(configurations=configs,seeds=legacy.SEEDS,objectives=legacy.OBJECTIVES,assets=14,features=6,neural_fits=492,ridge_fits=12,
        optimizer='Adam',learning_rate=.01,batch='full train666',max_epochs=500,patience=30,min_delta=1e-12,include_epoch0=True,dtype='float64',device='cpu',threads_per_worker=1,
        target='exact copies of paper_etf centered future20 covariance, ddof1',
        changes='14-dimensional outputs/full Idea2; centered LW; daily sample origins and 252 pre-2017 warmup returns; six features k3 unchanged',
        cab_difference='CAB retains Adam1e-4/batch128/max100/no epoch0; same task and test, not identical optimizer or input lengths',
        selection='all validation checkpoints locked before scoring; no test-driven configuration/seed selection',
        ridge='squared-error+L2 coefficients; alpha separately selected using matching F/E validation metric, NOT F/E-trained ridge',
        grid='all56 retained candidates reported; test ranking exploratory, validation-selected best also reported',
        code_sha256={str(p.relative_to(ROOT.parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.npz')},
        data_protocol_sha256=hashlib.sha256((OUT/'data_protocol.json').read_bytes()).hexdigest())
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol,'Changed protocol; refuse reuse'
    else:(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    tasks=list(itertools.product(configs,legacy.OBJECTIVES,legacy.SEEDS));records=[];started=time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool:
        futures={pool.submit(fit,t):t for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            r,seconds=future.result();records.append(r)
            progress=dict(completed=len(records),total=492,id=r['id'],objective=r['objective'],seed=r['seed'],best_epoch=r['best_epoch'],stopped_epoch=r['stopped_epoch'],seconds=seconds,elapsed=time.time()-started)
            (OUT/'progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
            print(json.dumps(progress),flush=True)
    for k,o,seed in itertools.product([4,10],legacy.OBJECTIVES,legacy.SEEDS):records.append(ridge(k,o,seed))
    records.sort(key=lambda r:(r['id'],r['objective'],r['seed']))
    (OUT/'selection.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('All 504 selections locked; evaluating TEST.',flush=True)
    a=dict(np.load(OUT/'inputs_test.npz'));dates=a['dates'];b={k:torch.tensor(v,dtype=torch.float64) for k,v in a.items() if k not in ['dates','origins']}
    rows=[legacy.evaluate(r,b,dates) for r in records]
    table=pd.DataFrame(rows);table.to_csv(OUT/'per_seed.csv',index=False);legacy.summarize(table).to_csv(OUT/'summary.csv',index=False)
    z=dict(np.load(OUT/'baselines.npz'));baseline=[]
    for k,name in enumerate(z['names']):
        baseline.append(dict(method=str(name),**{o:float(legacy.numpy_scores(z['prediction'][:,k],z['target'],o).mean()) for o in legacy.OBJECTIVES}))
    pd.DataFrame(baseline).to_csv(OUT/'baselines.csv',index=False)
    (OUT/'completion.json').write_text(json.dumps(dict(neural_fits=492,ridge_fits=12,summary_rows=168,seeds=legacy.SEEDS,test_samples=len(dates),seconds=time.time()-started),indent=2),encoding='utf-8')
    from summarize_etf_migration import main as summarize
    summarize()

if __name__=='__main__':main()

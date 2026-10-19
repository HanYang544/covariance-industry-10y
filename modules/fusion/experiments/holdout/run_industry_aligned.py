"""Retained models on the new industry protocol; old runner/source hashes untouched."""
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
import run_etf_migration as engine

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/industry_10y_aligned';legacy=engine.legacy

def create(c):
    if c['kind']=='joint':return engine.JointRefinementND(c['variant'],c['q'],c['fusion'],dimension=10)
    if c['kind']=='idea':return legacy.ideas.StructuredRiskModel(c['idea'],dimension=10,components=3).double()
    return (legacy.ConvexGate if c['fusion']=='covariance' else legacy.PrecisionGate)(c['experts'],c['calibrated'])

def initialize():
    engine.OUT=OUT;engine.create=create;engine.initialize()

def main():
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
    initialize();configs=legacy.configurations()
    files=[Path(__file__),Path(__file__).with_name('prepare_industry_aligned.py'),Path(engine.__file__),Path(legacy.__file__),legacy.IDEA_PATH,ROOT/'src/portfolio_risk/models/joint_refinement_nd.py',ROOT/'src/portfolio_risk/models/convex_gate.py',ROOT/'src/portfolio_risk/models/precision_gate.py']
    protocol=dict(configurations=configs,seeds=legacy.SEEDS,objectives=legacy.OBJECTIVES,assets=10,neural_fits=492,ridge_fits=12,
        optimizer='Adam',learning_rate=.01,max_epochs=500,patience=30,min_delta=1e-12,include_epoch0=True,batch='full training set',dtype='float64',
        selection='Lock all504 validation selections before test; fixed test parameters',ridge='Squared error+L2; matching F/E validation alpha selection',
        code_sha256={str(p.relative_to(ROOT.parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        data_protocol_sha256=hashlib.sha256((OUT/'data_protocol.json').read_bytes()).hexdigest())
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    else:p.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    records=[];t=time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=3,initializer=initialize) as pool:
        futures=[pool.submit(engine.fit,task) for task in itertools.product(configs,legacy.OBJECTIVES,legacy.SEEDS)]
        for f in concurrent.futures.as_completed(futures):
            r,seconds=f.result();records.append(r)
            progress=dict(completed=len(records),total=492,id=r['id'],objective=r['objective'],seed=r['seed'],seconds=time.time()-t)
            (OUT/'progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
            if len(records)%12==0:print(json.dumps(progress),flush=True)
    for k,o,seed in itertools.product([4,10],legacy.OBJECTIVES,legacy.SEEDS):records.append(engine.ridge(k,o,seed))
    records.sort(key=lambda r:(r['id'],r['objective'],r['seed']))
    (OUT/'selection.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('All504 selections locked; test evaluation starts',flush=True)
    a=dict(np.load(OUT/'inputs_test.npz'));b={k:torch.tensor(v,dtype=torch.float64) for k,v in a.items() if k not in ['dates','origins']}
    table=pd.DataFrame([legacy.evaluate(r,b,a['dates']) for r in records]);table.to_csv(OUT/'per_seed.csv',index=False)
    legacy.summarize(table).to_csv(OUT/'summary.csv',index=False)
    z=dict(np.load(OUT/'baselines.npz'));rows=[]
    for k,name in enumerate(z['names']):rows.append(dict(method=str(name),**{o:float(legacy.numpy_scores(z['prediction'][:,k],z['target'],o).mean()) for o in legacy.OBJECTIVES}))
    pd.DataFrame(rows).to_csv(OUT/'baselines.csv',index=False)
    (OUT/'completion.json').write_text(json.dumps(dict(neural_fits=492,ridge_fits=12,summary_rows=168,seconds=time.time()-t),indent=2),encoding='utf-8')
    print('Industry retained models complete',flush=True)

if __name__=='__main__':main()

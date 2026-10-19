"""Export validation-selected tests incrementally without interrupting training."""
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from portfolio_risk.models.direct_spd import DirectSPD

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/direct_spd_100'
SOURCE=ROOT/'outputs/industry_10y_aligned'
torch.set_num_threads(1)
protocol=json.loads((OUT/'protocol.json').read_text())
a=np.load(SOURCE/'inputs_test.npz')
scaler=np.load(SOURCE/'scaler.npz')
raw=np.genfromtxt(SOURCE/'raw_features.csv',delimiter=',',skip_header=1,usecols=range(1,7))
features=np.clip((raw-scaler['gate_mean'])/scaler['gate_scale'],-3,3)
x=torch.tensor(np.stack([features[t-99:t+1] for t in a['origins']]),dtype=torch.float64)
v,u=np.linalg.eigh(a['experts'][:,1]/protocol['unit'])
base=torch.tensor((u*np.sqrt(v)[:,None,:])@u.transpose(0,2,1),dtype=torch.float64)
deadline=time.monotonic()+7200
while True:
    changed=False
    for path in OUT.glob('*/*/selection.json'):
        destination=path.parent/'live_test.json'
        if destination.exists():continue
        try:r=json.loads(path.read_text())
        except json.JSONDecodeError:continue
        model=DirectSPD(r['method'],protocol['unit'])
        model.load_state_dict(torch.load(path.parent/'model.pt',map_location='cpu',weights_only=True));model.eval()
        with torch.no_grad():p=torch.cat([model(x[i:i+128],base[i:i+128]) for i in range(0,len(x),128)]).numpy()
        np.linalg.cholesky(p)
        error=p-a['target']
        if r['objective']=='euclidean':
            i,j=np.triu_indices(10);error=error[:,i,j]
        r['test']=float(np.linalg.norm(error.reshape(len(error),-1),axis=1).mean())
        r['evaluation']='CPU float64 incremental evaluation after this run validation selection; training unchanged'
        np.savez_compressed(path.parent/'live_predictions.npz',prediction=p,dates=a['dates'])
        destination.write_text(json.dumps(r,indent=2),encoding='utf-8')
        changed=True
        print(json.dumps(r),flush=True)
    records=[json.loads(p.read_text()) for p in OUT.glob('*/*/live_test.json')]
    if changed:
        (OUT/'live_results.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        subprocess.run([sys.executable,'-B',str(ROOT.parent.parent/'scripts/build_report.py')],check=True)
    if len(records)==10 or '--watch' not in sys.argv:break
    if time.monotonic()>deadline:raise TimeoutError('Incremental export timed out; saved results retained')
    time.sleep(5)

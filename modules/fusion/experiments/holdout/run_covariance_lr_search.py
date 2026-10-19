"""Frozen LR grid, validation-only selection across three seeds, additive output."""
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
import torch
from covariance_lr_worker import main as fit

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/covariance_lr_search'
SEEDS=[20261012,20261013,20261014]
RATES=[.01,.003,.001,.0001]
METHODS=['residual','cholesky','matrix_exp','spectral_softplus','variance_correlation']

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')

def base(method,loss,seed):
    parent=ROOT/'outputs/covariance_spd_100' if seed==20261012 else ROOT/'outputs/seed_completion'/str(seed)/'covariance_spd_100'
    return parent/method/loss

def folder(method,loss,rate,seed):return OUT/method/loss/str(rate)/str(seed)

def refresh():subprocess.run([sys.executable,'-B',str(ROOT.parent.parent/'scripts/build_report.py')],check=True)

def main():
    assert torch.cuda.is_available()
    OUT.mkdir(parents=True,exist_ok=True)
    # Validate source data/model equality with the baseline runs before reusing scores.
    paths=[ROOT/'outputs/industry_10y_aligned'/f'inputs_{s}.npz' for s in ['train','validation','test']]
    paths += [ROOT/'outputs/industry_10y_aligned/cab'/f'{s}.npz' for s in ['train','validation','test']]
    paths += [ROOT/'outputs/industry_10y_aligned/cab/scaler.npz']
    paths += [ROOT/'src/portfolio_risk/models'/f'{s}.py' for s in ['covariance_spd','direct_spd','temporal_gate']]
    digest={p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for seed in SEEDS:
        old=json.loads((base('residual','frobenius',seed).parents[1]/'protocol.json').read_text())
        old_hash=old['hashes']
        for p,h in digest.items():
            suffix=p.split('/modules/fusion/')[-1]
            matches=[v for k,v in old_hash.items() if k.replace('\\','/').endswith(suffix)]
            assert matches==[h],(seed,p)
    plan=dict(seeds=SEEDS,rates=RATES,methods=METHODS,objectives=['frobenius','euclidean'],new_fits=90,device=torch.cuda.get_device_name(0),selection='lowest three-seed mean validation; test only after lock per method/objective',max_epochs=500,patience=30,epoch0=True,hashes=digest,worker_sha256=hashlib.sha256(Path(__file__).with_name('covariance_lr_worker.py').read_bytes()).hexdigest())
    if (OUT/'plan.json').exists():assert json.loads((OUT/'plan.json').read_text())==plan
    else:save(OUT/'plan.json',plan)
    refresh();start=time.time()
    for method in METHODS:
        for loss in ['frobenius','euclidean']:
            lock=OUT/method/loss/'lr_selection.json'
            if not lock.exists():
                candidates=[]
                for rate in RATES:
                    validations=[]
                    for seed in SEEDS:
                        dest=folder(method,loss,rate,seed)
                        selection=base(method,loss,seed)/'selection.json' if rate==.01 else dest/method/loss/'selection.json'
                        if not selection.exists():
                            save(OUT/'progress.json',dict(status='training',method=method,loss=loss,lr=rate,seed=seed,seconds=time.time()-start))
                            fit(seed,rate,method,loss,dest,train_only=True)
                            refresh();torch.cuda.empty_cache()
                        validations.append(json.loads(selection.read_text())['validation'])
                    candidates.append(dict(lr=rate,validation=statistics.mean(validations),per_seed=validations))
                chosen=min(candidates,key=lambda r:r['validation'])
                save(lock,dict(**chosen,candidates=candidates,policy='locked before new test evaluation'))
            choice=json.loads(lock.read_text());rate=choice['lr'];rows=[]
            for seed in SEEDS:
                dest=folder(method,loss,rate,seed)
                path=base(method,loss,seed)/'result.json' if rate==.01 else dest/method/loss/'result.json'
                if not path.exists():fit(seed,rate,method,loss,dest,train_only=False)
                rows.append(json.loads(path.read_text()))
            save(OUT/method/loss/'selected_results.json',dict(lr=rate,records=rows))
            refresh();torch.cuda.empty_cache()
    save(OUT/'completion.json',dict(branches=10,new_fits=90,seconds=time.time()-start))
    save(OUT/'progress.json',dict(status='complete'));refresh()

if __name__=='__main__':main()

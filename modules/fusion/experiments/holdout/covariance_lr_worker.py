"""Covariance-history input; test/export after each validation-selected run."""
import csv
import hashlib
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from portfolio_risk.models.covariance_spd import CovarianceSPD

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'outputs/industry_10y_aligned'
OUT=ROOT/'outputs/covariance_spd_100'
METHODS=['cholesky','matrix_exp','spectral_softplus','variance_correlation','residual']

def save(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def scores(p,y,objective):
    error=p-y
    if objective=='euclidean':
        i,j=torch.triu_indices(10,10,device=p.device);error=error[:,i,j]
    return torch.linalg.vector_norm(error.flatten(1),dim=1)

def main(seed, learning_rate, method, objective_name, output, train_only=True):
    global OUT
    OUT=Path(output)
    methods=[method]
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    device=torch.device('cuda')
    scaler=np.load(SOURCE/'cab/scaler.npz')
    target=np.load(SOURCE/'inputs_train.npz')['target']
    unit=float(np.diagonal(target,axis1=-2,axis2=-1).mean())
    def load(split):
        a=np.load(SOURCE/f'inputs_{split}.npz')
        history=np.load(SOURCE/'cab'/f'{split}.npz')
        np.testing.assert_array_equal(history['dates'],a['dates'])
        np.testing.assert_array_equal(history['target'],a['target'])
        sequence=(history['history']-scaler['mean'])/scaler['scale']
        assert sequence.shape[1:]==(100,10,10) and np.isfinite(sequence).all()
        values,vectors=np.linalg.eigh(a['experts'][:,1]/unit)
        assert np.all(values>0)
        root=(vectors*np.sqrt(values)[:,None,:])@vectors.transpose(0,2,1)
        return {k:torch.tensor(v,dtype=torch.float64,device=device) for k,v in dict(sequence=sequence,root=root,target=a['target']).items()},a['dates']
    train,_=load('train');validation,_=load('validation')
    paths=[SOURCE/f'inputs_{s}.npz' for s in ['train','validation','test']]+[SOURCE/'cab/scaler.npz',Path(__file__),ROOT/'src/portfolio_risk/models/covariance_spd.py',ROOT/'src/portfolio_risk/models/direct_spd.py',ROOT/'src/portfolio_risk/models/temporal_gate.py']+[SOURCE/'cab'/f'{s}.npz' for s in ['train','validation','test']]
    protocol=dict(methods=methods,objectives=[objective_name],seed=seed,history=100,features=100,input_shape=[100,10,10],covariance_window=20,input_scaling='per-entry mean/std fitted on training histories only; no clipping',test_policy='select learning rate using three-seed mean validation only; score test only after LR locked',assets=10,hidden=32,heads=4,layers=1,ffn=64,dropout=0,pooling='last token',optimizer='Adam',lr=learning_rate,max_epochs=500,patience=30,epoch0=True,min_delta=1e-12,dtype='float64',batch='full gradient accumulated in128 microbatches',unit=unit,unit_definition='mean training-target diagonal, scalar output unit only; losses remain raw covariance units',residual_base='centered LW60 symmetric square root, fixed data-derived input',spectral_gradient='Loewner divided differences with sigmoid midpoint limit for gaps below1e-7',initialization='heads std0.001; pure heads near unit*I, residual zero head gives exact LW60',hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
    else:save(OUT/'protocol.json',protocol)
    def predict(model,batch):
        with torch.no_grad():return torch.cat([model(batch['sequence'][i:i+128],batch['root'][i:i+128]) for i in range(0,len(batch['target']),128)])
    def evaluate(record):
        if train_only: return record
        test,dates=load('test')
        folder=OUT/record['method']/record['objective']
        model=CovarianceSPD(record['method'],unit).to(device)
        model.load_state_dict(torch.load(folder/'model.pt',map_location=device,weights_only=True));model.eval()
        prediction=predict(model,test)
        torch.linalg.cholesky(prediction)
        record['test']=scores(prediction,test['target'],record['objective']).mean().item()
        record['min_eigenvalue']=torch.linalg.eigvalsh(prediction).min().item()
        np.savez_compressed(folder/'predictions.npz',prediction=prediction.cpu().numpy(),dates=dates)
        save(folder/'result.json',record)
        return record

    records=[];started=time.time()
    for method,objective in itertools.product(methods,[objective_name]):
        folder=OUT/method/objective;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'selection.json').exists():
            record=json.loads((folder/'selection.json').read_text())
            records.append(evaluate(record));save(OUT/'results.json',records)
            continue
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=CovarianceSPD(method,unit).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
        best=float('inf');best_epoch=0;history=[]
        for epoch in range(501):
            if epoch:
                model.train();optimizer.zero_grad();count=len(train['target'])
                for i in range(0,count,128):
                    p=model(train['sequence'][i:i+128],train['root'][i:i+128])
                    loss=scores(p,train['target'][i:i+128],objective).sum()/count
                    if not torch.isfinite(loss):raise FloatingPointError(f'{method}/{objective}: nonfinite training loss')
                    loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                optimizer.step()
            model.eval();prediction=predict(model,validation)
            value=scores(prediction,validation['target'],objective).mean().item()
            assert np.isfinite(value)
            history.append(dict(epoch=epoch,validation=value))
            if value<best-1e-12:
                best=value;best_epoch=epoch
                torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},folder/'model.pt')
            if epoch%10==0:
                progress=dict(method=method,objective=objective,seed=seed,lr=learning_rate,epoch=epoch,best_epoch=best_epoch,validation=value,selected=len(records),total=1,seconds=time.time()-started)
                save(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
            if epoch-best_epoch>=30:break
        with (folder/'training_history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['epoch','validation']);writer.writeheader();writer.writerows(history)
        record=dict(method=method,objective=objective,seed=seed,best_epoch=best_epoch,stop_epoch=epoch,validation=best,parameters=sum(p.numel() for p in model.parameters()))
        save(folder/'selection.json',record)
        records.append(evaluate(record))
        save(OUT/'results.json',records)
    
    save(OUT/'selection.json',records)
    save(OUT/'results.json',records)
    save(OUT/'completion.json',dict(runs=1,phase='validation_only' if train_only else 'test_exported',seconds=time.time()-started))

    print(f'Completed {method}/{objective_name}/seed{seed}/lr{learning_rate}; train_only={train_only}',flush=True)

if __name__=='__main__':raise SystemExit('Use run_covariance_lr_search.py')

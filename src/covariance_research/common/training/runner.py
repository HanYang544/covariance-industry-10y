"""Unified explicit new-run entry; archived workers remain immutable."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from . import EarlyStopping,matrix_scores,seed_everything
from ..preprocessing.frozen import FrozenData
from ...factory import create_model

def run(project,config,name,device='cuda'):
    destination=project/'outputs/new_runs'/name
    if not name or Path(name).name!=name or name in ['.','..']:raise ValueError('Single directory name required')
    if destination.exists():raise FileExistsError('Choose a new run name; frozen runs are never overwritten')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    paper=config['family']=='paper';seed=config.get('seed',20261012)
    lr=config.get('lr',.0001 if paper else .01)
    epochs=config.get('max_epochs',100 if paper else 500)
    objective=config.get('loss','frobenius')
    if objective not in ['frobenius','euclidean']:raise ValueError(objective)
    loader=FrozenData(project/'modules/fusion/outputs/industry_10y_aligned',config,device)
    train,_=loader.load('train');validation,_=loader.load('validation')
    seed_everything(seed);rng=np.random.default_rng(seed)
    model=create_model(config,loader.unit,loader.scaler).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    stopping=EarlyStopping();history=[];micro=16 if paper else 128
    destination.mkdir(parents=True)
    def save(name,value):(destination/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    files=list((project/'src/covariance_research').rglob('*.py'))
    data_root=project/'modules/fusion/outputs/industry_10y_aligned'
    data_files=[p for p in data_root.rglob('*') if p.is_file() and p.suffix in ['.npz','.csv'] and (p.parent==data_root or p.parent==data_root/'cab')]
    save('protocol.json',dict(config=config,seed=seed,lr=lr,max_epochs=epochs,patience=30,epoch0=not paper,source='new modular runner, not a historical rerun claim',hashes={str(p.relative_to(project)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},input_hashes={str(p.relative_to(project)):hashlib.sha256(p.read_bytes()).hexdigest() for p in data_files}))
    def forward(batch,index):
        if config['family']=='fusion':return model(*[batch[k][index] for k in ['sequence','experts','whiteners']])[0]
        if paper:return model(batch['sequence'][index])
        return model(batch['sequence'][index],batch['root'][index])
    def predict(batch):
        with torch.no_grad():return torch.cat([forward(batch,slice(i,i+micro)) for i in range(0,len(batch['target']),micro)])
    count=len(train['target'])
    for epoch in range(1 if paper else 0,epochs+1):
        if epoch:
            model.train()
            groups=[rng.permutation(count)] if paper else [np.arange(count)]
            if paper:groups=[groups[0][i:i+128] for i in range(0,count,128)]
            for group in groups:
                optimizer.zero_grad()
                for i in range(0,len(group),micro):
                    ix=group[i:i+micro]
                    loss=matrix_scores(forward(train,ix),train['target'][ix],objective).sum()/len(group)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                    loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                optimizer.step()
        model.eval()
        # Post-epoch train evaluation makes learning curves comparable to validation.
        train_value=matrix_scores(predict(train),train['target'],objective).mean().item()
        value=matrix_scores(predict(validation),validation['target'],objective).mean().item()
        assert np.isfinite(train_value) and np.isfinite(value)
        improved,stop=stopping.update(epoch,value)
        if improved:torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},destination/'model.pt')
        history.append(dict(epoch=epoch,train=train_value,validation=value))
        with (destination/'training_history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['epoch','train','validation']);writer.writeheader();writer.writerows(history)
        save('progress.json',dict(epoch=epoch,best_epoch=stopping.best_epoch,validation=value))
        if stop:break
    save('selection.json',dict(best_epoch=stopping.best_epoch,validation=stopping.best,seed=seed))
    model.load_state_dict(torch.load(destination/'model.pt',map_location=device,weights_only=True));model.eval()
    test,dates=loader.load('test');prediction=predict(test)
    assert torch.isfinite(prediction).all()
    if not paper:torch.linalg.cholesky(prediction)
    score=matrix_scores(prediction,test['target'],objective).mean().item()
    np.savez_compressed(destination/'predictions.npz',prediction=prediction.cpu().numpy(),dates=dates)
    save('result.json',dict(test=score,validation=stopping.best,best_epoch=stopping.best_epoch,seed=seed,objective=objective))
    print(destination,score)

"""GPU CAB, current suite objectives/early stopping, three predeclared seeds."""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from portfolio_risk.models.cab import CAB

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/cab_holdout'
SEEDS=[20261012,20261013,20261014]


def losses(p,y,o):
    e=p-y
    if o=='frobenius':return torch.linalg.matrix_norm(e,ord='fro')
    if o=='euclidean':
        i,j=torch.triu_indices(10,10,device=e.device)
        return torch.linalg.vector_norm(e[:,i,j],dim=-1)
    raise ValueError(o)


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--benchmark',action='store_true')
    args=parser.parse_args()
    assert torch.cuda.is_available(),'Use the existing GPU runtime'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda';scaler=np.load(OUT/'scaler.npz')
    b={}
    for name in ['train','validation','test']:
        z=np.load(OUT/f'{name}.npz')
        b[name]=(torch.tensor(z['history'],device=device),torch.tensor(z['target'],device=device))
    def create():return CAB(scaler['mean'],scaler['scale']).to(device)
    micro=32
    if args.benchmark:
        model=create();start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        p=model(b['train'][0][:micro]);losses(p,b['train'][1][:micro],'frobenius').mean().backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
        torch.cuda.synchronize()
        print(json.dumps(dict(batch=micro,seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in model.parameters()),
                              peak_mb=torch.cuda.max_memory_allocated()/2**20,loss=float(losses(p.detach(),b['train'][1][:micro],'frobenius').mean()))),flush=True)
        return
    files=[Path(__file__),ROOT/'src/portfolio_risk/models/cab.py',ROOT/'experiments/holdout/prepare_cab.py']
    protocol=dict(model='CAB adapted',paper='https://doi.org/10.1002/for.70110',seeds=SEEDS,objectives=['frobenius','euclidean'],
        input=[100,10,10],conv=dict(channels=[1,1],kernel=5,padding=2,stride=1,activation='none'),
        bilstm=dict(input_size=100,hidden_per_direction=128,layers=7,dropout=0),attention=dict(embed_dim=256,heads=16,dropout=0),
        output='time mean pooling -> Linear(256,100) -> symmetric -> inverse scaler -> zero-clipped eigenvalues -> .8 DL + .2 last input',
        optimizer='Adam',learning_rate=.01,max_epochs=500,patience=30,min_delta=1e-12,include_epoch0=True,
        batch='full train 651 through weighted gradient accumulation; one optimizer step per epoch',microbatch=micro,
        dtype='float32 neural backbone; float64 normalization, PSD projection, labels and losses; no AMP or TF32',
        paper_training_difference='Paper: Adam 1e-4, batch128, 100epochs. Here retain suite Adam .01/full-batch/max500/patience30 for protocol-matched architecture adaptation.',
        ambiguity_choices='train-history per-entry scaling; PyTorch default initialization/biases; no unspecified nonlinearities/dropout; PSD and blend inside training graph',
        evaluation='each objective and seed independent validation early stop; all six checkpoints locked before test evaluation; fixed parameters throughout test',
        runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name()),
        data_protocol_sha256=hashlib.sha256((OUT/'data_protocol.json').read_bytes()).hexdigest(),
        code_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Protocol changed; cannot reuse fits'
    else:path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    def score(model,split,o):
        model.eval();x,y=b[split];total=0.
        with torch.no_grad():
            for start in range(0,len(x),micro):total+=float(losses(model(x[start:start+micro]),y[start:start+micro],o).sum())
        return total/len(x)
    records=[]
    for o in protocol['objectives']:
        for seed in SEEDS:
            folder=OUT/o/str(seed);folder.mkdir(parents=True,exist_ok=True)
            selection=folder/'selection.json'
            if selection.exists():records.append(json.loads(selection.read_text()));continue
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=create();optimizer=torch.optim.Adam(model.parameters(),lr=.01)
            history=[];best=float('inf');best_epoch=0;started=time.perf_counter()
            for epoch in range(501):
                tick=time.perf_counter();train_before=0.;grad=0.
                if epoch:
                    model.train();optimizer.zero_grad();x,y=b['train']
                    for start in range(0,len(x),micro):
                        value=losses(model(x[start:start+micro]),y[start:start+micro],o).sum()/len(x)
                        train_before+=float(value.detach());value.backward()
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                    grad=float(sum(p.grad.double().square().sum() for p in model.parameters()).sqrt())
                    optimizer.step()
                else:train_before=score(model,'train',o)
                val=score(model,'validation',o)
                assert np.isfinite([train_before,val,grad]).all()
                history.append(dict(epoch=epoch,train_loss_before_step=train_before,validation_loss=val,gradient_norm=grad,seconds=time.perf_counter()-tick))
                if val<best-1e-12:
                    best=val;best_epoch=epoch
                    torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},folder/'model.pt')
                write_csv(folder/'training_history.csv',history)
                if epoch%5==0:print(f'{o} seed={seed} epoch={epoch} val={val:.9f} best={best_epoch} seconds={history[-1]["seconds"]:.2f}',flush=True)
                if epoch-best_epoch>=30:break
            r=dict(objective=o,seed=seed,best_epoch=best_epoch,stopped_epoch=epoch,validation_score=best,
                   parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-started)
            selection.write_text(json.dumps(r,indent=2),encoding='utf-8');records.append(r)
            print(f'Locked {o} seed={seed}: best/stop {best_epoch}/{epoch}',flush=True)
    (OUT/'selection.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('All six checkpoints locked; evaluating test.',flush=True)
    for r in records:
        folder=OUT/r['objective']/str(r['seed']);model=create()
        model.load_state_dict(torch.load(folder/'model.pt',map_location=device,weights_only=True));model.eval()
        with torch.no_grad():p,d=model(b['test'][0],True)
        assert torch.isfinite(p).all();torch.linalg.cholesky(p)
        values=losses(p,b['test'][1],r['objective']).cpu().numpy()
        r['test_score']=float(values.mean())
        dates=np.load(OUT/'test.npz')['dates']
        np.savez_compressed(folder/'test_predictions.npz',prediction=p.cpu().numpy(),target=b['test'][1].cpu().numpy(),dates=dates,
                            raw=d['raw'].cpu().numpy(),psd=d['psd'].cpu().numpy(),last_history=d['last_history'].cpu().numpy())
        write_csv(folder/'test_by_origin.csv',[dict(date=date,loss=float(v)) for date,v in zip(dates,values)])
        (folder/'result.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    write_csv(OUT/'per_seed.csv',records)
    summary=[]
    for o in protocol['objectives']:
        scores=np.array([r['test_score'] for r in records if r['objective']==o])
        assert len(scores)==3
        summary.append(dict(method='CAB-adapted',objective=o,n=3,mean=float(scores.mean()),std=float(scores.std(ddof=1))))
    write_csv(OUT/'summary.csv',summary)
    (OUT/'completion.json').write_text(json.dumps(dict(fits=6,seeds=SEEDS,summary=summary),indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()

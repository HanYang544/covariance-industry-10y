"""CAB with the ETF architecture/optimizer on aligned ten-year industry inputs."""
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import run_paper_etf as paper

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/industry_10y_aligned/cab'

def dump(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')

def main():
    assert torch.cuda.is_available();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    scaler=dict(np.load(OUT/'scaler.npz'));data={}
    for s in ['train','validation','test']:
        a=dict(np.load(OUT/f'{s}.npz'));data[s]=(torch.tensor(a['history'],device='cuda'),torch.tensor(a['target'],device='cuda'))
    create=lambda:paper.PaperCAB(scaler['mean'],scaler['scale']).cuda()
    protocol=dict(assets=10,seeds=paper.SEEDS,objectives=['frobenius','euclidean'],optimizer='Adam',lr=1e-4,batch=128,microbatch=16,max_epochs=100,patience=30,include_epoch0=False,
        architecture='Same PaperCAB as ETF, only asset dimension14->10',test='fixed holdout, all6 selections locked before scoring',
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(paper.__file__)]},
        inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.npz')})
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
    else:dump(OUT/'protocol.json',protocol)
    def predict(model,split):
        model.eval();x,_=data[split]
        with torch.no_grad():return torch.cat([model(x[s:s+16]) for s in range(0,len(x),16)])
    for o in protocol['objectives']:
        for seed in paper.SEEDS:
            folder=OUT/o/str(seed);folder.mkdir(parents=True,exist_ok=True)
            if (folder/'selection.json').exists():continue
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
            model=create();opt=torch.optim.Adam(model.parameters(),lr=1e-4)
            history=[];best=float('inf');best_epoch=0;x,y=data['train'];t=time.time()
            for epoch in range(1,101):
                model.train();order=rng.permutation(len(x));total=0.
                for start in range(0,len(x),128):
                    batch=order[start:start+128];opt.zero_grad()
                    for s in range(0,len(batch),16):
                        ix=batch[s:s+16];loss=paper.losses(model(x[ix]),y[ix],o).sum()/len(batch)
                        assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())*len(batch)
                    assert all(torch.isfinite(p.grad).all() for p in model.parameters());opt.step()
                val=float(paper.losses(predict(model,'validation'),data['validation'][1],o).mean())
                history.append(dict(epoch=epoch,train=total/len(x),validation=val,seconds=time.time()-t))
                if val<best-1e-12:best=val;best_epoch=epoch;torch.save(model.state_dict(),folder/'model.pt')
                with (folder/'training_history.csv').open('w',newline='',encoding='utf-8') as f:
                    w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
                dump(OUT/'progress.json',dict(objective=o,seed=seed,**history[-1]))
                if epoch==1 or epoch%10==0:print(o,seed,history[-1],flush=True)
                if epoch-best_epoch>=30:break
            dump(folder/'selection.json',dict(objective=o,seed=seed,best_epoch=best_epoch,stop_epoch=epoch,validation=best))
            del model,opt;torch.cuda.empty_cache()
    records=[]
    for o in protocol['objectives']:
        for seed in paper.SEEDS:
            folder=OUT/o/str(seed);r=json.loads((folder/'selection.json').read_text())
            model=create();model.load_state_dict(torch.load(folder/'model.pt',weights_only=True));p=predict(model,'test')
            assert torch.isfinite(p).all() and torch.linalg.eigvalsh(p).min()>-1e-10
            r['test']=float(paper.losses(p,data['test'][1],o).mean())
            np.savez_compressed(folder/'predictions.npz',prediction=p.cpu().numpy(),target=data['test'][1].cpu().numpy())
            dump(folder/'result.json',r);records.append(r)
    dump(OUT/'results.json',records);print('All6 aligned CAB fits complete',flush=True)

if __name__=='__main__':main()

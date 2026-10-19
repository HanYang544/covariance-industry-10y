"""Recompute all retained sequence scores and validate encoder gradients."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from portfolio_risk.models.sequence_gates import SequenceGate

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs'
torch.set_num_threads(1)
target=np.load(OUT/'industry_10y_aligned/inputs_test.npz')
assert not (OUT/'industry_10y_aligned/runs').exists()
count=0
for name in ['temporal_gate_100','sequence_mlp_100','sequence_cnn_100']:
    records=json.loads((OUT/name/'results.json').read_text())
    assert len(records)==12
    zeros=0
    for r in records:
        folder=OUT/name/r['fusion']/r['objective']/str(r['seed'])
        a=np.load(folder/'predictions.npz')
        np.testing.assert_array_equal(a['dates'],target['dates'])
        np.testing.assert_allclose(a['weights'].sum(-1),1,atol=1e-12)
        assert (a['weights']>0).all()
        np.linalg.cholesky(a['prediction'])
        error=a['prediction']-target['target']
        if r['objective']=='euclidean':
            i,j=np.triu_indices(10);error=error[:,i,j]
        loss=np.linalg.norm(error.reshape(len(error),-1),axis=1).mean()
        np.testing.assert_allclose(loss,r['test'],rtol=1e-10)
        history=list(csv.DictReader((folder/'training_history.csv').open()))
        best=min(history,key=lambda h:float(h['validation']))
        assert int(best['epoch'])==r['best_epoch']
        assert r['stop_epoch']-r['best_epoch']>=30 or r['stop_epoch']==500
        zeros+=r['best_epoch']==0
        count+=1
    print(name,'verified 12 runs; epoch0 selected',zeros)
for architecture in ['mlp','cnn']:
    for fusion in ['precision','covariance']:
        model=SequenceGate(fusion,architecture)
        x=torch.randn(2,100,6,dtype=torch.float64)
        a=torch.randn(2,10,10,10,dtype=torch.float64)
        c=a@a.transpose(-1,-2)+torch.eye(10)
        w=torch.linalg.cholesky(torch.linalg.inv(c))
        p,weights=model(x,c,w)
        p.square().mean().backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
        changed=x.clone();changed[:,:50]+=3
        assert not torch.allclose(weights,model(changed,c,w)[1])
        print(architecture,fusion,'parameters',sum(v.numel() for v in model.parameters()))
print('Verified',count,'retained sequence results and all four new encoder/fusion gradient paths.')

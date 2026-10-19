import numpy as np
import torch
from pathlib import Path
from portfolio_risk.models.covariance_spd import CovarianceSPD
torch.set_num_threads(1)
root=Path(__file__).resolve().parents[2]/'outputs/industry_10y_aligned'
timeline=np.load(root/'timeline.npz')
for split in ['train','validation','test']:
    a=np.load(root/'cab'/f'{split}.npz')
    other=np.load(root/f'inputs_{split}.npz')
    np.testing.assert_array_equal(a['target'],other['target'])
    np.testing.assert_array_equal(a['dates'],other['dates'])
    for j in [0,len(a['dates'])-1]:
        t=int(a['origins'][j])
        for step in [0,99]:
            end=t-99+step
            c=np.cov(timeline['returns'][end-19:end+1],rowvar=False,ddof=1)
            np.testing.assert_allclose(c,a['history'][j,step],rtol=1e-12,atol=1e-16)
        np.testing.assert_allclose(np.cov(timeline['returns'][t+1:t+21],rowvar=False,ddof=1),a['target'][j],rtol=1e-12,atol=1e-16)
for method in ['cholesky','matrix_exp','spectral_softplus','variance_correlation','residual']:
    model=CovarianceSPD(method,.0001)
    x=torch.randn(2,100,10,10,dtype=torch.float64)
    base=torch.eye(10,dtype=torch.float64).expand(2,10,10)
    prediction=model(x,base)
    torch.linalg.cholesky(prediction)
    prediction.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
print('All split labels/dates aligned; sampled rolling20 histories causal; five heads SPD and finite gradients.')

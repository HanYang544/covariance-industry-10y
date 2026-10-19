import torch
from portfolio_risk.models.direct_spd import DirectSPD,SpectralSoftplus
torch.set_num_threads(1)
torch.manual_seed(23)
for repeated in [False,True]:
    a=torch.eye(3,dtype=torch.float64) if repeated else torch.randn(3,3,dtype=torch.float64)
    a.requires_grad_()
    assert torch.autograd.gradcheck(lambda z:SpectralSoftplus.apply((z+z.T)/2),(a,),atol=1e-5,rtol=1e-4)
for method in ['cholesky','matrix_exp','spectral_softplus','variance_correlation','residual']:
    m=DirectSPD(method,.0001)
    x=torch.randn(2,100,6,dtype=torch.float64)
    root=torch.diag(torch.linspace(.5,2,10,dtype=torch.float64)).expand(2,10,10)
    y=m(x,root)
    torch.linalg.cholesky(y)
    assert y.shape==(2,10,10)
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    if method=='residual':torch.testing.assert_close(y,.0001*(root@root.transpose(-1,-2)))
    print(method,'SPD and finite gradients OK')
print('Spectral matrix-function gradcheck passed, including repeated eigenvalues.')

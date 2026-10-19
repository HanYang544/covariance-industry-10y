import torch
from portfolio_risk.models.residual_architectures import build_model,STRUCTURES
torch.set_num_threads(1);torch.manual_seed(7)
x=torch.randn(2,100,10,10,dtype=torch.float64)
root=torch.diag(torch.linspace(.5,2,10,dtype=torch.float64)).expand(2,10,10)
for name in STRUCTURES:
    m=build_model(name,.0001)
    p=m(x,root);torch.linalg.cholesky(p)
    torch.testing.assert_close(p,.0001*(root@root.transpose(-1,-2)))
    p.square().mean().backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters())
    with torch.no_grad():m.head.weight.normal_(0,.001)
    m.zero_grad();m(x,root).square().mean().backward()
    assert sum(v.grad.abs().sum().item() for n,v in m.named_parameters() if not n.startswith('head'))>0
    print(name,'passed; parameters',sum(v.numel() for v in m.parameters()))

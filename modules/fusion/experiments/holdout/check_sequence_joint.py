"""Check 42 architecture/refinement/fusion combinations before expensive fits."""
import itertools
import torch
from portfolio_risk.models.sequence_joint import SequenceJoint
torch.set_num_threads(1)
torch.manual_seed(19)
x=torch.randn(2,100,6,dtype=torch.float64)
a=torch.randn(2,10,10,10,dtype=torch.float64)
c=a@a.transpose(-1,-2)+torch.eye(10)
w=torch.linalg.cholesky(torch.linalg.inv(c))
variants=[(v,q) for v in ['spectral','learned'] for q in [1,3,5]]+[('idea2',10)]
for architecture,(variant,q),fusion in itertools.product(['mlp','cnn','transformer'],variants,['precision','covariance']):
    m=SequenceJoint(fusion,architecture,variant,q)
    prediction,weights=m(x,c,w)
    base=m.core.fusion((c,w),x)
    torch.testing.assert_close(prediction,base,rtol=1e-9,atol=1e-9)
    torch.linalg.cholesky(prediction)
    prediction.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert sum(p.grad.abs().sum().item() for p in m.core.fusion.parameters())>0
    assert m.core.variance.weight.grad.abs().sum()>0
    # Zero output heads intentionally block encoder gradients at initialization.
    with torch.no_grad():m.core.variance.weight.normal_(0,.01)
    m.zero_grad();m(x,c,w)[0].square().mean().backward()
    assert sum(p.grad.abs().sum().item() for p in m.core.encoder.parameters())>0
print('42 combinations passed: identity initialization, SPD, both branch gradients.')

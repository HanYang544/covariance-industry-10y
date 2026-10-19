"""Dimension-general migration; legacy 10-asset code/artifacts remain immutable."""
import torch
from torch import nn
from . import idea1, idea2
from covariance_research.fusion.refinement._initialization_precision import PrecisionGate
from covariance_research.fusion.refinement._initialization_convex import ConvexGate


class JointRefinementND(nn.Module):
    def __init__(self,variant,q=3,fusion='precision',dimension=14):
        super().__init__()
        if variant not in ['spectral','learned','idea2']:raise ValueError(variant)
        if not 1<=q<=dimension or (variant=='idea2' and q!=dimension):raise ValueError(q)
        if fusion not in ['precision','covariance']:raise ValueError(fusion)
        self.variant,self.q,self.dimension=variant,q,dimension
        self.fusion=(PrecisionGate if fusion=='precision' else ConvexGate)(10,False)
        self.encoder=nn.Sequential(nn.Linear(6,16),nn.ReLU()).double()
        self.variance=nn.Linear(16,q).double()
        self.change=nn.Linear(16,q*(q-1)//2).double() if q>1 else None
        for head in [h for h in [self.variance,self.change] if h is not None]:
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        if variant=='learned':self.directions=nn.Linear(16,dimension*q).double()

    def forward(self,matrices,state,diagnostics=False):
        base,weights=self.fusion(matrices,state,return_weights=True)
        base=(base+base.transpose(-1,-2))/2
        h=self.encoder(state);q=self.q
        multiplier=torch.exp(2*torch.tanh(self.variance(h)))
        t=torch.eye(q,dtype=state.dtype,device=state.device).expand(len(state),q,q).clone()
        if self.change is not None:
            rows,cols=torch.tril_indices(q,q,offset=-1,device=state.device)
            t[:,rows,cols]=.5*torch.tanh(self.change(h))
        gram=t@t.transpose(-1,-2);std=gram.diagonal(dim1=-2,dim2=-1).sqrt()
        correlation=gram/(std.unsqueeze(-1)*std.unsqueeze(-2))
        detail=dict(base=base,weights=weights)
        if self.variant in ['spectral','idea2']:
            values,u=torch.linalg.eigh(base);values=values.flip(-1);u=u.flip(-1)
            pivot=u.abs().argmax(dim=-2,keepdim=True)
            u=u*torch.where(torch.gather(u,-2,pivot)<0,-1.,1.)
            if self.variant=='idea2':
                prediction=idea2.rotate(base,values,u,multiplier,self.change(h),rows,cols,detail)
            else:
                prediction=idea1.spectral(values,u,multiplier,correlation,q,detail)
        else:
            raw=self.directions(h).reshape(-1,self.dimension,q)
            prediction=idea1.learned(base,raw,multiplier,correlation,q,detail)
        prediction=(prediction+prediction.transpose(-1,-2))/2
        return (prediction,detail) if diagnostics else prediction

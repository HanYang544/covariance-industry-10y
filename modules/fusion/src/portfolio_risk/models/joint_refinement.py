"""Fully differentiable uncalibrated ten-expert fusion and rank-three refinement."""
import torch
from torch import nn
from .precision_gate import PrecisionGate
from .convex_gate import ConvexGate

class JointRefinement(nn.Module):
    def __init__(self,variant,q=3,fusion='precision'):
        super().__init__()
        if variant not in ['spectral','learned','idea2']:raise ValueError(variant)
        self.variant=variant
        if q not in ([10] if variant=='idea2' else [1,3,5]):raise ValueError(q)
        self.q=q
        if fusion not in ['precision','covariance']:raise ValueError(fusion)
        self.fusion=(PrecisionGate if fusion=='precision' else ConvexGate)(10,False)
        self.encoder=nn.Sequential(nn.Linear(6,16),nn.ReLU()).double()
        self.variance=nn.Linear(16,q).double()
        self.change=nn.Linear(16,q*(q-1)//2).double() if q>1 else None
        for head in [h for h in [self.variance,self.change] if h is not None]:
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        if variant=='learned':self.directions=nn.Linear(16,10*q).double()

    def forward(self,whiteners,state,diagnostics=False):
        base,weights=self.fusion(whiteners,state,return_weights=True)
        base=(base+base.transpose(-1,-2))/2
        h=self.encoder(state)
        multiplier=torch.exp(2*torch.tanh(self.variance(h)))
        q=self.q
        t=torch.eye(q,dtype=state.dtype,device=state.device).expand(len(state),q,q).clone()
        if self.change is not None:
            rows,cols=torch.tril_indices(q,q,offset=-1,device=state.device)
            t[:,rows,cols]=.5*torch.tanh(self.change(h))
        gram=t@t.transpose(-1,-2)
        std=gram.diagonal(dim1=-2,dim2=-1).sqrt()
        correlation=gram/(std.unsqueeze(-1)*std.unsqueeze(-2))
        detail=dict(base=base,weights=weights)
        if self.variant in ['spectral','idea2']:
            values,u=torch.linalg.eigh(base)
            values=values.flip(-1);u=u.flip(-1)
            pivot=u.abs().argmax(dim=-2,keepdim=True)
            sign=torch.where(torch.gather(u,-2,pivot)<0,-1.,1.)
            u=u*sign
            if self.variant=='idea2':
                normalized=values/values.mean(-1,keepdim=True)
                gap=(normalized[:,rows]-normalized[:,cols]).abs()
                angles=gap*(.1*torch.tanh(self.change(h)))/(gap.square()+.01)
                lower=torch.zeros_like(base)
                lower[:,rows,cols]=angles
                rotation=torch.matrix_exp(lower-lower.transpose(-1,-2))
                updated=u@rotation
                prediction=(updated*(values*multiplier).unsqueeze(-2))@updated.transpose(-1,-2)
                detail['relative_boundary_gap']=((values[:,:-1]-values[:,1:])/values.mean(-1,keepdim=True)).min(-1).values
                detail['rotation']=rotation
                prediction=(prediction+prediction.transpose(-1,-2))/2
                return (prediction,detail) if diagnostics else prediction
            amplitude=(values[:,:q]*multiplier).sqrt()
            core=correlation*amplitude.unsqueeze(-1)*amplitude.unsqueeze(-2)
            selected=u[:,:,:q];rest=u[:,:,q:]
            prediction=selected@core@selected.transpose(-1,-2)+(rest*values[:,None,q:])@rest.transpose(-1,-2)
            detail['relative_boundary_gap']=(values[:,q-1]-values[:,q])/values.mean(-1)
        else:
            raw=self.directions(h).reshape(-1,10,q)
            v,r=torch.linalg.qr(raw,mode='reduced')
            sign=torch.where(r.diagonal(dim1=-2,dim2=-1)<0,-1.,1.)
            v=v*sign.unsqueeze(-2)
            amplitude=multiplier.sqrt()
            core=correlation*amplitude.unsqueeze(-1)*amplitude.unsqueeze(-2)
            identity=torch.eye(10,dtype=state.dtype,device=state.device)
            middle=identity+v@(core-torch.eye(q,dtype=state.dtype,device=state.device))@v.transpose(-1,-2)
            factor=torch.linalg.cholesky(base)
            prediction=factor@middle@factor.transpose(-1,-2)
            detail['qr_min_abs_diagonal']=r.diagonal(dim1=-2,dim2=-1).abs().min(-1).values
        prediction=(prediction+prediction.transpose(-1,-2))/2
        return (prediction,detail) if diagnostics else prediction

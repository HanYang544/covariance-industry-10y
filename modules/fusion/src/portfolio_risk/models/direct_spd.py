"""Five smooth SPD heads for the same historical-feature Transformer."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .temporal_gate import TemporalGate


class SpectralSoftplus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix):
        values, vectors = torch.linalg.eigh(matrix)
        ctx.save_for_backward(values, vectors)
        return (vectors * F.softplus(values).unsqueeze(-2)) @ vectors.transpose(-1, -2)

    @staticmethod
    def backward(ctx, gradient):
        values, vectors = ctx.saved_tensors
        left, right = values.unsqueeze(-1), values.unsqueeze(-2)
        gap = left-right
        close = gap.abs() < 1e-7
        quotient = (F.softplus(left)-F.softplus(right))/torch.where(close, torch.ones_like(gap), gap)
        loewner = torch.where(close, torch.sigmoid((left+right)/2), quotient)
        symmetric = (gradient+gradient.transpose(-1,-2))/2
        rotated = vectors.transpose(-1,-2) @ symmetric @ vectors
        return vectors @ (loewner*rotated) @ vectors.transpose(-1,-2)


class DirectSPD(nn.Module):
    def __init__(self, method, unit):
        super().__init__()
        original = TemporalGate()
        self.input, self.encoder = original.input, original.encoder
        self.register_buffer('position', original.position)
        self.register_buffer('unit', torch.tensor(float(unit), dtype=torch.float64))
        self.method = method
        self.head = nn.Linear(32,55).double()
        self.register_buffer('rows',torch.tril_indices(10,10)[0])
        self.register_buffer('cols',torch.tril_indices(10,10)[1])
        nn.init.normal_(self.head.weight,std=.001);nn.init.zeros_(self.head.bias)
        inverse_softplus = math.log(math.expm1(1.))
        with torch.no_grad():
            if method in ['cholesky','spectral_softplus']:
                self.head.bias[self.rows==self.cols]=inverse_softplus
            if method=='residual':nn.init.zeros_(self.head.weight)
        if method=='variance_correlation':
            self.volatility=nn.Linear(32,10).double()
            nn.init.normal_(self.volatility.weight,std=.001)
            nn.init.constant_(self.volatility.bias,inverse_softplus)
        self.double()

    def forward(self, sequence, base_root=None):
        hidden=self.encoder(self.input(sequence)+self.position)[:,-1]
        parameters=self.head(hidden)
        lower=torch.zeros(len(sequence),10,10,dtype=sequence.dtype,device=sequence.device)
        lower[:,self.rows,self.cols]=parameters
        diagonal=lower.diagonal(dim1=-2,dim2=-1)
        identity=torch.eye(10,dtype=sequence.dtype,device=sequence.device)
        symmetric=lower+lower.transpose(-1,-2)-torch.diag_embed(diagonal)
        if self.method=='cholesky':
            factor=lower-torch.diag_embed(diagonal)+torch.diag_embed(F.softplus(diagonal)+1e-6)
            result=factor@factor.transpose(-1,-2)
        elif self.method=='matrix_exp':result=torch.matrix_exp(symmetric)
        elif self.method=='spectral_softplus':result=SpectralSoftplus.apply(symmetric)+1e-6*identity
        elif self.method=='variance_correlation':
            p=torch.matrix_exp(symmetric)
            sd=p.diagonal(dim1=-2,dim2=-1).sqrt()
            correlation=p/(sd.unsqueeze(-1)*sd.unsqueeze(-2))
            volatility=F.softplus(self.volatility(hidden))+1e-6
            result=correlation*volatility.unsqueeze(-1)*volatility.unsqueeze(-2)
        elif self.method=='residual':
            assert base_root is not None
            result=base_root@torch.matrix_exp(symmetric)@base_root.transpose(-1,-2)
        else:raise ValueError(self.method)
        return self.unit*(result+result.transpose(-1,-2))/2

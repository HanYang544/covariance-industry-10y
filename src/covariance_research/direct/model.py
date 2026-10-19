"""Five smooth SPD heads for the same historical-feature Transformer."""
import math
import torch
from torch import nn
from covariance_research.direct.output_heads import transform
from covariance_research.direct.output_heads.spectral_softplus import SpectralSoftplus
from covariance_research.fusion.gating.transformer import TemporalGate


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
        volatility = self.volatility(hidden) if self.method=='variance_correlation' else None
        result=transform(self.method,lower,volatility,base_root)
        return self.unit*(result+result.transpose(-1,-2))/2

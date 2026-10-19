"""Architecture-only exploration with common LW60 exponential residual head."""
import torch
from torch import nn
from torch.nn import functional as F
from .covariance_spd import CovarianceSPD
from .temporal_gate import TemporalGate

STRUCTURES={
 'transformer':'original Linear100-32; sinusoidal PE; 1 layer 4 heads FFN64; last token',
 'mlp':'Flatten10000; Linear10000-64 ReLU; Linear64-32 ReLU',
 'cnn':'Conv1d100-32 k5 pad2 ReLU; Conv1d32-32 k5 pad2 ReLU; global temporal mean',
 'tcn':'Linear100-32; six residual causal Conv1d32-32 k3 dilation1,2,4,8,16,32 ReLU; last token',
 'gru':'single layer GRU input100 hidden32; last hidden',
 'patch_transformer':'nonoverlap patches5 ->20 tokens500; Linear500-32; sinusoidal PE;1 layer4 heads FFN64;last token'}

class ResidualArchitecture(nn.Module):
    def __init__(self,architecture,unit):
        super().__init__();self.architecture=architecture
        self.register_buffer('unit',torch.tensor(unit,dtype=torch.float64))
        if architecture=='mlp':self.encoder=nn.Sequential(nn.Flatten(1),nn.Linear(10000,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        elif architecture=='cnn':self.encoder=nn.Sequential(nn.Conv1d(100,32,5,padding=2),nn.ReLU(),nn.Conv1d(32,32,5,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        elif architecture=='gru':self.encoder=nn.GRU(100,32,batch_first=True)
        elif architecture=='tcn':
            self.input=nn.Linear(100,32)
            self.blocks=nn.ModuleList([nn.Conv1d(32,32,3,dilation=d) for d in [1,2,4,8,16,32]])
        elif architecture=='patch_transformer':
            template=TemporalGate();self.encoder=template.encoder
            self.input=nn.Linear(500,32);self.register_buffer('position',template.position[:20])
        else:raise ValueError(architecture)
        self.head=nn.Linear(32,55);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        indices=torch.tril_indices(10,10);self.register_buffer('rows',indices[0]);self.register_buffer('cols',indices[1])
        self.double()

    def forward(self,sequence,base_root):
        x=sequence.flatten(2)
        if self.architecture=='mlp':h=self.encoder(x)
        elif self.architecture=='cnn':h=self.encoder(x.transpose(1,2)).squeeze(-1)
        elif self.architecture=='gru':h=self.encoder(x)[1][-1]
        elif self.architecture=='patch_transformer':h=self.encoder(self.input(x.reshape(len(x),20,500))+self.position)[:,-1]
        else:
            z=self.input(x).transpose(1,2)
            for d,conv in zip([1,2,4,8,16,32],self.blocks):z=F.relu(z+conv(F.pad(z,(2*d,0))))
            h=z[:,:,-1]
        lower=torch.zeros(len(x),10,10,dtype=x.dtype,device=x.device)
        lower[:,self.rows,self.cols]=self.head(h)
        symmetric=lower+lower.transpose(-1,-2)-torch.diag_embed(lower.diagonal(dim1=-2,dim2=-1))
        prediction=self.unit*(base_root@torch.matrix_exp(symmetric)@base_root.transpose(-1,-2))
        return (prediction+prediction.transpose(-1,-2))/2

def build_model(architecture,unit):
    return CovarianceSPD('residual',unit) if architecture=='transformer' else ResidualArchitecture(architecture,unit)

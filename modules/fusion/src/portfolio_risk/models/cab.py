"""Reis et al. CAB architecture adapted to the suite's 10-asset risk target.

Not an author-code reproduction: scaling scope and defaults are disclosed in
docs/cab.md. No extra Conv activation, dropout, residual or Cholesky decoder.
"""
import numpy as np
import torch
from torch import nn


def rolling_risk(returns, window=20):
    """Causal uncentered daily second moments, deliberately matching the target."""
    returns=np.asarray(returns,dtype=np.float64)
    if returns.ndim!=2 or window<1 or not np.isfinite(returns).all():
        raise ValueError('Finite time-by-asset returns required')
    out=np.full((len(returns),returns.shape[1],returns.shape[1]),np.nan)
    for t in range(window-1,len(returns)):
        x=returns[t-window+1:t+1]
        out[t]=x.T@x/window
    return out


def history_sequences(returns,origins,length=100,window=20):
    origins=np.asarray(origins,dtype=int)
    if length<1 or origins.min()<length+window-2 or origins.max()>=len(returns):
        raise ValueError('History must contain length+window-1 available days')
    risk=rolling_risk(returns,window)
    return np.stack([risk[t-length+1:t+1] for t in origins])


class CAB(nn.Module):
    def __init__(self,mean,scale):
        super().__init__()
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float64).clone())
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float64).clone())
        if self.mean.shape!=(10,10) or self.scale.shape!=(10,10) or not (self.scale>0).all():
            raise ValueError('10x10 entrywise training scaler required')
        self.conv=nn.Conv3d(1,1,kernel_size=5,padding=2,stride=1)
        self.lstm=nn.LSTM(100,128,num_layers=7,bidirectional=True,batch_first=True,dropout=0)
        self.attention=nn.MultiheadAttention(256,16,dropout=0,batch_first=True)
        self.readout=nn.Linear(256,100)

    def forward(self,history,details=False):
        if history.ndim!=4 or history.shape[1:]!=(100,10,10):
            raise ValueError('Expected [batch,100,10,10]')
        normalized=((history.double()-self.mean)/self.scale).to(self.conv.weight.dtype)
        conv=self.conv(normalized.unsqueeze(1)).squeeze(1).flatten(2)
        temporal,_=self.lstm(conv)
        attended,_=self.attention(temporal,temporal,temporal,need_weights=False)
        normalized_output=self.readout(attended.mean(dim=1)).reshape(-1,10,10).double()
        normalized_output=(normalized_output+normalized_output.transpose(-1,-2))/2
        raw=normalized_output*self.scale+self.mean
        raw=(raw+raw.transpose(-1,-2))/2
        # Float64 small-matrix spectral projection; neural backbone stays float32.
        values,u=torch.linalg.eigh(raw)
        psd=(u*values.clamp_min(0).unsqueeze(-2))@u.transpose(-1,-2)
        prediction=.8*psd+.2*history[:,-1].double()
        prediction=(prediction+prediction.transpose(-1,-2))/2
        if details:
            return prediction,dict(raw=raw,psd=psd,last_history=history[:,-1].double(),
                                   negative_eigenvalues=(values<0).sum(-1))
        return prediction

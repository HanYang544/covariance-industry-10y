"""Paper CAB architecture adaptation; not author-code replication."""
import torch
from torch import nn

class PaperCAB(nn.Module):
    def __init__(self,mean,scale):
        super().__init__();self.n=len(mean)
        self.register_buffer('mean',torch.tensor(mean,dtype=torch.float64))
        self.register_buffer('scale',torch.tensor(scale,dtype=torch.float64))
        self.conv=nn.Conv3d(1,1,5,padding=2)
        self.lstm=nn.LSTM(self.n**2,128,7,batch_first=True,bidirectional=True)
        self.attention=nn.MultiheadAttention(256,16,batch_first=True)
        self.readout=nn.Linear(256,self.n**2)
    def forward(self,x):
        z=((x-self.mean)/self.scale).float()
        z=self.conv(z[:,None]).squeeze(1).flatten(2)
        z,_=self.lstm(z);z,_=self.attention(z,z,z,need_weights=False)
        z=self.readout(z.mean(1)).reshape(-1,self.n,self.n).double()
        z=(z+z.transpose(-1,-2))/2*self.scale+self.mean
        z=(z+z.transpose(-1,-2))/2
        v,u=torch.linalg.eigh(z)
        return .8*((u*v.clamp_min(0).unsqueeze(-2))@u.transpose(-1,-2))+.2*x[:,-1]

import torch
from torch.nn import functional as F

def transform(symmetric,volatility_logits):
    p=torch.matrix_exp(symmetric)
    sd=p.diagonal(dim1=-2,dim2=-1).sqrt()
    correlation=p/(sd.unsqueeze(-1)*sd.unsqueeze(-2))
    volatility=F.softplus(volatility_logits)+1e-6
    return correlation*volatility.unsqueeze(-1)*volatility.unsqueeze(-2)

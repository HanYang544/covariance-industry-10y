import torch
from torch.nn import functional as F

def transform(lower):
    diagonal=lower.diagonal(dim1=-2,dim2=-1)
    factor=lower-torch.diag_embed(diagonal)+torch.diag_embed(F.softplus(diagonal)+1e-6)
    return factor@factor.transpose(-1,-2)

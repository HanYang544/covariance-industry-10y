import torch
from torch.nn import functional as F

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

def transform(symmetric):
    identity=torch.eye(symmetric.shape[-1],dtype=symmetric.dtype,device=symmetric.device)
    return SpectralSoftplus.apply(symmetric)+1e-6*identity


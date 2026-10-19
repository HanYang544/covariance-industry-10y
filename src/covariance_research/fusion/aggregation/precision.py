"""Fuse Cholesky factors of inverse covariances, not precision matrices."""
import torch

def precision_factor_fusion(weights, whiteners):
    factor = torch.einsum('bk,bkij->bij', weights, whiteners)
    return torch.linalg.inv(factor @ factor.transpose(-1, -2))

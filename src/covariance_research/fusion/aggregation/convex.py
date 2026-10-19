"""Arithmetic convex combination of covariance experts."""
import torch

def convex_fusion(weights, experts):
    return torch.einsum('bk,bkij->bij', weights, experts)

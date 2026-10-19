"""Unweighted upper-triangle Euclidean distance, including the diagonal."""
import torch

def euclidean_loss(prediction, target):
    if prediction.shape != target.shape or prediction.shape[-1] != prediction.shape[-2]:
        raise ValueError('Matching square matrices required')
    n=prediction.shape[-1]
    i,j=torch.triu_indices(n,n,device=prediction.device)
    return torch.linalg.vector_norm((prediction-target)[...,i,j],dim=-1).mean()

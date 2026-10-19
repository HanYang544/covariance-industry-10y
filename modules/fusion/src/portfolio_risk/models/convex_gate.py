"""State-dependent arithmetic covariance fusion, optionally scale calibrated."""
import torch
from torch import nn


class ConvexGate(nn.Module):
    def __init__(self, experts, calibrated=False):
        super().__init__()
        self.calibrated=calibrated
        self.gate=nn.Sequential(nn.Linear(6,16),nn.ReLU(),nn.Linear(16,experts)).double()
        if calibrated:
            self.theta=nn.Parameter(torch.zeros(5,dtype=torch.float64))

    def forward(self, covariance, state, design=None, return_weights=False):
        weights=self.gate(state).softmax(-1)
        prediction=torch.einsum('bk,bkij->bij',weights,covariance)
        if self.calibrated:
            prediction=prediction*torch.exp(-design@self.theta)[:,None,None]
        return (prediction,weights) if return_weights else prediction

    def constrain(self):
        if self.calibrated:
            with torch.no_grad():
                self.theta[0].clamp_(-1,1)
                self.theta[1:].clamp_(-.25,.25)


def covariance_loss(prediction,target,objective):
    if objective=='frobenius':
        return torch.linalg.matrix_norm(prediction-target,ord='fro').mean()
    if objective!='nll': raise ValueError(objective)
    factor=torch.linalg.cholesky(prediction)
    logdet=2*factor.diagonal(dim1=-2,dim2=-1).log().sum(-1)
    trace=torch.cholesky_solve(target,factor).diagonal(dim1=-2,dim2=-1).sum(-1)
    return ((logdet+trace)/(2*target.shape[-1])).mean()

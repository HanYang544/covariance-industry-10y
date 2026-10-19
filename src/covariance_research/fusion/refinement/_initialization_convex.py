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



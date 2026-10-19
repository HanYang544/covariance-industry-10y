"""Dynamic precision-factor fusion with optional scale calibration."""
import torch
from .convex_gate import ConvexGate


class PrecisionGate(ConvexGate):
    def forward(self, whiteners, state, design=None, return_weights=False):
        weights=self.gate(state).softmax(-1)
        factor=torch.einsum('bk,bkij->bij',weights,whiteners)
        prediction=torch.linalg.inv(factor@factor.transpose(-1,-2))
        if self.calibrated:
            prediction=prediction*torch.exp(-design@self.theta)[:,None,None]
        return (prediction,weights) if return_weights else prediction

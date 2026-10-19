"""Conservative dynamic covariance fusion initialized at a train-only anchor."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class AnchoredRiskGate(nn.Module):
    def __init__(self, input_dim: int, anchor: np.ndarray, architecture: str,
                 hidden_dim: int, strength: float, calibrate: bool, log_scale_bound: float):
        super().__init__()
        self.register_buffer("anchor", torch.as_tensor(anchor, dtype=torch.float64))
        self.strength, self.calibrate, self.log_scale_bound = strength, calibrate, log_scale_bound
        output_dim = len(anchor) + int(calibrate)
        if architecture == "linear":
            self.network = nn.Sequential(nn.Linear(input_dim, output_dim))
        elif architecture == "mlp":
            self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                                         nn.Linear(hidden_dim, output_dim))
        else:
            raise ValueError(architecture)
        # Zero residual -> exactly fixed-fusion initialization, with scale=1.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        raw = self.network(features)
        dynamic = torch.softmax(torch.log(self.anchor.clamp_min(1e-12)) + raw[:, :len(self.anchor)], dim=-1)
        weights = (1 - self.strength) * self.anchor + self.strength * dynamic
        log_scale = (self.log_scale_bound * torch.tanh(raw[:, -1]) if self.calibrate
                     else torch.zeros_like(raw[:, 0]))
        return weights, log_scale

    def penalty(self, weights, log_scale):
        kl = (weights * (torch.log(weights.clamp_min(1e-12)) -
                         torch.log(self.anchor.clamp_min(1e-12)))).sum(-1)
        return (kl + log_scale.square()).mean()

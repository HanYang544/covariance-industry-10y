"""Softmax gating networks for covariance fusion."""

from __future__ import annotations

import torch
from torch import nn


class LinearGate(nn.Module):
    """Map features directly to convex covariance weights."""

    def __init__(self, input_dim: int, output_dim: int = 4) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.projection(features), dim=-1)


class MLPGate(nn.Module):
    """One-hidden-layer nonlinear covariance weighting model."""

    def __init__(self, input_dim: int, hidden_dim: int = 16, output_dim: int = 4) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.network(features), dim=-1)


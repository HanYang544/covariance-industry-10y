"""Structured risk extrapolation and gap-regularized spectral rotation."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn


def spectral_basis(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Descending eigensystem with deterministic per-column sign convention."""
    values, vectors = np.linalg.eigh((matrix + matrix.swapaxes(-1, -2)) / 2)
    values, vectors = values[..., ::-1].copy(), vectors[..., ::-1].copy()
    if np.any(values <= 0):
        raise ValueError('Strictly positive base spectrum required')
    pivot = np.argmax(np.abs(vectors), axis=-2)
    signs = np.take_along_axis(vectors, pivot[..., None, :], axis=-2).squeeze(-2)
    vectors *= np.where(signs < 0, -1, 1)[..., None, :]
    return values, vectors


def zero_head(inputs: int, outputs: int) -> nn.Linear:
    layer = nn.Linear(inputs, outputs)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class StructuredRiskModel(nn.Module):
    """Six state inputs; base spectrum is a deterministic historical input.

    Idea1: bounded variance factors and normalized SPD correlation in top m.
    Idea2: bounded positive spectrum and skew-exponential full-basis rotation.
    No additional scale-calibration branch or auxiliary loss is used.
    """
    def __init__(self, idea: int, dimension: int = 10, components: int = 3,
                 tau: float = .01):
        super().__init__()
        if idea not in (1, 2) or not 1 <= components <= dimension or tau <= 0:
            raise ValueError('Invalid architecture')
        self.idea, self.dimension, self.components, self.tau = idea, dimension, components, tau
        self.encoder = nn.Sequential(nn.Linear(6, 16), nn.ReLU())
        width = components if idea == 1 else dimension
        self.variance = zero_head(16, width)
        self.change = zero_head(16, width * (width - 1) // 2)
        pairs = torch.tril_indices(width, width, offset=-1)
        self.register_buffer('rows', pairs[0])
        self.register_buffer('cols', pairs[1])

    def forward(self, features: torch.Tensor, values: torch.Tensor,
                basis: torch.Tensor, diagnostics: bool = False):
        hidden = self.encoder(features)
        log_scale = 2 * torch.tanh(self.variance(hidden))
        change = self.change(hidden)
        batch = len(features)
        if self.idea == 1:
            m = self.components
            # Unit diagonal triangular factor -> SPD -> correlation normalization.
            factor = torch.eye(m, dtype=features.dtype, device=features.device).expand(batch, m, m).clone()
            factor[:, self.rows, self.cols] = .5 * torch.tanh(change)
            gram = factor @ factor.transpose(-1, -2)
            std = torch.diagonal(gram, dim1=-2, dim2=-1).sqrt()
            correlation = gram / (std.unsqueeze(-1) * std.unsqueeze(-2))
            predicted_values = values[:, :m] * torch.exp(log_scale)
            amplitude = predicted_values.sqrt()
            core = correlation * amplitude.unsqueeze(-1) * amplitude.unsqueeze(-2)
            selected = basis[:, :, :m]
            rest = basis[:, :, m:]
            matrix = selected @ core @ selected.transpose(-1, -2)
            matrix = matrix + (rest * values[:, None, m:]) @ rest.transpose(-1, -2)
            detail = {'variance_multiplier': torch.exp(log_scale), 'correlation': correlation}
        else:
            # Dimensionless gaps: equivalent to c=.1*mean(lambda)*tanh(head),
            # tau_ij=.01*mean(lambda)^2 in the unnormalized user formula.
            normalized = values / values.mean(-1, keepdim=True)
            gap = torch.abs(normalized[:, self.rows] - normalized[:, self.cols])
            angles = gap * (.1 * torch.tanh(change)) / (gap.square() + self.tau)
            lower = torch.zeros((batch, self.dimension, self.dimension), dtype=features.dtype, device=features.device)
            lower[:, self.rows, self.cols] = angles
            omega = lower - lower.transpose(-1, -2)
            rotation = torch.matrix_exp(omega)
            updated = basis @ rotation
            predicted_values = values * torch.exp(log_scale)
            matrix = (updated * predicted_values.unsqueeze(-2)) @ updated.transpose(-1, -2)
            detail = {'variance_multiplier': torch.exp(log_scale), 'rotation': rotation, 'omega': omega}
        matrix = (matrix + matrix.transpose(-1, -2)) / 2
        return (matrix, detail) if diagnostics else matrix


def matrix_loss(prediction: torch.Tensor, target: torch.Tensor, objective: str) -> torch.Tensor:
    if objective == 'frobenius':
        return torch.linalg.matrix_norm(prediction - target, ord='fro').mean()
    if objective != 'nll':
        raise ValueError(objective)
    factor = torch.linalg.cholesky(prediction)
    logdet = 2 * torch.diagonal(factor, dim1=-2, dim2=-1).log().sum(-1)
    trace = torch.diagonal(torch.cholesky_solve(target, factor), dim1=-2, dim2=-1).sum(-1)
    return ((logdet + trace) / (2 * target.shape[-1])).mean()

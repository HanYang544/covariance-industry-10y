"""Numerically stable Gaussian covariance loss."""

from __future__ import annotations

import torch


def gaussian_covariance_nll(
    covariance: torch.Tensor, realized_second_moment: torch.Tensor
) -> torch.Tensor:
    """Compute normalized Gaussian NLL without the additive constant."""
    dimension = covariance.shape[-1]
    cholesky = torch.linalg.cholesky(covariance)
    log_determinant = 2.0 * torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)).sum(-1)
    solved = torch.cholesky_solve(realized_second_moment, cholesky)
    trace_term = torch.diagonal(solved, dim1=-2, dim2=-1).sum(-1)
    return 0.5 * (log_determinant + trace_term) / dimension


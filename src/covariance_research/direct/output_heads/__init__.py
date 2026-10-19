import torch
from . import cholesky,matrix_exp,spectral_softplus,variance_correlation,spd_residual

def transform(method,lower,volatility=None,base_root=None):
    diagonal=lower.diagonal(dim1=-2,dim2=-1)
    symmetric=lower+lower.transpose(-1,-2)-torch.diag_embed(diagonal)
    if method=='cholesky':return cholesky.transform(lower)
    if method=='matrix_exp':return matrix_exp.transform(symmetric)
    if method=='spectral_softplus':return spectral_softplus.transform(symmetric)
    if method=='variance_correlation':return variance_correlation.transform(symmetric,volatility)
    if method=='residual':return spd_residual.transform(symmetric,base_root)
    raise ValueError(method)

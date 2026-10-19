from .convex import convex_fusion
from .precision import precision_factor_fusion

def aggregate(weights, experts, whiteners, method):
    if method == 'covariance': return convex_fusion(weights, experts)
    if method == 'precision': return precision_factor_fusion(weights, whiteners)
    raise ValueError(method)

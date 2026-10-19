import numpy as np

def historical_covariance(returns):
    """Centered daily sample covariance, denominator window-1."""
    return np.cov(returns,rowvar=False,ddof=1)

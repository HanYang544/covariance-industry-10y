import numpy as np
from sklearn.covariance import LedoitWolf

def ledoit_wolf_covariance(returns):
    """Current industry protocol: centered LW plus relative diagonal jitter."""
    c=LedoitWolf(assume_centered=False).fit(returns).covariance_
    return c+np.eye(c.shape[-1])*np.trace(c)/c.shape[-1]*1e-8

import numpy as np

def ewma_covariance(returns,decay):
    """Frozen protocol's zero-mean EWMA recursion, including initialization."""
    if not 0 < decay < 1: raise ValueError('decay must be in (0,1)')
    state=np.outer(returns[0],returns[0]);series=[]
    for row in returns:
        state=decay*state+(1-decay)*np.outer(row,row)
        series.append(state+np.eye(len(row))*np.trace(state)/len(row)*1e-8)
    return np.stack(series)

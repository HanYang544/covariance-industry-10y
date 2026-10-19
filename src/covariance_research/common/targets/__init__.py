from ..losses.frobenius import future_risk_targets

def future_covariance(returns,origins,horizon=20):
    """Only centered sample covariance is exposed by the current protocol."""
    return future_risk_targets(returns,origins,horizon,center=True,ddof=1)

"""Online convex likelihood update from labels that have fully matured."""

import numpy as np
from scipy.optimize import brentq

from portfolio_risk.covariance.iewma import covariance_from_whitener
from portfolio_risk.models.state_whitener import fit_convex_fusion


def mature_origin_indices(t, horizon, window):
    stop = t - horizon + 1
    start = stop - window
    if start < 0:
        raise ValueError("Insufficient matured labels.")
    return np.arange(start, stop)


def precision_offset(energy, penalty, bound):
    if not np.isfinite(energy) or energy <= 0 or penalty < 0 or bound <= 0:
        raise ValueError("Need positive finite energy and bounds, with nonnegative penalty.")
    gradient = lambda delta: .5 * (energy * np.exp(delta) - 1) + penalty * delta
    if gradient(-bound) >= 0:
        return -bound
    if gradient(bound) <= 0:
        return bound
    return float(brentq(gradient, -bound, bound, xtol=1e-12))


def online_prediction(parent, prepared, t, horizon, spec, config):
    mode = spec["parent_mode"]
    x = prepared["features"][mode]
    if spec["window"] == 0:
        prediction, weights, scale, _ = parent.predict(prepared["whiteners"][t:t + 1], x[t:t + 1], x[t:t + 1])
        return prediction[0], weights[0], float(scale[0]), 1.
    indices = mature_origin_indices(t, horizon, spec["window"])
    current_states = parent.state_probabilities(x[t:t + 1])
    anchor = (current_states @ parent.beta)[0]
    local_weights, _ = fit_convex_fusion(np.ones((len(indices), 1)),
        prepared["diagonal"][indices], prepared["gram"][indices], anchor, spec["anchor_penalty"])
    weights = local_weights[0]
    combined = np.einsum("k,kij->ij", weights, prepared["whiteners"][t])
    design_t = np.r_[1., np.clip((x[t] - parent.calibration_mean) / parent.calibration_scale, -parent.clip, parent.clip)]
    eta_t = float(design_t @ parent.theta)
    correction = 0.
    if spec["correct"]:
        design_history = np.column_stack([np.ones(len(indices)), np.clip(
            (x[indices] - parent.calibration_mean) / parent.calibration_scale, -parent.clip, parent.clip)])
        hypothetical_precision = np.exp(design_history @ parent.theta)
        energy = np.einsum("k,bkl,l->b", weights, prepared["gram"][indices], weights) * hypothetical_precision
        correction = precision_offset(float(energy.mean()), config["scalar_penalty"], config["max_abs_log_correction"])
    scale = np.exp(-eta_t - correction)
    prediction = covariance_from_whitener(combined) * scale
    return prediction, weights, float(scale), float(np.exp(-correction))

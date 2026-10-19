"""State-conditioned convex whitening fusion plus convex precision calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.cluster import KMeans

from portfolio_risk.covariance.iewma import covariance_from_whitener


def whitener_statistics(whiteners: np.ndarray, realized: np.ndarray):
    dimension = realized.shape[-1]
    diagonal = np.diagonal(whiteners, axis1=-2, axis2=-1)
    transformed = np.einsum("bij,bkjl->bkil", realized, whiteners)
    gram = np.einsum("bkij,blij->bkl", whiteners, transformed) / dimension
    return diagonal, (gram + gram.swapaxes(-1, -2)) / 2


def fusion_objective(flat: np.ndarray, states: np.ndarray, diagonal: np.ndarray,
                     gram: np.ndarray, anchor: np.ndarray, ridge: float):
    r, k = states.shape[1], diagonal.shape[1]
    beta = flat.reshape(r, k)
    weights = states @ beta
    d = np.einsum("bk,bki->bi", weights, diagonal)
    if np.any(d <= 0):
        return float("inf"), np.zeros_like(flat)
    q = np.einsum("bkl,bl->bk", gram, weights)
    value = np.mean(-np.log(d).mean(1) + .5 * np.sum(weights * q, axis=1))
    gradient_w = q - np.mean(diagonal / d[:, None, :], axis=2)
    gradient = states.T @ gradient_w / len(states)
    difference = beta - anchor
    value += ridge * np.sum(difference ** 2) / (2 * r)
    gradient += ridge * difference / r
    return float(value), gradient.ravel()


def fit_convex_fusion(states, diagonal, gram, anchor=None, ridge=0.):
    r, k = states.shape[1], diagonal.shape[1]
    if anchor is None:
        anchor = np.full(k, 1 / k)
    initial = np.broadcast_to(anchor, (r, k)).copy()
    equality_jac = np.zeros((r, r * k))
    for row in range(r):
        equality_jac[row, row * k:(row + 1) * k] = 1
    result = minimize(fusion_objective, initial.ravel(), args=(states, diagonal, gram, anchor, ridge),
        jac=True, method="SLSQP", bounds=[(0, 1)] * (r * k),
        constraints={"type": "eq", "fun": lambda b: b.reshape(r, k).sum(1) - 1,
                     "jac": lambda b: equality_jac},
        options={"ftol": 1e-10, "maxiter": 300})
    if not result.success:
        raise RuntimeError(f"Convex fusion solver failed: {result.message}")
    beta = np.maximum(result.x.reshape(r, k), 0)
    beta /= beta.sum(1, keepdims=True)
    return beta, dict(objective=float(result.fun), iterations=int(result.nit))


def calibration_objective(theta, design, energy, ridge):
    eta = design @ theta
    # Bound parameters during fitting; no clipping/changed objective inside this function.
    exponential = np.exp(eta)
    value = np.mean(.5 * (energy * exponential - eta)) + .5 * ridge * np.sum(theta[1:] ** 2)
    gradient = design.T @ (.5 * (energy * exponential - 1)) / len(design)
    gradient[1:] += ridge * theta[1:]
    return float(value), gradient


def fit_precision_calibration(design, energy, ridge):
    result = minimize(calibration_objective, np.zeros(design.shape[1]),
        args=(design, energy, ridge), method="L-BFGS-B", jac=True,
        bounds=[(-1, 1)] + [(-.25, .25)] * (design.shape[1] - 1),
        options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8})
    if not result.success:
        raise RuntimeError(f"Convex scale solver failed: {result.message}")
    return result.x, dict(objective=float(result.fun), iterations=int(result.nit))


@dataclass
class StateWhitenerModel:
    mean: np.ndarray
    scale: np.ndarray
    projection: np.ndarray
    centers: np.ndarray
    temperature: float
    beta: np.ndarray
    calibration_mean: np.ndarray
    calibration_scale: np.ndarray
    theta: np.ndarray
    clip: float

    def state_probabilities(self, features):
        if self.beta.shape[0] == 1:
            return np.ones((len(features), 1))
        standardized = np.clip((features - self.mean) / self.scale, -self.clip, self.clip)
        projected = standardized @ self.projection
        distance = np.sum((projected[:, None, :] - self.centers[None, :, :]) ** 2, axis=2)
        return softmax(-distance / (2 * self.temperature), axis=1)

    def predict(self, whiteners, state_features, calibration_features):
        probabilities = self.state_probabilities(state_features)
        weights = probabilities @ self.beta
        combined = np.einsum("bk,bkij->bij", weights, whiteners)
        design = np.column_stack([np.ones(len(weights)), np.clip(
            (calibration_features - self.calibration_mean) / self.calibration_scale, -self.clip, self.clip)])
        eta = design @ self.theta
        matrix = covariance_from_whitener(combined) * np.exp(-eta)[:, None, None]
        return matrix, weights, np.exp(-eta), probabilities

    def save(self, path: Path):
        np.savez(path, **self.__dict__)

    @classmethod
    def load(cls, path: Path):
        with np.load(path) as saved:
            values = {k: saved[k] for k in saved.files}
        values["temperature"], values["clip"] = float(values["temperature"]), float(values["clip"])
        return cls(**values)


def fit_state_model(whiteners, realized, state_features, calibration_features, spec, config,
                    statistics=None, anchor=None):
    diagonal, gram = whitener_statistics(whiteners, realized) if statistics is None else statistics
    mean, scale = state_features.mean(0), np.maximum(state_features.std(0), 1e-12)
    standardized = np.clip((state_features - mean) / scale, -config["feature_clip"], config["feature_clip"])
    if spec["states"] == 1:
        projection, centers, temperature = np.zeros((state_features.shape[1], 1)), np.zeros((1, 1)), 1.
        probabilities = np.ones((len(state_features), 1))
    else:
        _, _, vh = np.linalg.svd(standardized, full_matrices=False)
        projection = vh[:min(config["pca_components"], state_features.shape[1])].T
        projected = standardized @ projection
        clustering = KMeans(n_clusters=spec["states"], n_init=10, random_state=config["seed"]).fit(projected)
        centers = clustering.cluster_centers_
        distance = np.sum((projected[:, None] - centers[None]) ** 2, axis=2)
        temperature = max(float(distance.min(1).mean()), 1e-3)
        probabilities = softmax(-distance / (2 * temperature), axis=1)
    beta, fusion_info = fit_convex_fusion(probabilities, diagonal, gram, anchor, spec["ridge"])
    calibration_mean = calibration_features.mean(0)
    calibration_scale = np.maximum(calibration_features.std(0), 1e-12)
    design = np.column_stack([np.ones(len(realized)), np.clip(
        (calibration_features - calibration_mean) / calibration_scale, -config["feature_clip"], config["feature_clip"])])
    theta = np.zeros(design.shape[1])
    calibration_info = {}
    if spec["calibrate"]:
        weights = probabilities @ beta
        energy = np.einsum("bk,bkl,bl->b", weights, gram, weights)
        theta, calibration_info = fit_precision_calibration(design, energy, config["calibration_ridge"])
    model = StateWhitenerModel(mean, scale, projection, centers, temperature, beta,
                              calibration_mean, calibration_scale, theta, config["feature_clip"])
    return model, dict(fusion=fusion_info, calibration=calibration_info)

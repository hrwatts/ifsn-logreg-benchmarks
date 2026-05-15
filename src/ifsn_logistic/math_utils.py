"""Core mathematical primitives for logistic regression."""

from __future__ import annotations

import numpy as np

from ifsn_logistic.base import PROB_CLIP_EPS


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid implemented with NumPy only."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def logistic_probabilities(
    X: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Compute p_i = sigma(x_i^T beta) for every row of *X*."""
    return sigmoid(X @ beta)


def curvature_weights(p: np.ndarray) -> np.ndarray:
    """Curvature weights q_i = p_i (1 - p_i)."""
    return p * (1.0 - p)


def clip_probabilities(
    p: np.ndarray,
    eps: float = PROB_CLIP_EPS,
) -> np.ndarray:
    """Clip probabilities to [eps, 1 - eps] for safe log computation."""
    return np.clip(p, eps, 1.0 - eps)


def linear_scores(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Raw linear scores z_i = x_i^T beta."""
    return X @ beta


def subsampled_gradient(
    X_S: np.ndarray,
    y_S: np.ndarray,
    p_S: np.ndarray,
    beta: np.ndarray,
    alpha: float,
    pen_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Subsampled gradient of the penalized logistic loss."""
    s = X_S.shape[0]
    residual = y_S - p_S
    g = -(1.0 / s) * (X_S.T @ residual)
    if pen_weights is not None:
        g += alpha * (pen_weights * beta)
    else:
        g += alpha * beta
    return g


def full_gradient(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    alpha: float,
    pen_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Full-data gradient of the penalized logistic loss."""
    p = logistic_probabilities(X, beta)
    return subsampled_gradient(X, y, p, beta, alpha, pen_weights)

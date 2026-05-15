"""Logistic loss and related objective functions.

All functions assume binary labels y ∈ {0, 1} and work with the
standard negative-log-likelihood formulation plus optional L2 penalty.
"""

from __future__ import annotations

import numpy as np

from ifsn_logistic.math_utils import clip_probabilities, logistic_probabilities


def logistic_loss(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    alpha: float = 0.0,
    pen_weights: np.ndarray | None = None,
) -> float:
    """Penalised logistic loss (negative log-likelihood + L2).

    L(β) = −(1/n) Σ [ y_i log p_i + (1−y_i) log(1−p_i) ] + (α/2) β^T W β

    Parameters
    ----------
    X : (n, d)
    y : (n,)  binary labels in {0, 1}.
    beta : (d,)
    alpha : float  regularization strength.
    pen_weights : (d,) or None  per-coefficient penalty weights.

    Returns
    -------
    loss : float
    """
    p = logistic_probabilities(X, beta)
    p = clip_probabilities(p)
    n = X.shape[0]
    nll = -(1.0 / n) * (y @ np.log(p) + (1.0 - y) @ np.log(1.0 - p))
    if pen_weights is not None:
        reg = 0.5 * alpha * np.dot(pen_weights * beta, beta)
    else:
        reg = 0.5 * alpha * np.dot(beta, beta)
    return float(nll + reg)


def logistic_gradient(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    alpha: float = 0.0,
    pen_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Full-data gradient of the penalised logistic loss.

    g = −(1/n) X^T (y − p) + α W β

    Parameters
    ----------
    X : (n, d)
    y : (n,)
    beta : (d,)
    alpha : float
    pen_weights : (d,) or None

    Returns
    -------
    g : (d,)
    """
    p = logistic_probabilities(X, beta)
    n = X.shape[0]
    g = -(1.0 / n) * (X.T @ (y - p))
    if pen_weights is not None:
        g += alpha * (pen_weights * beta)
    else:
        g += alpha * beta
    return g

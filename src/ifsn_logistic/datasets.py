"""Synthetic data generators for testing and comparison experiments."""

from __future__ import annotations

import numpy as np


def _stable_sigmoid_from_linear(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def make_logistic_data(
    n: int = 1000,
    d: int = 20,
    *,
    beta_true: np.ndarray | None = None,
    intercept: float = 0.0,
    noise_scale: float = 1.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard well-conditioned logistic regression data."""
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n, d)) * noise_scale
    if beta_true is None:
        beta_true = rng.standard_normal(d)
    z = X @ beta_true + intercept
    p = _stable_sigmoid_from_linear(z)
    y = rng.binomial(1, p).astype(np.float64)
    return X, y, beta_true


def make_ill_conditioned_data(
    n: int = 1000,
    d: int = 20,
    *,
    condition_number: float = 1e4,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate data with a controlled condition number for X."""
    rng = np.random.default_rng(random_state)
    U, _ = np.linalg.qr(rng.standard_normal((n, d)))
    Vt, _ = np.linalg.qr(rng.standard_normal((d, d)))
    singular_values = np.geomspace(1.0, condition_number, num=d)
    X = (U * singular_values) @ Vt

    beta_true = rng.standard_normal(d)
    z = X @ beta_true
    p = _stable_sigmoid_from_linear(z)
    y = rng.binomial(1, p).astype(np.float64)
    return X, y, beta_true


def make_sparse_signal_data(
    n: int = 1000,
    d: int = 50,
    *,
    sparsity: int = 5,
    signal_strength: float = 3.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Data where only a small subset of features is active."""
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n, d))
    beta_true = np.zeros(d)
    active = rng.choice(d, size=sparsity, replace=False)
    beta_true[active] = rng.choice([-1.0, 1.0], size=sparsity) * signal_strength
    z = X @ beta_true
    p = _stable_sigmoid_from_linear(z)
    y = rng.binomial(1, p).astype(np.float64)
    return X, y, beta_true


def make_correlated_data(
    n: int = 1000,
    d: int = 20,
    *,
    rho: float = 0.9,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Data with AR(1)-correlated features."""
    rng = np.random.default_rng(random_state)
    idx = np.arange(d)
    cov = rho ** np.abs(idx[:, None] - idx[None, :])
    L = np.linalg.cholesky(cov)
    X = rng.standard_normal((n, d)) @ L.T

    beta_true = rng.standard_normal(d)
    z = X @ beta_true
    p = _stable_sigmoid_from_linear(z)
    y = rng.binomial(1, p).astype(np.float64)
    return X, y, beta_true


def make_high_dimensional_data(
    n: int = 200,
    d: int = 500,
    *,
    sparsity: int = 10,
    signal_strength: float = 2.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """High-dimensional regime where d > n."""
    return make_sparse_signal_data(
        n=n,
        d=d,
        sparsity=sparsity,
        signal_strength=signal_strength,
        random_state=random_state,
    )


def make_imbalanced_data(
    n: int = 1000,
    d: int = 20,
    *,
    positive_rate: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Data with substantial class imbalance."""
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n, d))
    beta_true = rng.standard_normal(d) * 0.5
    logit_rate = np.log(positive_rate / (1.0 - positive_rate))
    intercept = logit_rate - np.mean(X @ beta_true)
    z = X @ beta_true + intercept
    p = _stable_sigmoid_from_linear(z)
    y = rng.binomial(1, p).astype(np.float64)
    return X, y, beta_true


def make_near_separable_data(
    n: int = 500,
    d: int = 20,
    *,
    signal_strength: float = 10.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Data that is nearly linearly separable."""
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n, d))
    beta_true = rng.standard_normal(d) * signal_strength
    z = X @ beta_true
    p = _stable_sigmoid_from_linear(z)
    y = rng.binomial(1, p).astype(np.float64)
    return X, y, beta_true


def make_redundant_features_data(
    n: int = 500,
    d: int = 20,
    *,
    n_duplicates: int = 3,
    n_constant: int = 2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Data with duplicate and constant-valued feature columns."""
    rng = np.random.default_rng(random_state)
    d_orig = d - n_duplicates - n_constant
    if d_orig < 2:
        raise ValueError("d must be large enough to accommodate duplicates and constants")

    X_orig = rng.standard_normal((n, d_orig))
    beta_orig = rng.standard_normal(d_orig)
    X_dup = X_orig[:, :n_duplicates].copy()
    X_const = np.ones((n, n_constant)) * rng.uniform(-2, 2, n_constant)

    X = np.hstack([X_orig, X_dup, X_const])
    beta_true = np.concatenate([beta_orig, np.zeros(n_duplicates + n_constant)])

    z = X_orig @ beta_orig
    p = _stable_sigmoid_from_linear(z)
    y = rng.binomial(1, p).astype(np.float64)
    return X, y, beta_true

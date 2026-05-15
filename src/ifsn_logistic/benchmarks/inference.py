"""Inference utilities for benchmark result summaries."""

from __future__ import annotations

import math

import numpy as np


def bootstrap_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    random_state: int = 0,
) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval for the mean."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("bootstrap_confidence_interval expects a 1D array.")
    if x.size == 0:
        raise ValueError("bootstrap_confidence_interval requires at least one value.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1.")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100.")

    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, x.size, size=(n_bootstrap, x.size))
    sampled_means = x[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(sampled_means, alpha))
    hi = float(np.quantile(sampled_means, 1.0 - alpha))
    return lo, hi


def cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Return paired Cohen's d computed from elementwise differences x - y."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.ndim != 1 or ya.ndim != 1:
        raise ValueError("cohens_d_paired expects 1D arrays.")
    if xa.size != ya.size:
        raise ValueError("cohens_d_paired requires equal-length arrays.")
    if xa.size == 0:
        raise ValueError("cohens_d_paired requires at least one paired sample.")

    diff = xa - ya
    if diff.size == 1:
        return 0.0
    std = float(np.std(diff, ddof=1))
    if std < 1e-15:
        return 0.0
    return float(np.mean(diff) / std)


def paired_sign_test_two_sided(x: np.ndarray, y: np.ndarray) -> float:
    """Return an exact two-sided sign-test p-value for paired arrays x and y."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.ndim != 1 or ya.ndim != 1:
        raise ValueError("paired_sign_test_two_sided expects 1D arrays.")
    if xa.size != ya.size:
        raise ValueError("paired_sign_test_two_sided requires equal-length arrays.")
    if xa.size == 0:
        raise ValueError("paired_sign_test_two_sided requires at least one paired sample.")

    diff = xa - ya
    non_ties = diff[diff != 0.0]
    n = int(non_ties.size)
    if n == 0:
        return 1.0

    pos = int(np.sum(non_ties > 0.0))
    k = min(pos, n - pos)
    tail_prob = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return float(min(1.0, 2.0 * tail_prob))


def paired_direction(
    method_values: np.ndarray,
    baseline_values: np.ndarray,
    lower_is_better: bool,
    p_value: float,
    alpha: float = 0.05,
) -> str:
    """Return a human-readable paired direction label."""
    m = np.asarray(method_values, dtype=np.float64)
    b = np.asarray(baseline_values, dtype=np.float64)
    delta = float(np.mean(m - b))

    if p_value >= alpha:
        return "equal_p>=0.05"
    if lower_is_better:
        return "ifsn_better_p<0.05" if delta < 0.0 else "baseline_better_p<0.05"
    return "ifsn_better_p<0.05" if delta > 0.0 else "baseline_better_p<0.05"
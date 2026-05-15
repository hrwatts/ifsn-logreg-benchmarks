"""Subsampling utilities for the sequential subsampling solvers."""

from __future__ import annotations

import numpy as np


def subsample_indices(
    n: int,
    subsample_size: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a sorted subsample of row indices without replacement."""
    if subsample_size is None or subsample_size >= n:
        return np.arange(n)
    return np.sort(rng.choice(n, size=subsample_size, replace=False))


def resolve_iteration_indices(
    n: int,
    subsample_size: int | None,
    rng: np.random.Generator,
    sampling_scheme: str,
    fixed_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Return the row indices used at the current iteration."""
    if sampling_scheme == "fresh":
        return subsample_indices(n, subsample_size, rng)
    if sampling_scheme == "fixed":
        if fixed_indices is None:
            raise ValueError("fixed_indices must be provided when sampling_scheme='fixed'.")
        return fixed_indices
    raise ValueError(f"Unknown sampling_scheme {sampling_scheme!r}.")


def subsample_data(
    X: np.ndarray,
    y: np.ndarray,
    subsample_size: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a subsampled view of the data."""
    idx = subsample_indices(X.shape[0], subsample_size, rng)
    return X[idx], y[idx], idx


def stratified_subsample_indices(
    y: np.ndarray,
    subsample_size: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stratified subsampling that preserves class proportions."""
    n = len(y)
    if subsample_size is None or subsample_size >= n:
        return np.arange(n)

    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    frac_1 = len(idx_1) / n

    s1 = max(1, int(round(subsample_size * frac_1)))
    s0 = subsample_size - s1
    s0 = min(s0, len(idx_0))
    s1 = min(s1, len(idx_1))

    chosen_0 = rng.choice(idx_0, size=s0, replace=False)
    chosen_1 = rng.choice(idx_1, size=s1, replace=False)
    return np.sort(np.concatenate([chosen_0, chosen_1]))

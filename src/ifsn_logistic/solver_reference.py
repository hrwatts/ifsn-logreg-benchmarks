"""Reference sequential subsampling solver for binary logistic regression.

This solver implements the simplest contribution extracted from the larger IFSN
workspace:

- baseline: sequential Sherman-Morrison rebuild with a fresh subsample each step
- adaptation: the same sequential rebuild, but with one fixed subsample reused
  across all iterations

The update structure is identical in both modes. Only the sampling schedule
changes.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ifsn_logistic.config import SolverConfig
from ifsn_logistic.losses import logistic_loss
from ifsn_logistic.math_utils import (
    curvature_weights,
    logistic_probabilities,
    subsampled_gradient,
)
from ifsn_logistic.sampling import resolve_iteration_indices, subsample_indices

logger = logging.getLogger(__name__)


def build_inverse_hessian_sm(
    X_S: np.ndarray,
    q_S: np.ndarray,
    alpha: float,
    s: int,
    pen_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the inverse subsampled Hessian via sequential Sherman-Morrison."""
    d = X_S.shape[1]

    if pen_weights is not None:
        pw = pen_weights.copy()
        pw[pw < 1e-12] = 1e-12
        B = np.diag(1.0 / (alpha * pw))
    else:
        B = np.eye(d) / alpha

    info: dict[str, Any] = {
        "sm_skipped": 0,
        "min_denom_seen": np.inf,
        "max_denom_seen": -np.inf,
        "neg_denom_count": 0,
    }

    for i in range(len(q_S)):
        w_i = q_S[i] / s
        if w_i < 1e-30:
            info["sm_skipped"] += 1
            continue
        x_i = X_S[i]
        v = B @ x_i
        denom = 1.0 + w_i * (x_i @ v)

        info["min_denom_seen"] = min(info["min_denom_seen"], denom)
        info["max_denom_seen"] = max(info["max_denom_seen"], denom)
        if denom <= 0:
            info["neg_denom_count"] += 1

        B -= w_i * np.outer(v, v) / denom

    return B, info


def ifsn_reference_solve(
    X: np.ndarray,
    y: np.ndarray,
    config: SolverConfig | None = None,
    pen_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the reference sequential subsampling solver."""
    if config is None:
        config = SolverConfig()
    config.validate()

    n, d = X.shape
    alpha = config.alpha
    rng = np.random.default_rng(config.random_state)
    beta = np.zeros(d)

    fixed_indices = None
    if config.sampling_scheme == "fixed":
        fixed_indices = subsample_indices(n, config.subsample_size, rng)

    diagnostics: dict[str, Any] = {
        "loss": [],
        "grad_norm": [],
        "step_norm": [],
        "beta_history": [],
        "sm_skipped": [],
        "min_denom": [],
        "neg_denom_count": [],
        "symmetry_err": [],
        "sampling_scheme": config.sampling_scheme,
        "sample_size": n if config.subsample_size is None else min(config.subsample_size, n),
        "fixed_sample_indices": None if fixed_indices is None else fixed_indices.copy(),
        "sample_indices_history": [],
    }

    stall_count = 0
    best_loss = np.inf

    for k in range(config.max_iter):
        S = resolve_iteration_indices(
            n,
            config.subsample_size,
            rng,
            config.sampling_scheme,
            fixed_indices=fixed_indices,
        )
        diagnostics["sample_indices_history"].append(S.copy())
        X_S = X[S]
        y_S = y[S]
        s = len(S)

        p_S = logistic_probabilities(X_S, beta)
        q_S = curvature_weights(p_S)
        g = subsampled_gradient(X_S, y_S, p_S, beta, alpha, pen_weights)
        B, sm_info = build_inverse_hessian_sm(X_S, q_S, alpha, s, pen_weights)

        sym_err = float(np.linalg.norm(B - B.T) / max(np.linalg.norm(B), 1e-30))
        diagnostics["symmetry_err"].append(sym_err)
        diagnostics["sm_skipped"].append(sm_info["sm_skipped"])
        diagnostics["min_denom"].append(sm_info["min_denom_seen"])
        diagnostics["neg_denom_count"].append(sm_info["neg_denom_count"])

        if sym_err > 1e-8:
            logger.warning("iter %d: B symmetry error %.2e exceeds threshold", k, sym_err)
        if sm_info["neg_denom_count"] > 0:
            logger.warning(
                "iter %d: %d negative SM denominators detected (min=%.2e)",
                k,
                sm_info["neg_denom_count"],
                sm_info["min_denom_seen"],
            )

        d_k = -B @ g
        if not np.all(np.isfinite(d_k)):
            logger.error("iter %d: non-finite Newton direction", k)

        beta = beta + d_k

        loss = logistic_loss(X, y, beta, alpha, pen_weights)
        g_norm = float(np.linalg.norm(g))
        s_norm = float(np.linalg.norm(d_k))

        diagnostics["loss"].append(loss)
        diagnostics["grad_norm"].append(g_norm)
        diagnostics["step_norm"].append(s_norm)
        diagnostics["beta_history"].append(beta.copy())

        if loss < best_loss - 1e-12:
            best_loss = loss
            stall_count = 0
        else:
            stall_count += 1

        if config.verbose >= 1:
            print(
                f"[ref:{config.sampling_scheme}] iter {k:4d}  loss={loss:.8f}  "
                f"||g||={g_norm:.4e}  ||d||={s_norm:.4e}  "
                f"sym_err={sym_err:.1e}  sm_skip={sm_info['sm_skipped']}"
            )

        if g_norm < config.tol:
            diagnostics["converged"] = True
            diagnostics["n_iter"] = k + 1
            return beta, diagnostics

    diagnostics["converged"] = False
    diagnostics["n_iter"] = config.max_iter
    return beta, diagnostics

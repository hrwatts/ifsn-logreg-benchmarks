"""Numerically guarded sequential subsampling solver."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np

from ifsn_logistic.config import SafeSolverConfig
from ifsn_logistic.losses import logistic_loss
from ifsn_logistic.math_utils import (
    curvature_weights,
    full_gradient,
    logistic_probabilities,
    subsampled_gradient,
)
from ifsn_logistic.sampling import resolve_iteration_indices, subsample_indices

logger = logging.getLogger(__name__)


def build_inverse_hessian_sm_safe(
    X_S: np.ndarray,
    q_S: np.ndarray,
    alpha: float,
    s: int,
    pen_weights: np.ndarray | None = None,
    min_denom: float = 1e-12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build inverse Hessian with numerical safeguards."""
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

        if denom < min_denom:
            info["sm_skipped"] += 1
            logger.debug(
                "SM update %d skipped: denom=%.2e < floor=%.2e", i, denom, min_denom
            )
            continue

        B -= w_i * np.outer(v, v) / denom

    sym_err_pre = float(np.linalg.norm(B - B.T) / max(np.linalg.norm(B), 1e-30))
    info["symmetry_err_pre"] = sym_err_pre
    B = 0.5 * (B + B.T)
    return B, info


def _check_pd(B: np.ndarray) -> bool:
    """Quick positive-definiteness check via Cholesky."""
    try:
        np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def _armijo_line_search(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    d_k: np.ndarray,
    g: np.ndarray,
    alpha_reg: float,
    pen_weights: np.ndarray | None,
    c: float,
    rho: float,
    max_ls_iter: int,
) -> tuple[float, int]:
    """Find a step size satisfying the Armijo condition."""
    f0 = logistic_loss(X, y, beta, alpha_reg, pen_weights)
    slope = g @ d_k
    t = 1.0

    for ls_iter in range(max_ls_iter):
        f_new = logistic_loss(X, y, beta + t * d_k, alpha_reg, pen_weights)
        if f_new <= f0 + c * t * slope:
            return t, ls_iter + 1
        t *= rho

    return t, max_ls_iter


def ifsn_safe_solve(
    X: np.ndarray,
    y: np.ndarray,
    config: SafeSolverConfig | None = None,
    pen_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the numerically guarded sequential subsampling solver."""
    if config is None:
        config = SafeSolverConfig()
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
        "full_grad_norm": [],
        "step_norm": [],
        "step_size": [],
        "beta_history": [],
        "sm_skipped": [],
        "min_denom": [],
        "neg_denom_count": [],
        "symmetry_err": [],
        "pd_ok": [],
        "ls_iters": [],
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

        if not np.all(np.isfinite(g)):
            raise RuntimeError(f"Non-finite gradient at iteration {k}.")

        B, sm_info = build_inverse_hessian_sm_safe(
            X_S, q_S, alpha, s, pen_weights, min_denom=config.min_denominator
        )

        diagnostics["sm_skipped"].append(sm_info["sm_skipped"])
        diagnostics["min_denom"].append(sm_info["min_denom_seen"])
        diagnostics["neg_denom_count"].append(sm_info["neg_denom_count"])
        diagnostics["symmetry_err"].append(sm_info.get("symmetry_err_pre", 0.0))

        if not np.all(np.isfinite(B)):
            raise RuntimeError(f"Non-finite inverse Hessian at iteration {k}.")

        pd_ok = True
        if config.pd_check:
            pd_ok = _check_pd(B)
            if not pd_ok:
                warnings.warn(
                    f"Inverse Hessian lost positive-definiteness at iter {k}. "
                    "Falling back to gradient descent step.",
                    stacklevel=2,
                )

        d_k = -B @ g if pd_ok else -g

        step_norm = float(np.linalg.norm(d_k))
        if step_norm > config.max_step_norm:
            d_k = d_k * (config.max_step_norm / step_norm)
            step_norm = config.max_step_norm

        t = config.step_size
        ls_iters = 0
        if config.line_search:
            t, ls_iters = _armijo_line_search(
                X,
                y,
                beta,
                d_k,
                g,
                alpha,
                pen_weights,
                c=config.armijo_c,
                rho=config.armijo_rho,
                max_ls_iter=config.max_line_search_iter,
            )

        beta = beta + t * d_k
        if not np.all(np.isfinite(beta)):
            raise RuntimeError(f"Non-finite coefficients at iteration {k}.")

        loss = logistic_loss(X, y, beta, alpha, pen_weights)
        g_norm = float(np.linalg.norm(g))
        g_full = full_gradient(X, y, beta, alpha, pen_weights)
        g_full_norm = float(np.linalg.norm(g_full))

        diagnostics["loss"].append(loss)
        diagnostics["grad_norm"].append(g_norm)
        diagnostics["full_grad_norm"].append(g_full_norm)
        diagnostics["step_norm"].append(step_norm * t)
        diagnostics["step_size"].append(t)
        diagnostics["beta_history"].append(beta.copy())
        diagnostics["pd_ok"].append(pd_ok)
        diagnostics["ls_iters"].append(ls_iters)

        if loss < best_loss - 1e-12:
            best_loss = loss
            stall_count = 0
        else:
            stall_count += 1
        if stall_count >= 10:
            logger.warning(
                "iter %d: loss stalled for %d consecutive iterations", k, stall_count
            )

        if config.verbose >= 1:
            print(
                f"[safe:{config.sampling_scheme}] iter {k:4d}  loss={loss:.8f}  "
                f"||g_sub||={g_norm:.4e}  ||g_full||={g_full_norm:.4e}  "
                f"t={t:.4f}  ||d||={step_norm*t:.4e}  "
                f"skip={sm_info['sm_skipped']}  pd={'Y' if pd_ok else 'N'}"
            )

        if g_full_norm < config.tol:
            diagnostics["converged"] = True
            diagnostics["n_iter"] = k + 1
            return beta, diagnostics

    diagnostics["converged"] = False
    diagnostics["n_iter"] = config.max_iter
    return beta, diagnostics

"""Diagnostic utilities for inspecting solver behavior.

These functions consume the ``diagnostics`` dict returned by the
solvers and produce summaries, checks, and formatted output.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def convergence_summary(diagnostics: dict[str, Any]) -> str:
    """One-line human-readable convergence summary."""
    status = "CONVERGED" if diagnostics.get("converged") else "NOT CONVERGED"
    n_iter = diagnostics.get("n_iter", "?")
    final_loss = diagnostics["loss"][-1] if diagnostics["loss"] else float("nan")
    final_gnorm = diagnostics["grad_norm"][-1] if diagnostics["grad_norm"] else float("nan")
    return (
        f"{status} after {n_iter} iterations  "
        f"| loss={final_loss:.8f}  | ||g||={final_gnorm:.4e}"
    )


def check_no_nans(diagnostics: dict[str, Any]) -> bool:
    """Return True if no NaN/Inf appears in the tracked arrays."""
    for key in ("loss", "grad_norm", "step_norm"):
        vals = diagnostics.get(key, [])
        if any(not np.isfinite(v) for v in vals):
            return False
    for beta in diagnostics.get("beta_history", []):
        if not np.all(np.isfinite(beta)):
            return False
    return True


def loss_is_monotone(diagnostics: dict[str, Any], rtol: float = 1e-8) -> bool:
    """Check whether the loss decreased (or stayed flat) every iteration.

    Allows tiny increases up to *rtol* relative tolerance to account
    for floating-point noise.
    """
    losses = diagnostics.get("loss", [])
    for i in range(1, len(losses)):
        if losses[i] > losses[i - 1] * (1 + rtol):
            return False
    return True


def symmetry_error(B: np.ndarray) -> float:
    """Frobenius-norm measure of asymmetry: ||B - B^T||_F / ||B||_F."""
    return float(np.linalg.norm(B - B.T) / max(np.linalg.norm(B), 1e-30))


def condition_number_estimate(B: np.ndarray) -> float:
    """Condition number of B via singular values (expensive for large d)."""
    sv = np.linalg.svd(B, compute_uv=False)
    if sv[-1] < 1e-30:
        return float("inf")
    return float(sv[0] / sv[-1])


def detect_stall(diagnostics: dict[str, Any], window: int = 10, rtol: float = 1e-8) -> int:
    """Return the iteration at which loss stalled, or -1 if no stall.

    A stall is defined as ``window`` consecutive iterations where the loss
    does not decrease by more than ``rtol`` relative to the best-so-far.
    """
    losses = diagnostics.get("loss", [])
    best = np.inf
    run = 0
    for i, val in enumerate(losses):
        if val < best * (1 - rtol):
            best = val
            run = 0
        else:
            run += 1
        if run >= window:
            return i - window + 1
    return -1


def denominator_health(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Summarize Sherman-Morrison denominator behavior across iterations.

    Returns a dict with aggregate statistics useful for detecting
    pathological curvature conditions.
    """
    min_denoms = diagnostics.get("min_denom", [])
    neg_counts = diagnostics.get("neg_denom_count", [])
    sm_skipped = diagnostics.get("sm_skipped", [])

    if not min_denoms:
        return {"available": False}

    return {
        "available": True,
        "global_min_denom": float(min(min_denoms)),
        "total_neg_denoms": int(sum(neg_counts)),
        "total_sm_skipped": int(sum(sm_skipped)),
        "iters_with_neg_denom": int(sum(1 for c in neg_counts if c > 0)),
        "n_iters": len(min_denoms),
    }


def solver_health_report(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Comprehensive health check of a solver run.

    Returns a dict with boolean flags and numeric summaries that can be
    used to decide whether results are trustworthy.
    """
    report: dict[str, Any] = {
        "converged": diagnostics.get("converged", False),
        "n_iter": diagnostics.get("n_iter", None),
        "no_nans": check_no_nans(diagnostics),
        "loss_monotone": loss_is_monotone(diagnostics),
        "stall_iter": detect_stall(diagnostics),
    }

    # Loss reduction ratio
    losses = diagnostics.get("loss", [])
    if len(losses) >= 2:
        report["loss_reduction_ratio"] = float(losses[-1] / losses[0])
    else:
        report["loss_reduction_ratio"] = None

    # Symmetry error (from safe solver)
    sym_errs = diagnostics.get("symmetry_err", [])
    if sym_errs:
        report["max_symmetry_err"] = float(max(sym_errs))
    else:
        report["max_symmetry_err"] = None

    # PD checks (from safe solver)
    pd_flags = diagnostics.get("pd_ok", [])
    if pd_flags:
        report["pd_loss_count"] = int(sum(1 for f in pd_flags if not f))
    else:
        report["pd_loss_count"] = None

    # Denominator health
    report["denom_health"] = denominator_health(diagnostics)

    return report


def extract_convergence_table(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract a per-iteration table from diagnostics for CSV/markdown export."""
    rows = []
    n = len(diagnostics.get("loss", []))
    for i in range(n):
        row: dict[str, Any] = {"iter": i}
        for key in ("loss", "grad_norm", "full_grad_norm", "step_norm", "step_size"):
            vals = diagnostics.get(key, [])
            row[key] = vals[i] if i < len(vals) else None
        for key in ("sm_skipped", "pd_ok", "ls_iters", "min_denom",
                     "neg_denom_count", "symmetry_err"):
            vals = diagnostics.get(key, [])
            if vals:
                row[key] = vals[i] if i < len(vals) else None
        rows.append(row)
    return rows

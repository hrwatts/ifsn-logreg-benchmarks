"""Legacy synthetic benchmarking helpers.

These utilities support small synthetic diagnostics in ``experiments/``.
Official public benchmark claims should come from ``ifsn_logistic.benchmarks``
and the tracked outputs under ``results/``.
"""

from __future__ import annotations

import contextlib
import io
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ifsn_logistic.estimator import IFSNLogisticRegression
from ifsn_logistic.losses import logistic_loss

_SKLEARN_LOGISTIC_CLASS = None
_SKLEARN_IMPORT_ERROR = None
_SKLEARN_IMPORT_ATTEMPTED = False


@dataclass
class BenchmarkResult:
    """Result from a single solver run."""

    solver_name: str
    runtime_sec: float
    final_loss: float
    beta_error: float | None
    n_iter: int | None
    converged: bool | None
    accuracy: float | None
    extra: dict[str, Any] = field(default_factory=dict)


def sklearn_available() -> bool:
    """Return True when sklearn can be imported in the current environment."""
    logistic_cls = _load_sklearn_logistic()
    return logistic_cls is not None


def require_sklearn() -> None:
    """Raise a clear error when sklearn is unavailable in this interpreter."""
    logistic_cls = _load_sklearn_logistic()
    if logistic_cls is None:
        raise RuntimeError(
            "scikit-learn is unavailable in the current environment. "
            f"Import detail: {_SKLEARN_IMPORT_ERROR}"
        )


def _load_sklearn_logistic():
    """Lazily import sklearn to avoid polluting non-sklearn workflows."""
    global _SKLEARN_LOGISTIC_CLASS
    global _SKLEARN_IMPORT_ERROR
    global _SKLEARN_IMPORT_ATTEMPTED

    if not _SKLEARN_IMPORT_ATTEMPTED:
        _SKLEARN_IMPORT_ATTEMPTED = True
        stderr_buffer = io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stderr(stderr_buffer):
                try:  # pragma: no cover - environment dependent
                    from sklearn.linear_model import LogisticRegression as logistic_cls
                except Exception as exc:  # pragma: no cover - environment dependent
                    _SKLEARN_IMPORT_ERROR = exc
                    _SKLEARN_LOGISTIC_CLASS = None
                else:  # pragma: no cover - environment dependent
                    _SKLEARN_IMPORT_ERROR = None
                    _SKLEARN_LOGISTIC_CLASS = logistic_cls

    return _SKLEARN_LOGISTIC_CLASS


def _fit_sklearn_solver(
    X: np.ndarray,
    y: np.ndarray,
    solver: str,
    C_ours: float,
    max_iter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit an sklearn baseline using the C-scaling compatible with this repo."""
    logistic_cls = _load_sklearn_logistic()
    if logistic_cls is None:
        raise RuntimeError(f"scikit-learn is unavailable: {_SKLEARN_IMPORT_ERROR}")

    C_sklearn = C_ours / X.shape[0]
    clf = logistic_cls(
        solver=solver,
        C=C_sklearn,
        max_iter=max_iter,
        fit_intercept=False,
        tol=1e-8,
    )
    t0 = time.perf_counter()
    clf.fit(X, y)
    elapsed = time.perf_counter() - t0
    beta = clf.coef_.ravel()
    info = {
        "runtime_sec": elapsed,
        "n_iter": int(clf.n_iter_[0]),
        "converged": clf.n_iter_[0] < max_iter,
        "accuracy": float(clf.score(X, y)),
        "C_sklearn": C_sklearn,
    }
    return beta, info


def _fit_ifsn_solver(
    X: np.ndarray,
    y: np.ndarray,
    solver_variant: str,
    sampling_scheme: str,
    C: float,
    max_iter: int,
    subsample_size: int | None,
    random_state: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one of the sequential subsampling estimators."""
    clf = IFSNLogisticRegression(
        C=C,
        max_iter=max_iter,
        subsample_size=subsample_size,
        sampling_scheme=sampling_scheme,
        fit_intercept=False,
        solver=solver_variant,
        random_state=random_state,
        verbose=0,
    )
    t0 = time.perf_counter()
    clf.fit(X, y)
    elapsed = time.perf_counter() - t0
    beta = clf.coef_.ravel()
    info = {
        "runtime_sec": elapsed,
        "n_iter": clf.n_iter_,
        "converged": clf.converged_,
        "accuracy": float(clf.score(X, y)),
    }
    return beta, info


def run_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    beta_true: np.ndarray | None = None,
    *,
    C: float = 1.0,
    max_iter: int = 200,
    subsample_size: int | None = None,
    random_state: int | None = 42,
    sampling_schemes: list[str] | None = None,
    sklearn_solvers: list[str] | None = None,
    include_reference: bool = True,
    include_safe: bool = True,
) -> list[BenchmarkResult]:
    """Run fresh-vs-fixed comparisons and optional sklearn baselines."""
    if sampling_schemes is None:
        sampling_schemes = ["fresh", "fixed"]
    if sklearn_solvers is None:
        sklearn_solvers = ["lbfgs"]

    alpha = 1.0 / C
    results: list[BenchmarkResult] = []

    for solver_name in sklearn_solvers:
        try:
            beta, info = _fit_sklearn_solver(X, y, solver_name, C, max_iter)
            loss = logistic_loss(X, y, beta, alpha)
            beta_err = float(np.linalg.norm(beta - beta_true)) if beta_true is not None else None
            results.append(
                BenchmarkResult(
                    solver_name=f"sklearn-{solver_name}",
                    runtime_sec=info["runtime_sec"],
                    final_loss=loss,
                    beta_error=beta_err,
                    n_iter=info["n_iter"],
                    converged=info["converged"],
                    accuracy=info["accuracy"],
                    extra={"C_sklearn": info["C_sklearn"]},
                )
            )
        except Exception as exc:
            results.append(
                BenchmarkResult(
                    solver_name=f"sklearn-{solver_name}",
                    runtime_sec=float("nan"),
                    final_loss=float("nan"),
                    beta_error=None,
                    n_iter=None,
                    converged=False,
                    accuracy=None,
                    extra={"error": str(exc)},
                )
            )

    for sampling_scheme in sampling_schemes:
        if include_reference:
            try:
                beta, info = _fit_ifsn_solver(
                    X,
                    y,
                    "reference",
                    sampling_scheme,
                    C,
                    max_iter,
                    subsample_size,
                    random_state,
                )
                loss = logistic_loss(X, y, beta, alpha)
                beta_err = float(np.linalg.norm(beta - beta_true)) if beta_true is not None else None
                results.append(
                    BenchmarkResult(
                        solver_name=f"ifsn-reference-{sampling_scheme}",
                        runtime_sec=info["runtime_sec"],
                        final_loss=loss,
                        beta_error=beta_err,
                        n_iter=info["n_iter"],
                        converged=info["converged"],
                        accuracy=info["accuracy"],
                    )
                )
            except Exception as exc:
                results.append(
                    BenchmarkResult(
                        solver_name=f"ifsn-reference-{sampling_scheme}",
                        runtime_sec=float("nan"),
                        final_loss=float("nan"),
                        beta_error=None,
                        n_iter=None,
                        converged=False,
                        accuracy=None,
                        extra={"error": str(exc)},
                    )
                )

        if include_safe:
            try:
                beta, info = _fit_ifsn_solver(
                    X,
                    y,
                    "safe",
                    sampling_scheme,
                    C,
                    max_iter,
                    subsample_size,
                    random_state,
                )
                loss = logistic_loss(X, y, beta, alpha)
                beta_err = float(np.linalg.norm(beta - beta_true)) if beta_true is not None else None
                results.append(
                    BenchmarkResult(
                        solver_name=f"ifsn-safe-{sampling_scheme}",
                        runtime_sec=info["runtime_sec"],
                        final_loss=loss,
                        beta_error=beta_err,
                        n_iter=info["n_iter"],
                        converged=info["converged"],
                        accuracy=info["accuracy"],
                    )
                )
            except Exception as exc:
                results.append(
                    BenchmarkResult(
                        solver_name=f"ifsn-safe-{sampling_scheme}",
                        runtime_sec=float("nan"),
                        final_loss=float("nan"),
                        beta_error=None,
                        n_iter=None,
                        converged=False,
                        accuracy=None,
                        extra={"error": str(exc)},
                    )
                )

    return results


def results_to_markdown(results: list[BenchmarkResult]) -> str:
    """Format results as a Markdown table."""
    header = "| Solver | Time (s) | Loss | Beta Error | Iters | Conv. | Accuracy |"
    sep = "|--------|----------|------|------------|-------|-------|----------|"
    rows = [header, sep]
    for r in results:
        be = f"{r.beta_error:.4f}" if r.beta_error is not None else "-"
        ni = str(r.n_iter) if r.n_iter is not None else "-"
        cv = "Y" if r.converged else "N" if r.converged is not None else "-"
        ac = f"{r.accuracy:.4f}" if r.accuracy is not None else "-"
        rows.append(
            f"| {r.solver_name} | {r.runtime_sec:.4f} | {r.final_loss:.6f} "
            f"| {be} | {ni} | {cv} | {ac} |"
        )
    return "\n".join(rows)

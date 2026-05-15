"""Configuration dataclasses for the sequential subsampling solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


VALID_SAMPLING_SCHEMES = {"fresh", "fixed"}


@dataclass
class SolverConfig:
    """Configuration for the reference sequential subsampling solver.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on the tracked gradient norm.
    C : float
        Inverse regularization strength. The L2 strength is ``alpha = 1 / C``.
    subsample_size : int or None
        Number of rows used in the subsampled step. ``None`` means full data.
    sampling_scheme : {"fresh", "fixed"}
        ``"fresh"`` draws a new subsample each iteration.
        ``"fixed"`` draws one subsample and reuses it for the full run.
    fit_intercept : bool
        Retained for API symmetry with the estimator; the solvers expect any
        intercept augmentation to be done before they are called.
    random_state : int or None
        Seed for reproducible subsampling.
    verbose : int
        Verbosity level. ``0`` is silent.
    """

    max_iter: int = 200
    tol: float = 1e-6
    C: float = 1.0
    subsample_size: Optional[int] = None
    sampling_scheme: str = "fresh"
    fit_intercept: bool = True
    random_state: Optional[int] = None
    verbose: int = 0

    @property
    def alpha(self) -> float:
        """L2 regularization strength = 1 / C."""
        if self.C <= 0:
            raise ValueError("C must be strictly positive.")
        return 1.0 / self.C

    def validate(self) -> None:
        """Validate configuration values that materially affect the solver."""
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if self.tol < 0:
            raise ValueError("tol must be non-negative.")
        if self.subsample_size is not None and self.subsample_size <= 0:
            raise ValueError("subsample_size must be positive or None.")
        if self.sampling_scheme not in VALID_SAMPLING_SCHEMES:
            raise ValueError(
                "sampling_scheme must be one of "
                f"{sorted(VALID_SAMPLING_SCHEMES)}. "
                f"Got {self.sampling_scheme!r}."
            )


@dataclass
class SafeSolverConfig(SolverConfig):
    """Extended configuration for the numerically guarded solver."""

    step_size: float = 1.0
    min_denominator: float = 1e-12
    max_step_norm: float = 10.0
    line_search: bool = True
    armijo_c: float = 1e-4
    armijo_rho: float = 0.5
    max_line_search_iter: int = 20
    pd_check: bool = True

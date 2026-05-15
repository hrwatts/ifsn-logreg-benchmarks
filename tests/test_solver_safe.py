"""Tests for the engineering-safe IFSN solver."""

import numpy as np
import pytest

from ifsn_logistic.config import SafeSolverConfig
from ifsn_logistic.datasets import make_ill_conditioned_data, make_logistic_data
from ifsn_logistic.diagnostics import check_no_nans
from ifsn_logistic.solver_safe import ifsn_safe_solve


@pytest.fixture
def small_data():
    return make_logistic_data(n=200, d=10, random_state=42)


class TestSafeSolverBasic:
    def test_runs_without_error(self, small_data):
        X, y, _ = small_data
        config = SafeSolverConfig(max_iter=20, C=1.0, random_state=42)
        beta, diag = ifsn_safe_solve(X, y, config)
        assert beta.shape == (X.shape[1],)

    def test_no_nans(self, small_data):
        X, y, _ = small_data
        config = SafeSolverConfig(max_iter=50, C=1.0, random_state=42)
        beta, diag = ifsn_safe_solve(X, y, config)
        assert check_no_nans(diag)
        assert np.all(np.isfinite(beta))

    def test_loss_decreases_with_line_search(self):
        """With line search on full data, loss should be monotonically decreasing."""
        X, y, _ = make_logistic_data(n=200, d=5, random_state=0)
        config = SafeSolverConfig(
            max_iter=30, C=1.0, subsample_size=None,
            line_search=True, random_state=0,
        )
        _, diag = ifsn_safe_solve(X, y, config)
        losses = diag["loss"]
        # With Armijo line search, losses should be non-increasing (within tolerance)
        for i in range(1, len(losses)):
            assert losses[i] <= losses[i - 1] + 1e-10

    def test_converges_on_easy_problem(self):
        X, y, _ = make_logistic_data(n=500, d=5, random_state=11)
        config = SafeSolverConfig(
            max_iter=200, C=1.0, tol=1e-5,
            subsample_size=None, random_state=11,
        )
        beta, diag = ifsn_safe_solve(X, y, config)
        assert diag["converged"]


class TestSafeSolverGuards:
    def test_step_norm_cap(self, small_data):
        X, y, _ = small_data
        config = SafeSolverConfig(
            max_iter=10, C=1.0, max_step_norm=0.1, random_state=0,
        )
        _, diag = ifsn_safe_solve(X, y, config)
        for sn in diag["step_norm"]:
            assert sn <= 0.1 + 1e-10

    def test_pd_monitoring(self, small_data):
        X, y, _ = small_data
        config = SafeSolverConfig(
            max_iter=10, C=1.0, pd_check=True, random_state=0,
        )
        _, diag = ifsn_safe_solve(X, y, config)
        assert "pd_ok" in diag
        assert len(diag["pd_ok"]) == diag["n_iter"]

    def test_subsample_mode(self, small_data):
        X, y, _ = small_data
        config = SafeSolverConfig(
            max_iter=20, C=1.0, subsample_size=50, random_state=42,
        )
        beta, diag = ifsn_safe_solve(X, y, config)
        assert beta.shape == (X.shape[1],)
        assert check_no_nans(diag)

    def test_line_search_iterations_tracked(self, small_data):
        X, y, _ = small_data
        config = SafeSolverConfig(
            max_iter=10, C=1.0, line_search=True, random_state=0,
        )
        _, diag = ifsn_safe_solve(X, y, config)
        assert "ls_iters" in diag
        assert all(isinstance(x, int) for x in diag["ls_iters"])


class TestSafeSolverIllConditioned:
    @pytest.mark.slow
    def test_survives_ill_conditioning(self):
        """Safe solver should not blow up on ill-conditioned data."""
        X, y, _ = make_ill_conditioned_data(
            n=300, d=15, condition_number=1e6, random_state=0,
        )
        config = SafeSolverConfig(
            max_iter=100, C=1.0, subsample_size=100,
            line_search=True, max_step_norm=5.0, random_state=0,
        )
        beta, diag = ifsn_safe_solve(X, y, config)
        assert np.all(np.isfinite(beta))
        assert check_no_nans(diag)

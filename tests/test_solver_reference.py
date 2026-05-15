"""Tests for the reference sequential subsampling solver."""

import numpy as np

from ifsn_logistic.config import SolverConfig
from ifsn_logistic.datasets import make_logistic_data
from ifsn_logistic.diagnostics import check_no_nans
from ifsn_logistic.math_utils import curvature_weights, logistic_probabilities
from ifsn_logistic.solver_reference import build_inverse_hessian_sm, ifsn_reference_solve


def small_data():
    return make_logistic_data(n=200, d=10, random_state=42)


class TestBuildInverseHessianSM:
    def test_symmetric(self):
        X, y, beta = small_data()
        p = logistic_probabilities(X, beta)
        q = curvature_weights(p)
        B, info = build_inverse_hessian_sm(X, q, alpha=1.0, s=X.shape[0])
        np.testing.assert_allclose(B, B.T, atol=1e-10)
        assert info["neg_denom_count"] == 0

    def test_matches_direct_inverse(self):
        rng = np.random.default_rng(0)
        n, d = 20, 5
        X = rng.standard_normal((n, d))
        q = rng.uniform(0.05, 0.25, n)
        alpha = 0.5

        H = (1.0 / n) * (X.T @ np.diag(q) @ X) + alpha * np.eye(d)
        B_direct = np.linalg.inv(H)
        B_sm, sm_info = build_inverse_hessian_sm(X, q, alpha, s=n)

        np.testing.assert_allclose(B_sm, B_direct, atol=1e-8, rtol=1e-6)
        assert sm_info["sm_skipped"] == 0

    def test_positive_definite(self):
        X, y, beta = small_data()
        p = logistic_probabilities(X, beta)
        q = curvature_weights(p)
        B, _ = build_inverse_hessian_sm(X, q, alpha=1.0, s=X.shape[0])
        eigenvalues = np.linalg.eigvalsh(B)
        assert np.all(eigenvalues > -1e-10)


class TestReferenceSolve:
    def test_runs_without_error(self):
        X, y, _ = small_data()
        config = SolverConfig(max_iter=20, C=1.0, random_state=42)
        beta, diag = ifsn_reference_solve(X, y, config)
        assert beta.shape == (X.shape[1],)
        assert diag["sampling_scheme"] == "fresh"

    def test_no_nans(self):
        X, y, _ = small_data()
        config = SolverConfig(max_iter=50, C=1.0, random_state=42)
        beta, diag = ifsn_reference_solve(X, y, config)
        assert check_no_nans(diag)
        assert np.all(np.isfinite(beta))

    def test_loss_decreases_full_data(self):
        X, y, _ = make_logistic_data(n=100, d=5, random_state=0)
        config = SolverConfig(max_iter=30, C=1.0, subsample_size=None, random_state=0)
        _, diag = ifsn_reference_solve(X, y, config)
        assert diag["loss"][-1] < diag["loss"][0]

    def test_converges_on_easy_problem(self):
        X, y, _ = make_logistic_data(n=500, d=5, random_state=11)
        config = SolverConfig(
            max_iter=200, C=1.0, tol=1e-5, subsample_size=None, random_state=11
        )
        beta, diag = ifsn_reference_solve(X, y, config)
        assert beta.shape == (X.shape[1],)
        assert diag["converged"]

    def test_fixed_subsample_mode_runs(self):
        X, y, _ = small_data()
        config = SolverConfig(
            max_iter=30,
            C=1.0,
            subsample_size=50,
            sampling_scheme="fixed",
            random_state=42,
        )
        beta, diag = ifsn_reference_solve(X, y, config)
        assert beta.shape == (X.shape[1],)
        assert check_no_nans(diag)
        first = diag["sample_indices_history"][0]
        for sample in diag["sample_indices_history"][1:]:
            np.testing.assert_array_equal(first, sample)

    def test_pen_weights(self):
        X, y, _ = small_data()
        pen_weights = np.ones(X.shape[1])
        pen_weights[0] = 0.0
        config = SolverConfig(max_iter=20, C=1.0, random_state=0)
        beta, diag = ifsn_reference_solve(X, y, config, pen_weights=pen_weights)
        assert beta.shape == (X.shape[1],)

    def test_diagnostics_contain_sampling_metadata(self):
        X, y, _ = small_data()
        config = SolverConfig(max_iter=10, C=1.0, random_state=0, sampling_scheme="fixed")
        _, diag = ifsn_reference_solve(X, y, config)
        assert "sampling_scheme" in diag
        assert "sample_indices_history" in diag
        assert len(diag["sample_indices_history"]) == diag["n_iter"]

    def test_symmetry_error_stays_small(self):
        X, y, _ = small_data()
        config = SolverConfig(max_iter=20, C=1.0, random_state=42)
        _, diag = ifsn_reference_solve(X, y, config)
        for se in diag["symmetry_err"]:
            assert se < 1e-6

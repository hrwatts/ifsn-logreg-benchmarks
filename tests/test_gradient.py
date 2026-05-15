"""Tests for gradient computation correctness."""

import numpy as np
import pytest

from ifsn_logistic.datasets import make_logistic_data
from ifsn_logistic.losses import logistic_gradient, logistic_loss


@pytest.fixture
def data():
    return make_logistic_data(n=80, d=8, random_state=7)


class TestGradientFiniteDifference:
    """Check the analytic gradient against a finite-difference approximation."""

    def _numerical_gradient(self, X, y, beta, alpha, eps=1e-5):
        g = np.zeros_like(beta)
        for j in range(len(beta)):
            e_j = np.zeros_like(beta)
            e_j[j] = eps
            f_plus = logistic_loss(X, y, beta + e_j, alpha)
            f_minus = logistic_loss(X, y, beta - e_j, alpha)
            g[j] = (f_plus - f_minus) / (2 * eps)
        return g

    def test_gradient_matches_fd_at_zero(self, data):
        X, y, _ = data
        beta = np.zeros(X.shape[1])
        alpha = 0.1
        g_analytic = logistic_gradient(X, y, beta, alpha)
        g_numeric = self._numerical_gradient(X, y, beta, alpha)
        np.testing.assert_allclose(g_analytic, g_numeric, atol=1e-6, rtol=1e-4)

    def test_gradient_matches_fd_at_random(self, data):
        X, y, beta_true = data
        alpha = 0.05
        g_analytic = logistic_gradient(X, y, beta_true, alpha)
        g_numeric = self._numerical_gradient(X, y, beta_true, alpha)
        np.testing.assert_allclose(g_analytic, g_numeric, atol=1e-6, rtol=1e-4)

    def test_gradient_matches_fd_with_pen_weights(self, data):
        X, y, beta_true = data
        alpha = 0.1
        pen_weights = np.ones(X.shape[1])
        pen_weights[0] = 0.0  # simulate intercept
        g_analytic = logistic_gradient(X, y, beta_true, alpha, pen_weights)

        # Numeric with pen_weights
        g_numeric = np.zeros_like(beta_true)
        eps = 1e-5
        for j in range(len(beta_true)):
            e_j = np.zeros_like(beta_true)
            e_j[j] = eps
            f_plus = logistic_loss(X, y, beta_true + e_j, alpha, pen_weights)
            f_minus = logistic_loss(X, y, beta_true - e_j, alpha, pen_weights)
            g_numeric[j] = (f_plus - f_minus) / (2 * eps)

        np.testing.assert_allclose(g_analytic, g_numeric, atol=1e-6, rtol=1e-4)


class TestGradientProperties:
    def test_gradient_is_zero_at_optimum_approx(self):
        """After many iterations of our solver, the gradient should be small."""
        from ifsn_logistic.config import SafeSolverConfig
        from ifsn_logistic.solver_safe import ifsn_safe_solve

        X, y, _ = make_logistic_data(n=200, d=5, random_state=99)
        config = SafeSolverConfig(
            max_iter=200, tol=1e-10, C=1.0, random_state=0,
        )
        beta_opt, _ = ifsn_safe_solve(X, y, config)
        g = logistic_gradient(X, y, beta_opt, alpha=config.alpha)
        assert np.linalg.norm(g) < 1e-3

    def test_gradient_regularization_increases_norm(self, data):
        X, y, beta = data
        g0 = logistic_gradient(X, y, beta, alpha=0.0)
        g1 = logistic_gradient(X, y, beta, alpha=10.0)
        # With strong regularization and non-zero beta, gradient should differ
        assert not np.allclose(g0, g1)

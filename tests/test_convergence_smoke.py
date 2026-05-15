"""Convergence smoke tests."""

import numpy as np
import pytest

from ifsn_logistic.config import SafeSolverConfig, SolverConfig
from ifsn_logistic.datasets import make_logistic_data
from ifsn_logistic.solver_reference import ifsn_reference_solve
from ifsn_logistic.solver_safe import ifsn_safe_solve


class TestConvergenceSmoke:
    @pytest.mark.parametrize("subsample_size", [None, 80])
    @pytest.mark.parametrize("sampling_scheme", ["fresh", "fixed"])
    def test_reference_loss_improves(self, subsample_size, sampling_scheme):
        X, y, _ = make_logistic_data(n=200, d=8, random_state=0)
        config = SolverConfig(
            max_iter=40,
            C=1.0,
            subsample_size=subsample_size,
            sampling_scheme=sampling_scheme,
            random_state=0,
        )
        _, diag = ifsn_reference_solve(X, y, config)
        if subsample_size is None:
            assert diag["loss"][-1] < diag["loss"][0]
        else:
            assert diag["loss"][-1] < diag["loss"][0] * 1.10
        assert all(np.isfinite(v) for v in diag["loss"])

    @pytest.mark.parametrize("subsample_size", [None, 80])
    @pytest.mark.parametrize("sampling_scheme", ["fresh", "fixed"])
    def test_safe_loss_improves(self, subsample_size, sampling_scheme):
        X, y, _ = make_logistic_data(n=200, d=8, random_state=0)
        config = SafeSolverConfig(
            max_iter=40,
            C=1.0,
            subsample_size=subsample_size,
            sampling_scheme=sampling_scheme,
            random_state=0,
        )
        _, diag = ifsn_safe_solve(X, y, config)
        if subsample_size is None:
            assert diag["loss"][-1] < diag["loss"][0]
        else:
            assert diag["loss"][-1] <= diag["loss"][0] * 1.01
        assert all(np.isfinite(v) for v in diag["loss"])

    def test_reference_gradient_shrinks(self):
        X, y, _ = make_logistic_data(n=300, d=5, random_state=1)
        config = SolverConfig(
            max_iter=60, C=1.0, subsample_size=None, sampling_scheme="fresh", random_state=1
        )
        _, diag = ifsn_reference_solve(X, y, config)
        assert diag["grad_norm"][-1] < diag["grad_norm"][0]

    def test_safe_gradient_shrinks(self):
        X, y, _ = make_logistic_data(n=300, d=5, random_state=1)
        config = SafeSolverConfig(
            max_iter=60, C=1.0, subsample_size=None, sampling_scheme="fixed", random_state=1
        )
        _, diag = ifsn_safe_solve(X, y, config)
        assert diag["full_grad_norm"][-1] < diag["full_grad_norm"][0]

    @pytest.mark.slow
    def test_reference_close_to_sklearn(self):
        try:
            from sklearn.linear_model import LogisticRegression as sklearn_logistic
        except Exception as exc:
            pytest.skip(f"scikit-learn unavailable: {exc}")

        X, y, _ = make_logistic_data(n=500, d=8, random_state=5)
        C_sklearn = 1.0
        n = X.shape[0]
        config = SolverConfig(
            max_iter=300,
            C=n * C_sklearn,
            tol=1e-8,
            subsample_size=None,
            sampling_scheme="fresh",
            random_state=5,
        )
        beta_ifsn, _ = ifsn_reference_solve(X, y, config)

        clf = sklearn_logistic(C=C_sklearn, fit_intercept=False, max_iter=5000, tol=1e-10)
        clf.fit(X, y)
        beta_sklearn = clf.coef_.ravel()

        np.testing.assert_allclose(beta_ifsn, beta_sklearn, atol=0.05, rtol=0.05)

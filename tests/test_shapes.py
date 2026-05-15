"""Tests for output shapes and dtypes across all major functions."""

import numpy as np
import pytest

from ifsn_logistic.config import SafeSolverConfig, SolverConfig
from ifsn_logistic.datasets import make_logistic_data
from ifsn_logistic.estimator import IFSNLogisticRegression
from ifsn_logistic.losses import logistic_gradient, logistic_loss
from ifsn_logistic.math_utils import (
    curvature_weights,
    logistic_probabilities,
    sigmoid,
    subsampled_gradient,
)
from ifsn_logistic.solver_reference import build_inverse_hessian_sm, ifsn_reference_solve


@pytest.fixture
def data():
    X, y, beta_true = make_logistic_data(n=100, d=10, random_state=0)
    return X, y, beta_true


class TestMathShapes:
    def test_sigmoid_shape(self):
        z = np.random.randn(50)
        assert sigmoid(z).shape == (50,)

    def test_sigmoid_scalar(self):
        assert np.isscalar(sigmoid(np.float64(0.0))) or sigmoid(np.float64(0.0)).shape == ()

    def test_logistic_probabilities_shape(self, data):
        X, y, beta = data
        p = logistic_probabilities(X, beta)
        assert p.shape == (X.shape[0],)

    def test_curvature_weights_shape(self, data):
        X, _, beta = data
        p = logistic_probabilities(X, beta)
        q = curvature_weights(p)
        assert q.shape == p.shape

    def test_gradient_shape(self, data):
        X, y, beta = data
        g = logistic_gradient(X, y, beta, alpha=0.01)
        assert g.shape == (X.shape[1],)

    def test_subsampled_gradient_shape(self, data):
        X, y, beta = data
        p = logistic_probabilities(X, beta)
        g = subsampled_gradient(X, y, p, beta, alpha=0.01)
        assert g.shape == (X.shape[1],)

    def test_loss_is_scalar(self, data):
        X, y, beta = data
        loss = logistic_loss(X, y, beta, alpha=0.01)
        assert isinstance(loss, float)


class TestSolverShapes:
    def test_inverse_hessian_shape(self, data):
        X, y, beta = data
        p = logistic_probabilities(X, beta)
        q = curvature_weights(p)
        B, _ = build_inverse_hessian_sm(X, q, alpha=0.1, s=X.shape[0])
        assert B.shape == (X.shape[1], X.shape[1])

    def test_reference_solver_returns_correct_shape(self, data):
        X, y, _ = data
        config = SolverConfig(max_iter=5, C=1.0, random_state=0)
        beta, diag = ifsn_reference_solve(X, y, config)
        assert beta.shape == (X.shape[1],)
        assert isinstance(diag, dict)
        assert len(diag["loss"]) == 5 or diag["converged"]


class TestEstimatorShapes:
    def test_predict_proba_shape(self, data):
        X, y, _ = data
        clf = IFSNLogisticRegression(max_iter=10, random_state=0, fit_intercept=False, solver="reference")
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)

    def test_predict_shape(self, data):
        X, y, _ = data
        clf = IFSNLogisticRegression(max_iter=10, random_state=0, fit_intercept=False, solver="reference")
        clf.fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == (X.shape[0],)

    def test_decision_function_shape(self, data):
        X, y, _ = data
        clf = IFSNLogisticRegression(max_iter=10, random_state=0, fit_intercept=False, solver="reference")
        clf.fit(X, y)
        dec = clf.decision_function(X)
        assert dec.shape == (X.shape[0],)

    def test_coef_shape(self, data):
        X, y, _ = data
        clf = IFSNLogisticRegression(max_iter=10, random_state=0, fit_intercept=False, solver="reference")
        clf.fit(X, y)
        assert clf.coef_.shape == (1, X.shape[1])
        assert clf.intercept_.shape == (1,)

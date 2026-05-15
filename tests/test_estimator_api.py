"""Tests for the lightweight estimator API."""

import numpy as np
import pytest

from ifsn_logistic.datasets import make_logistic_data
from ifsn_logistic.estimator import IFSNLogisticRegression


@pytest.fixture
def data():
    return make_logistic_data(n=200, d=10, random_state=42)


class TestEstimatorBasicAPI:
    def test_fit_predict(self, data):
        X, y, _ = data
        clf = IFSNLogisticRegression(max_iter=30, random_state=0)
        clf.fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == (X.shape[0],)
        assert set(pred).issubset(set(clf.classes_))

    def test_predict_proba_sums_to_one(self, data):
        X, y, _ = data
        clf = IFSNLogisticRegression(max_iter=30, random_state=0)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_predict_proba_bounded(self, data):
        X, y, _ = data
        clf = IFSNLogisticRegression(max_iter=30, random_state=0)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_decision_function(self, data):
        X, y, _ = data
        clf = IFSNLogisticRegression(max_iter=30, random_state=0, fit_intercept=False)
        clf.fit(X, y)
        dec = clf.decision_function(X)
        expected = X @ clf.coef_.ravel() + clf.intercept_[0]
        np.testing.assert_allclose(dec, expected)

    def test_get_set_params(self):
        clf = IFSNLogisticRegression(C=2.0, max_iter=100, sampling_scheme="fixed")
        params = clf.get_params()
        assert params["C"] == 2.0
        assert params["max_iter"] == 100
        assert params["sampling_scheme"] == "fixed"

        clf.set_params(C=5.0, sampling_scheme="fresh")
        assert clf.C == 5.0
        assert clf.sampling_scheme == "fresh"

    def test_classes_attribute(self, data):
        X, y, _ = data
        clf = IFSNLogisticRegression(max_iter=10, random_state=0)
        clf.fit(X, y)
        assert len(clf.classes_) == 2

    def test_diagnostics_available(self, data):
        X, y, _ = data
        clf = IFSNLogisticRegression(max_iter=10, random_state=0, sampling_scheme="fixed")
        clf.fit(X, y)
        assert hasattr(clf, "diagnostics_")
        assert hasattr(clf, "n_iter_")
        assert hasattr(clf, "converged_")
        assert clf.diagnostics_["sampling_scheme"] == "fixed"

    def test_score(self, data):
        X, y, _ = data
        clf = IFSNLogisticRegression(max_iter=20, random_state=0)
        clf.fit(X, y)
        score = clf.score(X, y)
        assert 0.0 <= score <= 1.0


class TestEstimatorSolverVariants:
    @pytest.mark.parametrize("sampling_scheme", ["fresh", "fixed"])
    def test_reference_solver(self, data, sampling_scheme):
        X, y, _ = data
        clf = IFSNLogisticRegression(
            max_iter=20,
            solver="reference",
            sampling_scheme=sampling_scheme,
            random_state=0,
            fit_intercept=False,
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == (X.shape[0],)

    @pytest.mark.parametrize("sampling_scheme", ["fresh", "fixed"])
    def test_safe_solver(self, data, sampling_scheme):
        X, y, _ = data
        clf = IFSNLogisticRegression(
            max_iter=20,
            solver="safe",
            sampling_scheme=sampling_scheme,
            random_state=0,
            fit_intercept=False,
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == (X.shape[0],)

    def test_unknown_solver_raises(self, data):
        X, y, _ = data
        clf = IFSNLogisticRegression(solver="bogus")
        with pytest.raises(ValueError, match="Unknown solver"):
            clf.fit(X, y)

    def test_unknown_sampling_scheme_raises(self, data):
        X, y, _ = data
        clf = IFSNLogisticRegression(sampling_scheme="bogus")
        with pytest.raises(ValueError, match="sampling_scheme"):
            clf.fit(X, y)


class TestEstimatorIntercept:
    def test_fit_intercept_true(self, data):
        X, y, _ = data
        clf = IFSNLogisticRegression(max_iter=30, fit_intercept=True, random_state=0)
        clf.fit(X, y)
        assert clf.coef_.shape == (1, X.shape[1])
        assert clf.intercept_.shape == (1,)

    def test_fit_intercept_false(self, data):
        X, y, _ = data
        clf = IFSNLogisticRegression(max_iter=30, fit_intercept=False, random_state=0)
        clf.fit(X, y)
        assert clf.intercept_[0] == 0.0


class TestEstimatorMulticlassReject:
    def test_rejects_multiclass(self):
        X = np.random.randn(100, 5)
        y = np.array([0] * 33 + [1] * 33 + [2] * 34)
        clf = IFSNLogisticRegression(max_iter=10, random_state=0)
        with pytest.raises(ValueError, match="binary"):
            clf.fit(X, y)

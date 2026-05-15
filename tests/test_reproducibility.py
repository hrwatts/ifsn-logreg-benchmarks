"""Reproducibility tests."""

import numpy as np

from ifsn_logistic.config import SafeSolverConfig, SolverConfig
from ifsn_logistic.datasets import make_logistic_data
from ifsn_logistic.estimator import IFSNLogisticRegression
from ifsn_logistic.solver_reference import ifsn_reference_solve
from ifsn_logistic.solver_safe import ifsn_safe_solve


class TestReproducibilitySolvers:
    def test_reference_deterministic_fresh(self):
        X, y, _ = make_logistic_data(n=100, d=8, random_state=0)
        config1 = SolverConfig(
            max_iter=20, C=1.0, subsample_size=30, random_state=42, sampling_scheme="fresh"
        )
        config2 = SolverConfig(
            max_iter=20, C=1.0, subsample_size=30, random_state=42, sampling_scheme="fresh"
        )
        b1, d1 = ifsn_reference_solve(X, y, config1)
        b2, d2 = ifsn_reference_solve(X, y, config2)
        np.testing.assert_array_equal(b1, b2)
        np.testing.assert_array_equal(d1["loss"], d2["loss"])

    def test_reference_fixed_reuses_same_indices(self):
        X, y, _ = make_logistic_data(n=100, d=8, random_state=0)
        config = SolverConfig(
            max_iter=12, C=1.0, subsample_size=30, random_state=42, sampling_scheme="fixed"
        )
        _, diag = ifsn_reference_solve(X, y, config)
        first = diag["sample_indices_history"][0]
        for sample in diag["sample_indices_history"][1:]:
            np.testing.assert_array_equal(first, sample)

    def test_safe_deterministic_fixed(self):
        X, y, _ = make_logistic_data(n=100, d=8, random_state=0)
        config1 = SafeSolverConfig(
            max_iter=20, C=1.0, subsample_size=30, random_state=42, sampling_scheme="fixed"
        )
        config2 = SafeSolverConfig(
            max_iter=20, C=1.0, subsample_size=30, random_state=42, sampling_scheme="fixed"
        )
        b1, d1 = ifsn_safe_solve(X, y, config1)
        b2, d2 = ifsn_safe_solve(X, y, config2)
        np.testing.assert_array_equal(b1, b2)
        np.testing.assert_array_equal(d1["loss"], d2["loss"])

    def test_different_seeds_differ_for_fresh_sampling(self):
        X, y, _ = make_logistic_data(n=100, d=8, random_state=0)
        config1 = SolverConfig(
            max_iter=20, C=1.0, subsample_size=30, random_state=1, sampling_scheme="fresh"
        )
        config2 = SolverConfig(
            max_iter=20, C=1.0, subsample_size=30, random_state=2, sampling_scheme="fresh"
        )
        b1, _ = ifsn_reference_solve(X, y, config1)
        b2, _ = ifsn_reference_solve(X, y, config2)
        assert not np.allclose(b1, b2)


class TestReproducibilityEstimator:
    def test_estimator_deterministic_fixed(self):
        X, y, _ = make_logistic_data(n=100, d=8, random_state=0)
        clf1 = IFSNLogisticRegression(
            max_iter=20,
            subsample_size=30,
            sampling_scheme="fixed",
            random_state=42,
            fit_intercept=False,
        )
        clf2 = IFSNLogisticRegression(
            max_iter=20,
            subsample_size=30,
            sampling_scheme="fixed",
            random_state=42,
            fit_intercept=False,
        )
        clf1.fit(X, y)
        clf2.fit(X, y)
        np.testing.assert_array_equal(clf1.coef_, clf2.coef_)
        np.testing.assert_array_equal(
            clf1.diagnostics_["fixed_sample_indices"], clf2.diagnostics_["fixed_sample_indices"]
        )

    def test_dataset_generators_deterministic(self):
        X1, y1, b1 = make_logistic_data(n=50, d=5, random_state=99)
        X2, y2, b2 = make_logistic_data(n=50, d=5, random_state=99)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_array_equal(b1, b2)

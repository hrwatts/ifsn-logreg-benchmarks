"""Tests for sigmoid, curvature weights, and probability computations."""

import numpy as np
import pytest

from ifsn_logistic.math_utils import (
    clip_probabilities,
    curvature_weights,
    logistic_probabilities,
    sigmoid,
)


class TestSigmoid:
    def test_zero(self):
        assert sigmoid(np.float64(0.0)) == pytest.approx(0.5)

    def test_large_positive(self):
        val = sigmoid(np.float64(50.0))
        assert val == pytest.approx(1.0, abs=1e-10)

    def test_large_negative(self):
        val = sigmoid(np.float64(-50.0))
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_symmetry(self):
        z = np.linspace(-5, 5, 100)
        assert np.allclose(sigmoid(z) + sigmoid(-z), 1.0)

    def test_monotonic(self):
        z = np.linspace(-10, 10, 200)
        s = sigmoid(z)
        assert np.all(np.diff(s) > 0)

    def test_no_nans_extreme(self):
        z = np.array([-1000.0, -500.0, 0.0, 500.0, 1000.0])
        s = sigmoid(z)
        assert np.all(np.isfinite(s))

    def test_output_range(self):
        z = np.random.randn(1000)
        s = sigmoid(z)
        assert np.all(s > 0)
        assert np.all(s < 1)


class TestCurvatureWeights:
    def test_range(self):
        p = np.linspace(0.01, 0.99, 100)
        q = curvature_weights(p)
        assert np.all(q > 0)
        assert np.all(q <= 0.25)

    def test_max_at_half(self):
        p = np.array([0.5])
        q = curvature_weights(p)
        assert q[0] == pytest.approx(0.25)

    def test_symmetric(self):
        p = np.array([0.2, 0.8])
        q = curvature_weights(p)
        assert q[0] == pytest.approx(q[1])

    def test_zeros_at_boundary(self):
        p = np.array([0.0, 1.0])
        q = curvature_weights(p)
        assert q[0] == pytest.approx(0.0)
        assert q[1] == pytest.approx(0.0)


class TestLogisticProbabilities:
    def test_known_values(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        beta = np.array([0.0, 0.0])
        p = logistic_probabilities(X, beta)
        assert np.allclose(p, 0.5)

    def test_in_unit_interval(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 10))
        beta = rng.standard_normal(10)
        p = logistic_probabilities(X, beta)
        assert np.all(p > 0)
        assert np.all(p < 1)


class TestClipProbabilities:
    def test_clips_zeros(self):
        p = np.array([0.0, 0.5, 1.0])
        pc = clip_probabilities(p)
        assert pc[0] > 0
        assert pc[2] < 1
        assert pc[1] == pytest.approx(0.5)

    def test_custom_eps(self):
        p = np.array([0.0, 1.0])
        pc = clip_probabilities(p, eps=0.01)
        assert pc[0] == pytest.approx(0.01)
        assert pc[1] == pytest.approx(0.99)

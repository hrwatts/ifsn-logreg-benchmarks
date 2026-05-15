"""Tests for subsampling utilities."""

import numpy as np
import pytest

from ifsn_logistic.sampling import (
    stratified_subsample_indices,
    subsample_data,
    subsample_indices,
)


class TestSubsampleIndices:
    def test_returns_all_when_none(self):
        idx = subsample_indices(100, None, np.random.default_rng(0))
        np.testing.assert_array_equal(idx, np.arange(100))

    def test_returns_all_when_size_exceeds_n(self):
        idx = subsample_indices(50, 200, np.random.default_rng(0))
        np.testing.assert_array_equal(idx, np.arange(50))

    def test_correct_size(self):
        idx = subsample_indices(100, 30, np.random.default_rng(0))
        assert len(idx) == 30

    def test_no_duplicates(self):
        idx = subsample_indices(1000, 500, np.random.default_rng(42))
        assert len(np.unique(idx)) == 500

    def test_sorted(self):
        idx = subsample_indices(200, 50, np.random.default_rng(1))
        assert np.all(np.diff(idx) > 0)

    def test_within_range(self):
        idx = subsample_indices(100, 30, np.random.default_rng(0))
        assert np.all(idx >= 0)
        assert np.all(idx < 100)

    def test_deterministic(self):
        idx1 = subsample_indices(100, 20, np.random.default_rng(99))
        idx2 = subsample_indices(100, 20, np.random.default_rng(99))
        np.testing.assert_array_equal(idx1, idx2)


class TestSubsampleData:
    def test_shapes(self):
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100).astype(float)
        X_S, y_S, idx = subsample_data(X, y, 30, np.random.default_rng(0))
        assert X_S.shape == (30, 10)
        assert y_S.shape == (30,)
        assert idx.shape == (30,)

    def test_data_matches_indices(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))
        y = rng.integers(0, 2, 50).astype(float)
        X_S, y_S, idx = subsample_data(X, y, 20, np.random.default_rng(42))
        np.testing.assert_array_equal(X_S, X[idx])
        np.testing.assert_array_equal(y_S, y[idx])


class TestStratifiedSubsample:
    def test_preserves_class_proportions(self):
        rng = np.random.default_rng(0)
        y = np.array([0.0] * 90 + [1.0] * 10)  # 10% positive
        idx = stratified_subsample_indices(y, 20, rng)
        y_sub = y[idx]
        # Expect roughly 2 positives out of 20
        n_pos = np.sum(y_sub == 1)
        assert 1 <= n_pos <= 5  # loose bounds

    def test_returns_all_when_none(self):
        y = np.ones(30)
        idx = stratified_subsample_indices(y, None, np.random.default_rng(0))
        np.testing.assert_array_equal(idx, np.arange(30))

    def test_both_classes_present(self):
        rng = np.random.default_rng(7)
        y = np.array([0.0] * 80 + [1.0] * 20)
        idx = stratified_subsample_indices(y, 10, rng)
        y_sub = y[idx]
        assert 0.0 in y_sub
        assert 1.0 in y_sub

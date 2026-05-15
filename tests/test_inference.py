from __future__ import annotations

import numpy as np

from ifsn_logistic.benchmarks.inference import (
    bootstrap_confidence_interval,
    cohens_d_paired,
    paired_direction,
    paired_sign_test_two_sided,
)


def test_bootstrap_ci_is_deterministic_for_fixed_seed():
    values = np.array([0.10, 0.20, 0.15, 0.12, 0.18], dtype=np.float64)
    lo_a, hi_a = bootstrap_confidence_interval(values, n_bootstrap=1000, random_state=3)
    lo_b, hi_b = bootstrap_confidence_interval(values, n_bootstrap=1000, random_state=3)
    assert lo_a == lo_b
    assert hi_a == hi_b
    assert lo_a <= values.mean() <= hi_a


def test_cohens_d_paired_zero_for_equal_series():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    assert cohens_d_paired(x, y) == 0.0


def test_sign_test_small_p_for_consistent_direction():
    method = np.array([0.10, 0.11, 0.09, 0.10, 0.08, 0.09], dtype=np.float64)
    baseline = np.array([0.20, 0.22, 0.19, 0.21, 0.18, 0.20], dtype=np.float64)
    p_value = paired_sign_test_two_sided(method, baseline)
    assert p_value <= 0.05


def test_paired_direction_respects_lower_is_better():
    method = np.array([0.10, 0.11, 0.09, 0.10, 0.08], dtype=np.float64)
    baseline = np.array([0.20, 0.22, 0.19, 0.21, 0.18], dtype=np.float64)
    direction = paired_direction(method, baseline, lower_is_better=True, p_value=0.01)
    assert direction == "ifsn_better_p<0.05"
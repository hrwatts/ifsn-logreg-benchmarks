from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ifsn_logistic.benchmarks import data as benchmark_data
from ifsn_logistic.benchmarks.data import (
    BenchmarkDataset,
    MUSHROOM_COLUMNS,
    _parse_mushroom_csv,
    dataset_manifest,
    require_benchmark_stack,
)
from ifsn_logistic.benchmarks.preprocess import DenseOneHotEncoder
from ifsn_logistic.benchmarks.reporting import (
    aggregate_runs,
    aggregate_runs_with_inference,
    render_inference_markdown,
    render_summary_markdown,
)
from ifsn_logistic.benchmarks.run import _prepare_splits


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def test_mushroom_manifest_is_pinned():
    manifest = dataset_manifest()["mushroom"]
    assert manifest.source == "UCI Machine Learning Repository"
    assert manifest.license == "CC BY 4.0"
    assert manifest.source_url == "https://archive.ics.uci.edu/static/public/73/mushroom.zip"
    assert manifest.sha256 == "FACE32F32647E0D939F6233F36DD30DD5D619AE9F3F9B8E10BEA4AC7E1F60B1A"
    assert manifest.preprocess_mode == "categorical_dense_onehot_standardize"


def test_parse_mushroom_fixture_and_label_mapping():
    with open(FIXTURE_DIR / "mushroom_sample.csv", encoding="utf-8") as handle:
        X, y = _parse_mushroom_csv(handle)

    assert X.shape == (6, 22)
    assert y.tolist() == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert X[3, 10] == "?"


def test_dense_one_hot_reproducible_and_ignores_unknowns():
    with open(FIXTURE_DIR / "mushroom_sample.csv", encoding="utf-8") as handle:
        X, _ = _parse_mushroom_csv(handle)

    encoder = DenseOneHotEncoder().fit(X[:4], MUSHROOM_COLUMNS)
    transformed_a = encoder.transform(X[:4])
    transformed_b = encoder.transform(X[:4])
    np.testing.assert_allclose(transformed_a, transformed_b)

    unknown_row = X[[0]].copy()
    unknown_row[0, 0] = "not-a-category"
    transformed_unknown = encoder.transform(unknown_row)
    first_block_width = len(encoder.categories_[0])
    assert np.all(transformed_unknown[0, :first_block_width] == 0.0)


def test_prepare_splits_encodes_to_dense_numeric_arrays():
    with open(FIXTURE_DIR / "mushroom_sample.csv", encoding="utf-8") as handle:
        X, y = _parse_mushroom_csv(handle)

    dataset = BenchmarkDataset(
        name="mushroom",
        X=X,
        y=y,
        feature_names=MUSHROOM_COLUMNS.copy(),
        preprocess_mode="categorical_dense_onehot_standardize",
        metadata={},
    )
    train_idx = np.array([0, 1, 2])
    val_idx = np.array([3])
    test_idx = np.array([4, 5])

    X_train, X_val, X_test, meta = _prepare_splits(dataset, train_idx, val_idx, test_idx)
    assert X_train.dtype == np.float64
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    assert meta["encoded_feature_count"] == X_train.shape[1]


def test_require_benchmark_stack_raises_clear_error(monkeypatch):
    monkeypatch.setattr(benchmark_data, "benchmark_stack_status", lambda: (False, "boom"))
    with pytest.raises(RuntimeError, match="benchmark workflow requires"):
        require_benchmark_stack()


def test_aggregate_runs_and_markdown_summary():
    rows = [
        {
            "dataset": "mushroom",
            "method": "ifsn-safe-fixed-s64",
            "test_log_loss": 0.10,
            "test_roc_auc": 0.99,
            "test_accuracy": 0.98,
            "selection_runtime_sec": 0.5,
            "n_iter": 12,
            "converged": 1,
        },
        {
            "dataset": "mushroom",
            "method": "ifsn-safe-fixed-s64",
            "test_log_loss": 0.20,
            "test_roc_auc": 0.97,
            "test_accuracy": 0.96,
            "selection_runtime_sec": 0.7,
            "n_iter": 10,
            "converged": 1,
        },
    ]
    summary = aggregate_runs(rows)
    assert len(summary) == 1
    assert summary[0]["runs"] == 2
    rendered = render_summary_markdown(summary)
    assert "ifsn-safe-fixed-s64" in rendered
    assert "mushroom" in rendered


def test_aggregate_with_inference_and_markdown():
    rows = [
        {
            "dataset": "mushroom",
            "method": "sklearn-lbfgs",
            "seed": 0,
            "test_log_loss": 0.12,
            "test_roc_auc": 0.99,
            "test_accuracy": 0.98,
            "selection_runtime_sec": 0.3,
            "converged": 1,
        },
        {
            "dataset": "mushroom",
            "method": "sklearn-lbfgs",
            "seed": 1,
            "test_log_loss": 0.11,
            "test_roc_auc": 0.99,
            "test_accuracy": 0.98,
            "selection_runtime_sec": 0.4,
            "converged": 1,
        },
        {
            "dataset": "mushroom",
            "method": "ifsn-safe-fixed-s64",
            "seed": 0,
            "test_log_loss": 0.10,
            "test_roc_auc": 0.99,
            "test_accuracy": 0.98,
            "selection_runtime_sec": 0.5,
            "converged": 1,
        },
        {
            "dataset": "mushroom",
            "method": "ifsn-safe-fixed-s64",
            "seed": 1,
            "test_log_loss": 0.09,
            "test_roc_auc": 0.99,
            "test_accuracy": 0.98,
            "selection_runtime_sec": 0.6,
            "converged": 1,
        },
    ]
    summary = aggregate_runs_with_inference(rows, n_bootstrap=500, random_state=7)
    assert len(summary) == 2
    ifsn_row = next(r for r in summary if r["method"] == "ifsn-safe-fixed-s64")
    assert ifsn_row["ci_lo_log_loss"] <= ifsn_row["mean_log_loss"] <= ifsn_row["ci_hi_log_loss"]
    assert ifsn_row["vs_baseline_log_loss_direction"] in {
        "ifsn_better_p<0.05",
        "equal_p>=0.05",
        "baseline_better_p<0.05",
    }
    rendered = render_inference_markdown(summary)
    assert "Log-loss (95% CI)" in rendered
    assert "ifsn-safe-fixed-s64" in rendered

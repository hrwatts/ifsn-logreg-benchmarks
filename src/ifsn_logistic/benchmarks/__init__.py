"""Benchmark helpers for reproducible estimator comparisons."""

from ifsn_logistic.benchmarks.data import (
    BenchmarkDataset,
    DatasetManifestEntry,
    dataset_manifest,
    load_benchmark_dataset,
    require_benchmark_stack,
)

__all__ = [
    "BenchmarkDataset",
    "DatasetManifestEntry",
    "dataset_manifest",
    "load_benchmark_dataset",
    "require_benchmark_stack",
]

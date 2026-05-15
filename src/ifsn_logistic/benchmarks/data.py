"""Dataset manifest and loaders for benchmark suites."""

from __future__ import annotations

import csv
import hashlib
import importlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


MUSHROOM_COLUMNS = [
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]


@dataclass(frozen=True)
class DatasetManifestEntry:
    """Metadata needed to fetch and audit one benchmark dataset."""

    name: str
    source: str
    source_url: str | None
    citation: str
    license: str
    sha256: str | None
    cache_name: str | None
    parser_id: str
    target_column: str
    preprocess_mode: str
    description: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class BenchmarkDataset:
    """Loaded dataset and metadata consumed by the benchmark runner."""

    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    preprocess_mode: str
    metadata: dict[str, object]


def dataset_manifest() -> dict[str, DatasetManifestEntry]:
    """Return the benchmark dataset manifest keyed by dataset name."""
    return {
        "breast_cancer": DatasetManifestEntry(
            name="breast_cancer",
            source="scikit-learn built-in breast cancer dataset",
            source_url=None,
            citation=(
                "Breast Cancer Wisconsin (Diagnostic) [Dataset]. "
                "UCI Machine Learning Repository."
            ),
            license="See scikit-learn / UCI distribution terms.",
            sha256=None,
            cache_name=None,
            parser_id="sklearn_breast_cancer",
            target_column="target",
            preprocess_mode="numeric_standardize",
            description="Local smoke benchmark on a well-known binary classification task.",
            metadata={"task": "binary classification", "suite": "smoke,full"},
        ),
        "mushroom": DatasetManifestEntry(
            name="mushroom",
            source="UCI Machine Learning Repository",
            source_url="https://archive.ics.uci.edu/static/public/73/mushroom.zip",
            citation=(
                "Mushroom [Dataset]. (1981). UCI Machine Learning Repository. "
                "https://doi.org/10.24432/C5959T."
            ),
            license="CC BY 4.0",
            sha256="FACE32F32647E0D939F6233F36DD30DD5D619AE9F3F9B8E10BEA4AC7E1F60B1A",
            cache_name="mushroom.zip",
            parser_id="uci_mushroom_zip",
            target_column="class",
            preprocess_mode="categorical_dense_onehot_standardize",
            description=(
                "Categorical edibility classification with one missing-value code "
                "treated as its own category."
            ),
            metadata={"task": "binary classification", "suite": "full"},
        ),
    }


def benchmark_stack_status() -> tuple[bool, str]:
    """Return whether the pinned benchmark dependency stack is importable."""
    try:
        importlib.import_module("sklearn")
        importlib.import_module("scipy")
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, str(exc)
    return True, ""


def require_benchmark_stack() -> None:
    """Raise a clear error when the benchmark stack is unavailable."""
    ok, detail = benchmark_stack_status()
    if not ok:
        raise RuntimeError(
            "The benchmark workflow requires a compatible benchmark stack "
            "(scikit-learn + SciPy). Install the benchmark extra or use the "
            "local pinned dependency folder before running official benchmarks. "
            f"Import detail: {detail}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _download_with_checksum(entry: DatasetManifestEntry, cache_dir: Path) -> Path:
    if entry.source_url is None or entry.cache_name is None or entry.sha256 is None:
        raise ValueError(f"Dataset {entry.name!r} does not define a downloadable archive.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / entry.cache_name
    if not target.exists():
        urllib.request.urlretrieve(entry.source_url, target)

    observed = _sha256(target)
    if observed != entry.sha256.upper():
        raise RuntimeError(
            f"Checksum mismatch for {entry.name}: expected {entry.sha256}, observed {observed}."
        )
    return target


def _load_breast_cancer_dataset() -> BenchmarkDataset:
    require_benchmark_stack()
    from sklearn.datasets import load_breast_cancer

    bunch = load_breast_cancer()
    return BenchmarkDataset(
        name="breast_cancer",
        X=np.asarray(bunch.data, dtype=np.float64),
        y=np.asarray(bunch.target, dtype=np.float64),
        feature_names=list(bunch.feature_names),
        preprocess_mode="numeric_standardize",
        metadata={
            "n_samples": int(bunch.data.shape[0]),
            "raw_feature_count": int(bunch.data.shape[1]),
        },
    )


def _parse_mushroom_csv(handle: io.TextIOBase) -> tuple[np.ndarray, np.ndarray]:
    reader = csv.reader(handle)
    rows: list[list[str]] = []
    labels: list[float] = []
    label_map = {"e": 0.0, "p": 1.0}
    for row in reader:
        if not row:
            continue
        labels.append(label_map[row[0]])
        rows.append(row[1:])

    return np.asarray(rows, dtype=object), np.asarray(labels, dtype=np.float64)


def _load_mushroom_dataset(cache_dir: Path) -> BenchmarkDataset:
    entry = dataset_manifest()["mushroom"]
    archive_path = _download_with_checksum(entry, cache_dir)
    with zipfile.ZipFile(archive_path) as zf:
        with zf.open("agaricus-lepiota.data") as raw_handle:
            with io.TextIOWrapper(raw_handle, encoding="utf-8") as text_handle:
                X, y = _parse_mushroom_csv(text_handle)

    return BenchmarkDataset(
        name="mushroom",
        X=X,
        y=y,
        feature_names=MUSHROOM_COLUMNS.copy(),
        preprocess_mode="categorical_dense_onehot_standardize",
        metadata={
            "n_samples": int(X.shape[0]),
            "raw_feature_count": int(X.shape[1]),
            "missing_value_token": "?",
        },
    )


def load_benchmark_dataset(name: str, cache_dir: Path) -> BenchmarkDataset:
    """Load one dataset by manifest name."""
    if name == "breast_cancer":
        return _load_breast_cancer_dataset()
    if name == "mushroom":
        return _load_mushroom_dataset(cache_dir)
    raise KeyError(f"Unknown benchmark dataset {name!r}.")

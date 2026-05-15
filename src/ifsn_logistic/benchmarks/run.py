"""CLI for the reproducible benchmark suites."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ifsn_logistic.benchmarks.data import (
    BenchmarkDataset,
    dataset_manifest,
    load_benchmark_dataset,
    require_benchmark_stack,
)
from ifsn_logistic.benchmarks.env import RuntimeEnvironment
from ifsn_logistic.benchmarks.preprocess import DenseOneHotEncoder, DenseStandardScaler
from ifsn_logistic.benchmarks.reporting import (
    INFERENCE_FIELDS,
    SUMMARY_FIELDS,
    aggregate_runs,
    aggregate_runs_with_inference,
    render_inference_markdown,
    render_summary_markdown,
    write_csv,
)
from ifsn_logistic.estimator import IFSNLogisticRegression


RUN_FIELDS = [
    "suite",
    "dataset",
    "seed",
    "method",
    "family",
    "solver_variant",
    "sampling_scheme",
    "subsample_size",
    "selected_c",
    "encoded_feature_count",
    "val_log_loss",
    "test_log_loss",
    "test_roc_auc",
    "test_accuracy",
    "selection_runtime_sec",
    "best_fit_runtime_sec",
    "n_iter",
    "converged",
    "solver_seed",
    "c_normalization_note",
    "timestamp_utc",
    "python_version",
    "numpy_version",
    "sklearn_version",
]


@dataclass(frozen=True)
class MethodConfig:
    """One benchmarked estimator configuration family."""

    name: str
    family: str
    solver_variant: str
    sampling_scheme: str | None
    subsample_size: int | None
    sklearn_solver: str | None


def _stratified_train_val_test_split(
    y: np.ndarray,
    seed: int,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for cls in sorted(np.unique(y).tolist()):
        cls_idx = np.flatnonzero(y == cls)
        perm = rng.permutation(cls_idx)
        n_train = int(math.floor(len(perm) * train_frac))
        n_val = int(math.floor(len(perm) * val_frac))
        train_parts.append(perm[:n_train])
        val_parts.append(perm[n_train : n_train + n_val])
        test_parts.append(perm[n_train + n_val :])

    train_idx = np.sort(np.concatenate(train_parts))
    val_idx = np.sort(np.concatenate(val_parts))
    test_idx = np.sort(np.concatenate(test_parts))
    return train_idx, val_idx, test_idx


def _binary_log_loss(y_true: np.ndarray, prob_pos: np.ndarray) -> float:
    prob = np.clip(prob_pos, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y_true * np.log(prob) + (1.0 - y_true) * np.log(1.0 - prob)))


def _binary_accuracy(y_true: np.ndarray, prob_pos: np.ndarray) -> float:
    pred = (prob_pos >= 0.5).astype(np.float64)
    return float(np.mean(pred == y_true))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _roc_auc(y_true: np.ndarray, prob_pos: np.ndarray) -> float:
    pos = y_true == 1.0
    neg = y_true == 0.0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(prob_pos)
    rank_sum_pos = float(ranks[pos].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _prepare_splits(
    dataset: BenchmarkDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    X_train = dataset.X[train_idx]
    X_val = dataset.X[val_idx]
    X_test = dataset.X[test_idx]

    metadata: dict[str, Any] = {"raw_feature_count": dataset.X.shape[1]}
    if dataset.preprocess_mode == "numeric_standardize":
        scaler = DenseStandardScaler().fit(np.asarray(X_train, dtype=np.float64))
        X_train_p = scaler.transform(np.asarray(X_train, dtype=np.float64))
        X_val_p = scaler.transform(np.asarray(X_val, dtype=np.float64))
        X_test_p = scaler.transform(np.asarray(X_test, dtype=np.float64))
        metadata["encoded_feature_count"] = int(X_train_p.shape[1])
        return X_train_p, X_val_p, X_test_p, metadata

    if dataset.preprocess_mode == "categorical_dense_onehot_standardize":
        encoder = DenseOneHotEncoder().fit(np.asarray(X_train, dtype=object), dataset.feature_names)
        X_train_enc = encoder.transform(np.asarray(X_train, dtype=object))
        X_val_enc = encoder.transform(np.asarray(X_val, dtype=object))
        X_test_enc = encoder.transform(np.asarray(X_test, dtype=object))
        scaler = DenseStandardScaler().fit(X_train_enc)
        X_train_p = scaler.transform(X_train_enc)
        X_val_p = scaler.transform(X_val_enc)
        X_test_p = scaler.transform(X_test_enc)
        metadata["encoded_feature_count"] = int(X_train_p.shape[1])
        metadata["encoded_feature_names"] = encoder.feature_names_
        return X_train_p, X_val_p, X_test_p, metadata

    raise ValueError(f"Unsupported preprocess mode {dataset.preprocess_mode!r}.")


def _smoke_methods() -> list[MethodConfig]:
    return [
        MethodConfig(
            name="sklearn-lbfgs",
            family="sklearn",
            solver_variant="lbfgs",
            sampling_scheme=None,
            subsample_size=None,
            sklearn_solver="lbfgs",
        ),
        MethodConfig(
            name="ifsn-safe-fixed-s64",
            family="ifsn",
            solver_variant="safe",
            sampling_scheme="fixed",
            subsample_size=64,
            sklearn_solver=None,
        ),
        MethodConfig(
            name="ifsn-safe-fresh-s64",
            family="ifsn",
            solver_variant="safe",
            sampling_scheme="fresh",
            subsample_size=64,
            sklearn_solver=None,
        ),
    ]


def _full_methods() -> list[MethodConfig]:
    methods: list[MethodConfig] = []
    for solver_name in ["lbfgs", "liblinear", "saga"]:
        methods.append(
            MethodConfig(
                name=f"sklearn-{solver_name}",
                family="sklearn",
                solver_variant=solver_name,
                sampling_scheme=None,
                subsample_size=None,
                sklearn_solver=solver_name,
            )
        )

    for solver_variant in ["safe", "reference"]:
        for sampling_scheme in ["fixed", "fresh"]:
            for subsample_size in [32, 64, 128, None]:
                size_label = "full" if subsample_size is None else f"s{subsample_size}"
                methods.append(
                    MethodConfig(
                        name=f"ifsn-{solver_variant}-{sampling_scheme}-{size_label}",
                        family="ifsn",
                        solver_variant=solver_variant,
                        sampling_scheme=sampling_scheme,
                        subsample_size=subsample_size,
                        sklearn_solver=None,
                    )
                )
    return methods


def _fit_sklearn_model(
    solver_name: str,
    C_value: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
):
    require_benchmark_stack()
    from sklearn.linear_model import LogisticRegression

    adjusted_c = C_value / X_train.shape[0]
    clf = LogisticRegression(
        solver=solver_name,
        C=adjusted_c,
        max_iter=2000,
        tol=1e-8,
        fit_intercept=True,
        random_state=random_state,
    )
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    return clf, fit_time, adjusted_c, "sklearn_div_n"


def _fit_ifsn_model(
    method: MethodConfig,
    C_value: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
):
    solver_loggers = [
        logging.getLogger("ifsn_logistic.solver_safe"),
        logging.getLogger("ifsn_logistic.solver_reference"),
    ]
    old_levels = [logger.level for logger in solver_loggers]
    for logger in solver_loggers:
        logger.setLevel(logging.ERROR)

    clf = IFSNLogisticRegression(
        C=C_value,
        max_iter=200,
        tol=1e-6,
        subsample_size=method.subsample_size,
        sampling_scheme=method.sampling_scheme or "fresh",
        fit_intercept=True,
        solver=method.solver_variant,
        random_state=random_state,
        verbose=0,
    )
    try:
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0
    finally:
        for logger, old_level in zip(solver_loggers, old_levels):
            logger.setLevel(old_level)
    return clf, fit_time, C_value, "ifsn_raw"


def _fit_method_for_c(
    method: MethodConfig,
    c_value: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
):
    if method.family == "sklearn":
        return _fit_sklearn_model(
            method.sklearn_solver or "lbfgs",
            c_value,
            X_train,
            y_train,
            random_state,
        )
    return _fit_ifsn_model(method, c_value, X_train, y_train, random_state)


def _derive_solver_seed(seed: int, dataset_index: int, solver_seed_offset: int) -> int:
    return int(solver_seed_offset + dataset_index * 1000 + seed)


def _predict_positive_proba(model: Any, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    return np.asarray(proba[:, 1], dtype=np.float64)


def _n_iter(model: Any) -> int:
    if hasattr(model, "n_iter_"):
        value = getattr(model, "n_iter_")
        if isinstance(value, np.ndarray):
            return int(np.max(value))
        return int(value)
    return -1


def _converged(model: Any) -> bool:
    if hasattr(model, "converged_"):
        return bool(getattr(model, "converged_"))
    if hasattr(model, "n_iter_"):
        value = getattr(model, "n_iter_")
        if isinstance(value, np.ndarray):
            return bool(np.max(value) < 2000)
        return bool(value < 2000)
    return False


def _evaluate_method(
    suite: str,
    dataset: BenchmarkDataset,
    method: MethodConfig,
    seed: int,
    solver_seed: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    c_grid: list[float],
    env_info: dict[str, object],
) -> dict[str, object]:
    X_train, X_val, X_test, preprocess_meta = _prepare_splits(dataset, train_idx, val_idx, test_idx)
    y_train = dataset.y[train_idx]
    y_val = dataset.y[val_idx]
    y_test = dataset.y[test_idx]

    best: dict[str, object] | None = None
    total_runtime = 0.0
    for c_value in c_grid:
        model, fit_time, selected_c, c_norm_note = _fit_method_for_c(
            method,
            c_value,
            X_train,
            y_train,
            solver_seed,
        )
        total_runtime += fit_time
        val_prob = _predict_positive_proba(model, X_val)
        val_loss = _binary_log_loss(y_val, val_prob)
        if best is None or val_loss < float(best["val_log_loss"]):
            best = {
                "model": model,
                "selected_c": selected_c,
                "c_normalization_note": c_norm_note,
                "val_log_loss": val_loss,
                "best_fit_runtime_sec": fit_time,
            }

    assert best is not None
    model = best["model"]
    test_prob = _predict_positive_proba(model, X_test)
    return {
        "suite": suite,
        "dataset": dataset.name,
        "seed": seed,
        "method": method.name,
        "family": method.family,
        "solver_variant": method.solver_variant,
        "sampling_scheme": method.sampling_scheme or "",
        "subsample_size": "" if method.subsample_size is None else method.subsample_size,
        "selected_c": best["selected_c"],
        "encoded_feature_count": preprocess_meta["encoded_feature_count"],
        "val_log_loss": best["val_log_loss"],
        "test_log_loss": _binary_log_loss(y_test, test_prob),
        "test_roc_auc": _roc_auc(y_test, test_prob),
        "test_accuracy": _binary_accuracy(y_test, test_prob),
        "selection_runtime_sec": total_runtime,
        "best_fit_runtime_sec": best["best_fit_runtime_sec"],
        "n_iter": _n_iter(model),
        "converged": int(_converged(model)),
        "solver_seed": solver_seed,
        "c_normalization_note": best["c_normalization_note"],
        "timestamp_utc": env_info.get("timestamp_utc", ""),
        "python_version": env_info.get("python_version", ""),
        "numpy_version": env_info.get("numpy_version", ""),
        "sklearn_version": env_info.get("sklearn_version", ""),
    }


def _run_suite(
    suite: str,
    datasets: list[str],
    methods: list[MethodConfig],
    cache_dir: Path,
    c_grid: list[float],
    seeds: list[int],
    solver_seed_offset: int,
    env_info: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_index, dataset_name in enumerate(datasets):
        dataset = load_benchmark_dataset(dataset_name, cache_dir)
        for seed in seeds:
            train_idx, val_idx, test_idx = _stratified_train_val_test_split(dataset.y, seed)
            solver_seed = _derive_solver_seed(seed, dataset_index, solver_seed_offset)
            for method in methods:
                rows.append(
                    _evaluate_method(
                        suite,
                        dataset,
                        method,
                        seed,
                        solver_seed,
                        train_idx,
                        val_idx,
                        test_idx,
                        c_grid,
                        env_info,
                    )
                )
    return rows


def _write_dataset_manifest_snapshot(path: Path) -> None:
    manifest_rows = {}
    for name, entry in dataset_manifest().items():
        manifest_rows[name] = {
            "source": entry.source,
            "source_url": entry.source_url,
            "citation": entry.citation,
            "license": entry.license,
            "sha256": entry.sha256,
            "parser_id": entry.parser_id,
            "target_column": entry.target_column,
            "preprocess_mode": entry.preprocess_mode,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest_rows, handle, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the reproducible benchmark suites.")
    parser.add_argument("--suite", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--c-grid", nargs="*", type=float, default=[0.1, 1.0, 10.0])
    parser.add_argument("--solver-seed-offset", type=int, default=1000)
    parser.add_argument("--include-inference", action="store_true")
    parser.add_argument("--inference-n-bootstrap", type=int, default=2000)
    parser.add_argument(
        "--save-environment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write environment.json alongside benchmark outputs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    require_benchmark_stack()

    suite = args.suite
    cache_dir = Path(args.cache_dir)
    results_dir = Path(args.results_dir)

    if suite == "smoke":
        datasets = ["breast_cancer"]
        methods = _smoke_methods()
    else:
        datasets = ["breast_cancer", "mushroom"]
        methods = _full_methods()

    env_info = RuntimeEnvironment.capture().to_dict()

    rows = _run_suite(
        suite,
        datasets,
        methods,
        cache_dir,
        list(args.c_grid),
        list(args.seeds),
        int(args.solver_seed_offset),
        env_info,
    )
    summary_rows = aggregate_runs(rows)

    suite_dir = results_dir / suite
    write_csv(suite_dir / "runs.csv", rows, RUN_FIELDS)
    write_csv(suite_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    (suite_dir / "summary.md").write_text(render_summary_markdown(summary_rows), encoding="utf-8")

    if args.include_inference:
        inference_rows = aggregate_runs_with_inference(
            rows,
            n_bootstrap=int(args.inference_n_bootstrap),
            random_state=0,
        )
        write_csv(suite_dir / "summary_inference.csv", inference_rows, INFERENCE_FIELDS)
        (suite_dir / "summary_inference.md").write_text(
            render_inference_markdown(inference_rows),
            encoding="utf-8",
        )

    if args.save_environment:
        (suite_dir / "environment.json").write_text(
            json.dumps(env_info, indent=2),
            encoding="utf-8",
        )

    _write_dataset_manifest_snapshot(suite_dir / "dataset_manifest.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

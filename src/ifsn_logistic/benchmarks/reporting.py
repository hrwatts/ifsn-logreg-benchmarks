"""Benchmark result aggregation and rendering."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from ifsn_logistic.benchmarks.inference import (
    bootstrap_confidence_interval,
    cohens_d_paired,
    paired_direction,
    paired_sign_test_two_sided,
)


SUMMARY_FIELDS = [
    "dataset",
    "method",
    "runs",
    "mean_log_loss",
    "std_log_loss",
    "mean_roc_auc",
    "std_roc_auc",
    "mean_accuracy",
    "std_accuracy",
    "mean_runtime_sec",
    "std_runtime_sec",
    "mean_n_iter",
    "std_n_iter",
    "convergence_rate",
]

INFERENCE_FIELDS = [
    "dataset",
    "method",
    "runs",
    "mean_log_loss",
    "std_log_loss",
    "ci_lo_log_loss",
    "ci_hi_log_loss",
    "mean_roc_auc",
    "std_roc_auc",
    "ci_lo_roc_auc",
    "ci_hi_roc_auc",
    "mean_accuracy",
    "std_accuracy",
    "ci_lo_accuracy",
    "ci_hi_accuracy",
    "mean_runtime_sec",
    "std_runtime_sec",
    "convergence_rate",
    "vs_baseline_log_loss_mean_diff",
    "vs_baseline_log_loss_p_value",
    "vs_baseline_log_loss_effect_size",
    "vs_baseline_log_loss_direction",
]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) == 1:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def aggregate_runs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate per-seed run rows into dataset/method summaries."""
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["method"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for (dataset, method), items in sorted(grouped.items()):
        log_losses = [float(item["test_log_loss"]) for item in items]
        roc_aucs = [float(item["test_roc_auc"]) for item in items]
        accuracies = [float(item["test_accuracy"]) for item in items]
        runtimes = [float(item["selection_runtime_sec"]) for item in items]
        niters = [float(item["n_iter"]) for item in items]
        converged = [float(item["converged"]) for item in items]
        summary_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "runs": len(items),
                "mean_log_loss": _mean(log_losses),
                "std_log_loss": _std(log_losses),
                "mean_roc_auc": _mean(roc_aucs),
                "std_roc_auc": _std(roc_aucs),
                "mean_accuracy": _mean(accuracies),
                "std_accuracy": _std(accuracies),
                "mean_runtime_sec": _mean(runtimes),
                "std_runtime_sec": _std(runtimes),
                "mean_n_iter": _mean(niters),
                "std_n_iter": _std(niters),
                "convergence_rate": _mean(converged),
            }
        )
    return summary_rows


def aggregate_runs_with_inference(
    rows: list[dict[str, object]],
    baseline_method: str = "sklearn-lbfgs",
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    random_state: int = 0,
) -> list[dict[str, object]]:
    """Aggregate per-seed rows and append confidence intervals and paired comparisons."""
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    grouped_by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        dataset = str(row["dataset"])
        method = str(row["method"])
        grouped[(dataset, method)].append(row)
        grouped_by_dataset[dataset].append(row)

    result: list[dict[str, object]] = []
    for (dataset, method), items in sorted(grouped.items()):
        log_losses = np.asarray([float(item["test_log_loss"]) for item in items], dtype=np.float64)
        roc_aucs = np.asarray([float(item["test_roc_auc"]) for item in items], dtype=np.float64)
        accuracies = np.asarray([float(item["test_accuracy"]) for item in items], dtype=np.float64)
        runtimes = [float(item["selection_runtime_sec"]) for item in items]
        converged = [float(item["converged"]) for item in items]

        ci_lo_log_loss, ci_hi_log_loss = bootstrap_confidence_interval(
            log_losses,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        ci_lo_roc_auc, ci_hi_roc_auc = bootstrap_confidence_interval(
            roc_aucs,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        ci_lo_accuracy, ci_hi_accuracy = bootstrap_confidence_interval(
            accuracies,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )

        baseline_items = [
            r
            for r in grouped_by_dataset[dataset]
            if str(r["method"]) == baseline_method
        ]
        method_by_seed = {int(r["seed"]): r for r in items}
        baseline_by_seed = {int(r["seed"]): r for r in baseline_items}
        shared_seeds = sorted(set(method_by_seed.keys()) & set(baseline_by_seed.keys()))

        if method == baseline_method or not shared_seeds:
            mean_diff = 0.0
            p_value = 1.0
            effect_size = 0.0
            direction = "baseline"
        else:
            method_seed_losses = np.asarray(
                [float(method_by_seed[s]["test_log_loss"]) for s in shared_seeds],
                dtype=np.float64,
            )
            baseline_seed_losses = np.asarray(
                [float(baseline_by_seed[s]["test_log_loss"]) for s in shared_seeds],
                dtype=np.float64,
            )
            mean_diff = float(np.mean(method_seed_losses - baseline_seed_losses))
            p_value = paired_sign_test_two_sided(method_seed_losses, baseline_seed_losses)
            effect_size = cohens_d_paired(method_seed_losses, baseline_seed_losses)
            direction = paired_direction(
                method_seed_losses,
                baseline_seed_losses,
                lower_is_better=True,
                p_value=p_value,
            )

        result.append(
            {
                "dataset": dataset,
                "method": method,
                "runs": len(items),
                "mean_log_loss": _mean(log_losses.tolist()),
                "std_log_loss": _std(log_losses.tolist()),
                "ci_lo_log_loss": ci_lo_log_loss,
                "ci_hi_log_loss": ci_hi_log_loss,
                "mean_roc_auc": _mean(roc_aucs.tolist()),
                "std_roc_auc": _std(roc_aucs.tolist()),
                "ci_lo_roc_auc": ci_lo_roc_auc,
                "ci_hi_roc_auc": ci_hi_roc_auc,
                "mean_accuracy": _mean(accuracies.tolist()),
                "std_accuracy": _std(accuracies.tolist()),
                "ci_lo_accuracy": ci_lo_accuracy,
                "ci_hi_accuracy": ci_hi_accuracy,
                "mean_runtime_sec": _mean(runtimes),
                "std_runtime_sec": _std(runtimes),
                "convergence_rate": _mean(converged),
                "vs_baseline_log_loss_mean_diff": mean_diff,
                "vs_baseline_log_loss_p_value": p_value,
                "vs_baseline_log_loss_effect_size": effect_size,
                "vs_baseline_log_loss_direction": direction,
            }
        )
    return result


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_summary_markdown(rows: list[dict[str, object]]) -> str:
    """Render aggregate benchmark rows as a compact Markdown table."""
    header = (
        "| Dataset | Method | Runs | Log-loss | ROC-AUC | Accuracy | Runtime (s) | Conv. |"
    )
    sep = "|---|---|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {dataset} | {method} | {runs} | "
            "{mean_log_loss:.4f} +/- {std_log_loss:.4f} | "
            "{mean_roc_auc:.4f} +/- {std_roc_auc:.4f} | "
            "{mean_accuracy:.4f} +/- {std_accuracy:.4f} | "
            "{mean_runtime_sec:.4f} +/- {std_runtime_sec:.4f} | "
            "{convergence_rate:.2f} |".format(**row)
        )
    return "\n".join(lines)


def render_inference_markdown(rows: list[dict[str, object]]) -> str:
    """Render inferential summary rows as Markdown."""
    header = (
        "| Dataset | Method | Runs | Log-loss (95% CI) | ROC-AUC (95% CI) | "
        "Accuracy (95% CI) | Runtime (s) | vs sklearn-lbfgs | p-value |"
    )
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {dataset} | {method} | {runs} | "
            "{mean_log_loss:.4f} [{ci_lo_log_loss:.4f}, {ci_hi_log_loss:.4f}] | "
            "{mean_roc_auc:.4f} [{ci_lo_roc_auc:.4f}, {ci_hi_roc_auc:.4f}] | "
            "{mean_accuracy:.4f} [{ci_lo_accuracy:.4f}, {ci_hi_accuracy:.4f}] | "
            "{mean_runtime_sec:.4f} +/- {std_runtime_sec:.4f} | "
            "{vs_baseline_log_loss_direction} (d={vs_baseline_log_loss_effect_size:.3f}) | "
            "{vs_baseline_log_loss_p_value:.4f} |".format(**row)
        )
    return "\n".join(lines)

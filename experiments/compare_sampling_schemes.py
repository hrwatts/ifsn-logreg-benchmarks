"""Synthetic comparison of fresh and fixed sampling schedules.

This script is a local diagnostic helper. Official public benchmark evidence
should come from ``python -m ifsn_logistic.benchmarks.run`` and the tracked
outputs in ``results/``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ifsn_logistic.benchmarking import results_to_markdown, run_benchmark
from ifsn_logistic.datasets import make_logistic_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fresh and fixed subsampling schemes.")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y, beta_true = make_logistic_data(n=args.n, d=args.d, random_state=args.seed)
    results = run_benchmark(
        X,
        y,
        beta_true,
        C=args.C,
        max_iter=args.max_iter,
        subsample_size=args.subsample,
        random_state=args.seed,
        sampling_schemes=["fresh", "fixed"],
        sklearn_solvers=[],
    )

    print(results_to_markdown(results))

    out_dir = ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "compare_sampling_schemes.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["solver", "runtime_sec", "loss", "beta_error", "n_iter", "converged", "accuracy"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "solver": result.solver_name,
                    "runtime_sec": result.runtime_sec,
                    "loss": result.final_loss,
                    "beta_error": result.beta_error,
                    "n_iter": result.n_iter,
                    "converged": result.converged,
                    "accuracy": result.accuracy,
                }
            )
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()

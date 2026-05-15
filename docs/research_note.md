# Public Research Note

## Overview

This repository presents a reproducible implementation and benchmark study for
binary logistic regression with sequential subsampling Newton-style updates.
The central comparison is between two schedules:

- `fixed`: one subsample is drawn once and reused across iterations
- `fresh`: a new subsample is drawn at each iteration

Both variants are evaluated against standard sklearn logistic-regression
baselines under a shared protocol.

## What The Repository Provides

- a sklearn-style estimator interface (`IFSNLogisticRegression`)
- two solver implementations (`safe`, `reference`)
- reproducible benchmark execution with tracked result artifacts
- dataset provenance snapshots and run-environment metadata

## Evaluation Scope

- task type: binary classification
- benchmark datasets: `breast_cancer`, `mushroom`
- split protocol: stratified train/validation/test over multiple seeds
- model-selection grid: `C in {0.1, 1.0, 10.0}`

## Result Artifacts

Primary evidence is stored under `results/<suite>/`:

- `runs.csv`: per-run metrics
- `summary.csv`: aggregated descriptive metrics
- `summary_inference.csv` (optional): confidence intervals and paired baseline comparisons
- `dataset_manifest.json`: dataset provenance snapshot
- `environment.json`: runtime environment snapshot

## Reading The Results

- Use `summary.csv` for descriptive performance and runtime summaries.
- Use `summary_inference.csv` for protocol-scoped inferential comparisons.
- Use `runs.csv` when auditing per-seed behavior, convergence, and selected hyperparameters.

## Interpretation Boundaries

- The benchmark supports protocol-level comparisons for this repository.
- It does not claim universal superiority across all datasets or solvers.
- The implementation is intentionally binary-only at this stage.

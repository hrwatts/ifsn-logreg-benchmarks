# Benchmark Protocol

## Goal

This repository evaluates sequential subsampling logistic regression as an
**experimental external estimator** for binary classification. The benchmark is
designed to compare numerical behavior, predictive performance, and runtime
against common scikit-learn logistic regression baselines.

## Datasets

- `breast_cancer`: local smoke benchmark from scikit-learn's built-in loader
- `mushroom`: canonical UCI Mushroom dataset distributed under `CC BY 4.0`

Synthetic experiments remain useful for solver diagnostics, but they are not the
primary public evidence in this repo.

## Splits and Model Selection

- 5 random seeds: `0, 1, 2, 3, 4`
- Stratified `60/20/20` train/validation/test split for each seed
- Hyperparameter selection on validation log-loss over `C in {0.1, 1.0, 10.0}`
- Test metrics reported for the best validation setting
- Solver randomness is controlled with a derived per-run seed:
  `solver_seed = solver_seed_offset + dataset_index * 1000 + split_seed`

## Preprocessing

- Numeric datasets use train-only standardization, then the same transform is
  applied to validation and test splits.
- The mushroom dataset is categorical. It is converted to a **dense one-hot**
  representation fit on the training split only, followed by train-only
  standardization.
- The mushroom missing-value token `?` is treated as its own categorical level.

## Methods

External baselines:

- `sklearn-lbfgs`
- `sklearn-liblinear`
- `sklearn-saga`

IFSN variant comparisons:

- `ifsn-safe-fixed-*`
- `ifsn-safe-fresh-*`
- `ifsn-reference-fixed-*`
- `ifsn-reference-fresh-*`

Subsample ablations are run at `32`, `64`, `128`, and full-data mode.

## Metrics

- test log-loss
- test ROC-AUC
- test accuracy
- total hyperparameter-selection runtime
- selected-model iteration count
- convergence rate across seeds

## Statistical Analysis

- `summary.csv` provides descriptive aggregation (mean and std) over seeds.
- `summary_inference.csv` (optional, via `--include-inference`) adds:
  - 95% bootstrap confidence intervals for mean log-loss, ROC-AUC, and accuracy
  - paired per-seed comparison against `sklearn-lbfgs` on log-loss
  - paired effect size (Cohen's d on per-seed log-loss differences)
  - exact two-sided sign-test p-values for paired log-loss differences
- Inferential outputs are dataset-local and protocol-bound; they should not be
  interpreted as universal performance guarantees.

## Regularization Normalization

- The benchmark uses `C` as a nominal regularization grid.
- For sklearn baselines, the runner passes `C / n_train` to align objective
  scaling with the IFSN implementation for this protocol.
- For IFSN variants, the runner passes raw `C`.
- Run-level records include a `c_normalization_note` field so normalization is
  explicit and auditable.

## Results Contract

Official public benchmark claims must come from tracked files in `results/`.
Scratch experiment output belongs under ignored paths such as `artifacts/`.

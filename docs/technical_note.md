# Technical Note

## Summary

This repository studies binary logistic regression under **sequential
subsampling Newton-style updates** with a reproducible benchmark surface.

The two sampling schedules are:

- `fresh`: draw a new without-replacement subsample at every iteration
- `fixed`: draw one without-replacement subsample once and reuse it

Both schedules rebuild an inverse Hessian approximation from scratch each
iteration using sequential Sherman-Morrison rank-1 updates.

## Method

At each iteration:

1. choose a subsample according to `sampling_scheme`
2. compute logistic probabilities and curvature weights on that subsample
3. build an inverse Hessian approximation with sequential Sherman-Morrison updates
4. take a Newton-like step using the subsampled gradient

The `safe` solver adds numerical guards for denominator flooring, symmetry
enforcement, optional line search, positive-definiteness monitoring, and
fallback gradient steps.

## Benchmark Framing

The benchmark layer is designed to answer narrow empirical questions:

- how stable are fixed vs fresh sampling schedules on real binary tasks?
- how do subsample sizes affect loss, runtime, and convergence diagnostics?
- how do IFSN variants compare with common logistic-regression baselines on
  benchmark datasets?

## Deliverables

The repo provides:

- a binary estimator implementation
- explicit numerical diagnostics
- reproducible preprocessing and dataset provenance for benchmark runs
- tracked benchmark results under a fixed protocol

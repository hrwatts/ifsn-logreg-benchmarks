# Reproducibility Guide

## Benchmark Environment

The official benchmark workflow expects the pinned benchmark extra:

```bash
python -m pip install -e .[benchmark]
```

The benchmark extra pins a compatible stack for `numpy`, `scipy`,
`scikit-learn`, and `pandas`.

## Running the Benchmark Suites

Smoke suite:

```bash
python -m ifsn_logistic.benchmarks.run --suite smoke
```

Full suite:

```bash
python -m ifsn_logistic.benchmarks.run --suite full
```

Full suite with inferential summary artifacts:

```bash
python -m ifsn_logistic.benchmarks.run --suite full --include-inference
```

Outputs are written under `results/<suite>/`:

- `runs.csv`: one row per dataset/seed/method run
- `summary.csv`: aggregated metrics by dataset and method
- `summary.md`: Markdown rendering of the aggregate table
- `dataset_manifest.json`: snapshot of the dataset manifest used for the run
- `environment.json`: runtime provenance snapshot (OS, Python, package versions)

Optional inferential outputs:

- `summary_inference.csv`: 95% bootstrap CIs and paired baseline comparison fields
- `summary_inference.md`: Markdown rendering of inferential summaries

## Dataset Provenance

### UCI Mushroom

- Source: UCI Machine Learning Repository
- DOI: `10.24432/C5959T`
- License: `CC BY 4.0`
- Archive URL: `https://archive.ics.uci.edu/static/public/73/mushroom.zip`
- Pinned SHA256:
  `FACE32F32647E0D939F6233F36DD30DD5D619AE9F3F9B8E10BEA4AC7E1F60B1A`

### Breast Cancer Smoke Benchmark

- Loader: `sklearn.datasets.load_breast_cancer`
- Purpose: lightweight CI-facing smoke benchmark

## Release Notes

- The repo is intentionally binary-classification only.
- The benchmark runner fails fast when the scikit-learn / SciPy stack is
  unavailable or incompatible.
- Public-facing benchmark evidence is tracked under `results/`; ignored scratch
  output should not be cited as official evidence.

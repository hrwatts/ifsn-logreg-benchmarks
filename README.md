# IFSN Sequential Subsampling

`ifsn-sequential-subsampling` is a public research implementation and benchmark
for binary logistic regression under sequential subsampling Newton-style
updates.
The repository packages:

- a sklearn-style estimator, `IFSNLogisticRegression`
- reference and numerically guarded solver variants
- a reproducible benchmark runner with tracked result files
- pinned provenance for the canonical UCI Mushroom dataset

## Start Here

For the public-facing summary of the work, read
[docs/research_note.md](docs/research_note.md).

## Abstract

This repository studies sequential subsampling Newton-style updates for binary
logistic regression under a reproducible benchmark protocol.

The central comparison is between:

- `fixed` subsampling: one subsample reused across iterations
- `fresh` subsampling: a new subsample drawn at each iteration

The benchmark evaluates IFSN variants and sklearn baselines under shared data
splits, preprocessing, and model-selection rules, with tracked artifacts for
auditability.

## Method Summary

The estimator studies two subsampling schedules:

- `fresh`: draw a new subsample at every iteration
- `fixed`: draw one subsample once and reuse it

Both schedules rebuild an inverse Hessian approximation from scratch with
sequential Sherman-Morrison updates. The public API remains intentionally small:

```python
from ifsn_logistic import IFSNLogisticRegression

clf = IFSNLogisticRegression(
    solver="safe",
    sampling_scheme="fixed",
    subsample_size=64,
    random_state=0,
)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)
```

## Benchmark Suites

### Smoke suite

- dataset: `breast_cancer`
- purpose: lightweight reproducibility check
- methods: `sklearn-lbfgs`, `ifsn-safe-fixed-s64`, `ifsn-safe-fresh-s64`

### Full suite

- datasets: `breast_cancer`, `mushroom`
- seeds: `0, 1, 2, 3, 4`
- validation grid: `C in {0.1, 1.0, 10.0}`
- external baselines: `lbfgs`, `liblinear`, `saga`
- IFSN variant ablations: `safe` vs `reference`, `fixed` vs `fresh`, subsample
  sizes `32`, `64`, `128`, and full-data mode

Official outputs are committed under [results/full/summary.md](results/full/summary.md),
[results/full/summary.csv](results/full/summary.csv), and
[results/full/runs.csv](results/full/runs.csv).

## Snapshot Results

The current committed full benchmark snapshot contains `190` runs:

- `2` datasets
- `5` seeds per dataset
- `19` method configurations per dataset

Key findings from the tracked snapshot:

- full-data IFSN and sklearn baselines are closely aligned on log-loss and ROC-AUC
- aggressive subsampling can reduce quality and convergence reliability
- runtime and convergence behavior vary substantially by schedule and subsample size

Representative rows from `results/full/summary.csv`:

| Dataset | Method | Log-loss | ROC-AUC | Accuracy | Runtime (s) | Conv. |
|---|---|---:|---:|---:|---:|---:|
| `breast_cancer` | `sklearn-lbfgs` | `0.1373 +/- 0.0218` | `0.9948 +/- 0.0045` | `0.9652 +/- 0.0198` | `0.0183 +/- 0.0039` | `1.00` |
| `breast_cancer` | `ifsn-reference-fixed-full` | `0.1373 +/- 0.0218` | `0.9948 +/- 0.0045` | `0.9652 +/- 0.0198` | `0.0568 +/- 0.0020` | `1.00` |
| `mushroom` | `sklearn-lbfgs` | `0.0717 +/- 0.0009` | `1.0000 +/- 0.0000` | `0.9995 +/- 0.0005` | `0.0445 +/- 0.0087` | `1.00` |
| `mushroom` | `ifsn-safe-fresh-s128` | `0.0863 +/- 0.0017` | `0.9999 +/- 0.0000` | `0.9975 +/- 0.0012` | `17.2173 +/- 1.2969` | `0.00` |

For full method-by-dataset tables, see
[results/full/summary.md](results/full/summary.md) and
[results/full/summary.csv](results/full/summary.csv).

## Quick Reproduction

Install benchmark dependencies:

```bash
python -m pip install -e .[benchmark]
```

Run benchmark suites:

```bash
python -m ifsn_logistic.benchmarks.run --suite smoke
python -m ifsn_logistic.benchmarks.run --suite full
```

Generate inferential summaries:

```bash
python -m ifsn_logistic.benchmarks.run --suite full --include-inference
```

## Outputs

Each suite writes:

- `runs.csv`: per-seed run records
- `summary.csv`: aggregated descriptive metrics
- `summary.md`: Markdown rendering of descriptive summary
- `dataset_manifest.json`: dataset provenance snapshot
- `environment.json`: runtime provenance snapshot

Optional inferential outputs:

- `summary_inference.csv`: confidence intervals and paired baseline comparisons
- `summary_inference.md`: Markdown rendering of inferential summary

## Dataset Provenance

### UCI Mushroom

- source: UCI Machine Learning Repository
- DOI: `10.24432/C5959T`
- license: `CC BY 4.0`
- archive URL:
  `https://archive.ics.uci.edu/static/public/73/mushroom.zip`
- pinned SHA256:
  `FACE32F32647E0D939F6233F36DD30DD5D619AE9F3F9B8E10BEA4AC7E1F60B1A`

The mushroom benchmark uses a **dense one-hot encoding** fit on the training
split only. The missing-value token `?` is treated as its own category.

## Scope

- binary classification only
- dense numeric estimator interface
- benchmark claims are protocol-scoped to tracked results in this repository

## Further Reading

- [docs/research_note.md](docs/research_note.md)
- [docs/benchmark_protocol.md](docs/benchmark_protocol.md)
- [docs/reproducibility.md](docs/reproducibility.md)
- [docs/technical_note.md](docs/technical_note.md)

## How To Cite

Use `CITATION.cff` for the current software citation record.

When reporting benchmark outcomes, cite both:

- the software artifact (`CITATION.cff`)
- the method papers used for algorithmic context
- the underlying dataset sources (see `results/*/dataset_manifest.json`)

### Method References (BibTeX)

```bibtex
@article{kirkby2021inversion,
  title={An Inversion-Free Subsampling Newton's Method for Logistic Regression},
  author={Kirkby, J. Lars and Nguyen, Dang H. and Nguyen, Duy},
  journal={Statistical Papers},
  volume={63},
  pages={943--963},
  year={2022},
  doi={10.1007/s00362-021-01262-3}
}

@article{wang2018optimal,
  title={Optimal Subsampling for Large Sample Logistic Regression},
  author={Wang, HaiYing and Zhu, Rong and Ma, Ping},
  journal={Journal of the American Statistical Association},
  volume={113},
  number={522},
  pages={829--844},
  year={2018},
  doi={10.1080/01621459.2017.1292914}
}
```

If you are citing the theoretical background for logistic regression itself,
consider also citing standard references such as Bishop (2006) and Hastie,
Tibshirani, and Friedman (2009).

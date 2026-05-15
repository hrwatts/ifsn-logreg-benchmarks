# Internal Release Checklist

This checklist is maintainer-facing and not part of the public research presentation.

- Remove transient cache directories such as `__pycache__`, `.pytest_cache`, and `pytest-cache-files-*`
- Confirm `python -m pytest` passes
- Confirm `python -m ifsn_logistic.benchmarks.run --suite smoke` passes in the benchmark environment
- Confirm inferential outputs can be generated with `--include-inference`
- Confirm `environment.json` is emitted for benchmark suite runs
- Regenerate `results/` outputs before release if benchmark code changed
- Verify `CITATION.cff` and README links point at the public repository location
- Verify the dataset manifest snapshot matches the pinned dataset metadata
- Confirm no private workspace notes or unrelated parent-repo files are included in the public release

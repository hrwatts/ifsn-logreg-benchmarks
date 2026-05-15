from __future__ import annotations

from pathlib import Path

import pytest

from ifsn_logistic.benchmarks.data import benchmark_stack_status
from ifsn_logistic.benchmarks.run import build_parser, main as benchmark_main


def test_benchmark_cli_fails_without_stack(monkeypatch):
    monkeypatch.setattr(
        "ifsn_logistic.benchmarks.data.benchmark_stack_status",
        lambda: (False, "missing scipy"),
    )
    with pytest.raises(RuntimeError, match="Install the benchmark extra"):
        benchmark_main(["--suite", "smoke"])


@pytest.mark.slow
def test_smoke_benchmark_runs_when_stack_available(tmp_path: Path):
    ok, _ = benchmark_stack_status()
    if not ok:
        pytest.skip("benchmark stack unavailable in default interpreter")

    result_dir = tmp_path / "results"
    exit_code = benchmark_main(
        [
            "--suite",
            "smoke",
            "--results-dir",
            str(result_dir),
            "--seeds",
            "0",
            "--c-grid",
            "1.0",
        ]
    )
    assert exit_code == 0
    assert (result_dir / "smoke" / "runs.csv").exists()
    assert (result_dir / "smoke" / "summary.md").exists()
    assert (result_dir / "smoke" / "environment.json").exists()


def test_benchmark_parser_supports_stage1_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--suite",
            "smoke",
            "--solver-seed-offset",
            "777",
            "--include-inference",
            "--inference-n-bootstrap",
            "500",
            "--no-save-environment",
        ]
    )
    assert args.suite == "smoke"
    assert args.solver_seed_offset == 777
    assert args.include_inference is True
    assert args.inference_n_bootstrap == 500
    assert args.save_environment is False

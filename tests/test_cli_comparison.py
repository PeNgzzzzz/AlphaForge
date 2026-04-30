"""Tests for CLI run comparison helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from alphaforge.cli.comparison import (
    build_compare_artifact_metadata,
    build_compare_runs_report,
    rank_compare_runs,
)


def test_rank_compare_runs_supports_weighted_metrics(tmp_path: Path) -> None:
    """Weighted run comparison should rank runs from bundle-level metrics."""
    experiment_root = _write_compare_fixture(tmp_path)

    ranked = rank_compare_runs(
        experiment_root,
        rank_by=["summary_cumulative_return", "summary_mean_ic"],
        rank_weights={
            "summary_cumulative_return": 0.2,
            "summary_mean_ic": 0.8,
        },
    )

    assert ranked.loc[0, "run_id"] == "sweep-high-ic"
    assert ranked.loc[0, "weighted_rank_score"] == pytest.approx(1.2)
    assert ranked.loc[0, "weight_summary_cumulative_return"] == pytest.approx(0.2)
    assert ranked.loc[0, "weight_summary_mean_ic"] == pytest.approx(0.8)


def test_build_compare_runs_report_keeps_requested_order(tmp_path: Path) -> None:
    """Compare reports should honor explicit run-id order and include detail sections."""
    experiment_root = _write_compare_fixture(tmp_path)

    report = build_compare_runs_report(
        experiment_root,
        run_ids=["sweep-high-ic", "sweep-high-return"],
        sweep_top_k=1,
    )

    assert "Compared Runs" in report
    assert report.find("sweep-high-ic") < report.find("sweep-high-return")
    assert "Sweep Top Candidates: sweep-high-ic" in report
    assert "Candidates Shown: 1/2" in report


def test_build_compare_artifact_metadata_summarizes_best_runs() -> None:
    """Compare metadata should expose headline best-run diagnostics."""
    results = pd.DataFrame(
        {
            "run_id": ["run-low-return", "run-high-return", "run-high-ic"],
            "created_at": [
                "2024-01-02T00:00:00Z",
                "2024-01-03T00:00:00Z",
                "2024-01-04T00:00:00Z",
            ],
            "command": ["sweep-signal", "sweep-signal", "walk-forward-signal"],
            "summary_scope": [
                "best_in_sample_candidate",
                "best_in_sample_candidate",
                "overall_out_of_sample",
            ],
            "summary_cumulative_return": [0.05, 0.25, 0.10],
            "summary_sharpe_ratio": [0.6, 1.5, 0.7],
            "summary_mean_ic": [0.02, 0.03, 0.08],
        }
    )

    metadata = build_compare_artifact_metadata(
        experiment_root="artifacts/showcase_runs",
        run_ids=["run-low-return", "run-high-return", "run-high-ic"],
        selection_mode="explicit_run_ids",
        command_name_filter=None,
        parameter_filter=None,
        rank_by=None,
        rank_weight=None,
        sort_by=None,
        ascending=None,
        limit=None,
        results=results,
    )

    summary = metadata["comparison_summary"]
    assert metadata["row_count"] == 3
    assert summary["commands"] == ["sweep-signal", "walk-forward-signal"]
    assert summary["best_run_by_cumulative_return"]["run_id"] == "run-high-return"
    assert summary["best_run_by_mean_ic"]["run_id"] == "run-high-ic"


def _write_compare_fixture(tmp_path: Path) -> Path:
    experiment_root = tmp_path / "comparison_runs"
    experiment_root.mkdir()

    run_rows = [
        _write_sweep_run(
            experiment_root,
            run_id="sweep-high-return",
            created_at="2024-01-02T00:00:00Z",
            cumulative_returns=[0.20, 0.10],
            mean_ics=[0.02, 0.01],
        ),
        _write_sweep_run(
            experiment_root,
            run_id="sweep-high-ic",
            created_at="2024-01-03T00:00:00Z",
            cumulative_returns=[0.08, 0.06],
            mean_ics=[0.08, 0.04],
        ),
    ]
    pd.DataFrame(run_rows).to_csv(experiment_root / "runs.csv", index=False)
    return experiment_root


def _write_sweep_run(
    experiment_root: Path,
    *,
    run_id: str,
    created_at: str,
    cumulative_returns: list[float],
    mean_ics: list[float],
) -> dict[str, str | int | float]:
    run_dir = experiment_root / run_id
    run_dir.mkdir()

    results = pd.DataFrame(
        {
            "parameter_name": ["lookback", "lookback"],
            "parameter_value": [1, 2],
            "signal_column": ["momentum_signal_1d", "momentum_signal_2d"],
            "cumulative_return": cumulative_returns,
            "max_drawdown": [-0.05, -0.02],
            "sharpe_ratio": [1.1, 0.7],
            "average_turnover": [0.3, 0.2],
            "hit_rate": [0.6, 0.5],
            "mean_ic": mean_ics,
            "ic_ir": [0.4, 0.2],
            "joint_coverage_ratio": [0.9, 0.8],
        }
    )
    results_path = run_dir / "results.csv"
    results.to_csv(results_path, index=False)

    report_path = run_dir / "report.txt"
    report_path.write_text("report\n", encoding="utf-8")

    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "command": "sweep-signal",
                "config": "configs/momentum_example.toml",
                "parameter": "lookback",
                "values": [1, 2],
                "selection_metric": "cumulative_return",
                "row_count": 2,
                "overall_summary": {
                    "cumulative_return": max(cumulative_returns),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "run_id": run_id,
        "created_at": created_at,
        "command": "sweep-signal",
        "config": "configs/momentum_example.toml",
        "parameter": "lookback",
        "values": "1,2",
        "selection_metric": "cumulative_return",
        "train_periods": "",
        "test_periods": "",
        "row_count": 2,
        "overall_cumulative_return": max(cumulative_returns),
        "artifact_dir": str(run_dir),
        "results_path": str(results_path),
        "report_path": str(report_path),
        "metadata_path": str(metadata_path),
    }

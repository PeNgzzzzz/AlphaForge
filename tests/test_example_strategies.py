"""Reproducibility tests for bundled end-to-end example strategies."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from alphaforge.analytics import summarize_backtest
from alphaforge.cli.workflows import (
    build_dataset_from_config,
    build_report_text,
    run_backtest_from_config,
    write_report_artifact_bundle,
)
from alphaforge.common import load_pipeline_config

EXAMPLE_CONFIGS = (
    Path("configs/momentum_example.toml"),
    Path("configs/mean_reversion_example.toml"),
    Path("configs/trend_example.toml"),
    Path("configs/stage1_universe_example.toml"),
    Path("configs/stage2_execution_example.toml"),
    Path("configs/stage3_benchmark_example.toml"),
    Path("configs/stage4_flagship_example.toml"),
    Path("configs/market_cap_grouped_diagnostics_example.toml"),
)


@pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS)
def test_example_strategy_reports_run_reproducibly(config_path: Path) -> None:
    """Bundled example configs should render non-empty reports with diagnostics."""
    config = load_pipeline_config(config_path)

    report = build_report_text(config)

    assert "Research Workflow" in report
    assert "Data Quality Summary" in report
    assert "Portfolio Constraints" in report
    assert "Execution Assumptions" in report
    assert "Execution Summary" in report
    assert "Performance Summary" in report
    assert "Risk Summary" in report
    assert "Diagnostics Overview" in report
    assert "IC Summary" in report
    assert "Coverage Summary" in report
    assert "No quantile buckets produced" not in report
    if config.universe is not None:
        assert "Universe Rules" in report
        assert "Universe Summary" in report
    if config.benchmark is not None:
        assert "Benchmark Configuration" in report
        assert "Relative Performance Summary" in report
        assert "Benchmark Risk Summary" in report


@pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS)
def test_example_strategy_backtests_produce_finite_non_zero_results(
    config_path: Path,
) -> None:
    """Bundled example configs should produce finite, non-trivial backtest output."""
    config = load_pipeline_config(config_path)

    backtest = run_backtest_from_config(config)
    summary = summarize_backtest(backtest)

    assert not backtest.empty
    assert float(backtest["net_return"].abs().sum()) > 0.0
    assert pd.notna(summary["cumulative_return"])
    assert math.isfinite(float(summary["cumulative_return"]))


def test_market_cap_grouped_diagnostics_example_writes_grouped_artifacts(
    tmp_path: Path,
) -> None:
    """The market-cap example should exercise grouped diagnostics end to end."""
    config = load_pipeline_config(
        Path("configs/market_cap_grouped_diagnostics_example.toml")
    )

    dataset = build_dataset_from_config(config)
    artifact_paths = write_report_artifact_bundle(
        config,
        tmp_path / "market_cap_grouped_report",
    )
    metadata = json.loads(
        Path(artifact_paths["metadata_path"]).read_text(encoding="utf-8")
    )
    chart_ids = {
        chart["chart_id"] for chart in metadata["chart_bundle"]["charts"]
    }

    assert config.dataset.include_market_cap
    assert config.dataset.market_cap_bucket_count == 2
    assert config.diagnostics.group_columns == ("market_cap_bucket",)
    assert "market_cap_bucket" in dataset.columns
    assert set(dataset["market_cap_bucket"].dropna().astype(int)) == {1, 2}
    assert any(
        row["group_column"] == "market_cap_bucket"
        for row in metadata["grouped_coverage_summary"]["rows"]
    )
    assert any(
        row["group_column"] == "market_cap_bucket"
        for row in metadata["grouped_ic_summary"]["rows"]
    )
    assert "Grouped Coverage Summary" in metadata["report_sections"]
    assert "Grouped IC Summary" in metadata["report_sections"]
    assert "grouped_coverage_summary" in chart_ids
    assert "grouped_ic_summary" in chart_ids

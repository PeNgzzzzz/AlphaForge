"""Reproducibility tests for bundled end-to-end example strategies."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from alphaforge.analytics import summarize_backtest
from alphaforge.cli.workflows import build_report_text, run_backtest_from_config
from alphaforge.common import load_pipeline_config

EXAMPLE_CONFIGS = (
    Path("configs/momentum_example.toml"),
    Path("configs/mean_reversion_example.toml"),
    Path("configs/trend_example.toml"),
    Path("configs/stage1_universe_example.toml"),
    Path("configs/stage2_execution_example.toml"),
    Path("configs/stage3_benchmark_example.toml"),
    Path("configs/stage4_flagship_example.toml"),
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

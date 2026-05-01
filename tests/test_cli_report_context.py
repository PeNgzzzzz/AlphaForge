"""Tests for CLI report context assembly helpers."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from alphaforge.cli.errors import WorkflowError
from alphaforge.cli.report_context import (
    build_report_context,
    validate_diagnostics_column,
)
from alphaforge.common import load_pipeline_config


def test_build_report_context_attaches_benchmark_and_grouped_diagnostics() -> None:
    """Report context assembly should preserve benchmark and grouped diagnostics."""
    config = load_pipeline_config("configs/market_cap_grouped_diagnostics_example.toml")

    context = build_report_context(config)

    assert context["benchmark_data"] is not None
    assert context["relative_performance_summary"] is not None
    assert context["benchmark_risk_summary"] is not None
    assert {
        "benchmark_return",
        "excess_return",
        "benchmark_nav",
        "relative_return",
        "relative_nav",
    }.issubset(context["backtest"].columns)
    assert not context["grouped_ic_series"].empty
    assert set(context["grouped_ic_series"]["group_column"]) == {"market_cap_bucket"}
    assert not context["grouped_coverage_summary"].empty
    assert set(context["grouped_coverage_summary"]["group_column"]) == {
        "market_cap_bucket"
    }


def test_validate_diagnostics_column_rejects_missing_label() -> None:
    """Report and sweep paths should fail loudly for missing diagnostics labels."""
    config = load_pipeline_config("configs/stage3_benchmark_example.toml")
    bad_config = replace(
        config,
        diagnostics=replace(config.diagnostics, forward_return_column="missing_label"),
    )
    dataset = pd.DataFrame({"forward_return_1d": [0.01]})

    with pytest.raises(WorkflowError, match="diagnostics.forward_return_column"):
        validate_diagnostics_column(dataset, bad_config)

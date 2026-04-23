"""Tests for conservative walk-forward parameter selection."""

from __future__ import annotations

from pathlib import Path

import math

import pandas as pd
import pytest

from alphaforge.analytics import (
    WalkForwardError,
    format_walk_forward_report,
    validate_walk_forward_results,
)
from alphaforge.cli.workflows import WorkflowError, run_walk_forward_parameter_selection
from alphaforge.common import load_pipeline_config


def test_run_walk_forward_parameter_selection_returns_fold_table_and_summary() -> None:
    """Walk-forward selection should return ordered fold rows and OOS performance."""
    config = load_pipeline_config(Path("configs/momentum_example.toml"))

    fold_results, overall_summary = run_walk_forward_parameter_selection(
        config,
        parameter_name="lookback",
        values=[1, 2, 3],
        train_periods=4,
        test_periods=2,
    )

    assert fold_results["fold_index"].tolist() == [1.0, 2.0, 3.0]
    assert fold_results["parameter_name"].tolist() == ["lookback", "lookback", "lookback"]
    assert fold_results["selection_metric"].tolist() == [
        "cumulative_return",
        "cumulative_return",
        "cumulative_return",
    ]
    assert not fold_results.empty
    assert pd.notna(overall_summary["cumulative_return"])
    assert math.isfinite(float(overall_summary["cumulative_return"]))


def test_run_walk_forward_parameter_selection_supports_mean_ic_selection() -> None:
    """Walk-forward selection should support train-fold IC as the selection metric."""
    config = load_pipeline_config(Path("configs/trend_example.toml"))

    fold_results, _ = run_walk_forward_parameter_selection(
        config,
        parameter_name="short_window",
        values=[1, 2, 3],
        train_periods=4,
        test_periods=2,
        selection_metric="mean_ic",
    )

    assert fold_results["selection_metric"].tolist() == ["mean_ic", "mean_ic", "mean_ic"]
    assert fold_results["signal_column"].str.startswith("trend_signal_").all()


def test_run_walk_forward_parameter_selection_rejects_insufficient_dates() -> None:
    """Walk-forward folds should fail loudly when the panel is too short."""
    config = load_pipeline_config(Path("configs/momentum_example.toml"))

    with pytest.raises(WorkflowError, match="train_periods \\+ test_periods"):
        run_walk_forward_parameter_selection(
            config,
            parameter_name="lookback",
            values=[1, 2],
            train_periods=8,
            test_periods=4,
        )


def test_format_walk_forward_report_renders_sections() -> None:
    """Formatted walk-forward reports should include summary and fold sections."""
    fold_results = pd.DataFrame(
        {
            "fold_index": [1.0],
            "parameter_name": ["lookback"],
            "selected_parameter_value": [2.0],
            "signal_column": ["momentum_signal_2d"],
            "selection_metric": ["cumulative_return"],
            "train_start": ["2024-01-02"],
            "train_end": ["2024-01-05"],
            "test_start": ["2024-01-08"],
            "test_end": ["2024-01-09"],
            "train_selection_score": [0.12],
            "train_cumulative_return": [0.12],
            "train_mean_ic": [0.5],
            "test_cumulative_return": [0.03],
            "test_max_drawdown": [-0.01],
            "test_sharpe_ratio": [1.5],
            "test_mean_ic": [0.4],
            "test_joint_coverage_ratio": [0.75],
        }
    )
    overall_summary = pd.Series(
        {
            "periods": 2.0,
            "cumulative_return": 0.03,
            "annualized_return": 0.20,
            "annualized_volatility": 0.10,
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.01,
            "average_turnover": 0.2,
            "total_turnover": 0.4,
            "hit_rate": 0.5,
        }
    )

    rendered = format_walk_forward_report(fold_results, overall_summary)

    assert "Walk-Forward Summary" in rendered
    assert "Performance Summary" in rendered
    assert "Walk-Forward Folds" in rendered
    assert "1.50" in rendered


def test_validate_walk_forward_results_rejects_missing_columns() -> None:
    """Walk-forward result validation should fail loudly on incomplete frames."""
    frame = pd.DataFrame({"fold_index": [1.0]})

    with pytest.raises(WalkForwardError, match="missing required columns"):
        validate_walk_forward_results(frame)

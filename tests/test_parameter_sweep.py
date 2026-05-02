"""Tests for the simple signal parameter sweep extension."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from alphaforge.analytics import (
    ParameterSweepError,
    format_parameter_sweep_results,
    validate_parameter_sweep_results,
)
from alphaforge.cli.errors import WorkflowError
from alphaforge.cli.parameter_sweep import (
    build_sweep_artifact_metadata,
    run_signal_parameter_sweep,
)
from alphaforge.common import load_pipeline_config


def test_run_signal_parameter_sweep_returns_ordered_summary_rows() -> None:
    """Signal sweeps should return one summary row per requested candidate."""
    config = load_pipeline_config(Path("configs/momentum_example.toml"))

    results = run_signal_parameter_sweep(
        config,
        parameter_name="lookback",
        values=[1, 2, 3],
    )

    assert results["parameter_name"].tolist() == ["lookback", "lookback", "lookback"]
    assert results["parameter_value"].tolist() == [1.0, 2.0, 3.0]
    assert results["signal_column"].tolist() == [
        "momentum_signal_1d",
        "momentum_signal_2d",
        "momentum_signal_3d",
    ]
    assert results["cumulative_return"].abs().sum() > 0.0


def test_run_signal_parameter_sweep_rejects_invalid_signal_parameter() -> None:
    """Sweep parameter names should match the configured signal family."""
    config = load_pipeline_config(Path("configs/momentum_example.toml"))

    with pytest.raises(WorkflowError, match="only supports sweeping the 'lookback'"):
        run_signal_parameter_sweep(
            config,
            parameter_name="short_window",
            values=[2, 3],
        )


def test_run_signal_parameter_sweep_supports_trend_window_candidates() -> None:
    """Trend signal sweeps should support short or long window candidates."""
    config = load_pipeline_config(Path("configs/trend_example.toml"))

    results = run_signal_parameter_sweep(
        config,
        parameter_name="short_window",
        values=[1, 2, 3],
    )

    assert results["signal_column"].tolist() == [
        "trend_signal_1_4d",
        "trend_signal_2_4d",
        "trend_signal_3_4d",
    ]
    assert results["parameter_value"].tolist() == [1.0, 2.0, 3.0]


def test_run_signal_parameter_sweep_rejects_duplicate_values() -> None:
    """Sweep candidate lists should reject duplicate parameter values."""
    config = load_pipeline_config(Path("configs/momentum_example.toml"))

    with pytest.raises(WorkflowError, match="must be unique"):
        run_signal_parameter_sweep(
            config,
            parameter_name="lookback",
            values=[1, 1],
        )


def test_build_sweep_artifact_metadata_records_best_candidates() -> None:
    """Sweep artifact metadata should expose research context and ranked candidates."""
    config = load_pipeline_config(Path("configs/momentum_example.toml"))
    results = run_signal_parameter_sweep(
        config,
        parameter_name="lookback",
        values=[1, 2],
    )

    metadata = build_sweep_artifact_metadata(
        config,
        config_path="configs/momentum_example.toml",
        parameter_name="lookback",
        values=[1, 2],
        results=results,
    )

    assert metadata["command"] == "sweep-signal"
    assert metadata["parameter"] == "lookback"
    assert metadata["row_count"] == 2
    assert metadata["best_candidate"] is not None
    assert len(metadata["top_candidates"]) == 2
    assert "data_quality_summary" in metadata["research_context"]


def test_format_parameter_sweep_results_renders_plain_text_table() -> None:
    """Formatted sweep summaries should include a stable text header."""
    frame = pd.DataFrame(
        {
            "parameter_name": ["lookback"],
            "parameter_value": [2.0],
            "signal_column": ["momentum_signal_2d"],
            "cumulative_return": [0.1],
            "max_drawdown": [-0.02],
            "sharpe_ratio": [1.5],
            "average_turnover": [0.3],
            "hit_rate": [0.6],
            "mean_ic": [0.2],
            "ic_ir": [1.1],
            "joint_coverage_ratio": [0.7],
        }
    )

    rendered = format_parameter_sweep_results(frame)

    assert "Signal Parameter Sweep" in rendered
    assert "momentum_signal_2d" in rendered
    assert "10.00%" in rendered


def test_validate_parameter_sweep_results_rejects_missing_columns() -> None:
    """Sweep table validation should fail loudly on incomplete result frames."""
    frame = pd.DataFrame({"parameter_name": ["lookback"]})

    with pytest.raises(ParameterSweepError, match="missing required columns"):
        validate_parameter_sweep_results(frame)


def test_validate_parameter_sweep_results_rejects_invalid_numeric_values() -> None:
    """Sweep table validation should preserve invalid numeric fail-fast behavior."""
    frame = pd.DataFrame(
        {
            "parameter_name": ["lookback"],
            "parameter_value": ["bad"],
            "signal_column": ["momentum_signal_2d"],
            "cumulative_return": [0.1],
            "max_drawdown": [-0.02],
            "sharpe_ratio": [1.5],
            "average_turnover": [0.3],
            "hit_rate": [0.6],
            "mean_ic": [0.2],
            "ic_ir": [1.1],
            "joint_coverage_ratio": [0.7],
        }
    )

    with pytest.raises(
        ParameterSweepError,
        match="parameter sweep results contain invalid numeric values in 'parameter_value'",
    ):
        validate_parameter_sweep_results(frame)


def test_validate_parameter_sweep_results_rejects_empty_string_values() -> None:
    """Sweep table validation should reject blank required string fields."""
    frame = pd.DataFrame(
        {
            "parameter_name": [" "],
            "parameter_value": [2.0],
            "signal_column": ["momentum_signal_2d"],
            "cumulative_return": [0.1],
            "max_drawdown": [-0.02],
            "sharpe_ratio": [1.5],
            "average_turnover": [0.3],
            "hit_rate": [0.6],
            "mean_ic": [0.2],
            "ic_ir": [1.1],
            "joint_coverage_ratio": [0.7],
        }
    )

    with pytest.raises(
        ParameterSweepError,
        match="parameter sweep results contain missing or empty values in 'parameter_name'",
    ):
        validate_parameter_sweep_results(frame)

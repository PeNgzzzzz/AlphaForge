"""Tests for walk-forward CLI helper functions."""

from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.cli.errors import WorkflowError
from alphaforge.cli.walk_forward import (
    build_walk_forward_artifact_metadata,
    build_walk_forward_artifact_metadata_from_config,
    build_walk_forward_folds,
    extract_unique_dates,
    extract_walk_forward_selection_score,
    normalize_walk_forward_selection_metric,
    select_augmented_dates,
)
from alphaforge.common import load_pipeline_config


def test_build_walk_forward_folds_rolls_fixed_train_test_windows() -> None:
    """Walk-forward folds should roll by test window length without overlap leakage."""
    dates = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
        ]
    ).tolist()

    folds = build_walk_forward_folds(dates, train_periods=3, test_periods=2)

    assert folds == [
        {"train_dates": dates[0:3], "test_dates": dates[3:5]},
        {"train_dates": dates[2:5], "test_dates": dates[5:7]},
    ]


def test_build_walk_forward_folds_requires_enough_dates() -> None:
    """Walk-forward folds should fail loudly when the date panel is too short."""
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]).tolist()

    with pytest.raises(
        WorkflowError,
        match="at least train_periods \\+ test_periods unique dates",
    ):
        build_walk_forward_folds(dates, train_periods=2, test_periods=2)


def test_select_augmented_dates_preserves_backtest_history() -> None:
    """Daily delayed backtests need prior dates before filtering evaluation output."""
    dates = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ]
    ).tolist()
    frame = pd.DataFrame({"date": dates})

    selected = select_augmented_dates(
        frame,
        evaluation_dates=dates[3:5],
        history_periods=2,
    )

    assert selected == dates[1:5]


def test_extract_unique_dates_sorts_and_deduplicates() -> None:
    """Date extraction should be deterministic for dated panels."""
    frame = pd.DataFrame(
        {
            "date": [
                "2024-01-05",
                "2024-01-02",
                "2024-01-05",
                "2024-01-03",
            ]
        }
    )

    assert extract_unique_dates(frame) == pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-05"]
    ).tolist()


def test_selection_metric_validation_and_score_extraction() -> None:
    """Selection scoring should keep invalid metrics explicit and NaNs conservative."""
    assert normalize_walk_forward_selection_metric(" mean_ic ") == "mean_ic"

    with pytest.raises(WorkflowError, match="selection_metric must be one of"):
        normalize_walk_forward_selection_metric("turnover")

    evaluation = {
        "performance_summary": pd.Series(
            {"cumulative_return": 0.12, "sharpe_ratio": float("nan")}
        ),
        "ic_summary": pd.Series({"mean_ic": -0.03}),
    }

    assert extract_walk_forward_selection_score(
        evaluation,
        selection_metric="cumulative_return",
    ) == pytest.approx(0.12)
    assert extract_walk_forward_selection_score(
        evaluation,
        selection_metric="sharpe_ratio",
    ) == float("-inf")


def test_build_walk_forward_artifact_metadata_summarizes_selected_values() -> None:
    """Artifact metadata should summarize fold count and selected-parameter distribution."""
    fold_results = pd.DataFrame(
        {
            "selected_parameter_value": [2.0, 1.0, 2.0],
            "test_start": ["2024-01-05", "2024-01-09", "2024-01-11"],
            "test_end": ["2024-01-08", "2024-01-10", "2024-01-12"],
        }
    )
    overall_summary = pd.Series(
        {"cumulative_return": 0.08, "max_drawdown": -0.04}
    )

    metadata = build_walk_forward_artifact_metadata(
        config_path="configs/momentum_example.toml",
        parameter_name="lookback",
        values=[1, 2, 3],
        train_periods=4,
        test_periods=2,
        selection_metric="cumulative_return",
        fold_results=fold_results,
        overall_summary=overall_summary,
        research_context={"dataset_feature_metadata": []},
    )

    assert metadata["command"] == "walk-forward-signal"
    assert metadata["fold_count"] == 3
    assert metadata["selected_parameter_values"] == [1.0, 2.0]
    assert metadata["selection_distribution"] == {"1": 1, "2": 2}
    assert metadata["test_period_start"] == "2024-01-05"
    assert metadata["test_period_end"] == "2024-01-12"
    assert metadata["overall_summary"]["cumulative_return"] == pytest.approx(0.08)


def test_build_walk_forward_artifact_metadata_from_config_includes_research_context() -> None:
    """Config-aware metadata should attach the shared research context summary."""
    config = load_pipeline_config("configs/momentum_example.toml")
    fold_results = pd.DataFrame(
        {
            "selected_parameter_value": [2.0],
            "test_start": ["2024-01-08"],
            "test_end": ["2024-01-09"],
        }
    )
    overall_summary = pd.Series({"cumulative_return": 0.04})

    metadata = build_walk_forward_artifact_metadata_from_config(
        config,
        config_path="configs/momentum_example.toml",
        parameter_name="lookback",
        values=[1, 2, 3],
        train_periods=4,
        test_periods=2,
        selection_metric="cumulative_return",
        fold_results=fold_results,
        overall_summary=overall_summary,
    )

    assert metadata["command"] == "walk-forward-signal"
    assert metadata["research_context"]["signal_pipeline_metadata"]["factor"][
        "name"
    ] == "momentum"
    assert "data_quality_summary" in metadata["research_context"]

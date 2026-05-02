"""Formatting utilities for conservative walk-forward evaluation results."""

from __future__ import annotations

import pandas as pd

from alphaforge.analytics.performance import format_performance_summary
from alphaforge.common.errors import AlphaForgeError
from alphaforge.common.validation import (
    normalize_non_empty_string_series as _common_string_series,
    parse_numeric_series as _common_numeric_series,
    require_columns as _common_require_columns,
)


class WalkForwardError(AlphaForgeError):
    """Raised when walk-forward result tables or settings are invalid."""


_REQUIRED_WALK_FORWARD_COLUMNS = (
    "fold_index",
    "parameter_name",
    "selected_parameter_value",
    "signal_column",
    "selection_metric",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "train_selection_score",
    "train_cumulative_return",
    "train_mean_ic",
    "test_cumulative_return",
    "test_max_drawdown",
    "test_sharpe_ratio",
    "test_mean_ic",
    "test_joint_coverage_ratio",
)

_NUMERIC_WALK_FORWARD_COLUMNS = (
    "fold_index",
    "selected_parameter_value",
    "train_selection_score",
    "train_cumulative_return",
    "train_mean_ic",
    "test_cumulative_return",
    "test_max_drawdown",
    "test_sharpe_ratio",
    "test_mean_ic",
    "test_joint_coverage_ratio",
)

_STRING_WALK_FORWARD_COLUMNS = (
    "parameter_name",
    "signal_column",
    "selection_metric",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
)


def validate_walk_forward_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a walk-forward fold summary table."""
    if not isinstance(frame, pd.DataFrame):
        raise WalkForwardError("walk-forward results must be a pandas DataFrame.")

    _common_require_columns(
        frame.columns,
        _REQUIRED_WALK_FORWARD_COLUMNS,
        source="walk-forward results",
        error_factory=WalkForwardError,
        verb="are",
    )

    dataset = frame.loc[:, list(_REQUIRED_WALK_FORWARD_COLUMNS)].copy()
    if dataset.empty:
        raise WalkForwardError("walk-forward results must contain at least one row.")

    for column in _NUMERIC_WALK_FORWARD_COLUMNS:
        dataset[column] = _common_numeric_series(
            dataset[column],
            column_name=column,
            source="walk-forward results",
            error_factory=WalkForwardError,
            missing_values_are_invalid=True,
            verb="contain",
        )

    for column in _STRING_WALK_FORWARD_COLUMNS:
        dataset[column] = _common_string_series(
            dataset[column],
            column_name=column,
            source="walk-forward results",
            error_factory=WalkForwardError,
            verb="contain",
        )

    return dataset.reset_index(drop=True)


def format_walk_forward_report(
    fold_results: pd.DataFrame,
    overall_summary: pd.Series,
) -> str:
    """Format walk-forward fold results and combined OOS performance as plain text."""
    dataset = validate_walk_forward_results(fold_results)
    formatted = dataset.copy()

    formatted["fold_index"] = formatted["fold_index"].map(lambda value: str(int(value)))
    formatted["selected_parameter_value"] = formatted["selected_parameter_value"].map(
        lambda value: str(int(value))
    )
    for column in (
        "train_cumulative_return",
        "test_cumulative_return",
        "test_max_drawdown",
        "test_joint_coverage_ratio",
    ):
        formatted[column] = formatted[column].map(_format_percent)
    for column in ("train_mean_ic", "test_sharpe_ratio", "test_mean_ic"):
        formatted[column] = formatted[column].map(_format_number)

    selection_metric = dataset.loc[0, "selection_metric"]
    if selection_metric == "cumulative_return":
        formatted["train_selection_score"] = formatted["train_selection_score"].map(
            _format_percent
        )
    else:
        formatted["train_selection_score"] = formatted["train_selection_score"].map(
            _format_number
        )

    summary_lines = [
        "Walk-Forward Summary",
        f"Folds: {len(dataset)}",
        f"Parameter: {dataset.loc[0, 'parameter_name']}",
        f"Selection Metric: {selection_metric}",
    ]

    return "\n\n".join(
        [
            "\n".join(summary_lines),
            format_performance_summary(overall_summary),
            "Walk-Forward Folds\n" + formatted.to_string(index=False),
        ]
    )


def _format_percent(value: float) -> str:
    """Format a decimal value as a percentage."""
    if pd.isna(value):
        return "NaN"
    return f"{value:.2%}"


def _format_number(value: float) -> str:
    """Format a numeric scalar with two decimals."""
    if pd.isna(value):
        return "NaN"
    return f"{value:.2f}"

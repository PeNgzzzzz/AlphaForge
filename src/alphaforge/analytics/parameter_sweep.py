"""Formatting utilities for simple signal parameter sweep results."""

from __future__ import annotations

import pandas as pd

from alphaforge.common.errors import AlphaForgeError
from alphaforge.common.validation import (
    normalize_non_empty_string_series as _common_string_series,
    parse_numeric_series as _common_numeric_series,
    require_columns as _common_require_columns,
)


class ParameterSweepError(AlphaForgeError):
    """Raised when parameter sweep settings or result tables are invalid."""


_REQUIRED_SWEEP_COLUMNS = (
    "parameter_name",
    "parameter_value",
    "signal_column",
    "cumulative_return",
    "max_drawdown",
    "sharpe_ratio",
    "average_turnover",
    "hit_rate",
    "mean_ic",
    "ic_ir",
    "joint_coverage_ratio",
)

_NUMERIC_SWEEP_COLUMNS = (
    "parameter_value",
    "cumulative_return",
    "max_drawdown",
    "sharpe_ratio",
    "average_turnover",
    "hit_rate",
    "mean_ic",
    "ic_ir",
    "joint_coverage_ratio",
)


def validate_parameter_sweep_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a parameter sweep summary table."""
    if not isinstance(frame, pd.DataFrame):
        raise ParameterSweepError("parameter sweep results must be a pandas DataFrame.")

    _common_require_columns(
        frame.columns,
        _REQUIRED_SWEEP_COLUMNS,
        source="parameter sweep results",
        error_factory=ParameterSweepError,
        verb="are",
    )

    dataset = frame.loc[:, list(_REQUIRED_SWEEP_COLUMNS)].copy()
    if dataset.empty:
        raise ParameterSweepError("parameter sweep results must contain at least one row.")

    for column in _NUMERIC_SWEEP_COLUMNS:
        dataset[column] = _common_numeric_series(
            dataset[column],
            column_name=column,
            source="parameter sweep results",
            error_factory=ParameterSweepError,
            missing_values_are_invalid=True,
            verb="contain",
        )

    for column in ("parameter_name", "signal_column"):
        dataset[column] = _common_string_series(
            dataset[column],
            column_name=column,
            source="parameter sweep results",
            error_factory=ParameterSweepError,
            verb="contain",
        )

    return dataset.reset_index(drop=True)


def format_parameter_sweep_results(frame: pd.DataFrame) -> str:
    """Format a parameter sweep summary table as plain text."""
    dataset = validate_parameter_sweep_results(frame)
    formatted = dataset.copy()

    formatted["parameter_value"] = formatted["parameter_value"].map(lambda value: str(int(value)))
    for column in ("cumulative_return", "max_drawdown", "hit_rate", "joint_coverage_ratio"):
        formatted[column] = formatted[column].map(_format_percent)
    for column in ("sharpe_ratio", "average_turnover", "mean_ic", "ic_ir"):
        formatted[column] = formatted[column].map(_format_number)

    return "Signal Parameter Sweep\n" + formatted.to_string(index=False)


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

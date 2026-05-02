"""Formatting utilities for simple signal parameter sweep results."""

from __future__ import annotations

import pandas as pd

from alphaforge.common.errors import AlphaForgeError


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

    missing_columns = [
        column for column in _REQUIRED_SWEEP_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ParameterSweepError(
            f"parameter sweep results are missing required columns: {missing_text}."
        )

    dataset = frame.loc[:, list(_REQUIRED_SWEEP_COLUMNS)].copy()
    if dataset.empty:
        raise ParameterSweepError("parameter sweep results must contain at least one row.")

    for column in _NUMERIC_SWEEP_COLUMNS:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
        if dataset[column].isna().any():
            raise ParameterSweepError(
                f"parameter sweep results contain invalid numeric values in '{column}'."
            )

    for column in ("parameter_name", "signal_column"):
        values = dataset[column].astype("string").str.strip()
        if values.isna().any() or (values == "").any():
            raise ParameterSweepError(
                f"parameter sweep results contain missing or empty values in '{column}'."
            )
        dataset[column] = values

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

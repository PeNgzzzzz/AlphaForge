"""Internal validation helpers for daily data contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


class DataValidationError(ValueError):
    """Raised when input data fails schema or timing validation."""


def parse_non_empty_strings(
    values: pd.Series,
    *,
    source: str,
    column_name: str,
) -> pd.Series:
    """Normalize non-empty string identifiers."""
    parsed = values.astype("string").str.strip()
    if parsed.isna().any() or (parsed == "").any():
        raise DataValidationError(
            f"{source} contains missing or empty values in '{column_name}'."
        )
    return parsed


def parse_symbols(values: pd.Series, *, source: str) -> pd.Series:
    """Normalize symbol identifiers."""
    return parse_non_empty_strings(
        values,
        source=source,
        column_name="symbol",
    )


def parse_daily_dates(
    values: pd.Series,
    *,
    source: str,
    dataset_label: str,
) -> pd.Series:
    """Parse date-only values without silently truncating timestamps."""
    try:
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    except TypeError:
        parsed = values.map(lambda value: pd.to_datetime(value, errors="coerce"))
    if parsed.isna().any():
        raise DataValidationError(f"{source} contains missing or invalid date values.")

    tz = parsed.dt.tz
    if tz is not None:
        raise DataValidationError(
            f"{source} contains timezone-aware dates, which are ambiguous for daily {dataset_label}."
        )

    normalized = parsed.dt.normalize()
    if (parsed != normalized).any():
        raise DataValidationError(
            f"{source} contains intraday timestamps; daily {dataset_label} must use date-only values."
        )

    return normalized.astype("datetime64[ns]")


def parse_optional_daily_dates(
    values: pd.Series,
    *,
    source: str,
    dataset_label: str,
) -> pd.Series:
    """Parse optional date-only values while allowing missing entries."""
    text_values = values.astype("string")
    non_missing = values.notna() & text_values.str.strip().ne("")
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if not bool(non_missing.any()):
        return parsed

    parsed.loc[non_missing] = parse_daily_dates(
        values.loc[non_missing],
        source=source,
        dataset_label=dataset_label,
    )
    return parsed


def parse_numeric_column(
    values: pd.Series,
    *,
    column_name: str,
    source: str,
) -> pd.Series:
    """Parse numeric columns without silent coercion."""
    parsed = pd.to_numeric(values, errors="coerce")
    if parsed.isna().any():
        raise DataValidationError(
            f"{source} contains missing or invalid numeric values in '{column_name}'."
        )
    return parsed


def parse_optional_numeric_column(
    values: pd.Series,
    *,
    column_name: str,
    source: str,
) -> pd.Series:
    """Parse numeric columns while allowing blank or missing entries."""
    text_values = values.astype("string")
    non_missing = values.notna() & text_values.str.strip().ne("")
    parsed = pd.Series(float("nan"), index=values.index, dtype="float64")
    if not bool(non_missing.any()):
        return parsed

    parsed.loc[non_missing] = parse_numeric_column(
        values.loc[non_missing],
        column_name=column_name,
        source=source,
    ).astype("float64")
    return parsed


def parse_boolean_flags(
    values: pd.Series,
    *,
    source: str,
    column_name: str,
) -> pd.Series:
    """Normalize strict bool-like flags into a stable boolean dtype."""
    parsed: list[bool] = []
    for value in values.tolist():
        if pd.isna(value):
            raise DataValidationError(
                f"{source} contains missing boolean values in '{column_name}'."
            )
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
            continue
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            if value in {0, 1}:
                parsed.append(bool(value))
                continue
        if isinstance(value, (float, np.floating)):
            if np.isfinite(value) and value in {0.0, 1.0}:
                parsed.append(bool(int(value)))
                continue
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1"}:
                parsed.append(True)
                continue
            if normalized in {"false", "0"}:
                parsed.append(False)
                continue
        raise DataValidationError(
            f"{source} contains invalid boolean values in '{column_name}'; "
            "expected bool/0/1/true/false."
        )

    return pd.Series(parsed, index=values.index, dtype="bool")

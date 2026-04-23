"""Internal validation helpers for daily data contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]


class DataValidationError(ValueError):
    """Raised when input data fails schema or timing validation."""


def parse_symbols(values: pd.Series, *, source: str) -> pd.Series:
    """Normalize symbol identifiers."""
    parsed = values.astype("string").str.strip()
    if parsed.isna().any() or (parsed == "").any():
        raise DataValidationError(f"{source} contains missing or empty symbol values.")
    return parsed


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

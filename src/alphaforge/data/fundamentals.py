"""Fundamentals loading and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphaforge.data._validation import (
    DataValidationError,
    PathLike,
    parse_daily_dates,
    parse_non_empty_strings,
    parse_numeric_column,
    parse_symbols,
)

CANONICAL_FUNDAMENTALS_COLUMNS = (
    "symbol",
    "period_end_date",
    "release_date",
    "metric_name",
    "metric_value",
)


def load_fundamentals(
    path: PathLike,
    *,
    period_end_column: str = "period_end_date",
    release_date_column: str = "release_date",
    metric_name_column: str = "metric_name",
    metric_value_column: str = "metric_value",
) -> pd.DataFrame:
    """Load and validate long-form fundamentals from CSV or Parquet."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        frame = pd.read_csv(file_path)
    elif suffix == ".parquet":
        frame = pd.read_parquet(file_path)
    else:
        raise ValueError(
            f"Unsupported file format for {file_path}. Expected .csv or .parquet."
        )

    return validate_fundamentals(
        frame,
        period_end_column=period_end_column,
        release_date_column=release_date_column,
        metric_name_column=metric_name_column,
        metric_value_column=metric_value_column,
        source=str(file_path),
    )


def validate_fundamentals(
    frame: pd.DataFrame,
    *,
    period_end_column: str = "period_end_date",
    release_date_column: str = "release_date",
    metric_name_column: str = "metric_name",
    metric_value_column: str = "metric_value",
    source: str = "dataframe",
) -> pd.DataFrame:
    """Validate a long-form fundamentals frame and return a normalized copy."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("validate_fundamentals expects a pandas DataFrame.")

    for field_name, column_name in {
        "period_end_column": period_end_column,
        "release_date_column": release_date_column,
        "metric_name_column": metric_name_column,
        "metric_value_column": metric_value_column,
    }.items():
        if not isinstance(column_name, str) or not column_name.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")

    required_columns = [
        "symbol",
        period_end_column,
        release_date_column,
        metric_name_column,
        metric_value_column,
    ]
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise DataValidationError(
            f"{source} is missing required fundamentals columns: {missing_text}."
        )

    validated = frame.copy()
    validated["symbol"] = parse_symbols(validated["symbol"], source=source)
    validated["period_end_date"] = parse_daily_dates(
        validated[period_end_column],
        source=source,
        dataset_label="fundamentals data",
    )
    validated["release_date"] = parse_daily_dates(
        validated[release_date_column],
        source=source,
        dataset_label="fundamentals data",
    )
    validated["metric_name"] = parse_non_empty_strings(
        validated[metric_name_column],
        source=source,
        column_name=metric_name_column,
    )
    metric_values = parse_numeric_column(
        validated[metric_value_column],
        column_name=metric_value_column,
        source=source,
    )
    invalid_metric_values = ~np.isfinite(metric_values)
    if invalid_metric_values.any():
        raise DataValidationError(
            f"{source} contains metric values in '{metric_value_column}' that must be finite."
        )
    validated["metric_value"] = metric_values.astype("float64")

    invalid_release_dates = validated["release_date"].lt(validated["period_end_date"])
    if invalid_release_dates.any():
        raise DataValidationError(
            f"{source} contains rows where release_date is earlier than period_end_date."
        )

    duplicate_rows = validated.duplicated(
        subset=[
            "symbol",
            "period_end_date",
            "release_date",
            "metric_name",
        ],
        keep=False,
    )
    if duplicate_rows.any():
        duplicates = (
            validated.loc[
                duplicate_rows,
                ["symbol", "period_end_date", "release_date", "metric_name"],
            ]
            .drop_duplicates()
            .sort_values(
                ["symbol", "period_end_date", "release_date", "metric_name"],
                kind="mergesort",
            )
        )
        sample = ", ".join(
            (
                f"{row.symbol}@{row.period_end_date.date().isoformat()}"
                f"/{row.release_date.date().isoformat()}:{row.metric_name}"
            )
            for row in duplicates.itertuples(index=False)
        )
        raise DataValidationError(
            f"{source} contains duplicate fundamentals keys: {sample}."
        )

    extra_columns = [
        column
        for column in validated.columns
        if column
        not in {
            "symbol",
            period_end_column,
            release_date_column,
            metric_name_column,
            metric_value_column,
            "period_end_date",
            "release_date",
            "metric_name",
            "metric_value",
        }
    ]
    validated = validated.loc[:, [*CANONICAL_FUNDAMENTALS_COLUMNS, *extra_columns]]
    if validated.empty:
        raise DataValidationError("Fundamentals data must contain at least one row.")

    validated = validated.sort_values(
        ["symbol", "period_end_date", "release_date", "metric_name"],
        kind="mergesort",
    )
    return validated.reset_index(drop=True)

"""Benchmark return-series loading and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphaforge.data._validation import (
    DataValidationError,
    PathLike,
    parse_daily_dates,
    parse_numeric_column,
)

CANONICAL_BENCHMARK_COLUMNS = ("date", "benchmark_return")


def load_benchmark_returns(
    path: PathLike,
    *,
    return_column: str = "benchmark_return",
) -> pd.DataFrame:
    """Load and validate a daily benchmark return series from CSV or Parquet."""
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

    return validate_benchmark_returns(
        frame,
        return_column=return_column,
        source=str(file_path),
    )


def validate_benchmark_returns(
    frame: pd.DataFrame,
    *,
    return_column: str = "benchmark_return",
    source: str = "dataframe",
) -> pd.DataFrame:
    """Validate a daily benchmark return series and return a normalized copy."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("validate_benchmark_returns expects a pandas DataFrame.")
    if not isinstance(return_column, str) or not return_column.strip():
        raise ValueError("return_column must be a non-empty string.")

    required_columns = ("date", return_column)
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise DataValidationError(
            f"{source} is missing required benchmark columns: {missing_text}."
        )

    validated = frame.copy()
    validated["date"] = parse_daily_dates(
        validated["date"],
        source=source,
        dataset_label="benchmark return data",
    )
    benchmark_returns = parse_numeric_column(
        validated[return_column],
        column_name=return_column,
        source=source,
    )

    invalid_returns = (~np.isfinite(benchmark_returns)) | (
        benchmark_returns <= -1.0
    )
    if invalid_returns.any():
        raise DataValidationError(
            f"{source} contains benchmark returns in '{return_column}' that must be finite and greater than -1.0."
        )

    duplicate_rows = validated["date"].duplicated(keep=False)
    if duplicate_rows.any():
        duplicate_dates = (
            validated.loc[duplicate_rows, "date"].drop_duplicates().sort_values()
        )
        sample = ", ".join(timestamp.date().isoformat() for timestamp in duplicate_dates)
        raise DataValidationError(
            f"{source} contains duplicate benchmark dates: {sample}."
        )

    validated["benchmark_return"] = benchmark_returns
    extra_columns = [
        column
        for column in validated.columns
        if column not in {"date", return_column, "benchmark_return"}
    ]
    validated = validated.loc[:, [*CANONICAL_BENCHMARK_COLUMNS, *extra_columns]]
    if validated.empty:
        raise DataValidationError("Benchmark data must contain at least one row.")

    validated = validated.sort_values("date", kind="mergesort")
    return validated.reset_index(drop=True)

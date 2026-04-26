"""Shares outstanding loading and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphaforge.data._validation import (
    DataValidationError,
    PathLike,
    parse_daily_dates,
    parse_numeric_column,
    parse_symbols,
)

CANONICAL_SHARES_OUTSTANDING_COLUMNS = (
    "symbol",
    "effective_date",
    "shares_outstanding",
)


def load_shares_outstanding(
    path: PathLike,
    *,
    effective_date_column: str = "effective_date",
    shares_outstanding_column: str = "shares_outstanding",
) -> pd.DataFrame:
    """Load and validate shares-outstanding observations from CSV or Parquet."""
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

    return validate_shares_outstanding(
        frame,
        effective_date_column=effective_date_column,
        shares_outstanding_column=shares_outstanding_column,
        source=str(file_path),
    )


def validate_shares_outstanding(
    frame: pd.DataFrame,
    *,
    effective_date_column: str = "effective_date",
    shares_outstanding_column: str = "shares_outstanding",
    source: str = "dataframe",
) -> pd.DataFrame:
    """Validate shares-outstanding observations and return a normalized copy."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("validate_shares_outstanding expects a pandas DataFrame.")

    for field_name, column_name in {
        "effective_date_column": effective_date_column,
        "shares_outstanding_column": shares_outstanding_column,
    }.items():
        if not isinstance(column_name, str) or not column_name.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")

    required_columns = ["symbol", effective_date_column, shares_outstanding_column]
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise DataValidationError(
            f"{source} is missing required shares outstanding columns: {missing_text}."
        )

    validated = frame.copy()
    validated["symbol"] = parse_symbols(validated["symbol"], source=source)
    validated["effective_date"] = parse_daily_dates(
        validated[effective_date_column],
        source=source,
        dataset_label="shares outstanding data",
    )
    shares = parse_numeric_column(
        validated[shares_outstanding_column],
        column_name=shares_outstanding_column,
        source=source,
    )
    invalid_shares = ~np.isfinite(shares) | shares.le(0.0)
    if invalid_shares.any():
        raise DataValidationError(
            f"{source} contains invalid shares outstanding values in "
            f"'{shares_outstanding_column}'; values must be finite and positive."
        )
    validated["shares_outstanding"] = shares.astype("float64")

    duplicated = validated.duplicated(
        subset=["symbol", "effective_date"],
        keep=False,
    )
    if duplicated.any():
        duplicate_rows = (
            validated.loc[duplicated, ["symbol", "effective_date"]]
            .drop_duplicates()
            .sort_values(["symbol", "effective_date"], kind="mergesort")
        )
        sample = ", ".join(
            f"{row.symbol}@{row.effective_date.date().isoformat()}"
            for row in duplicate_rows.itertuples(index=False)
        )
        raise DataValidationError(
            f"{source} contains duplicate shares outstanding keys: {sample}."
        )

    extra_columns = [
        column
        for column in validated.columns
        if column
        not in {
            "symbol",
            effective_date_column,
            shares_outstanding_column,
            "effective_date",
            "shares_outstanding",
        }
    ]
    validated = validated.loc[
        :, [*CANONICAL_SHARES_OUTSTANDING_COLUMNS, *extra_columns]
    ]
    if validated.empty:
        raise DataValidationError("Shares outstanding data must contain at least one row.")

    validated = validated.sort_values(
        ["symbol", "effective_date"],
        kind="mergesort",
    )
    return validated.reset_index(drop=True)

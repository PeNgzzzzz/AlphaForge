"""Borrow availability loading and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphaforge.data._validation import (
    DataValidationError,
    PathLike,
    parse_boolean_flags,
    parse_daily_dates,
    parse_optional_numeric_column,
    parse_symbols,
)

CANONICAL_BORROW_AVAILABILITY_COLUMNS = (
    "symbol",
    "effective_date",
    "is_borrowable",
    "borrow_fee_bps",
)


def load_borrow_availability(
    path: PathLike,
    *,
    effective_date_column: str = "effective_date",
    is_borrowable_column: str = "is_borrowable",
    borrow_fee_bps_column: str = "borrow_fee_bps",
) -> pd.DataFrame:
    """Load and validate borrow availability events from CSV or Parquet."""
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

    return validate_borrow_availability(
        frame,
        effective_date_column=effective_date_column,
        is_borrowable_column=is_borrowable_column,
        borrow_fee_bps_column=borrow_fee_bps_column,
        source=str(file_path),
    )


def validate_borrow_availability(
    frame: pd.DataFrame,
    *,
    effective_date_column: str = "effective_date",
    is_borrowable_column: str = "is_borrowable",
    borrow_fee_bps_column: str = "borrow_fee_bps",
    source: str = "dataframe",
) -> pd.DataFrame:
    """Validate a borrow availability event frame and return a normalized copy."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("validate_borrow_availability expects a pandas DataFrame.")

    for field_name, column_name in {
        "effective_date_column": effective_date_column,
        "is_borrowable_column": is_borrowable_column,
        "borrow_fee_bps_column": borrow_fee_bps_column,
    }.items():
        if not isinstance(column_name, str) or not column_name.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")

    required_columns = [
        "symbol",
        effective_date_column,
        is_borrowable_column,
    ]
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise DataValidationError(
            f"{source} is missing required borrow availability columns: {missing_text}."
        )

    validated = frame.copy()
    validated["symbol"] = parse_symbols(validated["symbol"], source=source)
    validated["effective_date"] = parse_daily_dates(
        validated[effective_date_column],
        source=source,
        dataset_label="borrow availability data",
    )
    validated["is_borrowable"] = parse_boolean_flags(
        validated[is_borrowable_column],
        source=source,
        column_name=is_borrowable_column,
    )

    if borrow_fee_bps_column in validated.columns:
        borrow_fee_bps = parse_optional_numeric_column(
            validated[borrow_fee_bps_column],
            column_name=borrow_fee_bps_column,
            source=source,
        )
    else:
        borrow_fee_bps = pd.Series(float("nan"), index=validated.index, dtype="float64")

    invalid_borrow_fee_bps = borrow_fee_bps.notna() & (
        ~np.isfinite(borrow_fee_bps) | borrow_fee_bps.lt(0.0)
    )
    if invalid_borrow_fee_bps.any():
        raise DataValidationError(
            f"{source} contains invalid borrow fee values in "
            f"'{borrow_fee_bps_column}'; values must be finite and non-negative."
        )
    validated["borrow_fee_bps"] = borrow_fee_bps.astype("float64")

    duplicated = validated.duplicated(
        subset=["symbol", "effective_date"],
        keep=False,
    )
    if duplicated.any():
        duplicate_rows = (
            validated.loc[duplicated, ["symbol", "effective_date"]]
            .drop_duplicates()
            .sort_values(["symbol", "effective_date"])
        )
        sample = ", ".join(
            f"{row.symbol}@{row.effective_date.date().isoformat()}"
            for row in duplicate_rows.itertuples(index=False)
        )
        raise DataValidationError(
            f"{source} contains duplicate borrow availability keys: {sample}."
        )

    extra_columns = [
        column
        for column in validated.columns
        if column
        not in {
            "symbol",
            effective_date_column,
            is_borrowable_column,
            borrow_fee_bps_column,
            "effective_date",
            "is_borrowable",
            "borrow_fee_bps",
        }
    ]
    validated = validated.loc[
        :, [*CANONICAL_BORROW_AVAILABILITY_COLUMNS, *extra_columns]
    ]
    if validated.empty:
        raise DataValidationError("Borrow availability data must contain at least one row.")

    validated = validated.sort_values(
        ["symbol", "effective_date"],
        kind="mergesort",
    )
    return validated.reset_index(drop=True)

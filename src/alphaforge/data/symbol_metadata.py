"""Symbol metadata loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alphaforge.data._validation import (
    DataValidationError,
    PathLike,
    parse_daily_dates,
    parse_optional_daily_dates,
    parse_symbols,
)

CANONICAL_SYMBOL_METADATA_COLUMNS = (
    "symbol",
    "listing_date",
    "delisting_date",
)


def load_symbol_metadata(
    path: PathLike,
    *,
    listing_date_column: str = "listing_date",
    delisting_date_column: str = "delisting_date",
) -> pd.DataFrame:
    """Load and validate daily symbol metadata from CSV or Parquet."""
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

    return validate_symbol_metadata(
        frame,
        listing_date_column=listing_date_column,
        delisting_date_column=delisting_date_column,
        source=str(file_path),
    )


def validate_symbol_metadata(
    frame: pd.DataFrame,
    *,
    listing_date_column: str = "listing_date",
    delisting_date_column: str = "delisting_date",
    source: str = "dataframe",
) -> pd.DataFrame:
    """Validate a symbol metadata frame and return a normalized copy."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("validate_symbol_metadata expects a pandas DataFrame.")
    for field_name, column_name in {
        "listing_date_column": listing_date_column,
        "delisting_date_column": delisting_date_column,
    }.items():
        if not isinstance(column_name, str) or not column_name.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")

    required_columns = ["symbol", listing_date_column]
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise DataValidationError(
            f"{source} is missing required symbol metadata columns: {missing_text}."
        )

    validated = frame.copy()
    validated["symbol"] = parse_symbols(validated["symbol"], source=source)
    validated["listing_date"] = parse_daily_dates(
        validated[listing_date_column],
        source=source,
        dataset_label="symbol metadata",
    )
    if delisting_date_column in validated.columns:
        validated["delisting_date"] = parse_optional_daily_dates(
            validated[delisting_date_column],
            source=source,
            dataset_label="symbol metadata",
        )
    else:
        validated["delisting_date"] = pd.Series(
            pd.NaT,
            index=validated.index,
            dtype="datetime64[ns]",
        )

    duplicate_rows = validated["symbol"].duplicated(keep=False)
    if duplicate_rows.any():
        duplicate_symbols = (
            validated.loc[duplicate_rows, "symbol"].drop_duplicates().sort_values()
        )
        sample = ", ".join(str(symbol) for symbol in duplicate_symbols.tolist())
        raise DataValidationError(f"{source} contains duplicate symbols: {sample}.")

    invalid_delisting = (
        validated["delisting_date"].notna()
        & validated["delisting_date"].lt(validated["listing_date"])
    )
    if invalid_delisting.any():
        raise DataValidationError(
            f"{source} contains rows where delisting_date is earlier than listing_date."
        )

    extra_columns = [
        column
        for column in validated.columns
        if column
        not in {
            "symbol",
            listing_date_column,
            delisting_date_column,
            "listing_date",
            "delisting_date",
        }
    ]
    validated = validated.loc[:, [*CANONICAL_SYMBOL_METADATA_COLUMNS, *extra_columns]]
    if validated.empty:
        raise DataValidationError("Symbol metadata must contain at least one row.")

    validated = validated.sort_values(["symbol"], kind="mergesort")
    return validated.reset_index(drop=True)

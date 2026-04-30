"""Trading status loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alphaforge.data._validation import (
    DataValidationError,
    PathLike,
    parse_boolean_flags,
    parse_daily_dates,
    parse_symbols,
)

CANONICAL_TRADING_STATUS_COLUMNS = (
    "symbol",
    "effective_date",
    "is_tradable",
    "status_reason",
)


def load_trading_status(
    path: PathLike,
    *,
    effective_date_column: str = "effective_date",
    is_tradable_column: str = "is_tradable",
    status_reason_column: str = "status_reason",
) -> pd.DataFrame:
    """Load and validate trading status events from CSV or Parquet."""
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

    return validate_trading_status(
        frame,
        effective_date_column=effective_date_column,
        is_tradable_column=is_tradable_column,
        status_reason_column=status_reason_column,
        source=str(file_path),
    )


def validate_trading_status(
    frame: pd.DataFrame,
    *,
    effective_date_column: str = "effective_date",
    is_tradable_column: str = "is_tradable",
    status_reason_column: str = "status_reason",
    source: str = "dataframe",
) -> pd.DataFrame:
    """Validate a trading status event frame and return a normalized copy."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("validate_trading_status expects a pandas DataFrame.")

    for field_name, column_name in {
        "effective_date_column": effective_date_column,
        "is_tradable_column": is_tradable_column,
        "status_reason_column": status_reason_column,
    }.items():
        if not isinstance(column_name, str) or not column_name.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")

    required_columns = [
        "symbol",
        effective_date_column,
        is_tradable_column,
    ]
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise DataValidationError(
            f"{source} is missing required trading status columns: {missing_text}."
        )

    validated = frame.copy()
    validated["symbol"] = parse_symbols(validated["symbol"], source=source)
    validated["effective_date"] = parse_daily_dates(
        validated[effective_date_column],
        source=source,
        dataset_label="trading status data",
    )
    validated["is_tradable"] = parse_boolean_flags(
        validated[is_tradable_column],
        source=source,
        column_name=is_tradable_column,
    )

    if status_reason_column in validated.columns:
        validated["status_reason"] = _parse_optional_reason(
            validated[status_reason_column],
            source=source,
            column_name=status_reason_column,
        )
    else:
        validated["status_reason"] = pd.Series(
            pd.NA,
            index=validated.index,
            dtype="string",
        )

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
            f"{source} contains duplicate trading status keys: {sample}."
        )

    extra_columns = [
        column
        for column in validated.columns
        if column
        not in {
            "symbol",
            effective_date_column,
            is_tradable_column,
            status_reason_column,
            "effective_date",
            "is_tradable",
            "status_reason",
        }
    ]
    validated = validated.loc[:, [*CANONICAL_TRADING_STATUS_COLUMNS, *extra_columns]]
    if validated.empty:
        raise DataValidationError("Trading status data must contain at least one row.")

    validated = validated.sort_values(
        ["symbol", "effective_date"],
        kind="mergesort",
    )
    return validated.reset_index(drop=True)


def _parse_optional_reason(
    values: pd.Series,
    *,
    source: str,
    column_name: str,
) -> pd.Series:
    """Normalize optional status reason text while preserving missing values."""
    parsed = values.astype("string").str.strip()
    non_missing = values.notna() & parsed.ne("")
    output = pd.Series(pd.NA, index=values.index, dtype="string")
    output.loc[non_missing] = parsed.loc[non_missing]
    if parsed.loc[non_missing].isna().any():
        raise DataValidationError(
            f"{source} contains invalid text values in '{column_name}'."
        )
    return output

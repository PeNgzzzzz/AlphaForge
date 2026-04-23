"""Index membership loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alphaforge.data._validation import (
    DataValidationError,
    PathLike,
    parse_boolean_flags,
    parse_daily_dates,
    parse_non_empty_strings,
    parse_symbols,
)

CANONICAL_MEMBERSHIP_COLUMNS = (
    "symbol",
    "effective_date",
    "index_name",
    "is_member",
)


def load_memberships(
    path: PathLike,
    *,
    effective_date_column: str = "effective_date",
    index_column: str = "index_name",
    is_member_column: str = "is_member",
) -> pd.DataFrame:
    """Load and validate index membership events from CSV or Parquet."""
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

    return validate_memberships(
        frame,
        effective_date_column=effective_date_column,
        index_column=index_column,
        is_member_column=is_member_column,
        source=str(file_path),
    )


def validate_memberships(
    frame: pd.DataFrame,
    *,
    effective_date_column: str = "effective_date",
    index_column: str = "index_name",
    is_member_column: str = "is_member",
    source: str = "dataframe",
) -> pd.DataFrame:
    """Validate an index membership event frame and return a normalized copy."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("validate_memberships expects a pandas DataFrame.")

    for field_name, column_name in {
        "effective_date_column": effective_date_column,
        "index_column": index_column,
        "is_member_column": is_member_column,
    }.items():
        if not isinstance(column_name, str) or not column_name.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")

    required_columns = [
        "symbol",
        effective_date_column,
        index_column,
        is_member_column,
    ]
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise DataValidationError(
            f"{source} is missing required memberships columns: {missing_text}."
        )

    validated = frame.copy()
    validated["symbol"] = parse_symbols(validated["symbol"], source=source)
    validated["effective_date"] = parse_daily_dates(
        validated[effective_date_column],
        source=source,
        dataset_label="memberships data",
    )
    validated["index_name"] = parse_non_empty_strings(
        validated[index_column],
        source=source,
        column_name=index_column,
    )
    validated["is_member"] = parse_boolean_flags(
        validated[is_member_column],
        source=source,
        column_name=is_member_column,
    )

    duplicated = validated.duplicated(
        subset=["symbol", "effective_date", "index_name"],
        keep=False,
    )
    if duplicated.any():
        duplicate_rows = (
            validated.loc[duplicated, ["symbol", "effective_date", "index_name"]]
            .drop_duplicates()
            .sort_values(["symbol", "effective_date", "index_name"])
        )
        sample = ", ".join(
            (
                f"{row.symbol}@{row.effective_date.date().isoformat()}:"
                f"{row.index_name}"
            )
            for row in duplicate_rows.itertuples(index=False)
        )
        raise DataValidationError(
            f"{source} contains duplicate memberships keys: {sample}."
        )

    extra_columns = [
        column
        for column in validated.columns
        if column
        not in {
            "symbol",
            effective_date_column,
            index_column,
            is_member_column,
            "effective_date",
            "index_name",
            "is_member",
        }
    ]
    validated = validated.loc[:, [*CANONICAL_MEMBERSHIP_COLUMNS, *extra_columns]]
    if validated.empty:
        raise DataValidationError("Memberships data must contain at least one row.")

    validated = validated.sort_values(
        ["symbol", "index_name", "effective_date"],
        kind="mergesort",
    )
    return validated.reset_index(drop=True)

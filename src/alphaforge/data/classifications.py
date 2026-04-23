"""Sector and industry classification loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alphaforge.data._validation import (
    DataValidationError,
    PathLike,
    parse_daily_dates,
    parse_non_empty_strings,
    parse_symbols,
)

CANONICAL_CLASSIFICATION_COLUMNS = (
    "symbol",
    "effective_date",
    "sector",
    "industry",
)


def load_classifications(
    path: PathLike,
    *,
    effective_date_column: str = "effective_date",
    sector_column: str = "sector",
    industry_column: str = "industry",
) -> pd.DataFrame:
    """Load and validate daily sector/industry classifications from CSV or Parquet."""
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

    return validate_classifications(
        frame,
        effective_date_column=effective_date_column,
        sector_column=sector_column,
        industry_column=industry_column,
        source=str(file_path),
    )


def validate_classifications(
    frame: pd.DataFrame,
    *,
    effective_date_column: str = "effective_date",
    sector_column: str = "sector",
    industry_column: str = "industry",
    source: str = "dataframe",
) -> pd.DataFrame:
    """Validate a sector/industry classifications frame and return a normalized copy."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("validate_classifications expects a pandas DataFrame.")

    for field_name, column_name in {
        "effective_date_column": effective_date_column,
        "sector_column": sector_column,
        "industry_column": industry_column,
    }.items():
        if not isinstance(column_name, str) or not column_name.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")

    required_columns = [
        "symbol",
        effective_date_column,
        sector_column,
        industry_column,
    ]
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise DataValidationError(
            f"{source} is missing required classifications columns: {missing_text}."
        )

    validated = frame.copy()
    validated["symbol"] = parse_symbols(validated["symbol"], source=source)
    validated["effective_date"] = parse_daily_dates(
        validated[effective_date_column],
        source=source,
        dataset_label="classifications data",
    )
    validated["sector"] = parse_non_empty_strings(
        validated[sector_column],
        source=source,
        column_name=sector_column,
    )
    validated["industry"] = parse_non_empty_strings(
        validated[industry_column],
        source=source,
        column_name=industry_column,
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
            f"({row.symbol}, {row.effective_date.date().isoformat()})"
            for row in duplicate_rows.itertuples(index=False)
        )
        raise DataValidationError(
            f"{source} contains duplicate classifications keys: {sample}."
        )

    extra_columns = [
        column
        for column in validated.columns
        if column
        not in {
            "symbol",
            effective_date_column,
            sector_column,
            industry_column,
            "effective_date",
            "sector",
            "industry",
        }
    ]
    validated = validated.loc[:, [*CANONICAL_CLASSIFICATION_COLUMNS, *extra_columns]]
    if validated.empty:
        raise DataValidationError("Classifications must contain at least one row.")

    validated = validated.sort_values(
        ["symbol", "effective_date"],
        kind="mergesort",
    )
    return validated.reset_index(drop=True)

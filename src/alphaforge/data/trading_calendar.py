"""Trading calendar loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alphaforge.data._validation import (
    DataValidationError,
    PathLike,
    parse_daily_dates,
)

CANONICAL_TRADING_CALENDAR_COLUMNS = ("date",)


def load_trading_calendar(
    path: PathLike,
    *,
    date_column: str = "date",
) -> pd.DataFrame:
    """Load and validate a daily trading calendar from CSV or Parquet."""
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

    return validate_trading_calendar(
        frame,
        date_column=date_column,
        source=str(file_path),
    )


def validate_trading_calendar(
    frame: pd.DataFrame,
    *,
    date_column: str = "date",
    source: str = "dataframe",
) -> pd.DataFrame:
    """Validate a trading-calendar frame and return a normalized copy."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("validate_trading_calendar expects a pandas DataFrame.")
    if not isinstance(date_column, str) or not date_column.strip():
        raise ValueError("date_column must be a non-empty string.")
    if date_column not in frame.columns:
        raise DataValidationError(
            f"{source} is missing required trading calendar columns: {date_column}."
        )

    validated = frame.copy()
    validated["date"] = parse_daily_dates(
        validated[date_column],
        source=source,
        dataset_label="trading calendar data",
    )

    duplicate_rows = validated["date"].duplicated(keep=False)
    if duplicate_rows.any():
        duplicate_dates = (
            validated.loc[duplicate_rows, "date"].drop_duplicates().sort_values()
        )
        sample = ", ".join(timestamp.date().isoformat() for timestamp in duplicate_dates)
        raise DataValidationError(
            f"{source} contains duplicate trading calendar dates: {sample}."
        )

    extra_columns = [
        column for column in validated.columns if column not in {date_column, "date"}
    ]
    validated = validated.loc[:, [*CANONICAL_TRADING_CALENDAR_COLUMNS, *extra_columns]]
    if validated.empty:
        raise DataValidationError("Trading calendar must contain at least one row.")

    validated = validated.sort_values("date", kind="mergesort")
    return validated.reset_index(drop=True)


def ensure_dates_on_trading_calendar(
    dates: pd.Series,
    trading_calendar: pd.DataFrame,
    *,
    source: str,
) -> None:
    """Require every date in a series to exist in the configured trading calendar."""
    calendar_dates = pd.Index(
        validate_trading_calendar(
            trading_calendar,
            source="trading calendar input",
        )["date"]
    )
    observed_dates = pd.Index(dates.dropna().drop_duplicates().sort_values())
    missing_dates = observed_dates.difference(calendar_dates)
    if missing_dates.empty:
        return

    sample = ", ".join(timestamp.date().isoformat() for timestamp in missing_dates[:10])
    raise DataValidationError(
        f"{source} contains dates not present in the configured trading calendar: {sample}."
    )

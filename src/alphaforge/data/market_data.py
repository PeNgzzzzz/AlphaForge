"""Daily OHLCV market data loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alphaforge.data._validation import (
    DataValidationError,
    PathLike,
    parse_daily_dates,
    parse_numeric_column,
    parse_symbols,
)

CANONICAL_OHLCV_COLUMNS = (
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
_NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume")
_PRICE_COLUMNS = ("open", "high", "low", "close")


def load_ohlcv(path: PathLike) -> pd.DataFrame:
    """Load and validate daily OHLCV data from CSV or Parquet."""
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

    return validate_ohlcv(frame, source=str(file_path))


def validate_ohlcv(frame: pd.DataFrame, *, source: str = "dataframe") -> pd.DataFrame:
    """Validate a daily OHLCV frame and return a normalized copy."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("validate_ohlcv expects a pandas DataFrame.")

    missing_columns = [
        column for column in CANONICAL_OHLCV_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise DataValidationError(
            f"{source} is missing required OHLCV columns: {missing_text}."
        )

    validated = frame.copy()
    validated["date"] = parse_daily_dates(
        validated["date"],
        source=source,
        dataset_label="OHLCV data",
    )
    validated["symbol"] = parse_symbols(validated["symbol"], source=source)

    for column in _NUMERIC_COLUMNS:
        validated[column] = parse_numeric_column(
            validated[column], column_name=column, source=source
        )
    _validate_price_and_volume_integrity(validated, source=source)

    duplicate_rows = validated.duplicated(subset=["symbol", "date"], keep=False)
    if duplicate_rows.any():
        duplicate_keys = (
            validated.loc[duplicate_rows, ["symbol", "date"]]
            .drop_duplicates()
            .sort_values(["symbol", "date"])
        )
        sample = ", ".join(
            f"({row.symbol}, {row.date.date().isoformat()})"
            for row in duplicate_keys.itertuples(index=False)
        )
        raise DataValidationError(
            f"{source} contains duplicate symbol/date rows: {sample}."
        )

    ordered_columns = list(CANONICAL_OHLCV_COLUMNS) + [
        column for column in validated.columns if column not in CANONICAL_OHLCV_COLUMNS
    ]
    validated = validated.loc[:, ordered_columns]
    validated = validated.sort_values(["symbol", "date"], kind="mergesort")
    return validated.reset_index(drop=True)
def _validate_price_and_volume_integrity(
    frame: pd.DataFrame, *, source: str
) -> None:
    """Validate basic daily OHLCV integrity constraints."""
    non_positive_price_columns = [
        column for column in _PRICE_COLUMNS if (frame[column] <= 0.0).any()
    ]
    if non_positive_price_columns:
        columns_text = ", ".join(non_positive_price_columns)
        raise DataValidationError(
            f"{source} contains non-positive price values in: {columns_text}."
        )

    if (frame["volume"] < 0.0).any():
        raise DataValidationError(f"{source} contains negative volume values.")

    if (frame["high"] < frame["low"]).any():
        raise DataValidationError(f"{source} contains rows where high is below low.")

    open_close_outside_range = (
        frame["open"].gt(frame["high"])
        | frame["open"].lt(frame["low"])
        | frame["close"].gt(frame["high"])
        | frame["close"].lt(frame["low"])
    )
    if open_close_outside_range.any():
        raise DataValidationError(
            f"{source} contains rows where open/close fall outside the daily low/high range."
        )

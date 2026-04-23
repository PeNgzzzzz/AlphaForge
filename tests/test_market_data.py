"""Tests for the daily OHLCV market data layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from alphaforge.data import (
    CANONICAL_OHLCV_COLUMNS,
    DataValidationError,
    load_ohlcv,
    validate_ohlcv,
)


def test_validate_ohlcv_sorts_and_normalizes_values() -> None:
    """Validated data should be sorted by symbol/date with parsed dtypes."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-02", "2024-01-02"],
            "symbol": ["MSFT", "AAPL", "MSFT"],
            "open": ["372.5", "182.0", "370.0"],
            "high": [374.0, 183.1, 373.2],
            "low": [369.8, 181.5, 369.0],
            "close": [373.5, 182.9, 372.0],
            "volume": ["21000000", "30500000", "19800000"],
            "exchange": ["XNAS", "XNAS", "XNAS"],
        }
    )

    validated = validate_ohlcv(frame)

    assert list(validated.columns) == [*CANONICAL_OHLCV_COLUMNS, "exchange"]
    assert validated["symbol"].tolist() == ["AAPL", "MSFT", "MSFT"]
    assert validated["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-02",
        "2024-01-03",
    ]
    assert pd.api.types.is_datetime64_ns_dtype(validated["date"])
    assert validated.loc[0, "open"] == pytest.approx(182.0)


def test_validate_ohlcv_rejects_missing_required_columns() -> None:
    """Missing OHLCV columns should fail loudly."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "symbol": ["AAPL"],
            "open": [182.0],
            "high": [183.1],
            "low": [181.5],
            "close": [182.9],
        }
    )

    with pytest.raises(DataValidationError, match="volume"):
        validate_ohlcv(frame)


def test_validate_ohlcv_rejects_duplicate_symbol_date_rows() -> None:
    """Duplicate symbol/date keys should fail loudly."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "symbol": ["AAPL", "AAPL"],
            "open": [182.0, 182.1],
            "high": [183.1, 183.2],
            "low": [181.5, 181.6],
            "close": [182.9, 183.0],
            "volume": [30500000, 30550000],
        }
    )

    with pytest.raises(DataValidationError, match="duplicate symbol/date"):
        validate_ohlcv(frame)


def test_validate_ohlcv_rejects_intraday_timestamps() -> None:
    """Intraday timestamps should not be silently coerced to daily bars."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02 09:30:00"],
            "symbol": ["AAPL"],
            "open": [182.0],
            "high": [183.1],
            "low": [181.5],
            "close": [182.9],
            "volume": [30500000],
        }
    )

    with pytest.raises(DataValidationError, match="intraday"):
        validate_ohlcv(frame)


def test_validate_ohlcv_rejects_non_positive_prices() -> None:
    """Daily OHLCV prices must stay strictly positive."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "symbol": ["AAPL"],
            "open": [0.0],
            "high": [183.1],
            "low": [181.5],
            "close": [182.9],
            "volume": [30500000],
        }
    )

    with pytest.raises(DataValidationError, match="non-positive price"):
        validate_ohlcv(frame)


def test_validate_ohlcv_rejects_negative_volume() -> None:
    """Negative traded volume should fail validation."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "symbol": ["AAPL"],
            "open": [182.0],
            "high": [183.1],
            "low": [181.5],
            "close": [182.9],
            "volume": [-1],
        }
    )

    with pytest.raises(DataValidationError, match="negative volume"):
        validate_ohlcv(frame)


def test_validate_ohlcv_rejects_open_close_outside_daily_range() -> None:
    """Open and close must lie within the reported daily low/high range."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "symbol": ["AAPL"],
            "open": [182.0],
            "high": [181.0],
            "low": [180.5],
            "close": [182.9],
            "volume": [30500000],
        }
    )

    with pytest.raises(DataValidationError, match="daily low/high range"):
        validate_ohlcv(frame)


def test_load_ohlcv_reads_csv(tmp_path: Path) -> None:
    """CSV loading should route through validation."""
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "date,symbol,open,high,low,close,volume\n"
        "2024-01-02,AAPL,182.0,183.1,181.5,182.9,30500000\n",
        encoding="utf-8",
    )

    loaded = load_ohlcv(csv_path)

    assert loaded.shape == (1, 7)
    assert loaded.loc[0, "symbol"] == "AAPL"


def test_load_ohlcv_reads_parquet(tmp_path: Path) -> None:
    """Parquet loading should be supported when pyarrow is installed."""
    parquet_path = tmp_path / "sample.parquet"
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "symbol": ["MSFT"],
            "open": [370.0],
            "high": [373.2],
            "low": [369.0],
            "close": [372.0],
            "volume": [19800000],
        }
    )
    frame.to_parquet(parquet_path, index=False)

    loaded = load_ohlcv(parquet_path)

    assert loaded.shape == (1, 7)
    assert loaded.loc[0, "close"] == pytest.approx(372.0)


def test_load_ohlcv_rejects_unsupported_file_types(tmp_path: Path) -> None:
    """Only CSV and Parquet should be accepted."""
    unsupported_path = tmp_path / "sample.json"
    unsupported_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_ohlcv(unsupported_path)

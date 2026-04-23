"""Tests for the daily OHLCV market data layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from alphaforge.data import (
    CANONICAL_BENCHMARK_COLUMNS,
    CANONICAL_CORPORATE_ACTION_COLUMNS,
    CANONICAL_OHLCV_COLUMNS,
    CANONICAL_SYMBOL_METADATA_COLUMNS,
    CANONICAL_TRADING_CALENDAR_COLUMNS,
    DataValidationError,
    load_benchmark_returns,
    load_corporate_actions,
    load_ohlcv,
    load_symbol_metadata,
    load_trading_calendar,
    validate_benchmark_returns,
    validate_corporate_actions,
    validate_ohlcv,
    validate_symbol_metadata,
    validate_trading_calendar,
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


def test_validate_benchmark_returns_sorts_and_normalizes_custom_return_column() -> None:
    """Benchmark validation should canonicalize the configured return column."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-02"],
            "spx_return": ["0.01", "0.00"],
            "source_name": ["SPX", "SPX"],
        }
    )

    validated = validate_benchmark_returns(frame, return_column="spx_return")

    assert list(validated.columns) == [*CANONICAL_BENCHMARK_COLUMNS, "source_name"]
    assert validated["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-03",
    ]
    assert pd.api.types.is_datetime64_ns_dtype(validated["date"])
    assert validated["benchmark_return"].tolist() == pytest.approx([0.0, 0.01])


def test_validate_benchmark_returns_rejects_duplicate_dates() -> None:
    """Duplicate benchmark dates should fail loudly."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "benchmark_return": [0.01, 0.02],
        }
    )

    with pytest.raises(DataValidationError, match="duplicate benchmark dates"):
        validate_benchmark_returns(frame)


def test_validate_benchmark_returns_rejects_intraday_timestamps() -> None:
    """Intraday benchmark timestamps should not be silently truncated."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02 16:00:00"],
            "benchmark_return": [0.01],
        }
    )

    with pytest.raises(DataValidationError, match="intraday"):
        validate_benchmark_returns(frame)


@pytest.mark.parametrize("invalid_return", [-1.0, float("inf"), float("-inf")])
def test_validate_benchmark_returns_rejects_invalid_return_values(
    invalid_return: float,
) -> None:
    """Benchmark returns must stay finite and above -100%."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "benchmark_return": [invalid_return],
        }
    )

    with pytest.raises(DataValidationError, match="finite and greater than -1.0"):
        validate_benchmark_returns(frame)


def test_validate_symbol_metadata_sorts_and_normalizes_values() -> None:
    """Validated symbol metadata should canonicalize listing/delisting dates."""
    frame = pd.DataFrame(
        {
            "symbol": ["MSFT", "AAPL"],
            "ipo_date": ["1986-03-13", "1980-12-12"],
            "delisted_on": ["", ""],
            "exchange": ["XNAS", "XNAS"],
        }
    )

    validated = validate_symbol_metadata(
        frame,
        listing_date_column="ipo_date",
        delisting_date_column="delisted_on",
    )

    assert list(validated.columns) == [*CANONICAL_SYMBOL_METADATA_COLUMNS, "exchange"]
    assert validated["symbol"].tolist() == ["AAPL", "MSFT"]
    assert validated["listing_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "1980-12-12",
        "1986-03-13",
    ]
    assert validated["delisting_date"].isna().all()


def test_validate_symbol_metadata_allows_missing_delisting_date_column() -> None:
    """Active-only metadata should not require a physical delisting column."""
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "listing_date": ["1980-12-12"],
        }
    )

    validated = validate_symbol_metadata(frame)

    assert list(validated.columns) == list(CANONICAL_SYMBOL_METADATA_COLUMNS)
    assert validated["delisting_date"].isna().all()


def test_validate_symbol_metadata_rejects_duplicate_symbols() -> None:
    """Each symbol should have one metadata row in the current contract."""
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "listing_date": ["1980-12-12", "1980-12-12"],
        }
    )

    with pytest.raises(DataValidationError, match="duplicate symbols"):
        validate_symbol_metadata(frame)


def test_validate_symbol_metadata_rejects_delisting_before_listing() -> None:
    """Delisting dates must not precede listing dates."""
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "listing_date": ["1980-12-12"],
            "delisting_date": ["1980-12-11"],
        }
    )

    with pytest.raises(DataValidationError, match="delisting_date is earlier"):
        validate_symbol_metadata(frame)


def test_validate_corporate_actions_sorts_and_normalizes_values() -> None:
    """Corporate actions should canonicalize ex-date, type, and numeric payload columns."""
    frame = pd.DataFrame(
        {
            "symbol": ["MSFT", "AAPL"],
            "event_date": ["2024-01-05", "2024-01-04"],
            "kind": ["Cash_Dividend", "SPLIT"],
            "ratio": ["", "2.0"],
            "cash": ["0.62", ""],
            "source_name": ["vendor", "vendor"],
        }
    )

    validated = validate_corporate_actions(
        frame,
        ex_date_column="event_date",
        action_type_column="kind",
        split_ratio_column="ratio",
        cash_amount_column="cash",
    )

    assert list(validated.columns) == [*CANONICAL_CORPORATE_ACTION_COLUMNS, "source_name"]
    assert validated["symbol"].tolist() == ["AAPL", "MSFT"]
    assert validated["ex_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-04",
        "2024-01-05",
    ]
    assert validated["action_type"].tolist() == ["split", "cash_dividend"]
    assert validated["split_ratio"].tolist() == pytest.approx([2.0, float("nan")], nan_ok=True)
    assert validated["cash_amount"].tolist() == pytest.approx([float("nan"), 0.62], nan_ok=True)


def test_validate_corporate_actions_rejects_duplicate_keys() -> None:
    """Duplicate symbol/ex-date/action-type rows should fail loudly."""
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "ex_date": ["2024-01-04", "2024-01-04"],
            "action_type": ["split", "split"],
            "split_ratio": [2.0, 2.0],
        }
    )

    with pytest.raises(DataValidationError, match="duplicate corporate action keys"):
        validate_corporate_actions(frame)


def test_validate_corporate_actions_rejects_unsupported_action_types() -> None:
    """Only currently supported corporate-action types should be accepted."""
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "ex_date": ["2024-01-04"],
            "action_type": ["stock_dividend"],
        }
    )

    with pytest.raises(DataValidationError, match="unsupported corporate action types"):
        validate_corporate_actions(frame)


def test_validate_corporate_actions_rejects_invalid_split_rows() -> None:
    """Split rows must carry only a positive split ratio."""
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "ex_date": ["2024-01-04"],
            "action_type": ["split"],
            "split_ratio": [0.0],
            "cash_amount": [0.1],
        }
    )

    with pytest.raises(DataValidationError, match="split_ratio"):
        validate_corporate_actions(frame)


def test_validate_corporate_actions_rejects_invalid_cash_dividend_rows() -> None:
    """Cash-dividend rows must carry only a positive cash amount."""
    frame = pd.DataFrame(
        {
            "symbol": ["MSFT"],
            "ex_date": ["2024-01-05"],
            "action_type": ["cash_dividend"],
            "split_ratio": [2.0],
            "cash_amount": [0.0],
        }
    )

    with pytest.raises(DataValidationError, match="cash_amount"):
        validate_corporate_actions(frame)


def test_validate_corporate_actions_rejects_intraday_ex_dates() -> None:
    """Corporate-action dates must remain date-only."""
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "ex_date": ["2024-01-04 09:30:00"],
            "action_type": ["split"],
            "split_ratio": [2.0],
        }
    )

    with pytest.raises(DataValidationError, match="intraday"):
        validate_corporate_actions(frame)


def test_validate_trading_calendar_sorts_and_normalizes_values() -> None:
    """Validated trading calendars should canonicalize the date column."""
    frame = pd.DataFrame(
        {
            "session_date": ["2024-01-04", "2024-01-02", "2024-01-03"],
            "label": ["Thu", "Tue", "Wed"],
        }
    )

    validated = validate_trading_calendar(frame, date_column="session_date")

    assert list(validated.columns) == [*CANONICAL_TRADING_CALENDAR_COLUMNS, "label"]
    assert validated["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
    ]


def test_validate_trading_calendar_rejects_duplicate_dates() -> None:
    """Duplicate sessions should fail loudly."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
        }
    )

    with pytest.raises(DataValidationError, match="duplicate trading calendar dates"):
        validate_trading_calendar(frame)


def test_validate_trading_calendar_rejects_intraday_timestamps() -> None:
    """Trading calendars must remain date-only."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02 09:30:00"],
        }
    )

    with pytest.raises(DataValidationError, match="intraday"):
        validate_trading_calendar(frame)


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


def test_load_benchmark_returns_reads_csv_with_custom_return_column(
    tmp_path: Path,
) -> None:
    """Benchmark CSV loading should route through the shared contract."""
    csv_path = tmp_path / "benchmark.csv"
    csv_path.write_text(
        "date,spx_return,source_name\n"
        "2024-01-03,0.01,SPX\n"
        "2024-01-02,0.00,SPX\n",
        encoding="utf-8",
    )

    loaded = load_benchmark_returns(csv_path, return_column="spx_return")

    assert list(loaded.columns) == [*CANONICAL_BENCHMARK_COLUMNS, "source_name"]
    assert loaded["benchmark_return"].tolist() == pytest.approx([0.0, 0.01])


def test_load_symbol_metadata_reads_csv(tmp_path: Path) -> None:
    """Symbol metadata CSV loading should route through validation."""
    csv_path = tmp_path / "symbol_metadata.csv"
    csv_path.write_text(
        "symbol,listing_date,delisting_date,exchange\n"
        "MSFT,1986-03-13,,XNAS\n"
        "AAPL,1980-12-12,,XNAS\n",
        encoding="utf-8",
    )

    loaded = load_symbol_metadata(csv_path)

    assert list(loaded.columns) == [*CANONICAL_SYMBOL_METADATA_COLUMNS, "exchange"]
    assert loaded["symbol"].tolist() == ["AAPL", "MSFT"]
    assert loaded["delisting_date"].isna().all()


def test_load_corporate_actions_reads_csv(tmp_path: Path) -> None:
    """Corporate-action CSV loading should route through validation."""
    csv_path = tmp_path / "corporate_actions.csv"
    csv_path.write_text(
        "symbol,ex_date,action_type,split_ratio,cash_amount,source_name\n"
        "MSFT,2024-01-05,cash_dividend,,0.62,vendor\n"
        "AAPL,2024-01-04,split,2.0,,vendor\n",
        encoding="utf-8",
    )

    loaded = load_corporate_actions(csv_path)

    assert list(loaded.columns) == [*CANONICAL_CORPORATE_ACTION_COLUMNS, "source_name"]
    assert loaded["action_type"].tolist() == ["split", "cash_dividend"]


def test_load_trading_calendar_reads_csv(tmp_path: Path) -> None:
    """Trading calendar CSV loading should route through validation."""
    csv_path = tmp_path / "calendar.csv"
    csv_path.write_text(
        "date,label\n"
        "2024-01-03,Wed\n"
        "2024-01-02,Tue\n",
        encoding="utf-8",
    )

    loaded = load_trading_calendar(csv_path)

    assert list(loaded.columns) == [*CANONICAL_TRADING_CALENDAR_COLUMNS, "label"]
    assert loaded["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-03",
    ]


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


def test_load_benchmark_returns_rejects_unsupported_file_types(tmp_path: Path) -> None:
    """Only CSV and Parquet benchmark inputs should be accepted."""
    unsupported_path = tmp_path / "benchmark.json"
    unsupported_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_benchmark_returns(unsupported_path)


def test_load_symbol_metadata_rejects_unsupported_file_types(tmp_path: Path) -> None:
    """Only CSV and Parquet symbol metadata inputs should be accepted."""
    unsupported_path = tmp_path / "symbol_metadata.json"
    unsupported_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_symbol_metadata(unsupported_path)


def test_load_corporate_actions_rejects_unsupported_file_types(tmp_path: Path) -> None:
    """Only CSV and Parquet corporate-action inputs should be accepted."""
    unsupported_path = tmp_path / "corporate_actions.json"
    unsupported_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_corporate_actions(unsupported_path)


def test_load_trading_calendar_rejects_unsupported_file_types(tmp_path: Path) -> None:
    """Only CSV and Parquet trading calendar inputs should be accepted."""
    unsupported_path = tmp_path / "calendar.json"
    unsupported_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_trading_calendar(unsupported_path)

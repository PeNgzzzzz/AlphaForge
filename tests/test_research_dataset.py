"""Tests for research dataset construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.features import build_research_dataset


def test_build_research_dataset_computes_expected_returns_and_labels() -> None:
    """The dataset builder should compute returns, rolling features, and labels."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "open": [100.0, 110.0, 121.0, 133.1],
            "high": [101.0, 111.0, 122.0, 134.0],
            "low": [99.0, 109.0, 120.0, 132.0],
            "close": [100.0, 110.0, 121.0, 133.1],
            "volume": [10, 20, 30, 40],
        }
    )

    dataset = build_research_dataset(
        frame,
        forward_horizons=(1, 2),
        volatility_window=2,
        average_volume_window=2,
    )

    assert pd.isna(dataset.loc[0, "daily_return"])
    assert dataset.loc[1, "daily_return"] == pytest.approx(0.10)
    assert dataset.loc[2, "daily_return"] == pytest.approx(0.10)
    assert dataset.loc[1, "log_return"] == pytest.approx(np.log(1.10))
    assert dataset.loc[0, "forward_return_1d"] == pytest.approx(0.10)
    assert dataset.loc[1, "forward_return_2d"] == pytest.approx(0.21)
    assert pd.isna(dataset.loc[3, "forward_return_1d"])
    assert pd.isna(dataset.loc[2, "forward_return_2d"])
    assert pd.isna(dataset.loc[0, "rolling_volatility_2d"])
    assert dataset.loc[2, "rolling_volatility_2d"] == pytest.approx(0.0)
    assert dataset.loc[1, "rolling_average_volume_2d"] == pytest.approx(15.0)
    assert dataset.loc[3, "rolling_average_volume_2d"] == pytest.approx(35.0)


def test_build_research_dataset_sorts_input_and_keeps_symbol_boundaries() -> None:
    """Returns and labels should be computed within each symbol after sorting."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-02", "2024-01-03", "2024-01-02"],
            "symbol": ["MSFT", "AAPL", "AAPL", "MSFT"],
            "open": [210.0, 100.0, 110.0, 200.0],
            "high": [221.0, 101.0, 111.0, 201.0],
            "low": [209.0, 99.0, 109.0, 199.0],
            "close": [220.0, 100.0, 110.0, 200.0],
            "volume": [60, 10, 20, 50],
        }
    )

    dataset = build_research_dataset(
        frame,
        forward_horizons=1,
        volatility_window=2,
        average_volume_window=2,
    )

    assert dataset["symbol"].tolist() == ["AAPL", "AAPL", "MSFT", "MSFT"]
    assert dataset["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-02",
        "2024-01-03",
    ]
    assert dataset.loc[0, "forward_return_1d"] == pytest.approx(0.10)
    assert pd.isna(dataset.loc[1, "forward_return_1d"])
    assert dataset.loc[2, "forward_return_1d"] == pytest.approx(0.10)
    assert pd.isna(dataset.loc[3, "forward_return_1d"])


def test_build_research_dataset_validates_forward_horizons() -> None:
    """Forward return horizons must be positive, unique integers."""
    frame = _sample_frame()

    with pytest.raises(ValueError, match="at least one positive integer"):
        build_research_dataset(frame, forward_horizons=[])

    with pytest.raises(ValueError, match="positive integer horizons"):
        build_research_dataset(frame, forward_horizons=(1, 0))

    with pytest.raises(ValueError, match="duplicate horizons"):
        build_research_dataset(frame, forward_horizons=(1, 1))


def test_build_research_dataset_validates_window_inputs() -> None:
    """Rolling window arguments must be positive integers."""
    frame = _sample_frame()

    with pytest.raises(ValueError, match="volatility_window"):
        build_research_dataset(frame, volatility_window=0)

    with pytest.raises(ValueError, match="average_volume_window"):
        build_research_dataset(frame, average_volume_window=0)


def test_build_research_dataset_preserves_baseline_columns_without_universe_filters() -> None:
    """Universe filter columns should stay optional for backward compatibility."""
    dataset = build_research_dataset(_sample_frame())

    assert "is_universe_eligible" not in dataset.columns
    assert "universe_exclusion_reason" not in dataset.columns
    assert "universe_filter_date" not in dataset.columns


def test_build_research_dataset_adds_lagged_universe_filter_columns() -> None:
    """Tradability filters should expose lagged metrics, pass flags, and reasons."""
    frame = pd.DataFrame(
        {
            "date": [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
            ],
            "symbol": ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
            "open": [10.0, 20.0, 20.0, 20.0, 20.0],
            "high": [10.5, 20.5, 20.5, 20.5, 20.5],
            "low": [9.5, 19.5, 19.5, 19.5, 19.5],
            "close": [10.0, 20.0, 20.0, 20.0, 20.0],
            "volume": [100, 50, 100, 100, 100],
        }
    )

    dataset = build_research_dataset(
        frame,
        forward_horizons=1,
        volatility_window=2,
        average_volume_window=3,
        minimum_price=15.0,
        minimum_average_volume=80.0,
        minimum_average_dollar_volume=1600.0,
        minimum_listing_history_days=3,
        universe_lag=1,
        universe_average_volume_window=2,
        universe_average_dollar_volume_window=2,
    )

    assert "daily_dollar_volume" in dataset.columns
    assert "rolling_average_volume_3d" in dataset.columns
    assert "rolling_average_volume_2d" in dataset.columns
    assert "rolling_average_dollar_volume_2d" in dataset.columns
    assert "has_universe_history" in dataset.columns
    assert "universe_lagged_close" in dataset.columns
    assert "universe_lagged_listing_history_days" in dataset.columns
    assert "universe_lagged_average_volume_2d" in dataset.columns
    assert "universe_lagged_average_dollar_volume_2d" in dataset.columns
    assert "passes_universe_min_price" in dataset.columns
    assert "passes_universe_min_average_volume" in dataset.columns
    assert "passes_universe_min_average_dollar_volume" in dataset.columns
    assert "passes_universe_min_listing_history" in dataset.columns
    assert not bool(dataset.loc[0, "has_universe_history"])
    assert pd.isna(dataset.loc[0, "universe_filter_date"])
    assert not bool(dataset.loc[0, "is_universe_eligible"])
    assert dataset.loc[0, "universe_exclusion_reason"] == "insufficient_universe_history"

    second_row_reasons = set(str(dataset.loc[1, "universe_exclusion_reason"]).split(";"))
    assert second_row_reasons == {
        "below_min_price",
        "insufficient_average_volume_history",
        "insufficient_adv_history",
        "insufficient_listing_history",
    }
    assert dataset.loc[1, "universe_lagged_close"] == pytest.approx(10.0)
    assert dataset.loc[1, "universe_lagged_listing_history_days"] == pytest.approx(1.0)
    assert not bool(dataset.loc[1, "passes_universe_min_price"])
    assert not bool(dataset.loc[1, "passes_universe_min_average_volume"])
    assert not bool(dataset.loc[1, "passes_universe_min_average_dollar_volume"])
    assert not bool(dataset.loc[1, "passes_universe_min_listing_history"])

    assert dataset.loc[2, "universe_filter_date"].date().isoformat() == "2024-01-03"
    assert dataset.loc[2, "universe_lagged_average_volume_2d"] == pytest.approx(75.0)
    assert dataset.loc[2, "universe_lagged_average_dollar_volume_2d"] == pytest.approx(
        1000.0
    )
    third_row_reasons = set(str(dataset.loc[2, "universe_exclusion_reason"]).split(";"))
    assert third_row_reasons == {
        "below_min_average_volume",
        "below_min_average_dollar_volume",
        "insufficient_listing_history",
    }

    fourth_row_reasons = set(str(dataset.loc[3, "universe_exclusion_reason"]).split(";"))
    assert fourth_row_reasons == {
        "below_min_average_volume",
        "below_min_average_dollar_volume",
    }
    assert not bool(dataset.loc[3, "is_universe_eligible"])

    assert bool(dataset.loc[4, "has_universe_history"])
    assert dataset.loc[4, "universe_filter_date"].date().isoformat() == "2024-01-05"
    assert bool(dataset.loc[4, "passes_universe_min_price"])
    assert bool(dataset.loc[4, "passes_universe_min_average_volume"])
    assert bool(dataset.loc[4, "passes_universe_min_average_dollar_volume"])
    assert bool(dataset.loc[4, "passes_universe_min_listing_history"])
    assert bool(dataset.loc[4, "is_universe_eligible"])
    assert dataset.loc[4, "universe_exclusion_reason"] == ""


def _sample_frame() -> pd.DataFrame:
    """Build a minimal valid OHLCV frame for argument validation tests."""
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "symbol": ["AAPL", "AAPL"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [10, 20],
        }
    )

"""Tests for research dataset construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.data import DataValidationError
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


def test_build_research_dataset_uses_symbol_metadata_for_listing_history_days() -> None:
    """Listing history should use metadata-aware market-date counts when available."""
    frame = pd.DataFrame(
        {
            "date": [
                "2024-01-02",
                "2024-01-04",
                "2024-01-05",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ],
            "symbol": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT", "MSFT"],
            "open": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 23.0],
            "high": [10.5, 11.5, 12.5, 20.5, 21.5, 22.5, 23.5],
            "low": [9.5, 10.5, 11.5, 19.5, 20.5, 21.5, 22.5],
            "close": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 23.0],
            "volume": [100, 110, 120, 200, 210, 220, 230],
        }
    )
    symbol_metadata = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "listing_date": ["2024-01-02", "2024-01-02"],
        }
    )

    dataset = build_research_dataset(
        frame,
        symbol_metadata=symbol_metadata,
        minimum_listing_history_days=3,
        universe_lag=1,
    )

    aapl_last_row = dataset.loc[
        (dataset["symbol"] == "AAPL") & (dataset["date"] == pd.Timestamp("2024-01-05"))
    ].iloc[0]
    assert aapl_last_row["listing_date"] == pd.Timestamp("2024-01-02")
    assert pd.isna(aapl_last_row["delisting_date"])
    assert aapl_last_row["universe_lagged_listing_history_days"] == pytest.approx(3.0)
    assert bool(aapl_last_row["passes_universe_min_listing_history"])
    assert bool(aapl_last_row["is_universe_eligible"])


def test_build_research_dataset_rejects_rows_before_listing_date() -> None:
    """Market-data rows before listing_date should fail loudly."""
    frame = pd.DataFrame(
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
    symbol_metadata = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "listing_date": ["2024-01-03"],
        }
    )

    with pytest.raises(DataValidationError, match="before symbol listing_date"):
        build_research_dataset(frame, symbol_metadata=symbol_metadata)


def test_build_research_dataset_rejects_rows_after_delisting_date() -> None:
    """Market-data rows after delisting_date should fail loudly."""
    frame = pd.DataFrame(
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
    symbol_metadata = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "listing_date": ["2024-01-02"],
            "delisting_date": ["2024-01-02"],
        }
    )

    with pytest.raises(DataValidationError, match="after symbol delisting_date"):
        build_research_dataset(frame, symbol_metadata=symbol_metadata)


def test_build_research_dataset_uses_trading_calendar_for_listing_history_days() -> None:
    """Configured trading calendars should drive listing-history counts."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-04", "2024-01-05"],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "open": [10.0, 11.0, 12.0],
            "high": [10.5, 11.5, 12.5],
            "low": [9.5, 10.5, 11.5],
            "close": [10.0, 11.0, 12.0],
            "volume": [100, 110, 120],
        }
    )
    symbol_metadata = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "listing_date": ["2024-01-02"],
        }
    )
    trading_calendar = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        }
    )

    dataset = build_research_dataset(
        frame,
        trading_calendar=trading_calendar,
        symbol_metadata=symbol_metadata,
        minimum_listing_history_days=3,
        universe_lag=1,
    )

    last_row = dataset.loc[dataset["date"] == pd.Timestamp("2024-01-05")].iloc[0]
    assert last_row["universe_lagged_listing_history_days"] == pytest.approx(3.0)
    assert bool(last_row["passes_universe_min_listing_history"])
    assert bool(last_row["is_universe_eligible"])


def test_build_research_dataset_rejects_dates_outside_trading_calendar() -> None:
    """Market-data dates must fall on the configured trading calendar."""
    frame = pd.DataFrame(
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
    trading_calendar = pd.DataFrame(
        {
            "date": ["2024-01-02"],
        }
    )

    with pytest.raises(DataValidationError, match="configured trading calendar"):
        build_research_dataset(frame, trading_calendar=trading_calendar)


def test_build_research_dataset_attaches_fundamentals_on_next_session() -> None:
    """Date-only fundamentals releases should become usable on the next session."""
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
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [10, 11, 12, 13, 14],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "period_end_date": ["2023-09-30", "2023-09-30", "2023-12-31"],
            "release_date": ["2024-01-03", "2024-01-03", "2024-01-05"],
            "metric_name": ["revenue", "book value", "revenue"],
            "metric_value": [100.0, 50.0, 120.0],
        }
    )

    dataset = build_research_dataset(frame, fundamentals=fundamentals)

    assert "fundamental_revenue" in dataset.columns
    assert "fundamental_book_value" in dataset.columns
    assert pd.isna(
        dataset.loc[dataset["date"] == pd.Timestamp("2024-01-03"), "fundamental_revenue"]
    ).all()
    assert dataset.loc[
        dataset["date"] == pd.Timestamp("2024-01-04"),
        "fundamental_revenue",
    ].iloc[0] == pytest.approx(100.0)
    assert dataset.loc[
        dataset["date"] == pd.Timestamp("2024-01-04"),
        "fundamental_book_value",
    ].iloc[0] == pytest.approx(50.0)
    assert dataset.loc[
        dataset["date"] == pd.Timestamp("2024-01-05"),
        "fundamental_revenue",
    ].iloc[0] == pytest.approx(100.0)
    assert dataset.loc[
        dataset["date"] == pd.Timestamp("2024-01-08"),
        "fundamental_revenue",
    ].iloc[0] == pytest.approx(120.0)


def test_build_research_dataset_requires_fundamentals_for_metric_selection() -> None:
    """Explicit metric selection should not work without a fundamentals input."""
    with pytest.raises(ValueError, match="fundamental_metrics requires fundamentals"):
        build_research_dataset(
            _sample_frame(),
            fundamental_metrics=("revenue",),
        )


def test_build_research_dataset_attaches_valuation_metrics_on_next_session() -> None:
    """Valuation metrics should reuse the next-session-safe fundamentals join."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "open": [100.0, 110.0, 120.0],
            "high": [101.0, 111.0, 121.0],
            "low": [99.0, 109.0, 119.0],
            "close": [100.0, 110.0, 120.0],
            "volume": [10, 11, 12],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "period_end_date": ["2023-12-31"],
            "release_date": ["2024-01-02"],
            "metric_name": ["eps"],
            "metric_value": [5.5],
        }
    )

    dataset = build_research_dataset(
        frame,
        fundamentals=fundamentals,
        valuation_metrics=("eps",),
    )

    assert "fundamental_eps" in dataset.columns
    assert "valuation_eps_to_price" in dataset.columns
    assert pd.isna(dataset.loc[0, "valuation_eps_to_price"])
    assert dataset.loc[1, "valuation_eps_to_price"] == pytest.approx(5.5 / 110.0)
    assert dataset.loc[2, "valuation_eps_to_price"] == pytest.approx(5.5 / 120.0)


def test_build_research_dataset_requires_fundamentals_for_valuation_metrics() -> None:
    """Valuation feature selection should require a fundamentals input."""
    with pytest.raises(ValueError, match="valuation_metrics requires fundamentals"):
        build_research_dataset(
            _sample_frame(),
            valuation_metrics=("eps",),
        )


def test_build_research_dataset_attaches_quality_ratio_metrics_on_next_session() -> None:
    """Quality ratios should reuse the next-session-safe fundamentals join."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "open": [100.0, 110.0, 120.0],
            "high": [101.0, 111.0, 121.0],
            "low": [99.0, 109.0, 119.0],
            "close": [100.0, 110.0, 120.0],
            "volume": [10, 11, 12],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "period_end_date": ["2023-12-31", "2023-12-31"],
            "release_date": ["2024-01-02", "2024-01-02"],
            "metric_name": ["net_income", "total_assets"],
            "metric_value": [11.0, 110.0],
        }
    )

    dataset = build_research_dataset(
        frame,
        fundamentals=fundamentals,
        quality_ratio_metrics=(("net_income", "total_assets"),),
    )

    assert "fundamental_net_income" in dataset.columns
    assert "fundamental_total_assets" in dataset.columns
    assert "quality_net_income_to_total_assets" in dataset.columns
    assert pd.isna(dataset.loc[0, "quality_net_income_to_total_assets"])
    assert dataset.loc[1, "quality_net_income_to_total_assets"] == pytest.approx(0.1)
    assert dataset.loc[2, "quality_net_income_to_total_assets"] == pytest.approx(0.1)


def test_build_research_dataset_treats_nonpositive_quality_denominator_as_missing() -> None:
    """Quality ratios should not invert nonpositive balance-sheet denominators."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "symbol": ["AAPL", "AAPL"],
            "open": [100.0, 110.0],
            "high": [101.0, 111.0],
            "low": [99.0, 109.0],
            "close": [100.0, 110.0],
            "volume": [10, 11],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "period_end_date": ["2023-12-31", "2023-12-31"],
            "release_date": ["2024-01-02", "2024-01-02"],
            "metric_name": ["net_income", "total_assets"],
            "metric_value": [11.0, 0.0],
        }
    )

    dataset = build_research_dataset(
        frame,
        fundamentals=fundamentals,
        quality_ratio_metrics=(("net_income", "total_assets"),),
    )

    assert pd.isna(dataset.loc[1, "quality_net_income_to_total_assets"])


def test_build_research_dataset_requires_fundamentals_for_quality_ratio_metrics() -> None:
    """Quality ratio feature selection should require a fundamentals input."""
    with pytest.raises(ValueError, match="quality_ratio_metrics requires fundamentals"):
        build_research_dataset(
            _sample_frame(),
            quality_ratio_metrics=(("net_income", "total_assets"),),
        )


def test_build_research_dataset_attaches_growth_metrics_on_next_session() -> None:
    """Growth metrics should use adjacent fundamental periods and release timing."""
    frame = pd.DataFrame(
        {
            "date": [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ],
            "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "open": [100.0, 110.0, 120.0, 130.0],
            "high": [101.0, 111.0, 121.0, 131.0],
            "low": [99.0, 109.0, 119.0, 129.0],
            "close": [100.0, 110.0, 120.0, 130.0],
            "volume": [10, 11, 12, 13],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "period_end_date": ["2023-09-30", "2023-12-31"],
            "release_date": ["2024-01-02", "2024-01-04"],
            "metric_name": ["revenue", "revenue"],
            "metric_value": [100.0, 125.0],
        }
    )

    dataset = build_research_dataset(
        frame,
        fundamentals=fundamentals,
        growth_metrics=("revenue",),
    )

    assert "fundamental_revenue" in dataset.columns
    assert "growth_revenue" in dataset.columns
    assert pd.isna(dataset.loc[0, "growth_revenue"])
    assert pd.isna(dataset.loc[1, "growth_revenue"])
    assert pd.isna(dataset.loc[2, "growth_revenue"])
    assert dataset.loc[3, "growth_revenue"] == pytest.approx(0.25)
    assert dataset.loc[1, "fundamental_revenue"] == pytest.approx(100.0)
    assert dataset.loc[3, "fundamental_revenue"] == pytest.approx(125.0)


def test_build_research_dataset_treats_nonpositive_growth_prior_as_missing() -> None:
    """Growth metrics should not divide by nonpositive prior period values."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "open": [100.0, 110.0, 120.0],
            "high": [101.0, 111.0, 121.0],
            "low": [99.0, 109.0, 119.0],
            "close": [100.0, 110.0, 120.0],
            "volume": [10, 11, 12],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "period_end_date": ["2023-09-30", "2023-12-31"],
            "release_date": ["2024-01-02", "2024-01-03"],
            "metric_name": ["revenue", "revenue"],
            "metric_value": [0.0, 125.0],
        }
    )

    dataset = build_research_dataset(
        frame,
        fundamentals=fundamentals,
        growth_metrics=("revenue",),
    )

    assert "growth_revenue" in dataset.columns
    assert dataset["growth_revenue"].isna().all()


def test_build_research_dataset_requires_fundamentals_for_growth_metrics() -> None:
    """Growth feature selection should require a fundamentals input."""
    with pytest.raises(ValueError, match="growth_metrics requires fundamentals"):
        build_research_dataset(
            _sample_frame(),
            growth_metrics=("revenue",),
        )


def test_build_research_dataset_rejects_growth_metrics_with_restatements() -> None:
    """Growth metrics should fail when selected fundamentals have restatements."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [10, 11, 12],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "period_end_date": ["2023-09-30", "2023-09-30"],
            "release_date": ["2024-01-02", "2024-01-03"],
            "metric_name": ["revenue", "revenue"],
            "metric_value": [100.0, 101.0],
        }
    )

    with pytest.raises(DataValidationError, match="restatement lineage"):
        build_research_dataset(
            frame,
            fundamentals=fundamentals,
            growth_metrics=("revenue",),
        )


def test_build_research_dataset_rejects_same_session_fundamental_conflicts() -> None:
    """Ambiguous same-session fundamentals availability should fail loudly."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [10, 11, 12],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "period_end_date": ["2023-09-30", "2023-12-31"],
            "release_date": ["2024-01-03", "2024-01-03"],
            "metric_name": ["revenue", "revenue"],
            "metric_value": [90.0, 100.0],
        }
    )

    with pytest.raises(DataValidationError, match="same next-session date"):
        build_research_dataset(
            frame,
            fundamentals=fundamentals,
            fundamental_metrics=("revenue",),
        )


def test_build_research_dataset_attaches_classifications_on_effective_session() -> None:
    """Effective-date classifications should apply on the first valid market session."""
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
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [10, 11, 12, 13, 14],
        }
    )
    classifications = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "effective_date": ["2024-01-03", "2024-01-06"],
            "sector": ["Technology", "Consumer"],
            "industry": ["Hardware", "Retail"],
        }
    )
    trading_calendar = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"],
        }
    )

    dataset = build_research_dataset(
        frame,
        trading_calendar=trading_calendar,
        classifications=classifications,
        classification_fields=("sector", "industry"),
    )

    assert pd.isna(
        dataset.loc[
            dataset["date"] == pd.Timestamp("2024-01-02"),
            "classification_sector",
        ]
    ).all()
    assert dataset.loc[
        dataset["date"] == pd.Timestamp("2024-01-03"),
        "classification_sector",
    ].iloc[0] == "Technology"
    assert dataset.loc[
        dataset["date"] == pd.Timestamp("2024-01-03"),
        "classification_industry",
    ].iloc[0] == "Hardware"
    assert dataset.loc[
        dataset["date"] == pd.Timestamp("2024-01-05"),
        "classification_sector",
    ].iloc[0] == "Technology"
    assert dataset.loc[
        dataset["date"] == pd.Timestamp("2024-01-08"),
        "classification_sector",
    ].iloc[0] == "Consumer"
    assert dataset.loc[
        dataset["date"] == pd.Timestamp("2024-01-08"),
        "classification_industry",
    ].iloc[0] == "Retail"


def test_build_research_dataset_requires_classifications_for_field_selection() -> None:
    """Explicit classification field selection should require classifications input."""
    with pytest.raises(
        ValueError,
        match="classification_fields requires classifications",
    ):
        build_research_dataset(
            _sample_frame(),
            classification_fields=("sector",),
        )


def test_build_research_dataset_rejects_same_session_classification_conflicts() -> None:
    """Multiple changes mapping to one session should fail loudly."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-05", "2024-01-08"],
            "symbol": ["AAPL", "AAPL"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [10, 11],
        }
    )
    classifications = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "effective_date": ["2024-01-06", "2024-01-07"],
            "sector": ["Technology", "Consumer"],
            "industry": ["Hardware", "Retail"],
        }
    )
    trading_calendar = pd.DataFrame(
        {
            "date": ["2024-01-05", "2024-01-08"],
        }
    )

    with pytest.raises(DataValidationError, match="same market session"):
        build_research_dataset(
            frame,
            trading_calendar=trading_calendar,
            classifications=classifications,
            classification_fields=("sector",),
        )


def test_build_research_dataset_attaches_memberships_on_effective_session() -> None:
    """Effective-date memberships should apply on the first valid market session."""
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
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [10, 11, 12, 13, 14],
        }
    )
    memberships = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "effective_date": ["2024-01-03", "2024-01-06", "2024-01-04"],
            "index_name": ["S&P 500", "S&P 500", "NASDAQ 100"],
            "is_member": [True, False, True],
        }
    )
    trading_calendar = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"],
        }
    )

    dataset = build_research_dataset(
        frame,
        trading_calendar=trading_calendar,
        memberships=memberships,
        membership_indexes=("S&P 500", "NASDAQ 100"),
    )

    assert pd.isna(
        dataset.loc[
            dataset["date"] == pd.Timestamp("2024-01-02"),
            "membership_s_p_500",
        ]
    ).all()
    assert dataset.loc[
        dataset["date"] == pd.Timestamp("2024-01-03"),
        "membership_s_p_500",
    ].iloc[0]
    assert dataset.loc[
        dataset["date"] == pd.Timestamp("2024-01-05"),
        "membership_s_p_500",
    ].iloc[0]
    assert not dataset.loc[
        dataset["date"] == pd.Timestamp("2024-01-08"),
        "membership_s_p_500",
    ].iloc[0]
    assert pd.isna(
        dataset.loc[
            dataset["date"] == pd.Timestamp("2024-01-03"),
            "membership_nasdaq_100",
        ]
    ).all()
    assert dataset.loc[
        dataset["date"] == pd.Timestamp("2024-01-04"),
        "membership_nasdaq_100",
    ].iloc[0]


def test_build_research_dataset_requires_memberships_for_index_selection() -> None:
    """Explicit membership selection should require memberships input."""
    with pytest.raises(
        ValueError,
        match="membership_indexes requires memberships",
    ):
        build_research_dataset(
            _sample_frame(),
            membership_indexes=("S&P 500",),
        )


def test_build_research_dataset_rejects_same_session_membership_conflicts() -> None:
    """Multiple membership changes mapping to one session should fail loudly."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-05", "2024-01-08"],
            "symbol": ["AAPL", "AAPL"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [10, 11],
        }
    )
    memberships = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "effective_date": ["2024-01-06", "2024-01-07"],
            "index_name": ["S&P 500", "S&P 500"],
            "is_member": [True, False],
        }
    )
    trading_calendar = pd.DataFrame(
        {
            "date": ["2024-01-05", "2024-01-08"],
        }
    )

    with pytest.raises(DataValidationError, match="same market session"):
        build_research_dataset(
            frame,
            trading_calendar=trading_calendar,
            memberships=memberships,
            membership_indexes=("S&P 500",),
        )


def test_build_research_dataset_attaches_borrow_availability_on_effective_session() -> None:
    """Effective-date borrow events should apply on the first valid market session."""
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
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [10, 11, 12, 13, 14],
        }
    )
    borrow_availability = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "effective_date": ["2024-01-03", "2024-01-06"],
            "is_borrowable": [True, False],
            "borrow_fee_bps": [12.5, 25.0],
        }
    )
    trading_calendar = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"],
        }
    )

    dataset = build_research_dataset(
        frame,
        trading_calendar=trading_calendar,
        borrow_availability=borrow_availability,
        borrow_fields=("is_borrowable", "borrow_fee_bps"),
    )

    assert pd.isna(
        dataset.loc[
            dataset["date"] == pd.Timestamp("2024-01-02"),
            "borrow_is_borrowable",
        ]
    ).all()
    assert dataset.loc[
        dataset["date"] == pd.Timestamp("2024-01-03"),
        "borrow_is_borrowable",
    ].iloc[0]
    assert dataset.loc[
        dataset["date"] == pd.Timestamp("2024-01-03"),
        "borrow_fee_bps",
    ].iloc[0] == pytest.approx(12.5)
    assert not dataset.loc[
        dataset["date"] == pd.Timestamp("2024-01-08"),
        "borrow_is_borrowable",
    ].iloc[0]
    assert dataset.loc[
        dataset["date"] == pd.Timestamp("2024-01-08"),
        "borrow_fee_bps",
    ].iloc[0] == pytest.approx(25.0)


def test_build_research_dataset_requires_borrow_for_field_selection() -> None:
    """Explicit borrow selection should require borrow availability input."""
    with pytest.raises(
        ValueError,
        match="borrow_fields requires borrow_availability",
    ):
        build_research_dataset(
            _sample_frame(),
            borrow_fields=("is_borrowable",),
        )


def test_build_research_dataset_rejects_same_session_borrow_conflicts() -> None:
    """Multiple borrow changes mapping to one session should fail loudly."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-05", "2024-01-08"],
            "symbol": ["AAPL", "AAPL"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [10, 11],
        }
    )
    borrow_availability = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "effective_date": ["2024-01-06", "2024-01-07"],
            "is_borrowable": [True, False],
            "borrow_fee_bps": [10.0, 25.0],
        }
    )
    trading_calendar = pd.DataFrame(
        {
            "date": ["2024-01-05", "2024-01-08"],
        }
    )

    with pytest.raises(DataValidationError, match="same market session"):
        build_research_dataset(
            frame,
            trading_calendar=trading_calendar,
            borrow_availability=borrow_availability,
            borrow_fields=("is_borrowable",),
        )


def test_build_research_dataset_attaches_rolling_benchmark_statistics() -> None:
    """Rolling beta/correlation should use trailing aligned strategy and benchmark returns."""
    benchmark_returns = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "benchmark_return": [0.0, 0.01, -0.02, 0.03],
        }
    )
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "open": [100.0, 102.0, 97.92, 103.7952],
            "high": [101.0, 103.0, 98.92, 104.7952],
            "low": [99.0, 101.0, 96.92, 102.7952],
            "close": [100.0, 102.0, 97.92, 103.7952],
            "volume": [10, 11, 12, 13],
        }
    )

    dataset = build_research_dataset(
        frame,
        benchmark_returns=benchmark_returns,
        benchmark_rolling_window=3,
    )

    beta_column = "rolling_benchmark_beta_3d"
    correlation_column = "rolling_benchmark_correlation_3d"

    assert beta_column in dataset.columns
    assert correlation_column in dataset.columns
    assert pd.isna(dataset.loc[0, beta_column])
    assert pd.isna(dataset.loc[1, beta_column])
    assert dataset.loc[3, beta_column] == pytest.approx(2.0)
    assert dataset.loc[3, correlation_column] == pytest.approx(1.0)


def test_build_research_dataset_attaches_benchmark_residual_returns() -> None:
    """Benchmark residual returns should fit alpha/beta only on prior observations."""
    benchmark_returns = pd.DataFrame(
        {
            "date": [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
            ],
            "benchmark_return": [0.0, 0.01, -0.02, 0.03, 0.04],
        }
    )
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
            "open": [100.0, 103.0, 99.91, 107.9028, 117.614052],
            "high": [101.0, 104.0, 100.91, 108.9028, 118.614052],
            "low": [99.0, 102.0, 98.91, 106.9028, 116.614052],
            "close": [100.0, 103.0, 99.91, 107.9028, 117.614052],
            "volume": [10, 11, 12, 13, 14],
        }
    )

    dataset = build_research_dataset(
        frame,
        benchmark_returns=benchmark_returns,
        benchmark_residual_return_window=2,
    )

    column_name = "benchmark_residual_return_2d"

    assert column_name in dataset.columns
    assert "rolling_benchmark_beta_20d" not in dataset.columns
    assert pd.isna(dataset.loc[0, column_name])
    assert pd.isna(dataset.loc[1, column_name])
    assert pd.isna(dataset.loc[2, column_name])
    assert dataset.loc[3, column_name] == pytest.approx(0.01)


def test_build_research_dataset_attaches_garman_klass_volatility() -> None:
    """Garman-Klass volatility should use trailing OHLC ranges only."""
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
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [110.0, 120.0, 130.0, 140.0, 150.0],
            "low": [100.0, 100.0, 100.0, 100.0, 100.0],
            "close": [102.0, 104.0, 106.0, 108.0, 110.0],
            "volume": [10, 11, 12, 13, 14],
        }
    )

    dataset = build_research_dataset(
        frame,
        garman_klass_volatility_window=4,
    )

    column_name = "garman_klass_volatility_4d"
    expected_last = np.sqrt(
        np.mean(
            [
                0.004390528840879678,
                0.016026350397525954,
                0.03310593471596213,
                0.05431876284368194,
            ]
        )
    )

    assert column_name in dataset.columns
    assert dataset.loc[:2, column_name].isna().all()
    assert dataset.loc[3, column_name] == pytest.approx(expected_last)


def test_build_research_dataset_rejects_nonpositive_garman_klass_volatility_window() -> None:
    """Garman-Klass volatility windows should be positive integers."""
    with pytest.raises(
        ValueError,
        match="garman_klass_volatility_window must be a positive integer",
    ):
        build_research_dataset(
            _sample_frame(),
            garman_klass_volatility_window=0,
        )


def test_build_research_dataset_attaches_parkinson_volatility() -> None:
    """Parkinson volatility should use trailing high/low ranges only."""
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
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [110.0, 120.0, 130.0, 140.0, 150.0],
            "low": [100.0, 100.0, 100.0, 100.0, 100.0],
            "close": [105.0, 110.0, 115.0, 120.0, 125.0],
            "volume": [10, 11, 12, 13, 14],
        }
    )

    dataset = build_research_dataset(
        frame,
        parkinson_volatility_window=4,
    )

    column_name = "parkinson_volatility_4d"
    expected_last = np.sqrt(
        np.mean(
            (np.log(np.array([1.10, 1.20, 1.30, 1.40])) ** 2)
            / (4.0 * np.log(2.0))
        )
    )

    assert column_name in dataset.columns
    assert dataset.loc[:2, column_name].isna().all()
    assert dataset.loc[3, column_name] == pytest.approx(expected_last)


def test_build_research_dataset_rejects_nonpositive_parkinson_volatility_window() -> None:
    """Parkinson volatility windows should be positive integers."""
    with pytest.raises(
        ValueError,
        match="parkinson_volatility_window must be a positive integer",
    ):
        build_research_dataset(
            _sample_frame(),
            parkinson_volatility_window=0,
        )


def test_build_research_dataset_attaches_average_true_range() -> None:
    """Average true range should use trailing true-range observations only."""
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
            "open": [10.0, 11.0, 13.0, 13.0, 15.0],
            "high": [12.0, 14.0, 15.0, 16.0, 17.0],
            "low": [10.0, 11.0, 13.0, 12.0, 15.0],
            "close": [11.0, 13.0, 14.0, 15.0, 16.0],
            "volume": [10, 11, 12, 13, 14],
        }
    )

    dataset = build_research_dataset(
        frame,
        average_true_range_window=4,
    )

    column_name = "average_true_range_4d"
    expected_true_ranges = np.array([2.0, 3.0, 2.0, 4.0])
    expected_last = expected_true_ranges.mean()

    assert column_name in dataset.columns
    assert dataset.loc[:2, column_name].isna().all()
    assert dataset.loc[3, column_name] == pytest.approx(expected_last)


def test_build_research_dataset_rejects_nonpositive_average_true_range_window() -> None:
    """Average true range windows should be positive integers."""
    with pytest.raises(
        ValueError,
        match="average_true_range_window must be a positive integer",
    ):
        build_research_dataset(
            _sample_frame(),
            average_true_range_window=0,
        )


def test_build_research_dataset_attaches_normalized_average_true_range() -> None:
    """Normalized ATR should divide trailing ATR by the same-day close."""
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
            "open": [10.0, 11.0, 13.0, 13.0, 15.0],
            "high": [12.0, 14.0, 15.0, 16.0, 17.0],
            "low": [10.0, 11.0, 13.0, 12.0, 15.0],
            "close": [11.0, 13.0, 14.0, 15.0, 16.0],
            "volume": [10, 11, 12, 13, 14],
        }
    )

    dataset = build_research_dataset(
        frame,
        normalized_average_true_range_window=4,
    )

    column_name = "normalized_average_true_range_4d"
    expected_average_true_range = np.array([2.0, 3.0, 2.0, 4.0]).mean()
    expected_last = expected_average_true_range / 15.0

    assert column_name in dataset.columns
    assert dataset.loc[:2, column_name].isna().all()
    assert dataset.loc[3, column_name] == pytest.approx(expected_last)


def test_build_research_dataset_rejects_nonpositive_normalized_average_true_range_window() -> None:
    """Normalized ATR windows should be positive integers."""
    with pytest.raises(
        ValueError,
        match="normalized_average_true_range_window must be a positive integer",
    ):
        build_research_dataset(
            _sample_frame(),
            normalized_average_true_range_window=0,
        )


def test_build_research_dataset_attaches_amihud_illiquidity() -> None:
    """Amihud illiquidity should average abs(daily_return) over dollar volume."""
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
            "open": [100.0, 100.0, 110.0, 121.0, 133.1],
            "high": [101.0, 111.0, 122.0, 134.0, 147.0],
            "low": [99.0, 99.0, 109.0, 120.0, 132.0],
            "close": [100.0, 110.0, 121.0, 133.1, 146.41],
            "volume": [1000, 1000, 1000, 1000, 1000],
        }
    )

    dataset = build_research_dataset(
        frame,
        amihud_illiquidity_window=3,
    )

    column_name = "amihud_illiquidity_3d"
    daily_illiquidity = np.array(
        [
            0.10 / (110.0 * 1000.0),
            0.10 / (121.0 * 1000.0),
            0.10 / (133.1 * 1000.0),
        ]
    )

    assert column_name in dataset.columns
    assert dataset.loc[:2, column_name].isna().all()
    assert dataset.loc[3, column_name] == pytest.approx(daily_illiquidity.mean())


def test_build_research_dataset_amihud_illiquidity_treats_zero_dollar_volume_as_missing() -> None:
    """Amihud illiquidity should fail closed on zero-dollar-volume days."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "open": [100.0, 100.0, 110.0, 121.0],
            "high": [101.0, 111.0, 122.0, 134.0],
            "low": [99.0, 99.0, 109.0, 120.0],
            "close": [100.0, 110.0, 121.0, 133.1],
            "volume": [1000, 0, 1000, 1000],
        }
    )

    dataset = build_research_dataset(
        frame,
        amihud_illiquidity_window=2,
    )

    column_name = "amihud_illiquidity_2d"

    assert pd.isna(dataset.loc[0, column_name])
    assert pd.isna(dataset.loc[1, column_name])
    assert pd.isna(dataset.loc[2, column_name])
    assert dataset.loc[3, column_name] == pytest.approx(
        np.mean(
            [
                0.10 / (121.0 * 1000.0),
                0.10 / (133.1 * 1000.0),
            ]
        )
    )


def test_build_research_dataset_rejects_nonpositive_amihud_illiquidity_window() -> None:
    """Amihud illiquidity windows should be positive integers."""
    with pytest.raises(
        ValueError,
        match="amihud_illiquidity_window must be a positive integer",
    ):
        build_research_dataset(
            _sample_frame(),
            amihud_illiquidity_window=0,
        )


def test_build_research_dataset_attaches_dollar_volume_shock() -> None:
    """Dollar-volume shock should use a lagged log-dollar-volume baseline."""
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
            "open": [10.0, 10.0, 10.0, 10.0, 10.0],
            "high": [10.5, 10.5, 10.5, 10.5, 10.5],
            "low": [9.5, 9.5, 9.5, 9.5, 9.5],
            "close": [10.0, 10.0, 10.0, 10.0, 10.0],
            "volume": [10, 20, 40, 80, 160],
        }
    )

    dataset = build_research_dataset(
        frame,
        dollar_volume_shock_window=3,
    )

    column_name = "dollar_volume_shock_3d"
    first_window = np.log(np.array([100.0, 200.0, 400.0]))
    second_window = np.log(np.array([200.0, 400.0, 800.0]))

    assert column_name in dataset.columns
    assert dataset.loc[:2, column_name].isna().all()
    assert dataset.loc[3, column_name] == pytest.approx(
        np.log(800.0) - first_window.mean()
    )
    assert dataset.loc[4, column_name] == pytest.approx(
        np.log(1600.0) - second_window.mean()
    )


def test_build_research_dataset_rejects_nonpositive_dollar_volume_shock_window() -> None:
    """Dollar-volume shock windows should be positive integers."""
    with pytest.raises(
        ValueError,
        match="dollar_volume_shock_window must be a positive integer",
    ):
        build_research_dataset(
            _sample_frame(),
            dollar_volume_shock_window=0,
        )


def test_build_research_dataset_attaches_dollar_volume_zscore() -> None:
    """Dollar-volume z-score should use a lagged log-dollar-volume baseline."""
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
            "open": [10.0, 10.0, 10.0, 10.0, 10.0],
            "high": [10.5, 10.5, 10.5, 10.5, 10.5],
            "low": [9.5, 9.5, 9.5, 9.5, 9.5],
            "close": [10.0, 10.0, 10.0, 10.0, 10.0],
            "volume": [10, 20, 40, 80, 160],
        }
    )

    dataset = build_research_dataset(
        frame,
        dollar_volume_zscore_window=3,
    )

    column_name = "dollar_volume_zscore_3d"
    first_window = np.log(np.array([100.0, 200.0, 400.0]))
    second_window = np.log(np.array([200.0, 400.0, 800.0]))

    assert column_name in dataset.columns
    assert dataset.loc[:2, column_name].isna().all()
    assert dataset.loc[3, column_name] == pytest.approx(
        (np.log(800.0) - first_window.mean()) / first_window.std(ddof=1)
    )
    assert dataset.loc[4, column_name] == pytest.approx(
        (np.log(1600.0) - second_window.mean()) / second_window.std(ddof=1)
    )


def test_build_research_dataset_rejects_small_dollar_volume_zscore_window() -> None:
    """Dollar-volume z-score windows should support sample dispersion."""
    with pytest.raises(
        ValueError,
        match="dollar_volume_zscore_window must be at least 2",
    ):
        build_research_dataset(
            _sample_frame(),
            dollar_volume_zscore_window=1,
        )


def test_build_research_dataset_attaches_volume_shock() -> None:
    """Volume shock should compare log volume against a lagged log-volume baseline."""
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
            "open": [10.0, 10.0, 10.0, 10.0, 10.0],
            "high": [10.5, 10.5, 10.5, 10.5, 10.5],
            "low": [9.5, 9.5, 9.5, 9.5, 9.5],
            "close": [10.0, 10.0, 10.0, 10.0, 10.0],
            "volume": [10, 20, 40, 80, 160],
        }
    )

    dataset = build_research_dataset(
        frame,
        volume_shock_window=3,
    )

    column_name = "volume_shock_3d"
    first_window = np.log(np.array([10.0, 20.0, 40.0]))
    second_window = np.log(np.array([20.0, 40.0, 80.0]))

    assert column_name in dataset.columns
    assert dataset.loc[:2, column_name].isna().all()
    assert dataset.loc[3, column_name] == pytest.approx(
        np.log(80.0) - first_window.mean()
    )
    assert dataset.loc[4, column_name] == pytest.approx(
        np.log(160.0) - second_window.mean()
    )


def test_build_research_dataset_rejects_nonpositive_volume_shock_window() -> None:
    """Volume shock windows should be positive integers."""
    with pytest.raises(
        ValueError,
        match="volume_shock_window must be a positive integer",
    ):
        build_research_dataset(
            _sample_frame(),
            volume_shock_window=0,
        )


def test_build_research_dataset_attaches_relative_volume() -> None:
    """Relative volume should compare today against a lagged volume baseline."""
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
            "open": [10.0, 10.0, 10.0, 10.0, 10.0],
            "high": [10.5, 10.5, 10.5, 10.5, 10.5],
            "low": [9.5, 9.5, 9.5, 9.5, 9.5],
            "close": [10.0, 10.0, 10.0, 10.0, 10.0],
            "volume": [10, 20, 40, 80, 160],
        }
    )

    dataset = build_research_dataset(
        frame,
        relative_volume_window=3,
    )

    column_name = "relative_volume_3d"
    expected_ratio = 80.0 / np.mean([10.0, 20.0, 40.0])

    assert column_name in dataset.columns
    assert dataset.loc[:2, column_name].isna().all()
    assert dataset.loc[3, column_name] == pytest.approx(expected_ratio)
    assert dataset.loc[4, column_name] == pytest.approx(expected_ratio)


def test_build_research_dataset_rejects_nonpositive_relative_volume_window() -> None:
    """Relative volume windows should be positive integers."""
    with pytest.raises(
        ValueError,
        match="relative_volume_window must be a positive integer",
    ):
        build_research_dataset(
            _sample_frame(),
            relative_volume_window=0,
        )


def test_build_research_dataset_attaches_relative_dollar_volume() -> None:
    """Relative dollar volume should compare today against a lagged ADV baseline."""
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
            "open": [10.0, 10.0, 10.0, 10.0, 10.0],
            "high": [10.5, 10.5, 10.5, 10.5, 10.5],
            "low": [9.5, 9.5, 9.5, 9.5, 9.5],
            "close": [10.0, 10.0, 10.0, 10.0, 10.0],
            "volume": [10, 20, 40, 80, 160],
        }
    )

    dataset = build_research_dataset(
        frame,
        relative_dollar_volume_window=3,
    )

    column_name = "relative_dollar_volume_3d"
    expected_ratio = 800.0 / np.mean([100.0, 200.0, 400.0])

    assert column_name in dataset.columns
    assert dataset.loc[:2, column_name].isna().all()
    assert dataset.loc[3, column_name] == pytest.approx(expected_ratio)
    assert dataset.loc[4, column_name] == pytest.approx(expected_ratio)


def test_build_research_dataset_rejects_nonpositive_relative_dollar_volume_window() -> None:
    """Relative dollar volume windows should be positive integers."""
    with pytest.raises(
        ValueError,
        match="relative_dollar_volume_window must be a positive integer",
    ):
        build_research_dataset(
            _sample_frame(),
            relative_dollar_volume_window=0,
        )


def test_build_research_dataset_attaches_rogers_satchell_volatility() -> None:
    """Rogers-Satchell volatility should use trailing OHLC ranges only."""
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
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [110.0, 120.0, 130.0, 140.0, 150.0],
            "low": [95.0, 95.0, 95.0, 95.0, 95.0],
            "close": [102.0, 104.0, 106.0, 108.0, 110.0],
            "volume": [10, 11, 12, 13, 14],
        }
    )

    dataset = build_research_dataset(
        frame,
        rogers_satchell_volatility_window=4,
    )

    column_name = "rogers_satchell_volatility_4d"
    daily_variances = (
        np.log(np.array([1.10, 1.20, 1.30, 1.40]) / np.array([1.0, 1.0, 1.0, 1.0]))
        * np.log(np.array([1.10, 1.20, 1.30, 1.40]) / np.array([1.02, 1.04, 1.06, 1.08]))
        + np.log(np.array([0.95, 0.95, 0.95, 0.95]) / np.array([1.0, 1.0, 1.0, 1.0]))
        * np.log(np.array([0.95, 0.95, 0.95, 0.95]) / np.array([1.02, 1.04, 1.06, 1.08]))
    )
    expected_last = np.sqrt(np.mean(daily_variances))

    assert column_name in dataset.columns
    assert dataset.loc[:2, column_name].isna().all()
    assert dataset.loc[3, column_name] == pytest.approx(expected_last)


def test_build_research_dataset_rejects_nonpositive_rogers_satchell_volatility_window() -> None:
    """Rogers-Satchell volatility windows should be positive integers."""
    with pytest.raises(
        ValueError,
        match="rogers_satchell_volatility_window must be a positive integer",
    ):
        build_research_dataset(
            _sample_frame(),
            rogers_satchell_volatility_window=0,
        )


def test_build_research_dataset_attaches_yang_zhang_volatility() -> None:
    """Yang-Zhang volatility should use trailing overnight and OHLC components."""
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
            "open": [100.0, 103.0, 100.0, 106.0, 102.0],
            "high": [102.0, 106.0, 107.0, 109.0, 105.0],
            "low": [99.0, 101.0, 99.0, 102.0, 100.0],
            "close": [101.0, 102.0, 105.0, 103.0, 104.0],
            "volume": [10, 11, 12, 13, 14],
        }
    )

    dataset = build_research_dataset(
        frame,
        yang_zhang_volatility_window=4,
    )

    column_name = "yang_zhang_volatility_4d"
    overnight_returns = np.log(np.array([103.0 / 101.0, 100.0 / 102.0, 106.0 / 105.0, 102.0 / 103.0]))
    open_to_close_returns = np.log(np.array([102.0 / 103.0, 105.0 / 100.0, 103.0 / 106.0, 104.0 / 102.0]))
    rogers_satchell_variances = (
        np.log(np.array([106.0, 107.0, 109.0, 105.0]) / np.array([103.0, 100.0, 106.0, 102.0]))
        * np.log(np.array([106.0, 107.0, 109.0, 105.0]) / np.array([102.0, 105.0, 103.0, 104.0]))
        + np.log(np.array([101.0, 99.0, 102.0, 100.0]) / np.array([103.0, 100.0, 106.0, 102.0]))
        * np.log(np.array([101.0, 99.0, 102.0, 100.0]) / np.array([102.0, 105.0, 103.0, 104.0]))
    )
    weight = 0.34 / (1.34 + (4.0 + 1.0) / (4.0 - 1.0))
    expected_last = np.sqrt(
        overnight_returns.var(ddof=1)
        + weight * open_to_close_returns.var(ddof=1)
        + (1.0 - weight) * rogers_satchell_variances.mean()
    )

    assert column_name in dataset.columns
    assert dataset.loc[:3, column_name].isna().all()
    assert dataset.loc[4, column_name] == pytest.approx(expected_last)


def test_build_research_dataset_rejects_too_small_yang_zhang_volatility_window() -> None:
    """Yang-Zhang volatility needs at least two observations."""
    with pytest.raises(
        ValueError,
        match="yang_zhang_volatility_window must be at least 2",
    ):
        build_research_dataset(
            _sample_frame(),
            yang_zhang_volatility_window=1,
        )


def test_build_research_dataset_attaches_realized_volatility_family() -> None:
    """Realized volatility features should use trailing RMS daily returns only."""
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
            "open": [100.0, 110.0, 88.0, 114.4, 68.64],
            "high": [101.0, 111.0, 89.0, 115.4, 69.64],
            "low": [99.0, 109.0, 87.0, 113.4, 67.64],
            "close": [100.0, 110.0, 88.0, 114.4, 68.64],
            "volume": [10, 11, 12, 13, 14],
        }
    )

    dataset = build_research_dataset(
        frame,
        realized_volatility_window=4,
    )

    realized_column = "realized_volatility_4d"
    downside_column = "downside_realized_volatility_4d"
    upside_column = "upside_realized_volatility_4d"

    assert realized_column in dataset.columns
    assert downside_column in dataset.columns
    assert upside_column in dataset.columns
    assert dataset.loc[:3, realized_column].isna().all()
    assert dataset.loc[:3, downside_column].isna().all()
    assert dataset.loc[:3, upside_column].isna().all()
    assert dataset.loc[4, realized_column] == pytest.approx(np.sqrt(0.075))
    assert dataset.loc[4, downside_column] == pytest.approx(np.sqrt(0.05))
    assert dataset.loc[4, upside_column] == pytest.approx(np.sqrt(0.025))


def test_build_research_dataset_rejects_nonpositive_realized_volatility_window() -> None:
    """Realized volatility windows should be positive integers."""
    with pytest.raises(
        ValueError,
        match="realized_volatility_window must be a positive integer",
    ):
        build_research_dataset(
            _sample_frame(),
            realized_volatility_window=0,
        )


def test_build_research_dataset_attaches_rolling_higher_moments() -> None:
    """Rolling skew/kurtosis should use trailing strategy daily returns only."""
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
            "open": [100.0, 110.0, 132.0, 171.6, 240.24],
            "high": [101.0, 111.0, 133.0, 172.6, 241.24],
            "low": [99.0, 109.0, 131.0, 170.6, 239.24],
            "close": [100.0, 110.0, 132.0, 171.6, 240.24],
            "volume": [10, 11, 12, 13, 14],
        }
    )

    dataset = build_research_dataset(
        frame,
        higher_moments_window=4,
    )

    skew_column = "rolling_skew_4d"
    kurtosis_column = "rolling_kurtosis_4d"

    assert skew_column in dataset.columns
    assert kurtosis_column in dataset.columns
    assert dataset.loc[:3, skew_column].isna().all()
    assert dataset.loc[:3, kurtosis_column].isna().all()
    assert dataset.loc[4, skew_column] == pytest.approx(0.0)
    assert dataset.loc[4, kurtosis_column] == pytest.approx(-1.2)


def test_build_research_dataset_rejects_small_higher_moments_window() -> None:
    """Rolling skew/kurtosis should reject windows smaller than 4."""
    with pytest.raises(
        ValueError,
        match="higher_moments_window must be at least 4",
    ):
        build_research_dataset(
            _sample_frame(),
            higher_moments_window=3,
        )


def test_build_research_dataset_requires_benchmark_for_rolling_statistics() -> None:
    """Explicit rolling benchmark stats should require benchmark input."""
    with pytest.raises(
        ValueError,
        match="benchmark_rolling_window requires benchmark_returns",
    ):
        build_research_dataset(
            _sample_frame(),
            benchmark_rolling_window=3,
        )


def test_build_research_dataset_requires_benchmark_for_residual_returns() -> None:
    """Explicit benchmark residual returns should require benchmark input."""
    with pytest.raises(
        ValueError,
        match="benchmark_residual_return_window requires benchmark_returns",
    ):
        build_research_dataset(
            _sample_frame(),
            benchmark_residual_return_window=2,
        )


def test_build_research_dataset_rejects_small_benchmark_residual_window() -> None:
    """Benchmark residual returns need at least two prior observations for OLS."""
    benchmark_returns = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "benchmark_return": [0.0, 0.01],
        }
    )

    with pytest.raises(
        ValueError,
        match="benchmark_residual_return_window must be at least 2",
    ):
        build_research_dataset(
            _sample_frame(),
            benchmark_returns=benchmark_returns,
            benchmark_residual_return_window=1,
        )


def test_build_research_dataset_rejects_benchmark_date_misalignment() -> None:
    """Benchmark rolling stats should fail loudly on benchmark/date mismatches."""
    benchmark_returns = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-05"],
            "benchmark_return": [0.01, -0.02, 0.04],
        }
    )
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "open": [100.0, 102.0, 97.92],
            "high": [101.0, 103.0, 98.92],
            "low": [99.0, 101.0, 96.92],
            "close": [100.0, 102.0, 97.92],
            "volume": [10, 11, 12],
        }
    )

    with pytest.raises(
        ValueError,
        match="benchmark returns must align exactly to research dataset dates",
    ):
        build_research_dataset(
            frame,
            benchmark_returns=benchmark_returns,
            benchmark_rolling_window=2,
        )


def test_build_research_dataset_rejects_benchmark_residual_date_misalignment() -> None:
    """Benchmark residual returns should fail loudly on benchmark/date mismatches."""
    benchmark_returns = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-05"],
            "benchmark_return": [0.01, -0.02, 0.04],
        }
    )
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "open": [100.0, 102.0, 97.92],
            "high": [101.0, 103.0, 98.92],
            "low": [99.0, 101.0, 96.92],
            "close": [100.0, 102.0, 97.92],
            "volume": [10, 11, 12],
        }
    )

    with pytest.raises(
        ValueError,
        match="benchmark returns must align exactly to research dataset dates",
    ):
        build_research_dataset(
            frame,
            benchmark_returns=benchmark_returns,
            benchmark_residual_return_window=2,
        )


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

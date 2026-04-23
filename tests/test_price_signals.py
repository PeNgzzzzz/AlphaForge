"""Tests for reusable price-based signal functions."""

from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.signals import (
    add_mean_reversion_signal,
    add_momentum_signal,
    add_trend_signal,
)


def test_add_momentum_signal_computes_trailing_close_return() -> None:
    """Momentum should equal the trailing close return over the chosen lookback."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "open": [100.0, 110.0, 121.0],
            "high": [101.0, 111.0, 122.0],
            "low": [99.0, 109.0, 120.0],
            "close": [100.0, 110.0, 121.0],
            "volume": [10, 20, 30],
        }
    )

    signaled = add_momentum_signal(frame, lookback=1)

    assert pd.isna(signaled.loc[0, "momentum_signal_1d"])
    assert signaled.loc[1, "momentum_signal_1d"] == pytest.approx(0.10)
    assert signaled.loc[2, "momentum_signal_1d"] == pytest.approx(0.10)


def test_add_mean_reversion_signal_is_negative_of_short_term_return() -> None:
    """Mean reversion should invert the trailing close return."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "open": [100.0, 110.0, 121.0],
            "high": [101.0, 111.0, 122.0],
            "low": [99.0, 109.0, 120.0],
            "close": [100.0, 110.0, 121.0],
            "volume": [10, 20, 30],
        }
    )

    signaled = add_mean_reversion_signal(frame, lookback=1)

    assert pd.isna(signaled.loc[0, "mean_reversion_signal_1d"])
    assert signaled.loc[1, "mean_reversion_signal_1d"] == pytest.approx(-0.10)
    assert signaled.loc[2, "mean_reversion_signal_1d"] == pytest.approx(-0.10)


def test_add_trend_signal_computes_moving_average_spread() -> None:
    """Trend should compare the short and long moving averages."""
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

    signaled = add_trend_signal(frame, short_window=2, long_window=3)

    assert pd.isna(signaled.loc[0, "trend_signal_2_3d"])
    assert pd.isna(signaled.loc[1, "trend_signal_2_3d"])
    assert signaled.loc[2, "trend_signal_2_3d"] == pytest.approx(0.04682779456193353)
    assert signaled.loc[3, "trend_signal_2_3d"] == pytest.approx(0.04682779456193353)


def test_signal_functions_sort_input_and_keep_symbol_boundaries() -> None:
    """Signals should be computed per symbol after deterministic sorting."""
    frame = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-02", "2024-01-03", "2024-01-02"],
            "symbol": ["MSFT", "AAPL", "AAPL", "MSFT"],
            "open": [210.0, 100.0, 110.0, 200.0],
            "high": [231.0, 101.0, 111.0, 201.0],
            "low": [209.0, 99.0, 109.0, 199.0],
            "close": [230.0, 100.0, 110.0, 200.0],
            "volume": [60, 10, 20, 50],
        }
    )

    signaled = add_momentum_signal(frame, lookback=1)

    assert signaled["symbol"].tolist() == ["AAPL", "AAPL", "MSFT", "MSFT"]
    assert signaled["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-02",
        "2024-01-03",
    ]
    assert pd.isna(signaled.loc[0, "momentum_signal_1d"])
    assert pd.isna(signaled.loc[2, "momentum_signal_1d"])
    assert signaled.loc[1, "momentum_signal_1d"] == pytest.approx(0.10)
    assert signaled.loc[3, "momentum_signal_1d"] == pytest.approx(0.15)


def test_signal_functions_validate_parameters() -> None:
    """Signal parameter validation should fail loudly on invalid inputs."""
    frame = _sample_frame()

    with pytest.raises(ValueError, match="lookback"):
        add_momentum_signal(frame, lookback=0)

    with pytest.raises(ValueError, match="lookback"):
        add_mean_reversion_signal(frame, lookback=0)

    with pytest.raises(ValueError, match="short_window must be smaller"):
        add_trend_signal(frame, short_window=3, long_window=3)


def _sample_frame() -> pd.DataFrame:
    """Build a minimal valid OHLCV frame for parameter validation tests."""
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [10, 20, 30],
        }
    )

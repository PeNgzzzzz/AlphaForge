"""Reusable price-based signals built on daily OHLCV panels."""

from __future__ import annotations

import pandas as pd

from alphaforge.common.validation import normalize_positive_int as _common_positive_int
from alphaforge.data import validate_ohlcv


def add_momentum_signal(
    frame: pd.DataFrame,
    *,
    lookback: int = 20,
    signal_column: str | None = None,
) -> pd.DataFrame:
    """Append a close-to-close momentum signal.

    The score is the cumulative close return over the trailing ``lookback`` bars,
    anchored at the current row's close. It is a research signal, not a trade
    instruction, and should later be paired with an explicit execution delay.
    """
    lookback = _normalize_positive_int(lookback, parameter_name="lookback")
    signal_column = signal_column or f"momentum_signal_{lookback}d"

    dataset = validate_ohlcv(frame, source="momentum signal input").copy()
    close_by_symbol = dataset.groupby("symbol", sort=False)["close"]
    dataset[signal_column] = close_by_symbol.pct_change(periods=lookback)
    return dataset


def add_mean_reversion_signal(
    frame: pd.DataFrame,
    *,
    lookback: int = 5,
    signal_column: str | None = None,
) -> pd.DataFrame:
    """Append a short-horizon mean reversion signal.

    The score is the negative of the trailing close return over ``lookback`` bars.
    Positive recent performance therefore maps to a negative reversion score.
    """
    lookback = _normalize_positive_int(lookback, parameter_name="lookback")
    signal_column = signal_column or f"mean_reversion_signal_{lookback}d"

    dataset = validate_ohlcv(frame, source="mean reversion signal input").copy()
    close_by_symbol = dataset.groupby("symbol", sort=False)["close"]
    dataset[signal_column] = close_by_symbol.pct_change(periods=lookback).mul(-1.0)
    return dataset


def add_trend_signal(
    frame: pd.DataFrame,
    *,
    short_window: int = 20,
    long_window: int = 60,
    signal_column: str | None = None,
) -> pd.DataFrame:
    """Append a moving-average spread trend signal.

    The score is ``short_moving_average / long_moving_average - 1`` and uses
    prices observable through the current row's close only.
    """
    short_window = _normalize_positive_int(
        short_window, parameter_name="short_window"
    )
    long_window = _normalize_positive_int(long_window, parameter_name="long_window")
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window.")

    signal_column = signal_column or f"trend_signal_{short_window}_{long_window}d"

    dataset = validate_ohlcv(frame, source="trend signal input").copy()
    close_by_symbol = dataset.groupby("symbol", sort=False)["close"]
    short_average = close_by_symbol.transform(
        lambda values: values.rolling(
            window=short_window,
            min_periods=short_window,
        ).mean()
    )
    long_average = close_by_symbol.transform(
        lambda values: values.rolling(
            window=long_window,
            min_periods=long_window,
        ).mean()
    )
    dataset[signal_column] = short_average.div(long_average).sub(1.0)
    return dataset


def _normalize_positive_int(value: int, *, parameter_name: str) -> int:
    """Validate positive integer parameters used by signal definitions."""
    return _common_positive_int(value, parameter_name=parameter_name)

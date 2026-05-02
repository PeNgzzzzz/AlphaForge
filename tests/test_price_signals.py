"""Tests for reusable price-based signal functions."""

from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.signals import (
    apply_cross_sectional_signal_transform,
    add_mean_reversion_signal,
    add_momentum_signal,
    add_trend_signal,
    clip_signal_by_date,
    demean_signal_by_date,
    rank_normalize_signal_by_date,
    robust_zscore_signal_by_date,
    winsorize_signal_by_date,
    zscore_signal_by_date,
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


def test_winsorize_signal_by_date_clips_each_cross_section_independently() -> None:
    """Winsorization should clip scores within each date only."""
    frame = _cross_section_frame(
        signal_values=[0.0, 0.0, 0.0, 100.0, 1.0, 2.0, 3.0, 4.0],
    )

    transformed = winsorize_signal_by_date(
        frame,
        score_column="raw_signal",
        quantile=0.25,
    )

    date_one = transformed.loc[
        transformed["date"] == pd.Timestamp("2024-01-02"),
        "raw_signal_winsorized",
    ].tolist()
    date_two = transformed.loc[
        transformed["date"] == pd.Timestamp("2024-01-03"),
        "raw_signal_winsorized",
    ].tolist()

    assert date_one == pytest.approx([0.0, 0.0, 0.0, 25.0])
    assert date_two == pytest.approx([1.75, 2.0, 3.0, 3.25])


def test_clip_signal_by_date_uses_explicit_bounds_within_each_date() -> None:
    """Explicit clipping should preserve NaN and avoid cross-date state."""
    frame = _cross_section_frame(
        signal_values=[-5.0, -1.0, 1.0, 5.0, -10.0, float("nan"), 2.0, 10.0],
    )

    transformed = clip_signal_by_date(
        frame,
        score_column="raw_signal",
        lower_bound=-2.0,
        upper_bound=2.0,
    )

    date_one = transformed.loc[
        transformed["date"] == pd.Timestamp("2024-01-02"),
        "raw_signal_clipped",
    ].tolist()
    date_two = transformed.loc[
        transformed["date"] == pd.Timestamp("2024-01-03"),
        "raw_signal_clipped",
    ].tolist()

    assert date_one == pytest.approx([-2.0, -1.0, 1.0, 2.0])
    assert date_two[0] == pytest.approx(-2.0)
    assert pd.isna(date_two[1])
    assert date_two[2:] == pytest.approx([2.0, 2.0])


def test_zscore_signal_by_date_standardizes_each_date() -> None:
    """Z-scoring should use same-date mean and population dispersion only."""
    frame = _cross_section_frame(
        signal_values=[1.0, 3.0, 2.0, 4.0],
        symbols=["AAPL", "MSFT"],
        dates=["2024-01-02", "2024-01-03"],
    )

    transformed = zscore_signal_by_date(frame, score_column="raw_signal")

    assert transformed.loc[
        (transformed["date"] == pd.Timestamp("2024-01-02"))
        & (transformed["symbol"] == "AAPL"),
        "raw_signal_zscore",
    ].item() == pytest.approx(-1.0)
    assert transformed.loc[
        (transformed["date"] == pd.Timestamp("2024-01-02"))
        & (transformed["symbol"] == "MSFT"),
        "raw_signal_zscore",
    ].item() == pytest.approx(1.0)
    assert transformed.loc[
        (transformed["date"] == pd.Timestamp("2024-01-03"))
        & (transformed["symbol"] == "AAPL"),
        "raw_signal_zscore",
    ].item() == pytest.approx(-1.0)
    assert transformed.loc[
        (transformed["date"] == pd.Timestamp("2024-01-03"))
        & (transformed["symbol"] == "MSFT"),
        "raw_signal_zscore",
    ].item() == pytest.approx(1.0)


def test_robust_zscore_signal_by_date_uses_median_and_scaled_mad() -> None:
    """Robust z-scoring should use same-date median and MAD only."""
    frame = _cross_section_frame(
        signal_values=[0.0, 1.0, 2.0, 3.0, 10.0, 10.0, 10.0, 100.0],
    )

    transformed = robust_zscore_signal_by_date(frame, score_column="raw_signal")

    date_one = transformed.loc[
        transformed["date"] == pd.Timestamp("2024-01-02"),
        "raw_signal_robust_zscore",
    ].tolist()
    date_two = transformed.loc[
        transformed["date"] == pd.Timestamp("2024-01-03"),
        "raw_signal_robust_zscore",
    ]

    assert date_one == pytest.approx(
        [-1.5 / 1.4826, -0.5 / 1.4826, 0.5 / 1.4826, 1.5 / 1.4826]
    )
    assert date_two.isna().all()


def test_rank_normalize_signal_by_date_maps_average_ranks_to_unit_interval() -> None:
    """Rank normalization should map within-date average ranks to [0, 1]."""
    frame = _cross_section_frame(
        signal_values=[10.0, 20.0, 20.0, 40.0],
        symbols=["AAPL", "MSFT", "NVDA", "TSLA"],
        dates=["2024-01-02"],
    )

    transformed = rank_normalize_signal_by_date(frame, score_column="raw_signal")

    assert transformed["raw_signal_rank"].tolist() == pytest.approx([0.0, 0.5, 0.5, 1.0])


def test_zscore_signal_by_date_can_normalize_within_same_date_groups() -> None:
    """Grouped normalization should avoid pooling distinct same-date groups."""
    frame = _cross_section_frame(
        signal_values=[1.0, 3.0, 10.0, 20.0, 100.0],
        symbols=["AAPL", "MSFT", "NVDA", "TSLA", "ZZZ"],
        dates=["2024-01-02"],
    )
    frame["sector"] = ["Technology", "Technology", "Energy", "Energy", None]

    transformed = zscore_signal_by_date(
        frame,
        score_column="raw_signal",
        group_column="sector",
    )

    values = dict(
        zip(
            transformed["symbol"],
            transformed["raw_signal_zscore"],
            strict=True,
        )
    )
    assert values["AAPL"] == pytest.approx(-1.0)
    assert values["MSFT"] == pytest.approx(1.0)
    assert values["NVDA"] == pytest.approx(-1.0)
    assert values["TSLA"] == pytest.approx(1.0)
    assert pd.isna(values["ZZZ"])


def test_demean_signal_by_date_can_neutralize_same_date_groups() -> None:
    """Grouped de-meaning should remove same-date group means conservatively."""
    frame = _cross_section_frame(
        signal_values=[1.0, 3.0, 10.0, 20.0, 7.0, 100.0],
        symbols=["AAPL", "MSFT", "NVDA", "TSLA", "XOM", "ZZZ"],
        dates=["2024-01-02"],
    )
    frame["sector"] = [
        "Technology",
        "Technology",
        "Energy",
        "Energy",
        "Utilities",
        None,
    ]

    transformed = demean_signal_by_date(
        frame,
        score_column="raw_signal",
        group_column="sector",
    )

    values = dict(
        zip(
            transformed["symbol"],
            transformed["raw_signal_demeaned"],
            strict=True,
        )
    )
    assert values["AAPL"] == pytest.approx(-1.0)
    assert values["MSFT"] == pytest.approx(1.0)
    assert values["NVDA"] == pytest.approx(-5.0)
    assert values["TSLA"] == pytest.approx(5.0)
    assert pd.isna(values["XOM"])
    assert pd.isna(values["ZZZ"])


def test_apply_cross_sectional_signal_transform_composes_suffixes() -> None:
    """Configured transforms should preserve the raw signal and return the final column."""
    frame = _cross_section_frame(
        signal_values=[1.0, 2.0, 3.0, 100.0],
        symbols=["AAPL", "MSFT", "NVDA", "TSLA"],
        dates=["2024-01-02"],
    )

    transformed, signal_column = apply_cross_sectional_signal_transform(
        frame,
        score_column="raw_signal",
        winsorize_quantile=0.25,
        clip_lower_bound=1.8,
        clip_upper_bound=3.2,
        normalization=" Robust_ZScore ",
    )

    assert signal_column == "raw_signal_winsorized_clipped_robust_zscore"
    assert "raw_signal" in transformed.columns
    assert "raw_signal_winsorized" in transformed.columns
    assert "raw_signal_winsorized_clipped" in transformed.columns
    assert signal_column in transformed.columns


def test_apply_cross_sectional_signal_transform_demeans_before_normalization() -> None:
    """Configured de-meaning should run after clipping and before normalization."""
    frame = _cross_section_frame(
        signal_values=[1.0, 3.0, 10.0, 20.0],
        symbols=["AAPL", "MSFT", "NVDA", "TSLA"],
        dates=["2024-01-02"],
    )
    frame["sector"] = ["Technology", "Technology", "Energy", "Energy"]

    transformed, signal_column = apply_cross_sectional_signal_transform(
        frame,
        score_column="raw_signal",
        neutralize_group_column="sector",
        normalization="zscore",
    )

    assert signal_column == "raw_signal_demeaned_zscore"
    assert transformed["raw_signal_demeaned"].tolist() == pytest.approx(
        [-1.0, 1.0, -5.0, 5.0]
    )
    assert transformed[signal_column].tolist() == pytest.approx(
        [
            -1.0 / (13.0 ** 0.5),
            1.0 / (13.0 ** 0.5),
            -5.0 / (13.0 ** 0.5),
            5.0 / (13.0 ** 0.5),
        ]
    )


def test_cross_sectional_signal_transforms_validate_parameters() -> None:
    """Cross-sectional transform parameters should fail loudly on invalid inputs."""
    frame = _cross_section_frame(
        signal_values=[1.0, 2.0],
        symbols=["AAPL", "MSFT"],
        dates=["2024-01-02"],
    )

    with pytest.raises(ValueError, match="winsorize_quantile"):
        winsorize_signal_by_date(
            frame,
            score_column="raw_signal",
            quantile=0.5,
        )

    with pytest.raises(ValueError, match="normalization"):
        apply_cross_sectional_signal_transform(
            frame,
            score_column="raw_signal",
            normalization="robust",
        )

    with pytest.raises(ValueError, match="normalization_group_column"):
        apply_cross_sectional_signal_transform(
            frame,
            score_column="raw_signal",
            normalization_group_column="sector",
        )

    with pytest.raises(ValueError, match="group column 'sector'"):
        apply_cross_sectional_signal_transform(
            frame,
            score_column="raw_signal",
            normalization="zscore",
            normalization_group_column="sector",
        )

    with pytest.raises(ValueError, match="group column 'sector'"):
        apply_cross_sectional_signal_transform(
            frame,
            score_column="raw_signal",
            neutralize_group_column="sector",
        )

    with pytest.raises(ValueError, match="clip_lower_bound"):
        clip_signal_by_date(
            frame,
            score_column="raw_signal",
            lower_bound=1.0,
            upper_bound=1.0,
        )

    with pytest.raises(ValueError, match="clip_upper_bound"):
        apply_cross_sectional_signal_transform(
            frame,
            score_column="raw_signal",
            clip_lower_bound=-1.0,
        )


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


def _cross_section_frame(
    *,
    signal_values: list[float],
    symbols: list[str] | None = None,
    dates: list[str] | None = None,
) -> pd.DataFrame:
    """Build a valid OHLCV panel with an extra raw signal column."""
    if symbols is None:
        symbols = ["AAPL", "MSFT", "NVDA", "TSLA"]
    if dates is None:
        dates = ["2024-01-02", "2024-01-03"]

    rows: list[dict[str, object]] = []
    value_index = 0
    for date in dates:
        for symbol in symbols:
            if value_index >= len(signal_values):
                break
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000,
                    "raw_signal": signal_values[value_index],
                }
            )
            value_index += 1

    return pd.DataFrame(rows)

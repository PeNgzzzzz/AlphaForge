"""Tests for portfolio weight construction."""

from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.portfolio import (
    PortfolioConstructionError,
    build_long_only_weights,
    build_long_short_weights,
)


def test_build_long_only_weights_equal_weight_top_selection() -> None:
    """Long-only equal weights should allocate across the top-ranked names."""
    frame = _panel_with_signal(
        [
            ("2024-01-02", "AAPL", 3.0),
            ("2024-01-03", "AAPL", 2.0),
            ("2024-01-02", "MSFT", 2.0),
            ("2024-01-03", "MSFT", 4.0),
            ("2024-01-02", "NVDA", 1.0),
            ("2024-01-03", "NVDA", 1.0),
        ]
    )

    weighted = build_long_only_weights(
        frame,
        score_column="signal_score",
        top_n=2,
        weighting="equal",
        exposure=1.0,
    )

    first_day = weighted.loc[weighted["date"] == pd.Timestamp("2024-01-02")]
    second_day = weighted.loc[weighted["date"] == pd.Timestamp("2024-01-03")]

    assert first_day["portfolio_weight"].sum() == pytest.approx(1.0)
    assert second_day["portfolio_weight"].sum() == pytest.approx(1.0)
    assert first_day.loc[first_day["symbol"] == "AAPL", "portfolio_weight"].iloc[0] == pytest.approx(0.5)
    assert first_day.loc[first_day["symbol"] == "MSFT", "portfolio_weight"].iloc[0] == pytest.approx(0.5)
    assert first_day.loc[first_day["symbol"] == "NVDA", "portfolio_weight"].iloc[0] == pytest.approx(0.0)


def test_build_long_only_weights_score_weighting_normalizes_selected_scores() -> None:
    """Score weighting should overweight stronger selected names."""
    frame = _panel_with_signal(
        [
            ("2024-01-02", "AAPL", 5.0),
            ("2024-01-02", "MSFT", 4.0),
            ("2024-01-02", "NVDA", 1.0),
        ]
    )

    weighted = build_long_only_weights(
        frame,
        score_column="signal_score",
        top_n=2,
        weighting="score",
        exposure=1.0,
    )

    aapl_weight = weighted.loc[
        weighted["symbol"] == "AAPL",
        "portfolio_weight",
    ].iloc[0]
    msft_weight = weighted.loc[
        weighted["symbol"] == "MSFT",
        "portfolio_weight",
    ].iloc[0]

    assert aapl_weight == pytest.approx(2.0 / 3.0)
    assert msft_weight == pytest.approx(1.0 / 3.0)


def test_build_long_only_weights_applies_max_position_weight_with_redistribution() -> None:
    """A long-only position cap should redistribute leftover exposure across uncapped names."""
    frame = _panel_with_signal(
        [
            ("2024-01-02", "AAPL", 5.0),
            ("2024-01-02", "MSFT", 4.0),
            ("2024-01-02", "NVDA", 1.0),
        ]
    )

    weighted = build_long_only_weights(
        frame,
        score_column="signal_score",
        top_n=2,
        weighting="score",
        exposure=1.0,
        max_position_weight=0.6,
    )

    assert weighted.loc[weighted["symbol"] == "AAPL", "portfolio_weight"].iloc[0] == pytest.approx(0.6)
    assert weighted.loc[weighted["symbol"] == "MSFT", "portfolio_weight"].iloc[0] == pytest.approx(0.4)
    assert weighted["portfolio_weight"].sum() == pytest.approx(1.0)


def test_build_long_only_weights_applies_group_weight_cap() -> None:
    """A group cap should limit same-date exposure without re-expanding cash."""
    frame = _panel_with_signal(
        [
            ("2024-01-02", "AAPL", 5.0),
            ("2024-01-02", "MSFT", 4.0),
            ("2024-01-02", "TSLA", 3.0),
        ]
    )
    frame["classification_sector"] = ["Technology", "Technology", "Consumer"]

    weighted = build_long_only_weights(
        frame,
        score_column="signal_score",
        top_n=3,
        weighting="equal",
        exposure=1.0,
        group_column="classification_sector",
        max_group_weight=0.5,
    )

    tech_weight = weighted.loc[
        weighted["classification_sector"] == "Technology",
        "portfolio_weight",
    ].sum()
    consumer_weight = weighted.loc[
        weighted["classification_sector"] == "Consumer",
        "portfolio_weight",
    ].sum()

    assert tech_weight == pytest.approx(0.5)
    assert consumer_weight == pytest.approx(1.0 / 3.0)
    assert weighted["portfolio_weight"].sum() == pytest.approx(5.0 / 6.0)


def test_build_long_only_weights_zeros_missing_group_labels_when_capped() -> None:
    """Missing group labels should not bypass an explicit group constraint."""
    frame = _panel_with_signal(
        [
            ("2024-01-02", "AAPL", 5.0),
            ("2024-01-02", "MSFT", 4.0),
            ("2024-01-02", "TSLA", 3.0),
        ]
    )
    frame["classification_sector"] = ["Technology", None, " "]

    weighted = build_long_only_weights(
        frame,
        score_column="signal_score",
        top_n=3,
        weighting="equal",
        exposure=1.0,
        group_column="classification_sector",
        max_group_weight=1.0,
    )

    assert weighted.loc[weighted["symbol"] == "AAPL", "portfolio_weight"].iloc[0] == pytest.approx(1.0 / 3.0)
    assert weighted.loc[weighted["symbol"] == "MSFT", "portfolio_weight"].iloc[0] == pytest.approx(0.0)
    assert weighted.loc[weighted["symbol"] == "TSLA", "portfolio_weight"].iloc[0] == pytest.approx(0.0)


def test_build_long_short_weights_equal_weight_balances_both_sides() -> None:
    """Long-short equal weights should hit the configured side exposures."""
    frame = _panel_with_signal(
        [
            ("2024-01-02", "AAPL", 3.0),
            ("2024-01-02", "MSFT", 2.0),
            ("2024-01-02", "NVDA", -1.0),
            ("2024-01-02", "TSLA", -2.0),
        ]
    )

    weighted = build_long_short_weights(
        frame,
        score_column="signal_score",
        top_n=1,
        bottom_n=1,
        weighting="equal",
        long_exposure=1.0,
        short_exposure=1.0,
    )

    assert weighted["portfolio_weight"].sum() == pytest.approx(0.0)
    assert weighted.loc[weighted["symbol"] == "AAPL", "portfolio_weight"].iloc[0] == pytest.approx(1.0)
    assert weighted.loc[weighted["symbol"] == "TSLA", "portfolio_weight"].iloc[0] == pytest.approx(-1.0)


def test_build_long_short_weights_score_weighting_uses_side_relative_strength() -> None:
    """Score weighting should normalize long and short baskets independently."""
    frame = _panel_with_signal(
        [
            ("2024-01-02", "AAPL", 5.0),
            ("2024-01-02", "MSFT", 4.0),
            ("2024-01-02", "NVDA", -4.0),
            ("2024-01-02", "TSLA", -5.0),
        ]
    )

    weighted = build_long_short_weights(
        frame,
        score_column="signal_score",
        top_n=2,
        bottom_n=2,
        weighting="score",
        long_exposure=1.0,
        short_exposure=1.0,
    )

    assert weighted.loc[weighted["symbol"] == "AAPL", "portfolio_weight"].iloc[0] == pytest.approx(2.0 / 3.0)
    assert weighted.loc[weighted["symbol"] == "MSFT", "portfolio_weight"].iloc[0] == pytest.approx(1.0 / 3.0)
    assert weighted.loc[weighted["symbol"] == "TSLA", "portfolio_weight"].iloc[0] == pytest.approx(-2.0 / 3.0)
    assert weighted.loc[weighted["symbol"] == "NVDA", "portfolio_weight"].iloc[0] == pytest.approx(-1.0 / 3.0)


def test_build_long_short_weights_applies_max_position_weight_per_side() -> None:
    """Position caps should apply independently to the long and short books."""
    frame = _panel_with_signal(
        [
            ("2024-01-02", "AAPL", 5.0),
            ("2024-01-02", "MSFT", 4.0),
            ("2024-01-02", "NVDA", -4.0),
            ("2024-01-02", "TSLA", -5.0),
        ]
    )

    weighted = build_long_short_weights(
        frame,
        score_column="signal_score",
        top_n=2,
        bottom_n=2,
        weighting="score",
        long_exposure=1.0,
        short_exposure=1.0,
        max_position_weight=0.6,
    )

    assert weighted.loc[weighted["symbol"] == "AAPL", "portfolio_weight"].iloc[0] == pytest.approx(0.6)
    assert weighted.loc[weighted["symbol"] == "MSFT", "portfolio_weight"].iloc[0] == pytest.approx(0.4)
    assert weighted.loc[weighted["symbol"] == "TSLA", "portfolio_weight"].iloc[0] == pytest.approx(-0.6)
    assert weighted.loc[weighted["symbol"] == "NVDA", "portfolio_weight"].iloc[0] == pytest.approx(-0.4)


def test_build_long_short_weights_applies_group_weight_cap_per_side() -> None:
    """Long-short group caps should apply to long and short books separately."""
    frame = _panel_with_signal(
        [
            ("2024-01-02", "AAPL", 5.0),
            ("2024-01-02", "MSFT", 4.0),
            ("2024-01-02", "TSLA", -5.0),
            ("2024-01-02", "F", -4.0),
        ]
    )
    frame["classification_sector"] = [
        "Technology",
        "Technology",
        "Consumer",
        "Consumer",
    ]

    weighted = build_long_short_weights(
        frame,
        score_column="signal_score",
        top_n=2,
        bottom_n=2,
        weighting="equal",
        long_exposure=1.0,
        short_exposure=1.0,
        group_column="classification_sector",
        max_group_weight=0.6,
    )

    assert weighted.loc[weighted["symbol"] == "AAPL", "portfolio_weight"].iloc[0] == pytest.approx(0.3)
    assert weighted.loc[weighted["symbol"] == "MSFT", "portfolio_weight"].iloc[0] == pytest.approx(0.3)
    assert weighted.loc[weighted["symbol"] == "TSLA", "portfolio_weight"].iloc[0] == pytest.approx(-0.3)
    assert weighted.loc[weighted["symbol"] == "F", "portfolio_weight"].iloc[0] == pytest.approx(-0.3)
    assert weighted["portfolio_weight"].abs().sum() == pytest.approx(1.2)


def test_build_long_short_weights_zero_out_insufficient_universe_dates() -> None:
    """Dates without enough names for both sides should remain uninvested."""
    frame = _panel_with_signal(
        [
            ("2024-01-02", "AAPL", 3.0),
            ("2024-01-02", "MSFT", 2.0),
            ("2024-01-02", "NVDA", -1.0),
        ]
    )

    weighted = build_long_short_weights(
        frame,
        score_column="signal_score",
        top_n=2,
        bottom_n=2,
    )

    assert weighted["portfolio_weight"].sum() == pytest.approx(0.0)
    assert (weighted["portfolio_weight"] == 0.0).all()


def test_portfolio_weight_functions_validate_inputs() -> None:
    """Invalid portfolio construction settings should fail loudly."""
    frame = _panel_with_signal(
        [
            ("2024-01-02", "AAPL", 1.0),
            ("2024-01-02", "MSFT", 2.0),
        ]
    )

    with pytest.raises(PortfolioConstructionError, match="score column"):
        build_long_only_weights(frame, score_column="missing", top_n=1)

    with pytest.raises(PortfolioConstructionError, match="top_n"):
        build_long_only_weights(frame, score_column="signal_score", top_n=0)

    with pytest.raises(PortfolioConstructionError, match="weighting"):
        build_long_only_weights(
            frame,
            score_column="signal_score",
            top_n=1,
            weighting="unsupported",
        )

    with pytest.raises(PortfolioConstructionError, match="short_exposure"):
        build_long_short_weights(
            frame,
            score_column="signal_score",
            top_n=1,
            short_exposure=-1.0,
        )

    bad_frame = frame.copy()
    bad_frame["signal_score"] = bad_frame["signal_score"].astype("object")
    bad_frame.loc[0, "signal_score"] = "bad"
    with pytest.raises(PortfolioConstructionError, match="invalid numeric values"):
        build_long_only_weights(
            bad_frame,
            score_column="signal_score",
            top_n=1,
        )

    with pytest.raises(PortfolioConstructionError, match="exposure"):
        build_long_only_weights(
            frame,
            score_column="signal_score",
            top_n=1,
            exposure=float("nan"),
        )

    with pytest.raises(PortfolioConstructionError, match="max_position_weight"):
        build_long_only_weights(
            frame,
            score_column="signal_score",
            top_n=1,
            max_position_weight=0.0,
        )

    with pytest.raises(PortfolioConstructionError, match="group column"):
        build_long_only_weights(
            frame,
            score_column="signal_score",
            top_n=1,
            group_column="classification_sector",
            max_group_weight=0.5,
        )

    with pytest.raises(PortfolioConstructionError, match="configured together"):
        build_long_only_weights(
            frame,
            score_column="signal_score",
            top_n=1,
            max_group_weight=0.5,
        )

    frame_with_group = frame.assign(classification_sector="Technology")
    with pytest.raises(PortfolioConstructionError, match="max_group_weight"):
        build_long_only_weights(
            frame_with_group,
            score_column="signal_score",
            top_n=1,
            group_column="classification_sector",
            max_group_weight=0.0,
        )


def _panel_with_signal(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Build a minimal OHLCV panel with an attached signal column."""
    records = []
    for row_index, (date, symbol, signal_score) in enumerate(rows, start=1):
        close = float(100 + row_index)
        records.append(
            {
                "date": date,
                "symbol": symbol,
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1000 + row_index,
                "signal_score": signal_score,
            }
        )
    return pd.DataFrame(records)

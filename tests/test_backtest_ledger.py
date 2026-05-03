"""Tests for weight-based position ledger utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.backtest import BacktestError, build_position_ledger


def test_build_position_ledger_applies_signal_delay_without_lookahead() -> None:
    """A target observed on day t should only become a position after the delay."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
            ("2024-01-04", "AAPL", 121.0, 1.0),
        ]
    )

    ledger = build_position_ledger(frame, signal_delay=1)

    assert ledger["date"].tolist() == [pd.Timestamp("2024-01-04")]
    assert ledger["symbol"].tolist() == ["AAPL"]
    assert ledger["starting_weight"].tolist() == pytest.approx([0.0])
    assert ledger["trade_weight"].tolist() == pytest.approx([1.0])
    assert ledger["ending_weight"].tolist() == pytest.approx([1.0])
    assert ledger["asset_return"].tolist() == pytest.approx([0.10])
    assert ledger["position_return_contribution"].tolist() == pytest.approx([0.10])
    assert ledger["trade_side"].tolist() == ["buy"]
    assert ledger["position_side"].tolist() == ["long"]


def test_build_position_ledger_keeps_unfilled_weekly_rebalance_gap_visible() -> None:
    """Skipped rebalance days should show target gaps without fabricated trades."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
            ("2024-01-04", "AAPL", 121.0, 1.0),
            ("2024-01-05", "AAPL", 133.1, 1.0),
            ("2024-01-08", "AAPL", 146.41, 1.0),
            ("2024-01-09", "AAPL", 161.051, 1.0),
        ]
    )

    ledger = build_position_ledger(
        frame,
        signal_delay=1,
        rebalance_frequency="weekly",
    )

    assert ledger["date"].tolist() == [
        pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-08"),
        pd.Timestamp("2024-01-09"),
    ]
    assert ledger["trade_weight"].tolist() == pytest.approx([0.0, 0.0, 1.0, 0.0])
    assert ledger["ending_weight"].tolist() == pytest.approx([0.0, 0.0, 1.0, 1.0])
    assert ledger["target_position_gap"].tolist() == pytest.approx([1.0, 1.0, 0.0, 0.0])
    assert ledger["trade_side"].tolist() == ["hold", "hold", "buy", "hold"]
    assert ledger["position_side"].tolist() == ["flat", "flat", "long", "long"]


def test_build_position_ledger_exposes_turnover_limited_carry() -> None:
    """Turnover caps should show partial positions and remaining target gaps."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
            ("2024-01-04", "AAPL", 121.0, 1.0),
            ("2024-01-05", "AAPL", 133.1, 1.0),
        ]
    )

    ledger = build_position_ledger(
        frame,
        signal_delay=1,
        max_turnover=0.5,
    )

    assert ledger["date"].tolist() == [
        pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-05"),
    ]
    assert ledger["starting_weight"].tolist() == pytest.approx([0.0, 0.5])
    assert ledger["trade_weight"].tolist() == pytest.approx([0.5, 0.5])
    assert ledger["ending_weight"].tolist() == pytest.approx([0.5, 1.0])
    assert ledger["target_position_gap"].tolist() == pytest.approx([0.5, 0.0])
    assert ledger["turnover_contribution"].tolist() == pytest.approx([0.5, 0.5])
    assert ledger["turnover_limit_applied"].tolist() == [True, False]


def test_build_position_ledger_exposes_trade_limited_carry() -> None:
    """Row-level trade limits should show partial positions and remaining gaps."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
            ("2024-01-04", "AAPL", 121.0, 1.0),
            ("2024-01-05", "AAPL", 133.1, 1.0),
        ]
    )
    frame["max_trade_weight"] = [0.0, 0.0, 0.4, 0.4]

    ledger = build_position_ledger(
        frame,
        signal_delay=1,
        max_trade_weight_column="max_trade_weight",
    )

    assert ledger["date"].tolist() == [
        pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-05"),
    ]
    assert ledger["starting_weight"].tolist() == pytest.approx([0.0, 0.4])
    assert ledger["max_trade_weight"].tolist() == pytest.approx([0.4, 0.4])
    assert ledger["trade_weight"].tolist() == pytest.approx([0.4, 0.4])
    assert ledger["ending_weight"].tolist() == pytest.approx([0.4, 0.8])
    assert ledger["target_position_gap"].tolist() == pytest.approx([0.6, 0.2])
    assert ledger["turnover_contribution"].tolist() == pytest.approx([0.4, 0.4])
    assert ledger["trade_limit_applied"].tolist() == [True, True]
    assert ledger["turnover_limit_applied"].tolist() == [False, False]


def test_build_position_ledger_exposes_participation_limited_carry() -> None:
    """Participation caps should be visible in the weight-based ledger."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 100.0, 1.0),
            ("2024-01-04", "AAPL", 100.0, 1.0),
            ("2024-01-05", "AAPL", 100.0, 1.0),
        ]
    )

    ledger = build_position_ledger(
        frame,
        signal_delay=1,
        max_participation_rate=0.01,
        participation_notional=2500.0,
    )

    assert ledger["date"].tolist() == [
        pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-05"),
    ]
    assert ledger["participation_trade_weight_limit"].tolist() == pytest.approx(
        [0.4, 0.4]
    )
    assert ledger["max_trade_weight"].tolist() == pytest.approx([0.4, 0.4])
    assert ledger["trade_weight"].tolist() == pytest.approx([0.4, 0.4])
    assert ledger["ending_weight"].tolist() == pytest.approx([0.4, 0.8])
    assert ledger["target_position_gap"].tolist() == pytest.approx([0.6, 0.2])
    assert ledger["participation_limit_applied"].tolist() == [True, True]
    assert ledger["trade_limit_applied"].tolist() == [True, True]


def test_build_position_ledger_exposes_trade_clipped_carry() -> None:
    """Minimum trade clipping should leave a visible target gap in the ledger."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.00),
            ("2024-01-03", "AAPL", 100.0, 0.03),
            ("2024-01-04", "AAPL", 100.0, 0.06),
        ]
    )

    ledger = build_position_ledger(
        frame,
        signal_delay=1,
        min_trade_weight=0.05,
    )

    assert ledger["date"].tolist() == [pd.Timestamp("2024-01-04")]
    assert ledger["starting_weight"].tolist() == pytest.approx([0.0])
    assert ledger["trade_weight"].tolist() == pytest.approx([0.0])
    assert ledger["ending_weight"].tolist() == pytest.approx([0.0])
    assert ledger["target_position_gap"].tolist() == pytest.approx([0.03])
    assert ledger["trade_side"].tolist() == ["hold"]
    assert ledger["position_side"].tolist() == ["flat"]
    assert ledger["trade_clip_applied"].tolist() == [True]


def test_build_position_ledger_exposes_short_availability_limits() -> None:
    """Short availability constraints should be visible in the position ledger."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 90.0, -0.5),
            ("2024-01-04", "AAPL", 81.0, -0.5),
        ]
    )
    frame["is_shortable"] = [True, True, False]

    ledger = build_position_ledger(
        frame,
        signal_delay=1,
        shortable_column="is_shortable",
    )

    assert ledger["date"].tolist() == [pd.Timestamp("2024-01-04")]
    assert ledger["is_shortable"].tolist() == [False]
    assert ledger["short_constrained_target_weight"].tolist() == pytest.approx([0.0])
    assert ledger["trade_weight"].tolist() == pytest.approx([0.0])
    assert ledger["ending_weight"].tolist() == pytest.approx([0.0])
    assert ledger["target_position_gap"].tolist() == pytest.approx([-0.5])
    assert ledger["short_availability_limit_applied"].tolist() == [True]


def test_build_position_ledger_exposes_tradability_limits() -> None:
    """Untradable rows should carry prior positions and expose target gaps."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
            ("2024-01-04", "AAPL", 121.0, 0.0),
            ("2024-01-05", "AAPL", 133.1, 0.0),
        ]
    )
    frame["is_tradable"] = [True, True, True, False]

    ledger = build_position_ledger(
        frame,
        signal_delay=1,
        tradable_column="is_tradable",
    )

    untradable_row = ledger.loc[
        ledger["date"] == pd.Timestamp("2024-01-05")
    ].iloc[0]
    assert not bool(untradable_row["is_tradable"])
    assert untradable_row["starting_weight"] == pytest.approx(1.0)
    assert untradable_row["trade_weight"] == pytest.approx(0.0)
    assert untradable_row["ending_weight"] == pytest.approx(1.0)
    assert untradable_row["target_position_gap"] == pytest.approx(-1.0)
    assert bool(untradable_row["tradability_limit_applied"])


def test_build_position_ledger_supports_next_close_fill_timing() -> None:
    """Next-close fills should delay ledger positions by one close-to-close period."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
            ("2024-01-04", "AAPL", 121.0, 1.0),
            ("2024-01-05", "AAPL", 133.1, 1.0),
        ]
    )

    ledger = build_position_ledger(
        frame,
        signal_delay=1,
        fill_timing="next_close",
    )

    assert ledger["date"].tolist() == [pd.Timestamp("2024-01-05")]
    assert ledger["signal_delayed_target_weight"].tolist() == pytest.approx([1.0])
    assert ledger["delayed_target_weight"].tolist() == pytest.approx([1.0])
    assert ledger["fill_timing"].tolist() == ["next_close"]
    assert ledger["fill_delay_periods"].tolist() == [1]
    assert ledger["execution_delay_periods"].tolist() == [2]
    assert ledger["trade_weight"].tolist() == pytest.approx([1.0])
    assert ledger["ending_weight"].tolist() == pytest.approx([1.0])


def test_build_position_ledger_classifies_short_positions() -> None:
    """Negative effective weights should be recorded as short positions."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 90.0, -0.5),
            ("2024-01-04", "AAPL", 81.0, -0.5),
        ]
    )

    ledger = build_position_ledger(frame, signal_delay=1)

    assert ledger["trade_weight"].tolist() == pytest.approx([-0.5])
    assert ledger["ending_weight"].tolist() == pytest.approx([-0.5])
    assert ledger["position_return_contribution"].tolist() == pytest.approx([0.05])
    assert ledger["trade_side"].tolist() == ["sell"]
    assert ledger["position_side"].tolist() == ["short"]


def test_build_position_ledger_exposes_short_borrow_costs() -> None:
    """Borrow fees should be visible as position-level short carry diagnostics."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 90.0, -0.5),
            ("2024-01-04", "AAPL", 81.0, -0.5),
        ]
    )
    frame["borrow_fee_bps"] = [0.0, 0.0, 252.0]

    ledger = build_position_ledger(
        frame,
        signal_delay=1,
        borrow_fee_bps_column="borrow_fee_bps",
    )

    assert ledger["date"].tolist() == [pd.Timestamp("2024-01-04")]
    assert ledger["borrow_fee_bps"].tolist() == pytest.approx([252.0])
    assert ledger["short_exposure"].tolist() == pytest.approx([0.5])
    assert ledger["borrow_cost_contribution"].tolist() == pytest.approx([0.00005])


def test_build_position_ledger_filters_small_weight_rows() -> None:
    """The minimum position threshold should suppress tiny ledger rows."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 0.001),
            ("2024-01-04", "AAPL", 121.0, 0.001),
        ]
    )

    ledger = build_position_ledger(
        frame,
        signal_delay=1,
        min_position_weight=0.01,
    )

    assert ledger.empty


def test_build_position_ledger_validates_inputs() -> None:
    """Ledger helpers should preserve fail-fast backtest validation."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
        ]
    )

    with pytest.raises(BacktestError, match="min_position_weight"):
        build_position_ledger(frame, min_position_weight=-0.1)

    with pytest.raises(BacktestError, match="fill_timing"):
        build_position_ledger(frame, fill_timing="next_open")

    with pytest.raises(BacktestError, match="weight column"):
        build_position_ledger(frame.drop(columns=["portfolio_weight"]))

    with pytest.raises(BacktestError, match="max_trade_weight_column"):
        build_position_ledger(
            frame,
            max_trade_weight_column="missing_max_trade_weight",
        )

    with pytest.raises(BacktestError, match="borrow_fee_bps_column"):
        build_position_ledger(frame, borrow_fee_bps_column="missing_borrow_fee_bps")

    with pytest.raises(BacktestError, match="shortable_column"):
        build_position_ledger(frame, shortable_column="missing_is_shortable")

    with pytest.raises(BacktestError, match="tradable_column"):
        build_position_ledger(frame, tradable_column="missing_is_tradable")

    with pytest.raises(BacktestError, match="min_trade_weight"):
        build_position_ledger(frame, min_trade_weight=-0.01)


def _panel_with_weights(
    rows: list[tuple[str, str, float, float | None]],
) -> pd.DataFrame:
    """Build a minimal OHLCV panel with an attached target weight column."""
    records = []
    for date, symbol, close, weight in rows:
        records.append(
            {
                "date": date,
                "symbol": symbol,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1_000,
                "portfolio_weight": weight,
            }
        )
    return pd.DataFrame(records)

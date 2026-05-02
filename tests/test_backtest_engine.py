"""Tests for the conservative daily backtest engine."""

from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.backtest import (
    BacktestError,
    prepare_daily_backtest_panel,
    run_daily_backtest,
)


def test_prepare_daily_backtest_panel_applies_signal_delay_without_lookahead() -> None:
    """Weights observed on day t should affect returns only after the configured delay."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
            ("2024-01-04", "AAPL", 121.0, 1.0),
        ]
    )

    panel = prepare_daily_backtest_panel(frame, signal_delay=1)

    assert panel["effective_weight"].tolist() == pytest.approx([0.0, 0.0, 1.0])
    assert pd.isna(panel.loc[0, "asset_return"])
    assert panel.loc[1, "gross_return_contribution"] == pytest.approx(0.0)
    assert panel.loc[2, "asset_return"] == pytest.approx(0.10)
    assert panel.loc[2, "gross_return_contribution"] == pytest.approx(0.10)


def test_prepare_daily_backtest_panel_supports_next_close_fill_timing() -> None:
    """Next-close fills should add one conservative close-to-close delay."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
            ("2024-01-04", "AAPL", 121.0, 1.0),
            ("2024-01-05", "AAPL", 133.1, 1.0),
        ]
    )

    panel = prepare_daily_backtest_panel(
        frame,
        signal_delay=1,
        fill_timing=" next_close ",
    )

    assert panel["fill_timing"].tolist() == ["next_close"] * 4
    assert panel["fill_delay_periods"].tolist() == [1] * 4
    assert panel["execution_delay_periods"].tolist() == [2] * 4
    assert panel["signal_delayed_target_weight"].tolist() == pytest.approx(
        [0.0, 0.0, 1.0, 1.0]
    )
    assert panel["effective_weight"].tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 1.0]
    )
    assert panel.loc[2, "gross_return_contribution"] == pytest.approx(0.0)
    assert panel.loc[3, "asset_return"] == pytest.approx(0.10)
    assert panel.loc[3, "gross_return_contribution"] == pytest.approx(0.10)


def test_run_daily_backtest_charges_turnover_costs_on_entries_and_exits() -> None:
    """Transaction costs should scale with absolute weight changes."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
            ("2024-01-04", "AAPL", 121.0, 0.0),
            ("2024-01-05", "AAPL", 121.0, 0.0),
        ]
    )

    results = run_daily_backtest(frame, signal_delay=1, transaction_cost_bps=10.0)

    third_day = results.loc[results["date"] == pd.Timestamp("2024-01-04")].iloc[0]
    fourth_day = results.loc[results["date"] == pd.Timestamp("2024-01-05")].iloc[0]

    assert third_day["gross_return"] == pytest.approx(0.10)
    assert third_day["turnover"] == pytest.approx(1.0)
    assert third_day["transaction_cost"] == pytest.approx(0.001)
    assert third_day["net_return"] == pytest.approx(0.099)
    assert fourth_day["gross_return"] == pytest.approx(0.0)
    assert fourth_day["turnover"] == pytest.approx(1.0)
    assert fourth_day["transaction_cost"] == pytest.approx(0.001)


def test_prepare_daily_backtest_panel_supports_weekly_rebalance_frequency() -> None:
    """Weekly rebalancing should only update effective weights on weekly rebalance dates."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 1.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
            ("2024-01-04", "AAPL", 121.0, 1.0),
            ("2024-01-05", "AAPL", 133.1, 1.0),
            ("2024-01-08", "AAPL", 146.41, 1.0),
            ("2024-01-09", "AAPL", 161.051, 1.0),
        ]
    )

    panel = prepare_daily_backtest_panel(
        frame,
        signal_delay=1,
        rebalance_frequency=" weekly ",
    )

    assert panel["is_rebalance_date"].tolist() == [True, False, False, False, True, False]
    assert panel["effective_weight"].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])


def test_prepare_daily_backtest_panel_supports_monthly_rebalance_frequency() -> None:
    """Monthly rebalancing should trigger on the first available date of each month."""
    frame = _panel_with_weights(
        [
            ("2024-01-30", "AAPL", 100.0, 1.0),
            ("2024-01-31", "AAPL", 101.0, 1.0),
            ("2024-02-01", "AAPL", 102.0, 1.0),
            ("2024-02-02", "AAPL", 103.0, 1.0),
        ]
    )

    panel = prepare_daily_backtest_panel(
        frame,
        signal_delay=1,
        rebalance_frequency="monthly",
    )

    assert panel["is_rebalance_date"].tolist() == [True, False, True, False]


def test_run_daily_backtest_splits_commission_and_slippage_costs() -> None:
    """Commission and slippage should be tracked separately and sum into total cost."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
            ("2024-01-04", "AAPL", 121.0, 0.0),
            ("2024-01-05", "AAPL", 121.0, 0.0),
        ]
    )

    results = run_daily_backtest(
        frame,
        signal_delay=1,
        commission_bps=4.0,
        slippage_bps=6.0,
    )

    third_day = results.loc[results["date"] == pd.Timestamp("2024-01-04")].iloc[0]

    assert third_day["turnover"] == pytest.approx(1.0)
    assert third_day["commission_cost"] == pytest.approx(0.0004)
    assert third_day["slippage_cost"] == pytest.approx(0.0006)
    assert third_day["transaction_cost"] == pytest.approx(0.001)


def test_run_daily_backtest_applies_max_turnover_progressively() -> None:
    """A turnover cap should scale trades toward target weights instead of jumping fully."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
            ("2024-01-04", "AAPL", 121.0, 1.0),
            ("2024-01-05", "AAPL", 133.1, 1.0),
        ]
    )

    panel = prepare_daily_backtest_panel(
        frame,
        signal_delay=1,
        max_turnover=0.5,
    )
    results = run_daily_backtest(
        frame,
        signal_delay=1,
        max_turnover=0.5,
    )

    assert panel["executed_weight"].tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0])
    assert panel["turnover_limit_applied"].tolist() == [False, False, True, False]
    assert panel["target_effective_weight_gap"].tolist() == pytest.approx([0.0, 0.0, 0.5, 0.0])

    third_day = results.loc[results["date"] == pd.Timestamp("2024-01-04")].iloc[0]
    assert third_day["target_turnover"] == pytest.approx(1.0)
    assert third_day["turnover"] == pytest.approx(0.5)
    assert third_day["target_effective_weight_gap"] == pytest.approx(0.5)
    assert bool(third_day["turnover_limit_applied"])


def test_run_daily_backtest_aggregates_long_short_weights_and_nav() -> None:
    """Daily aggregation should preserve exposure and cumulative NAV behavior."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.5),
            ("2024-01-03", "AAPL", 110.0, 0.5),
            ("2024-01-02", "MSFT", 100.0, -0.5),
            ("2024-01-03", "MSFT", 90.0, -0.5),
        ]
    )

    results = run_daily_backtest(frame, signal_delay=1, transaction_cost_bps=0.0)
    second_day = results.loc[results["date"] == pd.Timestamp("2024-01-03")].iloc[0]

    assert second_day["gross_return"] == pytest.approx(0.10)
    assert second_day["net_return"] == pytest.approx(0.10)
    assert second_day["gross_exposure"] == pytest.approx(1.0)
    assert second_day["net_exposure"] == pytest.approx(0.0)
    assert second_day["holdings_count"] == 2
    assert second_day["gross_nav"] == pytest.approx(1.10)
    assert second_day["net_nav"] == pytest.approx(1.10)


def test_run_daily_backtest_treats_missing_weights_as_zero_exposure() -> None:
    """Missing weight values should be interpreted as flat positions."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, None),
            ("2024-01-03", "AAPL", 110.0, None),
        ]
    )

    results = run_daily_backtest(frame, signal_delay=1)

    assert (results["gross_return"] == 0.0).all()
    assert (results["turnover"] == 0.0).all()
    assert (results["gross_exposure"] == 0.0).all()


def test_run_daily_backtest_preserves_legacy_transaction_cost_compatibility() -> None:
    """Legacy transaction_cost_bps should match commission-only split cost settings."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
            ("2024-01-04", "AAPL", 121.0, 0.0),
        ]
    )

    legacy = run_daily_backtest(frame, signal_delay=1, transaction_cost_bps=10.0)
    split = run_daily_backtest(frame, signal_delay=1, commission_bps=10.0, slippage_bps=0.0)

    pd.testing.assert_frame_equal(legacy, split)


def test_backtest_functions_validate_inputs() -> None:
    """Invalid backtest settings and weight inputs should fail loudly."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 1.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
        ]
    )

    with pytest.raises(BacktestError, match="weight column"):
        run_daily_backtest(frame.drop(columns=["portfolio_weight"]))

    with pytest.raises(BacktestError, match="signal_delay"):
        run_daily_backtest(frame, signal_delay=0)

    with pytest.raises(BacktestError, match="transaction_cost_bps"):
        run_daily_backtest(frame, transaction_cost_bps=-1.0)

    with pytest.raises(BacktestError, match="rebalance_frequency"):
        run_daily_backtest(frame, rebalance_frequency="quarterly")

    with pytest.raises(BacktestError, match="fill_timing"):
        run_daily_backtest(frame, fill_timing="next_open")

    with pytest.raises(BacktestError, match="cannot be combined"):
        run_daily_backtest(
            frame,
            transaction_cost_bps=5.0,
            commission_bps=1.0,
        )

    with pytest.raises(BacktestError, match="max_turnover"):
        run_daily_backtest(frame, max_turnover=-0.1)

    with pytest.raises(BacktestError, match="initial_nav"):
        run_daily_backtest(frame, initial_nav=0.0)

    bad_frame = frame.copy()
    bad_frame["portfolio_weight"] = bad_frame["portfolio_weight"].astype("object")
    bad_frame.loc[0, "portfolio_weight"] = "bad"
    with pytest.raises(BacktestError, match="invalid numeric values"):
        prepare_daily_backtest_panel(bad_frame)


def _panel_with_weights(
    rows: list[tuple[str, str, float, float | None]]
) -> pd.DataFrame:
    """Build a minimal OHLCV panel with an attached target weight column."""
    records = []
    for date, symbol, close, portfolio_weight in rows:
        records.append(
            {
                "date": date,
                "symbol": symbol,
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1_000.0,
                "portfolio_weight": portfolio_weight,
            }
        )
    return pd.DataFrame(records)

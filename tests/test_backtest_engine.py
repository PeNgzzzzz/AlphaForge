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


def test_run_daily_backtest_supports_row_level_cost_bps_columns() -> None:
    """Row-level cost bps should price turnover using each executed row's settings."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
            ("2024-01-04", "AAPL", 121.0, 0.0),
            ("2024-01-05", "AAPL", 121.0, 0.0),
        ]
    )
    frame["row_commission_bps"] = [0.0, 0.0, 4.0, 20.0]
    frame["row_slippage_bps"] = [0.0, 0.0, 6.0, 30.0]

    results = run_daily_backtest(
        frame,
        signal_delay=1,
        commission_bps_column="row_commission_bps",
        slippage_bps_column="row_slippage_bps",
    )

    third_day = results.loc[results["date"] == pd.Timestamp("2024-01-04")].iloc[0]
    fourth_day = results.loc[results["date"] == pd.Timestamp("2024-01-05")].iloc[0]

    assert third_day["turnover"] == pytest.approx(1.0)
    assert third_day["commission_cost"] == pytest.approx(0.0004)
    assert third_day["slippage_cost"] == pytest.approx(0.0006)
    assert third_day["transaction_cost"] == pytest.approx(0.001)
    assert fourth_day["turnover"] == pytest.approx(1.0)
    assert fourth_day["commission_cost"] == pytest.approx(0.002)
    assert fourth_day["slippage_cost"] == pytest.approx(0.003)
    assert fourth_day["transaction_cost"] == pytest.approx(0.005)


def test_run_daily_backtest_supports_liquidity_bucket_slippage_model() -> None:
    """Explicit liquidity buckets should map to slippage bps for executed turnover."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
            ("2024-01-04", "AAPL", 121.0, 0.0),
            ("2024-01-05", "AAPL", 121.0, 0.0),
        ]
    )
    frame["liquidity_bucket"] = ["high", "high", "low", "low"]

    results = run_daily_backtest(
        frame,
        signal_delay=1,
        commission_bps=4.0,
        liquidity_bucket_column="liquidity_bucket",
        slippage_bps_by_liquidity_bucket={
            "high": 2.0,
            "low": 12.0,
        },
    )

    third_day = results.loc[results["date"] == pd.Timestamp("2024-01-04")].iloc[0]
    assert third_day["turnover"] == pytest.approx(1.0)
    assert third_day["commission_cost"] == pytest.approx(0.0004)
    assert third_day["slippage_cost"] == pytest.approx(0.0012)
    assert third_day["transaction_cost"] == pytest.approx(0.0016)


def test_run_daily_backtest_charges_borrow_cost_on_short_exposure() -> None:
    """Borrow fees should apply only to realized short exposure."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 90.0, -0.5),
            ("2024-01-04", "AAPL", 81.0, -0.5),
        ]
    )
    frame["borrow_fee_bps"] = [0.0, 0.0, 252.0]

    results = run_daily_backtest(
        frame,
        signal_delay=1,
        borrow_fee_bps_column="borrow_fee_bps",
    )

    third_day = results.loc[results["date"] == pd.Timestamp("2024-01-04")].iloc[0]
    assert third_day["short_exposure"] == pytest.approx(0.5)
    assert third_day["gross_return"] == pytest.approx(0.05)
    assert third_day["borrow_cost"] == pytest.approx(0.00005)
    assert third_day["transaction_cost"] == pytest.approx(0.00005)
    assert third_day["net_return"] == pytest.approx(0.04995)


def test_run_daily_backtest_blocks_short_targets_when_not_shortable() -> None:
    """Explicit shortable flags should prevent new realized short exposure."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 90.0, -0.5),
            ("2024-01-04", "AAPL", 81.0, -0.5),
        ]
    )
    frame["is_shortable"] = [True, True, False]

    panel = prepare_daily_backtest_panel(
        frame,
        signal_delay=1,
        shortable_column="is_shortable",
    )
    results = run_daily_backtest(
        frame,
        signal_delay=1,
        shortable_column="is_shortable",
    )

    third_panel_row = panel.loc[panel["date"] == pd.Timestamp("2024-01-04")].iloc[0]
    assert third_panel_row["delayed_target_weight"] == pytest.approx(-0.5)
    assert third_panel_row["short_constrained_target_weight"] == pytest.approx(0.0)
    assert third_panel_row["executed_weight"] == pytest.approx(0.0)
    assert third_panel_row["target_effective_weight_gap"] == pytest.approx(-0.5)
    assert bool(third_panel_row["short_availability_limit_applied"])

    third_day = results.loc[results["date"] == pd.Timestamp("2024-01-04")].iloc[0]
    assert third_day["target_turnover"] == pytest.approx(0.5)
    assert third_day["turnover"] == pytest.approx(0.0)
    assert third_day["gross_exposure"] == pytest.approx(0.0)
    assert third_day["target_effective_weight_gap"] == pytest.approx(0.5)
    assert bool(third_day["short_availability_limit_applied"])


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


def test_run_daily_backtest_applies_row_level_trade_weight_limits() -> None:
    """Explicit row-level trade limits should cap each executed weight change."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
            ("2024-01-04", "AAPL", 121.0, 1.0),
            ("2024-01-05", "AAPL", 133.1, 1.0),
        ]
    )
    frame["max_trade_weight"] = [0.0, 0.0, 0.4, 0.4]

    panel = prepare_daily_backtest_panel(
        frame,
        signal_delay=1,
        max_trade_weight_column="max_trade_weight",
    )
    results = run_daily_backtest(
        frame,
        signal_delay=1,
        max_trade_weight_column="max_trade_weight",
    )

    assert panel["executed_weight"].tolist() == pytest.approx([0.0, 0.0, 0.4, 0.8])
    assert panel["turnover_contribution"].tolist() == pytest.approx(
        [0.0, 0.0, 0.4, 0.4]
    )
    assert panel["target_effective_weight_gap"].tolist() == pytest.approx(
        [0.0, 0.0, 0.6, 0.2]
    )
    assert panel["trade_limit_applied"].tolist() == [False, False, True, True]

    third_day = results.loc[results["date"] == pd.Timestamp("2024-01-04")].iloc[0]
    fourth_day = results.loc[results["date"] == pd.Timestamp("2024-01-05")].iloc[0]
    assert third_day["target_turnover"] == pytest.approx(1.0)
    assert third_day["turnover"] == pytest.approx(0.4)
    assert third_day["target_effective_weight_gap"] == pytest.approx(0.6)
    assert bool(third_day["trade_limit_applied"])
    assert not bool(third_day["turnover_limit_applied"])
    assert fourth_day["turnover"] == pytest.approx(0.4)
    assert fourth_day["target_effective_weight_gap"] == pytest.approx(0.2)
    assert bool(fourth_day["trade_limit_applied"])


def test_run_daily_backtest_applies_participation_caps() -> None:
    """Participation caps should derive row-level trade limits from close and volume."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 100.0, 1.0),
            ("2024-01-04", "AAPL", 100.0, 1.0),
            ("2024-01-05", "AAPL", 100.0, 1.0),
        ]
    )

    panel = prepare_daily_backtest_panel(
        frame,
        signal_delay=1,
        max_participation_rate=0.01,
        participation_notional=2500.0,
    )
    results = run_daily_backtest(
        frame,
        signal_delay=1,
        max_participation_rate=0.01,
        participation_notional=2500.0,
    )

    assert panel["participation_trade_weight_limit"].tolist() == pytest.approx(
        [0.4, 0.4, 0.4, 0.4]
    )
    assert panel["max_trade_weight"].tolist() == pytest.approx([0.4, 0.4, 0.4, 0.4])
    assert panel["executed_weight"].tolist() == pytest.approx([0.0, 0.0, 0.4, 0.8])
    assert panel["participation_limit_applied"].tolist() == [
        False,
        False,
        True,
        True,
    ]
    assert panel["trade_limit_applied"].tolist() == [False, False, True, True]

    third_day = results.loc[results["date"] == pd.Timestamp("2024-01-04")].iloc[0]
    assert third_day["target_turnover"] == pytest.approx(1.0)
    assert third_day["turnover"] == pytest.approx(0.4)
    assert third_day["target_effective_weight_gap"] == pytest.approx(0.6)
    assert bool(third_day["participation_limit_applied"])
    assert bool(third_day["trade_limit_applied"])


def test_participation_caps_do_not_apply_on_skipped_rebalance_dates() -> None:
    """Non-rebalance dates should not report participation caps as applied."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 100.0, 1.0),
            ("2024-01-04", "AAPL", 100.0, 1.0),
        ]
    )

    panel = prepare_daily_backtest_panel(
        frame,
        signal_delay=1,
        rebalance_frequency="weekly",
        max_participation_rate=0.01,
        participation_notional=2500.0,
    )

    third_day = panel.loc[panel["date"] == pd.Timestamp("2024-01-04")].iloc[0]
    assert not bool(third_day["is_rebalance_date"])
    assert third_day["desired_weight_change"] == pytest.approx(1.0)
    assert third_day["participation_trade_weight_limit"] == pytest.approx(0.4)
    assert not bool(third_day["participation_limit_applied"])
    assert not bool(third_day["trade_limit_applied"])


def test_explicit_trade_limits_can_be_tighter_than_participation_caps() -> None:
    """Participation diagnostics should reflect whether that cap is the tighter cap."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 100.0, 1.0),
            ("2024-01-04", "AAPL", 100.0, 1.0),
        ]
    )
    frame["max_trade_weight"] = 0.2

    panel = prepare_daily_backtest_panel(
        frame,
        signal_delay=1,
        max_trade_weight_column="max_trade_weight",
        max_participation_rate=0.01,
        participation_notional=2500.0,
    )

    third_day = panel.loc[panel["date"] == pd.Timestamp("2024-01-04")].iloc[0]
    assert third_day["participation_trade_weight_limit"] == pytest.approx(0.4)
    assert third_day["max_trade_weight"] == pytest.approx(0.2)
    assert third_day["executed_weight"] == pytest.approx(0.2)
    assert not bool(third_day["participation_limit_applied"])
    assert bool(third_day["trade_limit_applied"])


def test_run_daily_backtest_clips_small_trade_weights() -> None:
    """Minimum trade weights should drop nonzero trades below the threshold."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.00),
            ("2024-01-03", "AAPL", 100.0, 0.03),
            ("2024-01-04", "AAPL", 100.0, 0.06),
            ("2024-01-05", "AAPL", 100.0, 0.06),
        ]
    )

    panel = prepare_daily_backtest_panel(
        frame,
        signal_delay=1,
        min_trade_weight=0.05,
    )
    results = run_daily_backtest(
        frame,
        signal_delay=1,
        min_trade_weight=0.05,
    )

    assert panel["desired_weight_change"].tolist() == pytest.approx(
        [0.0, 0.0, 0.03, 0.06]
    )
    assert panel["executed_weight"].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.06])
    assert panel["turnover_contribution"].tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 0.06]
    )
    assert panel["target_effective_weight_gap"].tolist() == pytest.approx(
        [0.0, 0.0, 0.03, 0.0]
    )
    assert panel["trade_clip_applied"].tolist() == [False, False, True, False]

    third_day = results.loc[results["date"] == pd.Timestamp("2024-01-04")].iloc[0]
    fourth_day = results.loc[results["date"] == pd.Timestamp("2024-01-05")].iloc[0]
    assert third_day["target_turnover"] == pytest.approx(0.03)
    assert third_day["turnover"] == pytest.approx(0.0)
    assert third_day["target_effective_weight_gap"] == pytest.approx(0.03)
    assert bool(third_day["trade_clip_applied"])
    assert fourth_day["turnover"] == pytest.approx(0.06)
    assert not bool(fourth_day["trade_clip_applied"])


def test_trade_clipping_runs_after_row_level_trade_limits() -> None:
    """A row-level cap below the minimum trade size should leave the trade unfilled."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 100.0, 1.0),
            ("2024-01-04", "AAPL", 100.0, 1.0),
        ]
    )
    frame["max_trade_weight"] = 0.04

    panel = prepare_daily_backtest_panel(
        frame,
        signal_delay=1,
        max_trade_weight_column="max_trade_weight",
        min_trade_weight=0.05,
    )

    third_day = panel.loc[panel["date"] == pd.Timestamp("2024-01-04")].iloc[0]
    assert third_day["desired_weight_change"] == pytest.approx(1.0)
    assert third_day["max_trade_weight"] == pytest.approx(0.04)
    assert third_day["executed_weight"] == pytest.approx(0.0)
    assert third_day["target_effective_weight_gap"] == pytest.approx(1.0)
    assert bool(third_day["trade_limit_applied"])
    assert bool(third_day["trade_clip_applied"])


def test_run_daily_backtest_combines_trade_and_turnover_limits() -> None:
    """Daily turnover limits should scale trades after row-level limits are applied."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
            ("2024-01-02", "MSFT", 100.0, 0.0),
            ("2024-01-03", "MSFT", 110.0, 1.0),
            ("2024-01-04", "AAPL", 121.0, 1.0),
            ("2024-01-04", "MSFT", 121.0, 1.0),
        ]
    )
    frame["max_trade_weight"] = [0.0, 0.0, 0.0, 0.0, 0.8, 0.8]

    results = run_daily_backtest(
        frame,
        signal_delay=1,
        max_trade_weight_column="max_trade_weight",
        max_turnover=1.0,
    )

    third_day = results.loc[results["date"] == pd.Timestamp("2024-01-04")].iloc[0]
    assert third_day["target_turnover"] == pytest.approx(2.0)
    assert third_day["turnover"] == pytest.approx(1.0)
    assert third_day["target_effective_weight_gap"] == pytest.approx(1.0)
    assert bool(third_day["trade_limit_applied"])
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

    with pytest.raises(BacktestError, match="commission_bps_column"):
        run_daily_backtest(frame, commission_bps_column="missing_commission_bps")

    with pytest.raises(BacktestError, match="cannot be combined"):
        run_daily_backtest(
            frame.assign(row_commission_bps=1.0),
            commission_bps=1.0,
            commission_bps_column="row_commission_bps",
        )

    with pytest.raises(BacktestError, match="cannot be combined"):
        run_daily_backtest(
            frame.assign(row_slippage_bps=1.0),
            transaction_cost_bps=5.0,
            slippage_bps_column="row_slippage_bps",
        )

    with pytest.raises(BacktestError, match="row_commission_bps"):
        run_daily_backtest(
            frame.assign(row_commission_bps=[1.0, None]),
            commission_bps_column="row_commission_bps",
        )

    with pytest.raises(BacktestError, match="row_slippage_bps"):
        run_daily_backtest(
            frame.assign(row_slippage_bps=[1.0, -1.0]),
            slippage_bps_column="row_slippage_bps",
        )

    with pytest.raises(BacktestError, match="liquidity_bucket_column"):
        run_daily_backtest(
            frame,
            liquidity_bucket_column="missing_liquidity_bucket",
            slippage_bps_by_liquidity_bucket={"high": 2.0},
        )

    with pytest.raises(BacktestError, match="unmapped liquidity bucket"):
        run_daily_backtest(
            frame.assign(liquidity_bucket=["high", "low"]),
            liquidity_bucket_column="liquidity_bucket",
            slippage_bps_by_liquidity_bucket={"high": 2.0},
        )

    with pytest.raises(BacktestError, match="liquidity-bucket slippage"):
        run_daily_backtest(
            frame.assign(liquidity_bucket=["high", "high"]),
            slippage_bps=1.0,
            liquidity_bucket_column="liquidity_bucket",
            slippage_bps_by_liquidity_bucket={"high": 2.0},
        )

    with pytest.raises(BacktestError, match="borrow_fee_bps_column"):
        run_daily_backtest(frame, borrow_fee_bps_column="missing_borrow_fee_bps")

    with pytest.raises(BacktestError, match="borrow_fee_bps"):
        run_daily_backtest(
            frame.assign(borrow_fee_bps=[0.0, None]),
            borrow_fee_bps_column="borrow_fee_bps",
        )

    with pytest.raises(BacktestError, match="borrow_fee_bps"):
        run_daily_backtest(
            frame.assign(borrow_fee_bps=[0.0, -1.0]),
            borrow_fee_bps_column="borrow_fee_bps",
        )

    with pytest.raises(BacktestError, match="shortable_column"):
        run_daily_backtest(frame, shortable_column="missing_is_shortable")

    with pytest.raises(BacktestError, match="shortable_column"):
        run_daily_backtest(
            frame.assign(is_shortable=[True, None]),
            shortable_column="is_shortable",
        )

    with pytest.raises(BacktestError, match="shortable_column"):
        run_daily_backtest(
            frame.assign(is_shortable=[True, "yes"]),
            shortable_column="is_shortable",
        )

    with pytest.raises(BacktestError, match="max_trade_weight_column"):
        run_daily_backtest(frame, max_trade_weight_column="missing_max_trade_weight")

    with pytest.raises(BacktestError, match="max_trade_weight"):
        run_daily_backtest(
            frame.assign(max_trade_weight=[1.0, None]),
            max_trade_weight_column="max_trade_weight",
        )

    with pytest.raises(BacktestError, match="max_trade_weight"):
        run_daily_backtest(
            frame.assign(max_trade_weight=[1.0, -0.1]),
            max_trade_weight_column="max_trade_weight",
        )

    with pytest.raises(BacktestError, match="configured together"):
        run_daily_backtest(frame, max_participation_rate=0.10)

    with pytest.raises(BacktestError, match="max_participation_rate"):
        run_daily_backtest(
            frame,
            max_participation_rate=1.10,
            participation_notional=1000.0,
        )

    with pytest.raises(BacktestError, match="participation_notional"):
        run_daily_backtest(
            frame,
            max_participation_rate=0.10,
            participation_notional=0.0,
        )

    with pytest.raises(BacktestError, match="min_trade_weight"):
        run_daily_backtest(frame, min_trade_weight=-0.01)

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

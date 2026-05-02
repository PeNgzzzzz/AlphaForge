"""Tests for target-weight order diagnostics."""

from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.backtest import BacktestError, generate_target_weight_orders


def test_generate_target_weight_orders_applies_signal_delay_without_lookahead() -> None:
    """A target observed on day t should not create an order until after the delay."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
            ("2024-01-04", "AAPL", 121.0, 1.0),
        ]
    )

    orders = generate_target_weight_orders(frame, signal_delay=1)

    assert orders["date"].tolist() == [pd.Timestamp("2024-01-04")]
    assert orders["symbol"].tolist() == ["AAPL"]
    assert orders["desired_order_weight"].tolist() == pytest.approx([1.0])
    assert orders["executed_order_weight"].tolist() == pytest.approx([1.0])
    assert orders["previous_weight"].tolist() == pytest.approx([0.0])
    assert orders["executed_weight"].tolist() == pytest.approx([1.0])
    assert orders["desired_order_side"].tolist() == ["buy"]
    assert orders["executed_order_side"].tolist() == ["buy"]


def test_generate_target_weight_orders_exposes_skipped_rebalance_orders() -> None:
    """Non-rebalance dates should keep target gaps visible without executing trades."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
            ("2024-01-04", "AAPL", 121.0, 1.0),
        ]
    )

    orders = generate_target_weight_orders(
        frame,
        signal_delay=1,
        rebalance_frequency="weekly",
    )

    assert orders["date"].tolist() == [pd.Timestamp("2024-01-04")]
    assert orders["desired_order_weight"].tolist() == pytest.approx([1.0])
    assert orders["executed_order_weight"].tolist() == pytest.approx([0.0])
    assert orders["unfilled_order_weight"].tolist() == pytest.approx([1.0])
    assert orders["desired_order_side"].tolist() == ["buy"]
    assert orders["executed_order_side"].tolist() == ["hold"]
    assert not bool(orders["is_rebalance_date"].iloc[0])


def test_generate_target_weight_orders_exposes_turnover_limited_orders() -> None:
    """Turnover caps should show both the executed trade and remaining target gap."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
            ("2024-01-04", "AAPL", 121.0, 1.0),
            ("2024-01-05", "AAPL", 133.1, 1.0),
        ]
    )

    orders = generate_target_weight_orders(
        frame,
        signal_delay=1,
        max_turnover=0.5,
    )

    assert orders["date"].tolist() == [
        pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-05"),
    ]
    assert orders["desired_order_weight"].tolist() == pytest.approx([1.0, 0.5])
    assert orders["executed_order_weight"].tolist() == pytest.approx([0.5, 0.5])
    assert orders["unfilled_order_weight"].tolist() == pytest.approx([0.5, 0.0])
    assert orders["realized_turnover_contribution"].tolist() == pytest.approx(
        [0.5, 0.5]
    )
    assert orders["turnover_limit_applied"].tolist() == [True, False]


def test_generate_target_weight_orders_filters_small_diagnostics() -> None:
    """The minimum order threshold should suppress tiny desired and executed deltas."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 0.001),
            ("2024-01-04", "AAPL", 121.0, 0.001),
        ]
    )

    orders = generate_target_weight_orders(
        frame,
        signal_delay=1,
        min_order_weight=0.01,
    )

    assert orders.empty


def test_generate_target_weight_orders_validates_inputs() -> None:
    """Order diagnostics should preserve fail-fast backtest input validation."""
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
        ]
    )

    with pytest.raises(BacktestError, match="min_order_weight"):
        generate_target_weight_orders(frame, min_order_weight=-0.1)

    with pytest.raises(BacktestError, match="weight column"):
        generate_target_weight_orders(frame.drop(columns=["portfolio_weight"]))


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

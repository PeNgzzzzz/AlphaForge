"""Target-weight order diagnostics for daily backtests."""

from __future__ import annotations

import pandas as pd

from alphaforge.backtest.engine import BacktestError, prepare_daily_backtest_panel
from alphaforge.common.validation import (
    normalize_non_negative_float as _common_non_negative_float,
)


def generate_target_weight_orders(
    frame: pd.DataFrame,
    *,
    weight_column: str = "portfolio_weight",
    signal_delay: int = 1,
    fill_timing: str = "close",
    rebalance_frequency: str = "daily",
    shortable_column: str | None = None,
    tradable_column: str | None = None,
    max_trade_weight_column: str | None = None,
    max_participation_rate: float | None = None,
    participation_notional: float | None = None,
    min_trade_weight: float | None = None,
    max_turnover: float | None = None,
    min_order_weight: float = 0.0,
) -> pd.DataFrame:
    """Generate target-weight order diagnostics from portfolio weights.

    The output is a weight-delta order view, not a share-level broker order book.
    Timing and turnover semantics intentionally reuse ``prepare_daily_backtest_panel``.
    """
    min_order_weight = _normalize_non_negative_float(
        min_order_weight,
        parameter_name="min_order_weight",
    )
    panel = prepare_daily_backtest_panel(
        frame,
        weight_column=weight_column,
        signal_delay=signal_delay,
        fill_timing=fill_timing,
        rebalance_frequency=rebalance_frequency,
        shortable_column=shortable_column,
        tradable_column=tradable_column,
        max_trade_weight_column=max_trade_weight_column,
        max_participation_rate=max_participation_rate,
        participation_notional=participation_notional,
        min_trade_weight=min_trade_weight,
        max_turnover=max_turnover,
    )

    orders = panel.loc[
        :,
        [
            "date",
            "symbol",
            "target_weight",
            "signal_delayed_target_weight",
            "delayed_target_weight",
            "is_shortable",
            "is_tradable",
            "short_constrained_target_weight",
            "tradability_constrained_target_weight",
            "fill_timing",
            "fill_delay_periods",
            "execution_delay_periods",
            "previous_effective_weight",
            "participation_trade_weight_limit",
            "max_trade_weight",
            "desired_weight_change",
            "weight_change",
            "executed_weight",
            "target_effective_weight_gap",
            "target_turnover_contribution",
            "turnover_contribution",
            "is_rebalance_date",
            "short_availability_limit_applied",
            "tradability_limit_applied",
            "participation_limit_applied",
            "trade_limit_applied",
            "trade_clip_applied",
            "turnover_limit_applied",
        ],
    ].copy()
    orders = orders.rename(
        columns={
            "previous_effective_weight": "previous_weight",
            "desired_weight_change": "desired_order_weight",
            "weight_change": "executed_order_weight",
            "target_effective_weight_gap": "unfilled_order_weight",
            "target_turnover_contribution": "desired_turnover_contribution",
            "turnover_contribution": "realized_turnover_contribution",
        }
    )
    orders["desired_order_side"] = _classify_order_side(
        orders["desired_order_weight"]
    )
    orders["executed_order_side"] = _classify_order_side(
        orders["executed_order_weight"]
    )

    active_order_mask = (
        orders["desired_order_weight"].abs().gt(min_order_weight)
        | orders["executed_order_weight"].abs().gt(min_order_weight)
        | orders["unfilled_order_weight"].abs().gt(min_order_weight)
    )
    return (
        orders.loc[active_order_mask]
        .sort_values(["date", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )


def _normalize_non_negative_float(value: float, *, parameter_name: str) -> float:
    """Validate non-negative float parameters."""
    return _common_non_negative_float(
        value,
        parameter_name=parameter_name,
        error_factory=BacktestError,
    )


def _classify_order_side(order_weight: pd.Series) -> pd.Series:
    """Classify signed target-weight deltas into simple side labels."""
    side = pd.Series("hold", index=order_weight.index, dtype="object")
    side.loc[order_weight.gt(0.0)] = "buy"
    side.loc[order_weight.lt(0.0)] = "sell"
    return side

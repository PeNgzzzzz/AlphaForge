"""Weight-based position ledger utilities for daily backtests."""

from __future__ import annotations

import pandas as pd

from alphaforge.backtest.engine import (
    BacktestError,
    _attach_borrow_cost_fields,
    prepare_daily_backtest_panel,
)
from alphaforge.common.validation import (
    normalize_non_negative_float as _common_non_negative_float,
)


def build_position_ledger(
    frame: pd.DataFrame,
    *,
    weight_column: str = "portfolio_weight",
    signal_delay: int = 1,
    fill_timing: str = "close",
    rebalance_frequency: str = "daily",
    rebalance_stagger_column: str | None = None,
    rebalance_stagger_count: int | None = None,
    shortable_column: str | None = None,
    tradable_column: str | None = None,
    can_buy_column: str | None = None,
    can_sell_column: str | None = None,
    max_trade_weight_column: str | None = None,
    max_participation_rate: float | None = None,
    participation_notional: float | None = None,
    min_trade_weight: float | None = None,
    max_turnover: float | None = None,
    borrow_fee_bps_column: str | None = None,
    min_position_weight: float = 0.0,
) -> pd.DataFrame:
    """Build a weight-based position ledger from target portfolio weights.

    The ledger is intentionally weight based. It does not model share counts,
    fill prices, cash accounting, or intraday execution.
    """
    min_position_weight = _normalize_non_negative_float(
        min_position_weight,
        parameter_name="min_position_weight",
    )
    panel = prepare_daily_backtest_panel(
        frame,
        weight_column=weight_column,
        signal_delay=signal_delay,
        fill_timing=fill_timing,
        rebalance_frequency=rebalance_frequency,
        rebalance_stagger_column=rebalance_stagger_column,
        rebalance_stagger_count=rebalance_stagger_count,
        shortable_column=shortable_column,
        tradable_column=tradable_column,
        can_buy_column=can_buy_column,
        can_sell_column=can_sell_column,
        max_trade_weight_column=max_trade_weight_column,
        max_participation_rate=max_participation_rate,
        participation_notional=participation_notional,
        min_trade_weight=min_trade_weight,
        max_turnover=max_turnover,
    )
    panel = _attach_borrow_cost_fields(
        panel,
        borrow_fee_bps_column=borrow_fee_bps_column,
        source="position ledger input",
    )

    ledger = panel.loc[
        :,
        [
            "date",
            "symbol",
            "target_weight",
            "signal_delayed_target_weight",
            "delayed_target_weight",
            "is_shortable",
            "is_tradable",
            "is_buyable",
            "is_sellable",
            "short_constrained_target_weight",
            "tradability_constrained_target_weight",
            "direction_constrained_target_weight",
            "fill_timing",
            "fill_delay_periods",
            "execution_delay_periods",
            "rebalance_stagger_bucket",
            "active_rebalance_stagger_bucket",
            "previous_effective_weight",
            "participation_trade_weight_limit",
            "max_trade_weight",
            "weight_change",
            "effective_weight",
            "asset_return",
            "gross_return_contribution",
            "borrow_fee_bps",
            "short_exposure",
            "borrow_cost_contribution",
            "target_effective_weight_gap",
            "target_effective_weight_gap_abs",
            "turnover_contribution",
            "is_base_rebalance_date",
            "is_rebalance_date",
            "short_availability_limit_applied",
            "tradability_limit_applied",
            "buy_limit_applied",
            "sell_limit_applied",
            "rebalance_stagger_skipped",
            "participation_limit_applied",
            "trade_limit_applied",
            "trade_clip_applied",
            "turnover_limit_applied",
        ],
    ].copy()
    ledger = ledger.rename(
        columns={
            "previous_effective_weight": "starting_weight",
            "weight_change": "trade_weight",
            "effective_weight": "ending_weight",
            "gross_return_contribution": "position_return_contribution",
            "target_effective_weight_gap": "target_position_gap",
            "target_effective_weight_gap_abs": "target_position_gap_abs",
        }
    )
    ledger["trade_side"] = _classify_trade_side(ledger["trade_weight"])
    ledger["position_side"] = _classify_position_side(ledger["ending_weight"])

    active_mask = (
        ledger["starting_weight"].abs().gt(min_position_weight)
        | ledger["trade_weight"].abs().gt(min_position_weight)
        | ledger["ending_weight"].abs().gt(min_position_weight)
        | ledger["target_position_gap"].abs().gt(min_position_weight)
    )
    return (
        ledger.loc[active_mask]
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


def _classify_trade_side(trade_weight: pd.Series) -> pd.Series:
    """Classify signed weight trades into simple side labels."""
    side = pd.Series("hold", index=trade_weight.index, dtype="object")
    side.loc[trade_weight.gt(0.0)] = "buy"
    side.loc[trade_weight.lt(0.0)] = "sell"
    return side


def _classify_position_side(weight: pd.Series) -> pd.Series:
    """Classify signed ending weights into simple position labels."""
    side = pd.Series("flat", index=weight.index, dtype="object")
    side.loc[weight.gt(0.0)] = "long"
    side.loc[weight.lt(0.0)] = "short"
    return side

"""Conservative daily backtesting utilities."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from alphaforge.common.errors import AlphaForgeError
from alphaforge.common.validation import (
    normalize_choice_string as _common_choice_string,
    normalize_non_negative_float as _common_non_negative_float,
    normalize_non_empty_string as _common_non_empty_string,
    normalize_optional_non_negative_float as _common_optional_non_negative_float,
    normalize_positive_float as _common_positive_float,
    normalize_positive_int as _common_positive_int,
)
from alphaforge.data import validate_ohlcv

_BORROW_COST_TRADING_DAYS_PER_YEAR = 252.0


class BacktestError(AlphaForgeError):
    """Raised when daily backtest inputs or settings are invalid."""


def prepare_daily_backtest_panel(
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
) -> pd.DataFrame:
    """Prepare per-symbol daily backtest fields from target portfolio weights.

    Timing convention:
    - ``asset_return`` on date ``t`` is the close-to-close return from ``t-1`` to ``t``
    - ``target_weight`` on date ``t`` is observed at the close of ``t``
    - ``signal_delayed_target_weight`` on date ``t`` is the signal-delayed target
    - ``delayed_target_weight`` on date ``t`` is the fill-timing-adjusted desired allocation
    - ``is_base_rebalance_date`` is the date-level rebalance schedule
    - ``is_rebalance_date`` is the row-level schedule after optional staggering
    - ``short_constrained_target_weight`` clips negative desired allocations
      to zero when an explicit shortable flag says the row is not shortable
    - ``tradability_constrained_target_weight`` keeps the previous effective
      weight when an explicit tradable flag says the row cannot trade
    - ``direction_constrained_target_weight`` keeps the previous effective
      weight when explicit buy/sell flags block the desired trade direction
    - ``executed_weight`` on date ``t`` applies the rebalance schedule,
      short availability, tradability, direction-specific execution flags,
      row-level trade / participation limits, trade clipping, and turnover limit
    - ``effective_weight`` on date ``t`` is the executed allocation used for return ``t``
    """
    signal_delay = _normalize_positive_int(signal_delay, parameter_name="signal_delay")
    fill_timing = _normalize_fill_timing(fill_timing)
    rebalance_frequency = _normalize_rebalance_frequency(rebalance_frequency)
    rebalance_stagger_column = _normalize_optional_column_name(
        rebalance_stagger_column,
        parameter_name="rebalance_stagger_column",
    )
    rebalance_stagger_count = _resolve_rebalance_stagger_inputs(
        rebalance_stagger_column=rebalance_stagger_column,
        rebalance_stagger_count=rebalance_stagger_count,
    )
    shortable_column = _normalize_optional_column_name(
        shortable_column,
        parameter_name="shortable_column",
    )
    tradable_column = _normalize_optional_column_name(
        tradable_column,
        parameter_name="tradable_column",
    )
    can_buy_column = _normalize_optional_column_name(
        can_buy_column,
        parameter_name="can_buy_column",
    )
    can_sell_column = _normalize_optional_column_name(
        can_sell_column,
        parameter_name="can_sell_column",
    )
    max_trade_weight_column = _normalize_optional_column_name(
        max_trade_weight_column,
        parameter_name="max_trade_weight_column",
    )
    max_participation_rate, participation_notional = _resolve_participation_inputs(
        max_participation_rate=max_participation_rate,
        participation_notional=participation_notional,
    )
    min_trade_weight = _normalize_optional_non_negative_float(
        min_trade_weight,
        parameter_name="min_trade_weight",
    )
    max_turnover = _normalize_optional_non_negative_float(
        max_turnover,
        parameter_name="max_turnover",
    )

    panel = _prepare_backtest_input(
        frame,
        weight_column=weight_column,
        source="daily backtest input",
    )
    shortable_values = _prepare_shortable_values(
        panel,
        column_name=shortable_column,
        source="daily backtest input",
    )
    tradable_values = _prepare_tradable_values(
        panel,
        column_name=tradable_column,
        source="daily backtest input",
    )
    can_buy_values = _prepare_can_buy_values(
        panel,
        column_name=can_buy_column,
        source="daily backtest input",
    )
    can_sell_values = _prepare_can_sell_values(
        panel,
        column_name=can_sell_column,
        source="daily backtest input",
    )
    rebalance_stagger_values = _prepare_rebalance_stagger_values(
        panel,
        column_name=rebalance_stagger_column,
        rebalance_stagger_count=rebalance_stagger_count,
        source="daily backtest input",
    )
    explicit_max_trade_weight = _prepare_max_trade_weight_values(
        panel,
        column_name=max_trade_weight_column,
        source="daily backtest input",
    )
    panel["participation_trade_weight_limit"] = (
        _prepare_participation_trade_weight_limit(
            panel,
            max_participation_rate=max_participation_rate,
            participation_notional=participation_notional,
        )
    )
    panel["max_trade_weight"] = pd.concat(
        [explicit_max_trade_weight, panel["participation_trade_weight_limit"]],
        axis=1,
    ).min(axis=1)
    close_by_symbol = panel.groupby("symbol", sort=False)["close"]
    target_weight_by_symbol = panel.groupby("symbol", sort=False)[weight_column]

    panel["asset_return"] = close_by_symbol.pct_change()
    panel["target_weight"] = panel[weight_column]
    panel["signal_delayed_target_weight"] = target_weight_by_symbol.shift(
        signal_delay
    ).fillna(0.0)
    fill_delay_periods = _fill_delay_periods(fill_timing)
    execution_delay = signal_delay + fill_delay_periods
    panel["fill_timing"] = fill_timing
    panel["fill_delay_periods"] = fill_delay_periods
    panel["execution_delay_periods"] = execution_delay
    panel["delayed_target_weight"] = target_weight_by_symbol.shift(
        execution_delay
    ).fillna(
        0.0
    )
    panel["is_shortable"] = shortable_values.to_numpy()
    panel["is_tradable"] = tradable_values.to_numpy()
    panel["is_buyable"] = can_buy_values.to_numpy()
    panel["is_sellable"] = can_sell_values.to_numpy()
    panel["rebalance_stagger_bucket"] = rebalance_stagger_values.to_numpy()
    panel["short_constrained_target_weight"] = panel["delayed_target_weight"].mask(
        panel["delayed_target_weight"].lt(0.0) & ~panel["is_shortable"],
        0.0,
    )
    panel["tradability_constrained_target_weight"] = panel[
        "short_constrained_target_weight"
    ]
    panel["direction_constrained_target_weight"] = panel[
        "tradability_constrained_target_weight"
    ]
    rebalance_dates = _build_rebalance_date_lookup(
        panel["date"],
        rebalance_frequency=rebalance_frequency,
    )
    panel["is_base_rebalance_date"] = panel["date"].map(rebalance_dates).astype(bool)
    active_rebalance_stagger_buckets = _build_active_rebalance_stagger_lookup(
        rebalance_dates,
        rebalance_stagger_count=rebalance_stagger_count,
    )
    panel["active_rebalance_stagger_bucket"] = (
        panel["date"].map(active_rebalance_stagger_buckets).fillna(-1).astype("int64")
    )
    panel["is_rebalance_date"] = panel["is_base_rebalance_date"] & panel[
        "rebalance_stagger_bucket"
    ].eq(panel["active_rebalance_stagger_bucket"])
    panel["previous_effective_weight"] = 0.0
    panel["executed_weight"] = 0.0
    panel["effective_weight"] = 0.0
    panel["desired_weight_change"] = 0.0
    panel["weight_change"] = 0.0
    panel["target_turnover_contribution"] = 0.0
    panel["turnover_contribution"] = 0.0
    panel["target_effective_weight_gap"] = 0.0
    panel["target_effective_weight_gap_abs"] = 0.0
    panel["short_availability_limit_applied"] = False
    panel["tradability_limit_applied"] = False
    panel["buy_limit_applied"] = False
    panel["sell_limit_applied"] = False
    panel["rebalance_stagger_skipped"] = False
    panel["participation_limit_applied"] = False
    panel["trade_limit_applied"] = False
    panel["trade_clip_applied"] = False
    panel["turnover_limit_applied"] = False

    previous_effective_by_symbol: dict[str, float] = {}
    for date in panel["date"].drop_duplicates().sort_values():
        date_mask = panel["date"] == date
        day_symbols = panel.loc[date_mask, "symbol"]
        previous_effective_weight = (
            day_symbols.map(previous_effective_by_symbol).fillna(0.0).astype(float)
        )
        delayed_target_weight = panel.loc[date_mask, "delayed_target_weight"].astype(
            float
        )
        short_constrained_target_weight = panel.loc[
            date_mask,
            "short_constrained_target_weight",
        ].astype(float)
        desired_weight_change = delayed_target_weight - previous_effective_weight
        is_base_rebalance_date = bool(
            panel.loc[date_mask, "is_base_rebalance_date"].iloc[0]
        )
        can_rebalance = panel.loc[date_mask, "is_rebalance_date"].astype(bool)
        short_availability_limit_applied = pd.Series(
            False,
            index=desired_weight_change.index,
            dtype="bool",
        )
        if is_base_rebalance_date:
            short_availability_limit_applied = (
                can_rebalance
                & delayed_target_weight.lt(0.0)
                & ~panel.loc[date_mask, "is_shortable"]
            )
        tradability_limit_applied = pd.Series(
            False,
            index=desired_weight_change.index,
            dtype="bool",
        )
        tradability_constrained_target_weight = short_constrained_target_weight.copy()
        if is_base_rebalance_date:
            tradability_limit_applied = (
                can_rebalance
                & ~panel.loc[date_mask, "is_tradable"]
                & short_constrained_target_weight.ne(previous_effective_weight)
            )
            tradability_constrained_target_weight = (
                short_constrained_target_weight.mask(
                    tradability_limit_applied,
                    previous_effective_weight,
                )
            )
        buy_limit_applied = pd.Series(
            False,
            index=desired_weight_change.index,
            dtype="bool",
        )
        sell_limit_applied = pd.Series(
            False,
            index=desired_weight_change.index,
            dtype="bool",
        )
        direction_constrained_target_weight = tradability_constrained_target_weight.copy()
        if is_base_rebalance_date:
            direction_desired_weight_change = (
                direction_constrained_target_weight - previous_effective_weight
            )
            buy_limit_applied = (
                can_rebalance
                & direction_desired_weight_change.gt(0.0)
                & ~panel.loc[date_mask, "is_buyable"]
            )
            sell_limit_applied = (
                can_rebalance
                & direction_desired_weight_change.lt(0.0)
                & ~panel.loc[date_mask, "is_sellable"]
            )
            direction_limit_applied = buy_limit_applied | sell_limit_applied
            direction_constrained_target_weight = (
                direction_constrained_target_weight.mask(
                    direction_limit_applied,
                    previous_effective_weight,
                )
            )
        rebalance_stagger_skipped = (
            is_base_rebalance_date
            & ~can_rebalance
            & delayed_target_weight.ne(previous_effective_weight)
        )
        direction_constrained_target_weight = direction_constrained_target_weight.mask(
            ~can_rebalance,
            previous_effective_weight,
        )
        execution_desired_weight_change = (
            direction_constrained_target_weight - previous_effective_weight
        )
        participation_limit_applied = pd.Series(
            False,
            index=desired_weight_change.index,
            dtype="bool",
        )
        if is_base_rebalance_date:
            participation_trade_weight_limit = panel.loc[
                date_mask,
                "participation_trade_weight_limit",
            ]
            participation_binds = participation_trade_weight_limit.le(
                panel.loc[date_mask, "max_trade_weight"]
            )
            participation_limit_applied = (
                can_rebalance
                & execution_desired_weight_change.abs().gt(
                    participation_trade_weight_limit
                )
                & participation_binds
            )
        trade_limited_target_weight, trade_limit_applied = _apply_trade_weight_limit(
            previous_effective_weight,
            direction_constrained_target_weight,
            max_trade_weight=panel.loc[date_mask, "max_trade_weight"],
            allow_rebalance=is_base_rebalance_date,
        )
        clipped_target_weight, trade_clip_applied = _apply_min_trade_weight_clip(
            previous_effective_weight,
            trade_limited_target_weight,
            min_trade_weight=min_trade_weight,
            allow_rebalance=is_base_rebalance_date,
        )
        executed_weight, turnover_limit_applied = _apply_turnover_limit(
            previous_effective_weight,
            clipped_target_weight,
            max_turnover=max_turnover,
            allow_rebalance=is_base_rebalance_date,
        )

        weight_change = executed_weight - previous_effective_weight
        target_effective_weight_gap = delayed_target_weight - executed_weight

        panel.loc[date_mask, "previous_effective_weight"] = previous_effective_weight.to_numpy()
        panel.loc[date_mask, "executed_weight"] = executed_weight.to_numpy()
        panel.loc[date_mask, "effective_weight"] = executed_weight.to_numpy()
        panel.loc[date_mask, "desired_weight_change"] = desired_weight_change.to_numpy()
        panel.loc[date_mask, "weight_change"] = weight_change.to_numpy()
        panel.loc[date_mask, "tradability_constrained_target_weight"] = (
            tradability_constrained_target_weight.to_numpy()
        )
        panel.loc[date_mask, "direction_constrained_target_weight"] = (
            direction_constrained_target_weight.to_numpy()
        )
        panel.loc[date_mask, "target_turnover_contribution"] = desired_weight_change.abs().to_numpy()
        panel.loc[date_mask, "turnover_contribution"] = weight_change.abs().to_numpy()
        panel.loc[date_mask, "target_effective_weight_gap"] = target_effective_weight_gap.to_numpy()
        panel.loc[date_mask, "target_effective_weight_gap_abs"] = target_effective_weight_gap.abs().to_numpy()
        panel.loc[date_mask, "short_availability_limit_applied"] = short_availability_limit_applied.to_numpy()
        panel.loc[date_mask, "tradability_limit_applied"] = tradability_limit_applied.to_numpy()
        panel.loc[date_mask, "buy_limit_applied"] = buy_limit_applied.to_numpy()
        panel.loc[date_mask, "sell_limit_applied"] = sell_limit_applied.to_numpy()
        panel.loc[date_mask, "rebalance_stagger_skipped"] = (
            rebalance_stagger_skipped.to_numpy()
        )
        panel.loc[date_mask, "participation_limit_applied"] = participation_limit_applied.to_numpy()
        panel.loc[date_mask, "trade_limit_applied"] = trade_limit_applied.to_numpy()
        panel.loc[date_mask, "trade_clip_applied"] = trade_clip_applied.to_numpy()
        panel.loc[date_mask, "turnover_limit_applied"] = turnover_limit_applied

        previous_effective_by_symbol.update(
            zip(day_symbols.tolist(), executed_weight.astype(float).tolist())
        )

    panel["gross_return_contribution"] = (
        panel["effective_weight"] * panel["asset_return"].fillna(0.0)
    )

    return panel


def run_daily_backtest(
    frame: pd.DataFrame,
    *,
    weight_column: str = "portfolio_weight",
    signal_delay: int = 1,
    fill_timing: str = "close",
    rebalance_frequency: str = "daily",
    rebalance_stagger_column: str | None = None,
    rebalance_stagger_count: int | None = None,
    transaction_cost_bps: float | None = None,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    commission_bps_column: str | None = None,
    slippage_bps_column: str | None = None,
    liquidity_bucket_column: str | None = None,
    slippage_bps_by_liquidity_bucket: Mapping[str, float] | None = None,
    market_impact_bps_per_turnover: float = 0.0,
    borrow_fee_bps_column: str | None = None,
    shortable_column: str | None = None,
    tradable_column: str | None = None,
    can_buy_column: str | None = None,
    can_sell_column: str | None = None,
    max_trade_weight_column: str | None = None,
    max_participation_rate: float | None = None,
    participation_notional: float | None = None,
    min_trade_weight: float | None = None,
    max_turnover: float | None = None,
    initial_nav: float = 1.0,
) -> pd.DataFrame:
    """Run a conservative daily close-to-close backtest from target weights."""
    signal_delay = _normalize_positive_int(signal_delay, parameter_name="signal_delay")
    fill_timing = _normalize_fill_timing(fill_timing)
    rebalance_frequency = _normalize_rebalance_frequency(rebalance_frequency)
    (
        commission_bps,
        slippage_bps,
        commission_bps_column,
        slippage_bps_column,
        liquidity_bucket_column,
        slippage_bps_by_liquidity_bucket,
        market_impact_bps_per_turnover,
    ) = _resolve_cost_bps(
        transaction_cost_bps=transaction_cost_bps,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        commission_bps_column=commission_bps_column,
        slippage_bps_column=slippage_bps_column,
        liquidity_bucket_column=liquidity_bucket_column,
        slippage_bps_by_liquidity_bucket=slippage_bps_by_liquidity_bucket,
        market_impact_bps_per_turnover=market_impact_bps_per_turnover,
    )
    borrow_fee_bps_column = _normalize_optional_column_name(
        borrow_fee_bps_column,
        parameter_name="borrow_fee_bps_column",
    )
    max_turnover = _normalize_optional_non_negative_float(
        max_turnover,
        parameter_name="max_turnover",
    )
    initial_nav = _normalize_positive_float(initial_nav, parameter_name="initial_nav")

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
    commission_bps_values = _prepare_cost_bps_values(
        panel,
        column_name=commission_bps_column,
        default_bps=commission_bps,
        parameter_name="commission_bps_column",
        source="daily backtest input",
    )
    slippage_bps_values = _prepare_cost_bps_values(
        panel,
        column_name=slippage_bps_column,
        default_bps=slippage_bps,
        parameter_name="slippage_bps_column",
        source="daily backtest input",
    )
    if liquidity_bucket_column is not None:
        slippage_bps_values = _prepare_liquidity_bucket_slippage_bps_values(
            panel,
            column_name=liquidity_bucket_column,
            slippage_bps_by_bucket=slippage_bps_by_liquidity_bucket,
            source="daily backtest input",
        )
    panel["commission_cost_contribution"] = (
        panel["turnover_contribution"] * commission_bps_values / 10_000.0
    )
    panel["slippage_cost_contribution"] = (
        panel["turnover_contribution"] * slippage_bps_values / 10_000.0
    )
    panel["market_impact_cost_contribution"] = (
        panel["turnover_contribution"]
        * (market_impact_bps_per_turnover * panel["turnover_contribution"])
        / 10_000.0
    )
    panel["transaction_cost_contribution"] = (
        panel["commission_cost_contribution"]
        + panel["slippage_cost_contribution"]
        + panel["market_impact_cost_contribution"]
    )
    panel = _attach_borrow_cost_fields(
        panel,
        borrow_fee_bps_column=borrow_fee_bps_column,
        source="daily backtest input",
    )
    panel["transaction_cost_contribution"] = (
        panel["transaction_cost_contribution"] + panel["borrow_cost_contribution"]
    )
    daily = (
        panel.groupby("date", sort=True)
        .agg(
            gross_return=("gross_return_contribution", "sum"),
            target_turnover=("target_turnover_contribution", "sum"),
            turnover=("turnover_contribution", "sum"),
            commission_cost=("commission_cost_contribution", "sum"),
            slippage_cost=("slippage_cost_contribution", "sum"),
            market_impact_cost=("market_impact_cost_contribution", "sum"),
            borrow_cost=("borrow_cost_contribution", "sum"),
            gross_exposure=("effective_weight", lambda values: values.abs().sum()),
            short_exposure=("short_exposure", "sum"),
            net_exposure=("effective_weight", "sum"),
            gross_target_exposure=(
                "delayed_target_weight",
                lambda values: values.abs().sum(),
            ),
            target_net_exposure=("delayed_target_weight", "sum"),
            holdings_count=(
                "effective_weight",
                lambda values: int(values.ne(0.0).sum()),
            ),
            target_holdings_count=(
                "delayed_target_weight",
                lambda values: int(values.ne(0.0).sum()),
            ),
            target_effective_weight_gap=("target_effective_weight_gap_abs", "sum"),
            is_base_rebalance_date=("is_base_rebalance_date", "max"),
            is_rebalance_date=("is_rebalance_date", "max"),
            short_availability_limit_applied=(
                "short_availability_limit_applied",
                "max",
            ),
            tradability_limit_applied=("tradability_limit_applied", "max"),
            buy_limit_applied=("buy_limit_applied", "max"),
            sell_limit_applied=("sell_limit_applied", "max"),
            rebalance_stagger_skipped=("rebalance_stagger_skipped", "max"),
            participation_limit_applied=("participation_limit_applied", "max"),
            trade_limit_applied=("trade_limit_applied", "max"),
            trade_clip_applied=("trade_clip_applied", "max"),
            turnover_limit_applied=("turnover_limit_applied", "max"),
        )
        .reset_index()
    )

    daily["transaction_cost"] = (
        daily["commission_cost"]
        + daily["slippage_cost"]
        + daily["market_impact_cost"]
        + daily["borrow_cost"]
    )
    daily["net_return"] = daily["gross_return"] - daily["transaction_cost"]
    daily["gross_nav"] = initial_nav * (1.0 + daily["gross_return"]).cumprod()
    daily["net_nav"] = initial_nav * (1.0 + daily["net_return"]).cumprod()

    return daily


def _prepare_backtest_input(
    frame: pd.DataFrame,
    *,
    weight_column: str,
    source: str,
) -> pd.DataFrame:
    """Validate the OHLCV panel and parse the selected weight column."""
    if weight_column not in frame.columns:
        raise BacktestError(f"{source} is missing the weight column '{weight_column}'.")

    panel = validate_ohlcv(frame, source=source).copy()
    parsed_weights = pd.to_numeric(panel[weight_column], errors="coerce")
    invalid_weights = panel[weight_column].notna() & parsed_weights.isna()
    if invalid_weights.any():
        raise BacktestError(
            f"{source} contains invalid numeric values in '{weight_column}'."
        )

    panel[weight_column] = parsed_weights.fillna(0.0)
    return panel


def _normalize_positive_int(value: int, *, parameter_name: str) -> int:
    """Validate positive integer parameters."""
    return _common_positive_int(
        value,
        parameter_name=parameter_name,
        error_factory=BacktestError,
    )


def _normalize_non_negative_float(value: float, *, parameter_name: str) -> float:
    """Validate non-negative float parameters."""
    return _common_non_negative_float(
        value,
        parameter_name=parameter_name,
        error_factory=BacktestError,
    )


def _normalize_positive_float(value: float, *, parameter_name: str) -> float:
    """Validate strictly positive float parameters."""
    return _common_positive_float(
        value,
        parameter_name=parameter_name,
        error_factory=BacktestError,
    )


def _normalize_optional_non_negative_float(
    value: float | None, *, parameter_name: str
) -> float | None:
    """Validate optional non-negative float parameters."""
    return _common_optional_non_negative_float(
        value,
        parameter_name=parameter_name,
        error_factory=BacktestError,
    )


def _normalize_optional_participation_rate(
    value: float | None,
    *,
    parameter_name: str,
) -> float | None:
    """Validate optional participation rates constrained to [0.0, 1.0]."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise BacktestError(f"{parameter_name} must be a float in [0.0, 1.0].")
    rate = _normalize_non_negative_float(value, parameter_name=parameter_name)
    if rate > 1.0:
        raise BacktestError(f"{parameter_name} must be a float in [0.0, 1.0].")
    return rate


def _normalize_optional_column_name(
    value: str | None,
    *,
    parameter_name: str,
) -> str | None:
    """Validate optional input column names."""
    if value is None:
        return None
    return _common_non_empty_string(
        value,
        parameter_name=parameter_name,
        error_factory=BacktestError,
    )


def _resolve_participation_inputs(
    *,
    max_participation_rate: float | None,
    participation_notional: float | None,
) -> tuple[float | None, float | None]:
    """Validate participation-cap inputs that must be configured together."""
    max_participation_rate = _normalize_optional_participation_rate(
        max_participation_rate,
        parameter_name="max_participation_rate",
    )
    participation_notional = (
        _normalize_positive_float(
            participation_notional,
            parameter_name="participation_notional",
        )
        if participation_notional is not None
        else None
    )
    if (max_participation_rate is None) != (participation_notional is None):
        raise BacktestError(
            "max_participation_rate and participation_notional must be configured together."
        )
    return max_participation_rate, participation_notional


def _resolve_rebalance_stagger_inputs(
    *,
    rebalance_stagger_column: str | None,
    rebalance_stagger_count: int | None,
) -> int:
    """Validate optional staggered-rebalance settings configured as a pair."""
    if (rebalance_stagger_column is None) != (rebalance_stagger_count is None):
        raise BacktestError(
            "rebalance_stagger_column and rebalance_stagger_count "
            "must be configured together."
        )
    if rebalance_stagger_count is None:
        return 1
    count = _normalize_positive_int(
        rebalance_stagger_count,
        parameter_name="rebalance_stagger_count",
    )
    if count < 2:
        raise BacktestError("rebalance_stagger_count must be at least 2.")
    return count


def _normalize_rebalance_frequency(value: str) -> str:
    """Validate supported rebalance schedule choices."""
    return _common_choice_string(
        value,
        parameter_name="rebalance_frequency",
        choices={"daily", "weekly", "monthly"},
        error_factory=BacktestError,
    )


def _normalize_fill_timing(value: str) -> str:
    """Validate supported daily close-to-close fill timing choices."""
    return _common_choice_string(
        value,
        parameter_name="fill_timing",
        choices={"close", "next_close"},
        error_factory=BacktestError,
    )


def _fill_delay_periods(fill_timing: str) -> int:
    """Map close-to-close fill timing labels to extra target delay periods."""
    if fill_timing == "close":
        return 0
    if fill_timing == "next_close":
        return 1
    raise BacktestError(f"Unsupported fill_timing: {fill_timing}.")


def _resolve_cost_bps(
    *,
    transaction_cost_bps: float | None,
    commission_bps: float,
    slippage_bps: float,
    commission_bps_column: str | None,
    slippage_bps_column: str | None,
    liquidity_bucket_column: str | None,
    slippage_bps_by_liquidity_bucket: Mapping[str, float] | None,
    market_impact_bps_per_turnover: float,
) -> tuple[
    float,
    float,
    str | None,
    str | None,
    str | None,
    dict[str, float] | None,
    float,
]:
    """Resolve legacy and split transaction cost inputs."""
    commission_bps_column = _normalize_optional_column_name(
        commission_bps_column,
        parameter_name="commission_bps_column",
    )
    slippage_bps_column = _normalize_optional_column_name(
        slippage_bps_column,
        parameter_name="slippage_bps_column",
    )
    liquidity_bucket_column = _normalize_optional_column_name(
        liquidity_bucket_column,
        parameter_name="liquidity_bucket_column",
    )
    slippage_bps_by_liquidity_bucket = (
        _normalize_slippage_bps_by_liquidity_bucket(
            slippage_bps_by_liquidity_bucket,
        )
    )
    commission_bps = _normalize_non_negative_float(
        commission_bps,
        parameter_name="commission_bps",
    )
    slippage_bps = _normalize_non_negative_float(
        slippage_bps,
        parameter_name="slippage_bps",
    )
    market_impact_bps_per_turnover = _normalize_non_negative_float(
        market_impact_bps_per_turnover,
        parameter_name="market_impact_bps_per_turnover",
    )

    if transaction_cost_bps is not None:
        if (
            commission_bps != 0.0
            or slippage_bps != 0.0
            or commission_bps_column is not None
            or slippage_bps_column is not None
            or liquidity_bucket_column is not None
            or slippage_bps_by_liquidity_bucket is not None
            or market_impact_bps_per_turnover != 0.0
        ):
            raise BacktestError(
                "transaction_cost_bps cannot be combined with commission_bps, "
                "slippage_bps, commission_bps_column, slippage_bps_column, "
                "liquidity_bucket_column, slippage_bps_by_liquidity_bucket, "
                "or market_impact_bps_per_turnover."
            )
        return (
            _normalize_non_negative_float(
                transaction_cost_bps,
                parameter_name="transaction_cost_bps",
            ),
            0.0,
            None,
            None,
            None,
            None,
            0.0,
        )

    if commission_bps_column is not None and commission_bps != 0.0:
        raise BacktestError(
            "commission_bps cannot be combined with commission_bps_column."
        )
    if slippage_bps_column is not None and slippage_bps != 0.0:
        raise BacktestError(
            "slippage_bps cannot be combined with slippage_bps_column."
        )
    if (liquidity_bucket_column is None) != (
        slippage_bps_by_liquidity_bucket is None
    ):
        raise BacktestError(
            "liquidity_bucket_column and slippage_bps_by_liquidity_bucket "
            "must be configured together."
        )
    if liquidity_bucket_column is not None and (
        slippage_bps != 0.0 or slippage_bps_column is not None
    ):
        raise BacktestError(
            "liquidity-bucket slippage cannot be combined with slippage_bps "
            "or slippage_bps_column."
        )

    return (
        commission_bps,
        slippage_bps,
        commission_bps_column,
        slippage_bps_column,
        liquidity_bucket_column,
        slippage_bps_by_liquidity_bucket,
        market_impact_bps_per_turnover,
    )


def _normalize_slippage_bps_by_liquidity_bucket(
    values: Mapping[str, float] | None,
) -> dict[str, float] | None:
    """Validate explicit liquidity-bucket slippage bps settings."""
    if values is None:
        return None
    if not isinstance(values, Mapping) or not values:
        raise BacktestError(
            "slippage_bps_by_liquidity_bucket must be a non-empty mapping."
        )

    normalized: dict[str, float] = {}
    for raw_bucket, raw_bps in values.items():
        bucket = _common_non_empty_string(
            raw_bucket,
            parameter_name="slippage_bps_by_liquidity_bucket key",
            error_factory=BacktestError,
        )
        if bucket in normalized:
            raise BacktestError(
                "slippage_bps_by_liquidity_bucket contains duplicate "
                f"bucket '{bucket}'."
            )
        normalized[bucket] = _normalize_non_negative_float(
            raw_bps,
            parameter_name=f"slippage_bps_by_liquidity_bucket.{bucket}",
        )
    return normalized


def _prepare_cost_bps_values(
    panel: pd.DataFrame,
    *,
    column_name: str | None,
    default_bps: float,
    parameter_name: str,
    source: str,
) -> pd.Series:
    """Return scalar or row-level bps values for a transaction-cost component."""
    if column_name is None:
        return pd.Series(default_bps, index=panel.index, dtype="float64")
    if column_name not in panel.columns:
        raise BacktestError(f"{source} is missing {parameter_name} '{column_name}'.")

    parsed = pd.to_numeric(panel[column_name], errors="coerce")
    invalid_values = (
        parsed.isna()
        | (parsed < 0.0)
        | (parsed == float("inf"))
        | (parsed == float("-inf"))
    )
    if invalid_values.any():
        raise BacktestError(
            f"{source} contains missing, invalid, or negative values in "
            f"{parameter_name} '{column_name}'."
        )
    return parsed.astype(float)


def _prepare_liquidity_bucket_slippage_bps_values(
    panel: pd.DataFrame,
    *,
    column_name: str,
    slippage_bps_by_bucket: Mapping[str, float] | None,
    source: str,
) -> pd.Series:
    """Return slippage bps from explicit row-level liquidity bucket labels."""
    if slippage_bps_by_bucket is None:
        raise BacktestError(
            "slippage_bps_by_liquidity_bucket is required when "
            "liquidity_bucket_column is set."
        )
    if column_name not in panel.columns:
        raise BacktestError(
            f"{source} is missing liquidity_bucket_column '{column_name}'."
        )

    parsed: list[float] = []
    for value in panel[column_name].tolist():
        if pd.isna(value):
            raise BacktestError(
                f"{source} contains missing values in "
                f"liquidity_bucket_column '{column_name}'."
            )
        bucket = str(value).strip()
        if not bucket:
            raise BacktestError(
                f"{source} contains empty values in "
                f"liquidity_bucket_column '{column_name}'."
            )
        if bucket not in slippage_bps_by_bucket:
            raise BacktestError(
                f"{source} contains unmapped liquidity bucket '{bucket}' in "
                f"liquidity_bucket_column '{column_name}'."
            )
        parsed.append(float(slippage_bps_by_bucket[bucket]))
    return pd.Series(parsed, index=panel.index, dtype="float64")


def _prepare_shortable_values(
    panel: pd.DataFrame,
    *,
    column_name: str | None,
    source: str,
) -> pd.Series:
    """Return strict row-level flags for whether short targets are allowed."""
    return _prepare_bool_values(
        panel,
        column_name=column_name,
        parameter_name="shortable_column",
        source=source,
        default=True,
    )


def _prepare_tradable_values(
    panel: pd.DataFrame,
    *,
    column_name: str | None,
    source: str,
) -> pd.Series:
    """Return strict row-level flags for whether trades can execute."""
    return _prepare_bool_values(
        panel,
        column_name=column_name,
        parameter_name="tradable_column",
        source=source,
        default=True,
    )


def _prepare_can_buy_values(
    panel: pd.DataFrame,
    *,
    column_name: str | None,
    source: str,
) -> pd.Series:
    """Return strict row-level flags for whether positive trades are allowed."""
    return _prepare_bool_values(
        panel,
        column_name=column_name,
        parameter_name="can_buy_column",
        source=source,
        default=True,
    )


def _prepare_can_sell_values(
    panel: pd.DataFrame,
    *,
    column_name: str | None,
    source: str,
) -> pd.Series:
    """Return strict row-level flags for whether negative trades are allowed."""
    return _prepare_bool_values(
        panel,
        column_name=column_name,
        parameter_name="can_sell_column",
        source=source,
        default=True,
    )


def _prepare_rebalance_stagger_values(
    panel: pd.DataFrame,
    *,
    column_name: str | None,
    rebalance_stagger_count: int,
    source: str,
) -> pd.Series:
    """Return explicit zero-based stagger buckets for row-level rebalancing."""
    if column_name is None:
        return pd.Series(0, index=panel.index, dtype="int64")
    if column_name not in panel.columns:
        raise BacktestError(
            f"{source} is missing rebalance_stagger_column '{column_name}'."
        )

    parsed = pd.to_numeric(panel[column_name], errors="coerce")
    finite = np.isfinite(parsed)
    integer_like = parsed.mod(1).eq(0)
    invalid_values = (
        parsed.isna()
        | ~finite
        | ~integer_like
        | parsed.lt(0)
        | parsed.ge(rebalance_stagger_count)
    )
    if invalid_values.any():
        upper = rebalance_stagger_count - 1
        raise BacktestError(
            f"{source} contains missing or invalid values in "
            f"rebalance_stagger_column '{column_name}'; expected integers "
            f"from 0 to {upper}."
        )
    return parsed.astype("int64")


def _prepare_bool_values(
    panel: pd.DataFrame,
    *,
    column_name: str | None,
    parameter_name: str,
    source: str,
    default: bool,
) -> pd.Series:
    """Return strict row-level boolean values for optional execution flags."""
    if column_name is None:
        return pd.Series(default, index=panel.index, dtype="bool")
    if column_name not in panel.columns:
        raise BacktestError(f"{source} is missing {parameter_name} '{column_name}'.")

    parsed: list[bool] = []
    for value in panel[column_name].tolist():
        if pd.isna(value):
            raise BacktestError(
                f"{source} contains missing boolean values in "
                f"{parameter_name} '{column_name}'."
            )
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
            continue
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            if value in {0, 1}:
                parsed.append(bool(value))
                continue
        if isinstance(value, (float, np.floating)):
            if np.isfinite(value) and value in {0.0, 1.0}:
                parsed.append(bool(int(value)))
                continue
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1"}:
                parsed.append(True)
                continue
            if normalized in {"false", "0"}:
                parsed.append(False)
                continue
        raise BacktestError(
            f"{source} contains invalid boolean values in "
            f"{parameter_name} '{column_name}'; expected bool/0/1/true/false."
        )
    return pd.Series(parsed, index=panel.index, dtype="bool")


def _attach_borrow_cost_fields(
    panel: pd.DataFrame,
    *,
    borrow_fee_bps_column: str | None,
    source: str,
) -> pd.DataFrame:
    """Attach explicit annualized short borrow fee diagnostics to a panel."""
    borrow_fee_bps_column = _normalize_optional_column_name(
        borrow_fee_bps_column,
        parameter_name="borrow_fee_bps_column",
    )
    borrow_fee_bps_values = _prepare_cost_bps_values(
        panel,
        column_name=borrow_fee_bps_column,
        default_bps=0.0,
        parameter_name="borrow_fee_bps_column",
        source=source,
    )
    enriched = panel.copy()
    enriched["borrow_fee_bps"] = borrow_fee_bps_values.to_numpy()
    enriched["short_exposure"] = enriched["effective_weight"].clip(upper=0.0).abs()
    enriched["borrow_cost_contribution"] = (
        enriched["short_exposure"]
        * borrow_fee_bps_values
        / 10_000.0
        / _BORROW_COST_TRADING_DAYS_PER_YEAR
    )
    return enriched


def _prepare_participation_trade_weight_limit(
    panel: pd.DataFrame,
    *,
    max_participation_rate: float | None,
    participation_notional: float | None,
) -> pd.Series:
    """Convert daily realized volume participation into a weight-level trade cap."""
    if max_participation_rate is None:
        return pd.Series(float("inf"), index=panel.index, dtype="float64")
    if participation_notional is None:
        raise BacktestError(
            "participation_notional is required when max_participation_rate is set."
        )
    dollar_volume = panel["close"] * panel["volume"]
    return (dollar_volume * max_participation_rate / participation_notional).astype(
        float
    )


def _prepare_max_trade_weight_values(
    panel: pd.DataFrame,
    *,
    column_name: str | None,
    source: str,
) -> pd.Series:
    """Return explicit row-level absolute trade-weight limits."""
    if column_name is None:
        return pd.Series(float("inf"), index=panel.index, dtype="float64")
    if column_name not in panel.columns:
        raise BacktestError(
            f"{source} is missing max_trade_weight_column '{column_name}'."
        )

    parsed = pd.to_numeric(panel[column_name], errors="coerce")
    invalid_values = (
        parsed.isna()
        | (parsed < 0.0)
        | (parsed == float("inf"))
        | (parsed == float("-inf"))
    )
    if invalid_values.any():
        raise BacktestError(
            f"{source} contains missing, invalid, or negative values in "
            f"max_trade_weight_column '{column_name}'."
        )
    return parsed.astype(float)


def _build_rebalance_date_lookup(
    dates: pd.Series,
    *,
    rebalance_frequency: str,
) -> dict[pd.Timestamp, bool]:
    """Map each unique date to whether it is a rebalance date."""
    unique_dates = pd.Series(pd.Index(dates).unique()).sort_values().reset_index(drop=True)
    if rebalance_frequency == "daily":
        return {pd.Timestamp(date): True for date in unique_dates}

    if rebalance_frequency == "weekly":
        grouped = unique_dates.groupby(unique_dates.dt.to_period("W-SUN"), sort=False)
    else:
        grouped = unique_dates.groupby(unique_dates.dt.to_period("M"), sort=False)

    rebalance_dates = {pd.Timestamp(group.iloc[0]) for _, group in grouped}
    return {pd.Timestamp(date): pd.Timestamp(date) in rebalance_dates for date in unique_dates}


def _build_active_rebalance_stagger_lookup(
    rebalance_dates: Mapping[pd.Timestamp, bool],
    *,
    rebalance_stagger_count: int,
) -> dict[pd.Timestamp, int]:
    """Map each date to the active stagger bucket, or -1 on non-rebalance dates."""
    active_buckets: dict[pd.Timestamp, int] = {}
    rebalance_ordinal = 0
    for date in sorted(rebalance_dates):
        if rebalance_dates[date]:
            active_buckets[pd.Timestamp(date)] = (
                rebalance_ordinal % rebalance_stagger_count
            )
            rebalance_ordinal += 1
        else:
            active_buckets[pd.Timestamp(date)] = -1
    return active_buckets


def _apply_trade_weight_limit(
    previous_weight: pd.Series,
    delayed_target_weight: pd.Series,
    *,
    max_trade_weight: pd.Series,
    allow_rebalance: bool,
) -> tuple[pd.Series, pd.Series]:
    """Move toward target weights subject to per-row absolute trade limits."""
    trade_limit_applied = pd.Series(False, index=previous_weight.index, dtype="bool")
    if not allow_rebalance:
        return previous_weight.copy(), trade_limit_applied

    desired_trade = delayed_target_weight - previous_weight
    trade_limit_applied = desired_trade.abs().gt(max_trade_weight)
    limited_trade = desired_trade.clip(
        lower=-max_trade_weight,
        upper=max_trade_weight,
    )
    return previous_weight + limited_trade, trade_limit_applied


def _apply_min_trade_weight_clip(
    previous_weight: pd.Series,
    target_weight: pd.Series,
    *,
    min_trade_weight: float | None,
    allow_rebalance: bool,
) -> tuple[pd.Series, pd.Series]:
    """Drop nonzero trade-weight changes below the minimum execution threshold."""
    trade_clip_applied = pd.Series(False, index=previous_weight.index, dtype="bool")
    if not allow_rebalance or min_trade_weight is None or min_trade_weight == 0.0:
        return target_weight.copy(), trade_clip_applied

    desired_trade = target_weight - previous_weight
    trade_clip_applied = desired_trade.abs().lt(min_trade_weight) & desired_trade.ne(
        0.0
    )
    clipped_trade = desired_trade.mask(trade_clip_applied, 0.0)
    return previous_weight + clipped_trade, trade_clip_applied


def _apply_turnover_limit(
    previous_weight: pd.Series,
    delayed_target_weight: pd.Series,
    *,
    max_turnover: float | None,
    allow_rebalance: bool,
) -> tuple[pd.Series, bool]:
    """Move toward target weights subject to rebalance schedule and turnover limit."""
    if not allow_rebalance:
        return previous_weight.copy(), False

    desired_trade = delayed_target_weight - previous_weight
    desired_turnover = float(desired_trade.abs().sum())
    if (
        max_turnover is None
        or desired_turnover <= max_turnover
        or desired_turnover == 0.0
    ):
        return delayed_target_weight.copy(), False

    scaling = max_turnover / desired_turnover
    executed_weight = previous_weight + desired_trade * scaling
    return executed_weight, True

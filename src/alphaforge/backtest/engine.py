"""Conservative daily backtesting utilities."""

from __future__ import annotations

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


class BacktestError(AlphaForgeError):
    """Raised when daily backtest inputs or settings are invalid."""


def prepare_daily_backtest_panel(
    frame: pd.DataFrame,
    *,
    weight_column: str = "portfolio_weight",
    signal_delay: int = 1,
    fill_timing: str = "close",
    rebalance_frequency: str = "daily",
    max_turnover: float | None = None,
) -> pd.DataFrame:
    """Prepare per-symbol daily backtest fields from target portfolio weights.

    Timing convention:
    - ``asset_return`` on date ``t`` is the close-to-close return from ``t-1`` to ``t``
    - ``target_weight`` on date ``t`` is observed at the close of ``t``
    - ``signal_delayed_target_weight`` on date ``t`` is the signal-delayed target
    - ``delayed_target_weight`` on date ``t`` is the fill-timing-adjusted desired allocation
    - ``executed_weight`` on date ``t`` applies the rebalance schedule and turnover limit
    - ``effective_weight`` on date ``t`` is the executed allocation used for return ``t``
    """
    signal_delay = _normalize_positive_int(signal_delay, parameter_name="signal_delay")
    fill_timing = _normalize_fill_timing(fill_timing)
    rebalance_frequency = _normalize_rebalance_frequency(rebalance_frequency)
    max_turnover = _normalize_optional_non_negative_float(
        max_turnover,
        parameter_name="max_turnover",
    )

    panel = _prepare_backtest_input(
        frame,
        weight_column=weight_column,
        source="daily backtest input",
    )
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
    rebalance_dates = _build_rebalance_date_lookup(
        panel["date"],
        rebalance_frequency=rebalance_frequency,
    )
    panel["is_rebalance_date"] = panel["date"].map(rebalance_dates).astype(bool)
    panel["previous_effective_weight"] = 0.0
    panel["executed_weight"] = 0.0
    panel["effective_weight"] = 0.0
    panel["desired_weight_change"] = 0.0
    panel["weight_change"] = 0.0
    panel["target_turnover_contribution"] = 0.0
    panel["turnover_contribution"] = 0.0
    panel["target_effective_weight_gap"] = 0.0
    panel["target_effective_weight_gap_abs"] = 0.0
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
        desired_weight_change = delayed_target_weight - previous_effective_weight
        is_rebalance_date = bool(panel.loc[date_mask, "is_rebalance_date"].iloc[0])
        executed_weight, turnover_limit_applied = _apply_turnover_limit(
            previous_effective_weight,
            delayed_target_weight,
            max_turnover=max_turnover,
            allow_rebalance=is_rebalance_date,
        )

        weight_change = executed_weight - previous_effective_weight
        target_effective_weight_gap = delayed_target_weight - executed_weight

        panel.loc[date_mask, "previous_effective_weight"] = previous_effective_weight.to_numpy()
        panel.loc[date_mask, "executed_weight"] = executed_weight.to_numpy()
        panel.loc[date_mask, "effective_weight"] = executed_weight.to_numpy()
        panel.loc[date_mask, "desired_weight_change"] = desired_weight_change.to_numpy()
        panel.loc[date_mask, "weight_change"] = weight_change.to_numpy()
        panel.loc[date_mask, "target_turnover_contribution"] = desired_weight_change.abs().to_numpy()
        panel.loc[date_mask, "turnover_contribution"] = weight_change.abs().to_numpy()
        panel.loc[date_mask, "target_effective_weight_gap"] = target_effective_weight_gap.to_numpy()
        panel.loc[date_mask, "target_effective_weight_gap_abs"] = target_effective_weight_gap.abs().to_numpy()
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
    transaction_cost_bps: float | None = None,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    commission_bps_column: str | None = None,
    slippage_bps_column: str | None = None,
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
    ) = _resolve_cost_bps(
        transaction_cost_bps=transaction_cost_bps,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        commission_bps_column=commission_bps_column,
        slippage_bps_column=slippage_bps_column,
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
    panel["commission_cost_contribution"] = (
        panel["turnover_contribution"] * commission_bps_values / 10_000.0
    )
    panel["slippage_cost_contribution"] = (
        panel["turnover_contribution"] * slippage_bps_values / 10_000.0
    )
    panel["transaction_cost_contribution"] = (
        panel["commission_cost_contribution"] + panel["slippage_cost_contribution"]
    )
    daily = (
        panel.groupby("date", sort=True)
        .agg(
            gross_return=("gross_return_contribution", "sum"),
            target_turnover=("target_turnover_contribution", "sum"),
            turnover=("turnover_contribution", "sum"),
            commission_cost=("commission_cost_contribution", "sum"),
            slippage_cost=("slippage_cost_contribution", "sum"),
            gross_exposure=("effective_weight", lambda values: values.abs().sum()),
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
            is_rebalance_date=("is_rebalance_date", "max"),
            turnover_limit_applied=("turnover_limit_applied", "max"),
        )
        .reset_index()
    )

    daily["transaction_cost"] = daily["commission_cost"] + daily["slippage_cost"]
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
) -> tuple[float, float, str | None, str | None]:
    """Resolve legacy and split transaction cost inputs."""
    commission_bps_column = _normalize_optional_column_name(
        commission_bps_column,
        parameter_name="commission_bps_column",
    )
    slippage_bps_column = _normalize_optional_column_name(
        slippage_bps_column,
        parameter_name="slippage_bps_column",
    )
    commission_bps = _normalize_non_negative_float(
        commission_bps,
        parameter_name="commission_bps",
    )
    slippage_bps = _normalize_non_negative_float(
        slippage_bps,
        parameter_name="slippage_bps",
    )

    if transaction_cost_bps is not None:
        if (
            commission_bps != 0.0
            or slippage_bps != 0.0
            or commission_bps_column is not None
            or slippage_bps_column is not None
        ):
            raise BacktestError(
                "transaction_cost_bps cannot be combined with commission_bps, "
                "slippage_bps, commission_bps_column, or slippage_bps_column."
            )
        return (
            _normalize_non_negative_float(
                transaction_cost_bps,
                parameter_name="transaction_cost_bps",
            ),
            0.0,
            None,
            None,
        )

    if commission_bps_column is not None and commission_bps != 0.0:
        raise BacktestError(
            "commission_bps cannot be combined with commission_bps_column."
        )
    if slippage_bps_column is not None and slippage_bps != 0.0:
        raise BacktestError(
            "slippage_bps cannot be combined with slippage_bps_column."
        )

    return (commission_bps, slippage_bps, commission_bps_column, slippage_bps_column)


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

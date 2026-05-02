"""Config-driven pipeline assembly helpers for CLI workflows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from alphaforge.backtest import run_daily_backtest
from alphaforge.cli.data_loading import (
    dataset_membership_indexes,
    dataset_requires_benchmark_returns,
    dataset_requires_fundamentals,
    dataset_requires_memberships,
    dataset_requires_shares_outstanding,
    dataset_requires_trading_status,
    load_benchmark_returns_from_config,
    load_borrow_availability_from_config,
    load_classifications_from_config,
    load_fundamentals_from_config,
    load_market_data_from_config,
    load_memberships_from_config,
    load_shares_outstanding_from_config,
    load_symbol_metadata_from_config,
    load_trading_calendar_from_config,
    load_trading_status_from_config,
)
from alphaforge.cli.errors import WorkflowError
from alphaforge.common import AlphaForgeConfig
from alphaforge.features import build_research_dataset
from alphaforge.portfolio import build_long_only_weights, build_long_short_weights
from alphaforge.signals import (
    apply_cross_sectional_signal_transform,
    build_factor_signal,
)

__all__ = [
    "add_signal_from_config",
    "build_dataset_from_config",
    "build_dataset_from_market_data",
    "build_weights_from_config",
    "align_benchmark_to_backtest",
    "maybe_attach_benchmark_to_backtest",
    "run_backtest_from_config",
    "run_backtest_with_config",
    "require_backtest_config",
    "require_portfolio_config",
    "require_signal_config",
    "signal_parameters_from_config",
]


def build_dataset_from_config(
    config: AlphaForgeConfig,
    *,
    market_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the research dataset from config-driven dataset settings."""
    if market_data is None:
        market_data = load_market_data_from_config(config)
    trading_calendar = (
        load_trading_calendar_from_config(config)
        if config.calendar is not None
        else None
    )
    symbol_metadata = (
        load_symbol_metadata_from_config(config)
        if config.symbol_metadata is not None
        else None
    )
    fundamentals = (
        load_fundamentals_from_config(config)
        if dataset_requires_fundamentals(config)
        else None
    )
    classifications = (
        load_classifications_from_config(config)
        if config.dataset.classification_fields
        else None
    )
    memberships = (
        load_memberships_from_config(config)
        if dataset_requires_memberships(config)
        else None
    )
    borrow_availability = (
        load_borrow_availability_from_config(config)
        if config.dataset.borrow_fields
        else None
    )
    trading_status = (
        load_trading_status_from_config(config)
        if dataset_requires_trading_status(config)
        else None
    )
    shares_outstanding = (
        load_shares_outstanding_from_config(config)
        if dataset_requires_shares_outstanding(config)
        else None
    )
    benchmark_returns = (
        load_benchmark_returns_from_config(config)
        if dataset_requires_benchmark_returns(config)
        else None
    )
    return build_dataset_from_market_data(
        market_data,
        config=config,
        trading_calendar=trading_calendar,
        symbol_metadata=symbol_metadata,
        fundamentals=fundamentals,
        classifications=classifications,
        memberships=memberships,
        borrow_availability=borrow_availability,
        trading_status=trading_status,
        shares_outstanding=shares_outstanding,
        benchmark_returns=benchmark_returns,
    )


def build_dataset_from_market_data(
    market_data: pd.DataFrame,
    *,
    config: AlphaForgeConfig,
    trading_calendar: pd.DataFrame | None = None,
    symbol_metadata: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    classifications: pd.DataFrame | None = None,
    memberships: pd.DataFrame | None = None,
    borrow_availability: pd.DataFrame | None = None,
    trading_status: pd.DataFrame | None = None,
    shares_outstanding: pd.DataFrame | None = None,
    benchmark_returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the research dataset from already-loaded market data."""
    universe_config = config.universe
    return build_research_dataset(
        market_data,
        trading_calendar=trading_calendar,
        symbol_metadata=symbol_metadata,
        fundamentals=fundamentals,
        classifications=classifications,
        memberships=memberships,
        borrow_availability=borrow_availability,
        trading_status=trading_status,
        shares_outstanding=shares_outstanding,
        benchmark_returns=benchmark_returns,
        include_market_cap=config.dataset.include_market_cap,
        market_cap_bucket_count=config.dataset.market_cap_bucket_count,
        average_true_range_window=config.dataset.average_true_range_window,
        normalized_average_true_range_window=(
            config.dataset.normalized_average_true_range_window
        ),
        amihud_illiquidity_window=config.dataset.amihud_illiquidity_window,
        dollar_volume_shock_window=config.dataset.dollar_volume_shock_window,
        dollar_volume_zscore_window=config.dataset.dollar_volume_zscore_window,
        volume_shock_window=config.dataset.volume_shock_window,
        relative_volume_window=config.dataset.relative_volume_window,
        relative_dollar_volume_window=config.dataset.relative_dollar_volume_window,
        fundamental_metrics=(
            config.dataset.fundamental_metrics if fundamentals is not None else None
        ),
        valuation_metrics=(
            config.dataset.valuation_metrics
            if fundamentals is not None and config.dataset.valuation_metrics
            else None
        ),
        quality_ratio_metrics=(
            config.dataset.quality_ratio_metrics
            if fundamentals is not None and config.dataset.quality_ratio_metrics
            else None
        ),
        growth_metrics=(
            config.dataset.growth_metrics
            if fundamentals is not None and config.dataset.growth_metrics
            else None
        ),
        stability_ratio_metrics=(
            config.dataset.stability_ratio_metrics
            if fundamentals is not None and config.dataset.stability_ratio_metrics
            else None
        ),
        classification_fields=(
            config.dataset.classification_fields if classifications is not None else None
        ),
        membership_indexes=(
            dataset_membership_indexes(config) if memberships is not None else None
        ),
        universe_required_membership_indexes=(
            universe_config.required_membership_indexes or None
            if universe_config is not None
            else None
        ),
        borrow_fields=(
            config.dataset.borrow_fields
            if borrow_availability is not None
            else None
        ),
        garman_klass_volatility_window=config.dataset.garman_klass_volatility_window,
        parkinson_volatility_window=config.dataset.parkinson_volatility_window,
        rogers_satchell_volatility_window=(
            config.dataset.rogers_satchell_volatility_window
        ),
        yang_zhang_volatility_window=config.dataset.yang_zhang_volatility_window,
        realized_volatility_window=config.dataset.realized_volatility_window,
        higher_moments_window=config.dataset.higher_moments_window,
        benchmark_residual_return_window=(
            config.dataset.benchmark_residual_return_window
        ),
        benchmark_rolling_window=config.dataset.benchmark_rolling_window,
        forward_horizons=config.dataset.forward_horizons,
        volatility_window=config.dataset.volatility_window,
        average_volume_window=config.dataset.average_volume_window,
        minimum_price=(
            universe_config.min_price if universe_config is not None else None
        ),
        minimum_average_volume=(
            universe_config.min_average_volume if universe_config is not None else None
        ),
        minimum_average_dollar_volume=(
            universe_config.min_average_dollar_volume
            if universe_config is not None
            else None
        ),
        minimum_listing_history_days=(
            universe_config.min_listing_history_days
            if universe_config is not None
            else None
        ),
        universe_require_tradable=(
            universe_config.require_tradable
            if universe_config is not None
            else False
        ),
        universe_lag=universe_config.lag if universe_config is not None else 1,
        universe_average_volume_window=(
            universe_config.average_volume_window if universe_config is not None else None
        ),
        universe_average_dollar_volume_window=(
            universe_config.average_dollar_volume_window
            if universe_config is not None
            else None
        ),
    )


def add_signal_from_config(
    dataset: pd.DataFrame,
    config: AlphaForgeConfig,
) -> tuple[pd.DataFrame, str]:
    """Append the configured signal to the research dataset."""
    signal_config = require_signal_config(config)
    signaled, signal_column = build_factor_signal(
        dataset,
        name=signal_config.name,
        parameters=signal_parameters_from_config(signal_config),
    )
    masked = _mask_ineligible_signal_rows(
        signaled,
        signal_column=signal_column,
    )
    return _apply_signal_transforms_from_config(
        masked,
        signal_column=signal_column,
        config=config,
    )


def build_weights_from_config(
    frame: pd.DataFrame,
    *,
    score_column: str,
    config: AlphaForgeConfig,
) -> pd.DataFrame:
    """Build portfolio weights from the configured portfolio settings."""
    portfolio_config = require_portfolio_config(config)

    if portfolio_config.construction == "long_only":
        return build_long_only_weights(
            frame,
            score_column=score_column,
            top_n=portfolio_config.top_n,
            weighting=portfolio_config.weighting,
            exposure=portfolio_config.exposure,
            max_position_weight=portfolio_config.max_position_weight,
            position_cap_column=portfolio_config.position_cap_column,
            group_column=portfolio_config.group_column,
            max_group_weight=portfolio_config.max_group_weight,
            factor_exposure_bounds=_portfolio_factor_exposure_bounds(
                portfolio_config,
            ),
        )

    return build_long_short_weights(
        frame,
        score_column=score_column,
        top_n=portfolio_config.top_n,
        bottom_n=portfolio_config.bottom_n,
        weighting=portfolio_config.weighting,
        long_exposure=portfolio_config.long_exposure,
        short_exposure=portfolio_config.short_exposure,
        max_position_weight=portfolio_config.max_position_weight,
        position_cap_column=portfolio_config.position_cap_column,
        group_column=portfolio_config.group_column,
        max_group_weight=portfolio_config.max_group_weight,
        factor_exposure_bounds=_portfolio_factor_exposure_bounds(portfolio_config),
    )


def run_backtest_from_config(config: AlphaForgeConfig) -> pd.DataFrame:
    """Run the configured research, signal, weighting, and backtest workflow."""
    dataset = build_dataset_from_config(config)
    signaled, signal_column = add_signal_from_config(dataset, config)
    weighted = build_weights_from_config(
        signaled,
        score_column=signal_column,
        config=config,
    )
    backtest = run_backtest_with_config(weighted, config=config)
    return maybe_attach_benchmark_to_backtest(backtest, config=config)


def maybe_attach_benchmark_to_backtest(
    backtest: pd.DataFrame,
    *,
    config: AlphaForgeConfig,
    benchmark_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach aligned benchmark returns and relative NAV diagnostics when configured."""
    if config.benchmark is None:
        return backtest

    aligned_benchmark = (
        benchmark_data
        if benchmark_data is not None
        else load_benchmark_returns_from_config(config)
    )
    attached = align_benchmark_to_backtest(backtest, aligned_benchmark)
    attached["excess_return"] = attached["net_return"] - attached["benchmark_return"]

    benchmark_invalid = attached["benchmark_return"] <= -1.0
    if benchmark_invalid.any():
        raise WorkflowError("benchmark_return values must be greater than -1.0.")

    initial_nav = require_backtest_config(config).initial_nav
    attached["benchmark_nav"] = (
        initial_nav * (1.0 + attached["benchmark_return"]).cumprod()
    )
    attached["relative_return"] = (
        (1.0 + attached["net_return"])
        .div(1.0 + attached["benchmark_return"])
        .sub(1.0)
    )
    attached["relative_nav"] = (
        initial_nav * (1.0 + attached["relative_return"]).cumprod()
    )
    return attached


def align_benchmark_to_backtest(
    backtest: pd.DataFrame,
    benchmark_data: pd.DataFrame,
) -> pd.DataFrame:
    """Require exact benchmark/date alignment before augmenting backtest output."""
    if "date" not in backtest.columns:
        raise WorkflowError("Backtest results are missing 'date'.")

    strategy_dates = (
        pd.to_datetime(backtest["date"], errors="coerce")
        .sort_values(kind="mergesort")
        .reset_index(drop=True)
    )
    if strategy_dates.isna().any():
        raise WorkflowError("Backtest results contain invalid date values.")

    benchmark_dates = benchmark_data["date"].reset_index(drop=True)
    if not strategy_dates.equals(benchmark_dates):
        strategy_set = set(strategy_dates.tolist())
        benchmark_set = set(benchmark_dates.tolist())
        missing_dates = strategy_set - benchmark_set
        extra_dates = benchmark_set - strategy_set
        problems: list[str] = []
        if missing_dates:
            problems.append(f"missing {len(missing_dates)} date(s)")
        if extra_dates:
            problems.append(f"extra {len(extra_dates)} date(s)")
        detail = ""
        if problems:
            detail = " (" + "; ".join(problems) + ")"
        raise WorkflowError(
            "benchmark returns must align exactly to backtest dates" + detail + "."
        )

    attached = (
        backtest.sort_values("date", kind="mergesort")
        .reset_index(drop=True)
        .copy()
    )
    attached["benchmark_return"] = benchmark_data["benchmark_return"].to_numpy()
    return attached


def run_backtest_with_config(
    frame: pd.DataFrame,
    *,
    config: AlphaForgeConfig,
) -> pd.DataFrame:
    """Run the shared backtest workflow using config-driven execution settings."""
    backtest_config = require_backtest_config(config)
    return run_daily_backtest(
        frame,
        signal_delay=backtest_config.signal_delay,
        fill_timing=backtest_config.fill_timing,
        rebalance_frequency=backtest_config.rebalance_frequency,
        transaction_cost_bps=backtest_config.transaction_cost_bps,
        commission_bps=backtest_config.commission_bps,
        slippage_bps=backtest_config.slippage_bps,
        max_turnover=backtest_config.max_turnover,
        initial_nav=backtest_config.initial_nav,
    )


def require_signal_config(config: AlphaForgeConfig):
    """Require a signal section for signal-dependent workflows."""
    if config.signal is None:
        raise WorkflowError(
            "The config must include a [signal] section for this command."
        )
    return config.signal


def require_portfolio_config(config: AlphaForgeConfig):
    """Require a portfolio section for portfolio-dependent workflows."""
    if config.portfolio is None:
        raise WorkflowError(
            "The config must include a [portfolio] section for this command."
        )
    return config.portfolio


def require_backtest_config(config: AlphaForgeConfig):
    """Require a backtest section for backtest-dependent workflows."""
    if config.backtest is None:
        raise WorkflowError(
            "The config must include a [backtest] section for this command."
        )
    return config.backtest


def signal_parameters_from_config(signal_config: Any) -> dict[str, int]:
    """Extract explicit factor parameters from a validated signal config."""
    parameters: dict[str, int] = {}
    for parameter_name in ("lookback", "short_window", "long_window"):
        value = getattr(signal_config, parameter_name, None)
        if value is not None:
            parameters[parameter_name] = value
    return parameters


def _portfolio_factor_exposure_bounds(
    portfolio_config: Any,
) -> tuple[tuple[str, float | None, float | None], ...]:
    """Convert config objects into portfolio-builder exposure bound tuples."""
    return tuple(
        (
            bound.column,
            bound.min_exposure,
            bound.max_exposure,
        )
        for bound in portfolio_config.factor_exposure_bounds
    )


def _mask_ineligible_signal_rows(
    frame: pd.DataFrame,
    *,
    signal_column: str,
) -> pd.DataFrame:
    """Set signal values to missing when the lagged universe filter rejects a row."""
    if (
        signal_column not in frame.columns
        or "is_universe_eligible" not in frame.columns
    ):
        return frame

    masked = frame.copy()
    eligible = masked["is_universe_eligible"].fillna(False).astype(bool)
    masked.loc[~eligible, signal_column] = float("nan")
    return masked


def _apply_signal_transforms_from_config(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    config: AlphaForgeConfig,
) -> tuple[pd.DataFrame, str]:
    """Apply any configured within-date signal transforms after universe masking."""
    signal_config = require_signal_config(config)
    if (
        signal_config.winsorize_quantile is None
        and signal_config.clip_lower_bound is None
        and signal_config.clip_upper_bound is None
        and not signal_config.cross_sectional_residualize_columns
        and signal_config.cross_sectional_neutralize_group_column is None
        and signal_config.cross_sectional_normalization == "none"
        and signal_config.cross_sectional_group_column is None
    ):
        return frame, signal_column

    return apply_cross_sectional_signal_transform(
        frame,
        score_column=signal_column,
        winsorize_quantile=signal_config.winsorize_quantile,
        clip_lower_bound=signal_config.clip_lower_bound,
        clip_upper_bound=signal_config.clip_upper_bound,
        residualize_columns=signal_config.cross_sectional_residualize_columns,
        neutralize_group_column=(
            signal_config.cross_sectional_neutralize_group_column
        ),
        normalization=signal_config.cross_sectional_normalization,
        normalization_group_column=signal_config.cross_sectional_group_column,
    )

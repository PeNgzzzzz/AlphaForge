"""Research metadata assembly helpers for config-driven CLI workflows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from alphaforge.cli.data_loading import (
    dataset_membership_indexes,
    dataset_requires_shares_outstanding,
    load_benchmark_returns_from_config,
    load_market_data_from_config,
    load_shares_outstanding_from_config,
)
from alphaforge.cli.pipeline import (
    build_dataset_from_market_data,
    require_signal_config,
    signal_parameters_from_config,
)
from alphaforge.cli.reports import (
    _summarize_benchmark_data,
    _summarize_data_quality,
    _summarize_market_data,
    _summarize_universe_eligibility,
)
from alphaforge.common import AlphaForgeConfig
from alphaforge.features import (
    build_research_dataset_feature_metadata,
    build_research_feature_cache_metadata,
)
from alphaforge.signals import build_signal_pipeline_metadata

__all__ = [
    "build_config_snapshot",
    "build_dataset_feature_metadata_from_config",
    "build_research_context_metadata",
    "build_research_metadata_from_config",
    "build_signal_pipeline_metadata_from_config",
    "dataframe_records",
    "scalar_or_none",
    "series_to_metadata_dict",
]


def build_research_context_metadata(config: AlphaForgeConfig) -> dict[str, Any]:
    """Build lightweight research-context metadata for artifact bundles."""
    market_data = load_market_data_from_config(config)
    benchmark_data = (
        load_benchmark_returns_from_config(config)
        if config.benchmark is not None
        else None
    )
    shares_outstanding = (
        load_shares_outstanding_from_config(config)
        if dataset_requires_shares_outstanding(config)
        else None
    )
    dataset = (
        build_dataset_from_market_data(
            market_data,
            config=config,
            shares_outstanding=shares_outstanding,
        )
        if config.universe is not None
        else None
    )
    research_metadata = build_research_metadata_from_config(config)
    return {
        "workflow_configuration": build_config_snapshot(config),
        **research_metadata,
        "data_summary": _summarize_market_data(market_data),
        "data_quality_summary": _summarize_data_quality(market_data),
        "benchmark_summary": (
            _summarize_benchmark_data(benchmark_data)
            if benchmark_data is not None
            else None
        ),
        "universe_summary": (
            _summarize_universe_eligibility(dataset)
            if dataset is not None
            else None
        ),
    }


def build_config_snapshot(config: AlphaForgeConfig) -> dict[str, Any]:
    """Build a metadata-friendly snapshot of the active research configuration."""
    snapshot: dict[str, Any] = {
        "data": {
            "path": str(config.data.path),
            "price_adjustment": config.data.price_adjustment,
        },
        "dataset": {
            "forward_horizons": list(config.dataset.forward_horizons),
            "volatility_window": config.dataset.volatility_window,
            "average_volume_window": config.dataset.average_volume_window,
            "average_true_range_window": config.dataset.average_true_range_window,
            "normalized_average_true_range_window": (
                config.dataset.normalized_average_true_range_window
            ),
            "amihud_illiquidity_window": config.dataset.amihud_illiquidity_window,
            "dollar_volume_shock_window": config.dataset.dollar_volume_shock_window,
            "dollar_volume_zscore_window": config.dataset.dollar_volume_zscore_window,
            "volume_shock_window": config.dataset.volume_shock_window,
            "relative_volume_window": config.dataset.relative_volume_window,
            "relative_dollar_volume_window": config.dataset.relative_dollar_volume_window,
            "garman_klass_volatility_window": config.dataset.garman_klass_volatility_window,
            "parkinson_volatility_window": config.dataset.parkinson_volatility_window,
            "rogers_satchell_volatility_window": (
                config.dataset.rogers_satchell_volatility_window
            ),
            "yang_zhang_volatility_window": config.dataset.yang_zhang_volatility_window,
            "realized_volatility_window": config.dataset.realized_volatility_window,
            "higher_moments_window": config.dataset.higher_moments_window,
            "benchmark_residual_return_window": (
                config.dataset.benchmark_residual_return_window
            ),
            "benchmark_rolling_window": config.dataset.benchmark_rolling_window,
            "fundamental_metrics": list(config.dataset.fundamental_metrics),
            "valuation_metrics": list(config.dataset.valuation_metrics),
            "quality_ratio_metrics": [
                list(metric_pair) for metric_pair in config.dataset.quality_ratio_metrics
            ],
            "growth_metrics": list(config.dataset.growth_metrics),
            "stability_ratio_metrics": [
                list(metric_pair)
                for metric_pair in config.dataset.stability_ratio_metrics
            ],
            "classification_fields": list(config.dataset.classification_fields),
            "membership_indexes": list(config.dataset.membership_indexes),
            "borrow_fields": list(config.dataset.borrow_fields),
            "include_market_cap": config.dataset.include_market_cap,
            "market_cap_bucket_count": config.dataset.market_cap_bucket_count,
        },
        "signal": None,
        "portfolio": None,
        "backtest": None,
        "benchmark": None,
        "universe": None,
        "diagnostics": {
            "forward_return_column": config.diagnostics.forward_return_column,
            "ic_method": config.diagnostics.ic_method,
            "n_quantiles": config.diagnostics.n_quantiles,
            "min_observations": config.diagnostics.min_observations,
            "rolling_ic_window": config.diagnostics.rolling_ic_window,
            "group_columns": list(config.diagnostics.group_columns),
            "exposure_columns": list(config.diagnostics.exposure_columns),
        },
    }

    if config.signal is not None:
        snapshot["signal"] = {
            "name": config.signal.name,
            "lookback": config.signal.lookback,
            "short_window": config.signal.short_window,
            "long_window": config.signal.long_window,
            "winsorize_quantile": config.signal.winsorize_quantile,
            "clip_lower_bound": config.signal.clip_lower_bound,
            "clip_upper_bound": config.signal.clip_upper_bound,
            "cross_sectional_residualize_columns": list(
                config.signal.cross_sectional_residualize_columns
            ),
            "cross_sectional_neutralize_group_column": (
                config.signal.cross_sectional_neutralize_group_column
            ),
            "cross_sectional_normalization": (
                config.signal.cross_sectional_normalization
            ),
            "cross_sectional_group_column": (
                config.signal.cross_sectional_group_column
            ),
        }
    if config.portfolio is not None:
        snapshot["portfolio"] = {
            "construction": config.portfolio.construction,
            "top_n": config.portfolio.top_n,
            "bottom_n": config.portfolio.bottom_n,
            "weighting": config.portfolio.weighting,
            "exposure": config.portfolio.exposure,
            "long_exposure": config.portfolio.long_exposure,
            "short_exposure": config.portfolio.short_exposure,
            "max_position_weight": config.portfolio.max_position_weight,
            "position_cap_column": config.portfolio.position_cap_column,
            "group_column": config.portfolio.group_column,
            "max_group_weight": config.portfolio.max_group_weight,
            "factor_exposure_bounds": [
                {
                    "column": bound.column,
                    "min": bound.min_exposure,
                    "max": bound.max_exposure,
                }
                for bound in config.portfolio.factor_exposure_bounds
            ],
        }
    if config.backtest is not None:
        snapshot["backtest"] = {
            "signal_delay": config.backtest.signal_delay,
            "fill_timing": config.backtest.fill_timing,
            "rebalance_frequency": config.backtest.rebalance_frequency,
            "transaction_cost_bps": config.backtest.transaction_cost_bps,
            "commission_bps": config.backtest.commission_bps,
            "slippage_bps": config.backtest.slippage_bps,
            "max_turnover": config.backtest.max_turnover,
            "initial_nav": config.backtest.initial_nav,
        }
    if config.symbol_metadata is not None:
        snapshot["symbol_metadata"] = {
            "path": str(config.symbol_metadata.path),
            "listing_date_column": config.symbol_metadata.listing_date_column,
            "delisting_date_column": config.symbol_metadata.delisting_date_column,
        }
    if config.fundamentals is not None:
        snapshot["fundamentals"] = {
            "path": str(config.fundamentals.path),
            "period_end_column": config.fundamentals.period_end_column,
            "release_date_column": config.fundamentals.release_date_column,
            "metric_name_column": config.fundamentals.metric_name_column,
            "metric_value_column": config.fundamentals.metric_value_column,
        }
    if config.shares_outstanding is not None:
        snapshot["shares_outstanding"] = {
            "path": str(config.shares_outstanding.path),
            "effective_date_column": (
                config.shares_outstanding.effective_date_column
            ),
            "shares_outstanding_column": (
                config.shares_outstanding.shares_outstanding_column
            ),
        }
    if config.classifications is not None:
        snapshot["classifications"] = {
            "path": str(config.classifications.path),
            "effective_date_column": config.classifications.effective_date_column,
            "sector_column": config.classifications.sector_column,
            "industry_column": config.classifications.industry_column,
        }
    if config.memberships is not None:
        snapshot["memberships"] = {
            "path": str(config.memberships.path),
            "effective_date_column": config.memberships.effective_date_column,
            "index_column": config.memberships.index_column,
            "is_member_column": config.memberships.is_member_column,
        }
    if config.borrow_availability is not None:
        snapshot["borrow_availability"] = {
            "path": str(config.borrow_availability.path),
            "effective_date_column": config.borrow_availability.effective_date_column,
            "is_borrowable_column": (
                config.borrow_availability.is_borrowable_column
            ),
            "borrow_fee_bps_column": (
                config.borrow_availability.borrow_fee_bps_column
            ),
        }
    if config.trading_status is not None:
        snapshot["trading_status"] = {
            "path": str(config.trading_status.path),
            "effective_date_column": config.trading_status.effective_date_column,
            "is_tradable_column": config.trading_status.is_tradable_column,
            "status_reason_column": config.trading_status.status_reason_column,
        }
    if config.calendar is not None:
        snapshot["calendar"] = {
            "path": str(config.calendar.path),
            "name": config.calendar.name,
            "date_column": config.calendar.date_column,
        }
    if config.benchmark is not None:
        snapshot["benchmark"] = {
            "path": str(config.benchmark.path),
            "name": config.benchmark.name,
            "return_column": config.benchmark.return_column,
            "rolling_window": config.benchmark.rolling_window,
        }
    if config.universe is not None:
        snapshot["universe"] = {
            "min_price": config.universe.min_price,
            "min_average_volume": config.universe.min_average_volume,
            "min_average_dollar_volume": config.universe.min_average_dollar_volume,
            "min_listing_history_days": config.universe.min_listing_history_days,
            "required_membership_indexes": list(
                config.universe.required_membership_indexes
            ),
            "require_tradable": config.universe.require_tradable,
            "lag": config.universe.lag,
            "average_volume_window": config.universe.average_volume_window,
            "average_dollar_volume_window": config.universe.average_dollar_volume_window,
        }
    return snapshot


def build_dataset_feature_metadata_from_config(
    config: AlphaForgeConfig,
) -> list[dict[str, Any]]:
    """Build dataset feature provenance metadata from validated config."""
    universe_config = config.universe
    return build_research_dataset_feature_metadata(
        forward_horizons=config.dataset.forward_horizons,
        volatility_window=config.dataset.volatility_window,
        average_volume_window=config.dataset.average_volume_window,
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
        fundamental_metrics=config.dataset.fundamental_metrics,
        valuation_metrics=config.dataset.valuation_metrics,
        quality_ratio_metrics=config.dataset.quality_ratio_metrics,
        growth_metrics=config.dataset.growth_metrics,
        stability_ratio_metrics=config.dataset.stability_ratio_metrics,
        classification_fields=config.dataset.classification_fields,
        membership_indexes=dataset_membership_indexes(config),
        borrow_fields=config.dataset.borrow_fields,
        include_market_cap=config.dataset.include_market_cap,
        market_cap_bucket_count=config.dataset.market_cap_bucket_count,
        universe_enabled=universe_config is not None,
        universe_lag=universe_config.lag if universe_config is not None else None,
        universe_average_volume_window=(
            universe_config.average_volume_window
            if universe_config is not None
            else None
        ),
        universe_average_dollar_volume_window=(
            universe_config.average_dollar_volume_window
            if universe_config is not None
            else None
        ),
        universe_required_membership_indexes=(
            universe_config.required_membership_indexes
            if universe_config is not None
            else ()
        ),
        universe_require_tradable=(
            universe_config.require_tradable
            if universe_config is not None
            else False
        ),
    )


def build_signal_pipeline_metadata_from_config(
    config: AlphaForgeConfig,
) -> dict[str, Any]:
    """Build configured factor and transform metadata from validated config."""
    signal_config = require_signal_config(config)
    return build_signal_pipeline_metadata(
        factor_name=signal_config.name,
        factor_parameters=signal_parameters_from_config(signal_config),
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


def build_research_metadata_from_config(config: AlphaForgeConfig) -> dict[str, Any]:
    """Build shared research-plan metadata for report and experiment artifacts."""
    dataset_feature_metadata = build_dataset_feature_metadata_from_config(config)
    signal_pipeline_metadata = build_signal_pipeline_metadata_from_config(config)
    return {
        "dataset_feature_metadata": dataset_feature_metadata,
        "signal_pipeline_metadata": signal_pipeline_metadata,
        "feature_cache_metadata": build_research_feature_cache_metadata(
            dataset_feature_metadata=dataset_feature_metadata,
            signal_pipeline_metadata=signal_pipeline_metadata,
        ),
    }


def series_to_metadata_dict(summary: pd.Series) -> dict[str, Any]:
    """Convert a summary series into JSON-safe scalar metadata."""
    return {str(key): scalar_or_none(value) for key, value in summary.items()}


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a small DataFrame into JSON-safe record dictionaries."""
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        records.append({str(key): scalar_or_none(value) for key, value in row.items()})
    return records


def scalar_or_none(value: Any) -> Any:
    """Convert pandas/numpy scalars into JSON-safe Python values."""
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value

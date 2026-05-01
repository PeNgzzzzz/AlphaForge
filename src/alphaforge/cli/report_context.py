"""Shared report context assembly for config-driven CLI workflows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from alphaforge.analytics import (
    compute_grouped_ic_series,
    compute_ic_decay_series,
    compute_ic_decay_summary,
    compute_ic_series,
    compute_quantile_bucket_returns,
    compute_quantile_cumulative_returns,
    compute_quantile_spread_series,
    compute_rolling_ic_series,
    compute_signal_coverage_by_date,
    compute_signal_coverage_by_date_and_group,
    summarize_backtest,
    summarize_grouped_ic,
    summarize_ic,
    summarize_quantile_spread_stability,
    summarize_relative_performance,
    summarize_rolling_ic,
    summarize_signal_coverage,
    summarize_signal_coverage_by_group,
)
from alphaforge.cli.data_loading import (
    load_benchmark_returns_from_config,
    load_market_data_from_config,
)
from alphaforge.cli.errors import WorkflowError
from alphaforge.cli.pipeline import (
    add_signal_from_config,
    build_dataset_from_config,
    build_weights_from_config,
    maybe_attach_benchmark_to_backtest,
    run_backtest_with_config,
)
from alphaforge.common import AlphaForgeConfig
from alphaforge.risk import (
    compute_rolling_benchmark_risk,
    summarize_group_exposure,
    summarize_numeric_exposures,
    summarize_portfolio_diversification,
    summarize_risk,
    summarize_rolling_benchmark_risk,
)

__all__ = [
    "build_report_context",
    "diagnostics_forward_return_columns",
    "validate_diagnostics_column",
]


def build_report_context(config: AlphaForgeConfig) -> dict[str, Any]:
    """Build the shared report context once so text and metadata stay aligned."""
    market_data = load_market_data_from_config(config)
    benchmark_data = (
        load_benchmark_returns_from_config(config)
        if config.benchmark is not None
        else None
    )
    dataset = build_dataset_from_config(config, market_data=market_data)
    signaled, signal_column = add_signal_from_config(dataset, config)
    validate_diagnostics_column(signaled, config)
    weighted = build_weights_from_config(
        signaled,
        score_column=signal_column,
        config=config,
    )
    portfolio_diversification_summary = summarize_portfolio_diversification(
        weighted,
        weight_column="portfolio_weight",
    )
    portfolio_group_exposure_summary = _summarize_portfolio_group_exposure(
        weighted,
        config=config,
    )
    portfolio_numeric_exposure_summary = _summarize_portfolio_numeric_exposures(
        weighted,
        config=config,
    )
    backtest = run_backtest_with_config(weighted, config=config)
    backtest = maybe_attach_benchmark_to_backtest(
        backtest,
        config=config,
        benchmark_data=benchmark_data,
    )

    performance_summary = summarize_backtest(backtest)
    relative_performance_summary = (
        summarize_relative_performance(backtest)
        if benchmark_data is not None
        else None
    )
    risk_summary = summarize_risk(backtest)
    benchmark_risk_summary = None
    rolling_benchmark_risk = None
    if benchmark_data is not None:
        rolling_benchmark_risk = compute_rolling_benchmark_risk(
            backtest.loc[:, ["date", "net_return"]],
            benchmark_data,
            return_column="net_return",
            benchmark_return_column="benchmark_return",
            window=config.benchmark.rolling_window,
        )
        benchmark_risk_summary = summarize_rolling_benchmark_risk(
            rolling_benchmark_risk
        )

    ic_series = compute_ic_series(
        signaled,
        signal_column=signal_column,
        forward_return_column=config.diagnostics.forward_return_column,
        method=config.diagnostics.ic_method,
        min_observations=config.diagnostics.min_observations,
    )
    ic_summary = summarize_ic(ic_series)
    rolling_ic_series = compute_rolling_ic_series(
        ic_series,
        window=config.diagnostics.rolling_ic_window,
    )
    rolling_ic_summary = summarize_rolling_ic(rolling_ic_series)
    diagnostics_forward_columns = diagnostics_forward_return_columns(config)
    ic_decay_summary = compute_ic_decay_summary(
        signaled,
        signal_column=signal_column,
        forward_return_columns=diagnostics_forward_columns,
        method=config.diagnostics.ic_method,
        min_observations=config.diagnostics.min_observations,
    )
    ic_decay_series = compute_ic_decay_series(
        signaled,
        signal_column=signal_column,
        forward_return_columns=diagnostics_forward_columns,
        method=config.diagnostics.ic_method,
        min_observations=config.diagnostics.min_observations,
    )
    grouped_ic_series = _compute_grouped_ic_series_from_config(
        signaled,
        signal_column=signal_column,
        config=config,
    )
    grouped_ic_summary = summarize_grouped_ic(grouped_ic_series)
    quantile_summary = compute_quantile_bucket_returns(
        signaled,
        signal_column=signal_column,
        forward_return_column=config.diagnostics.forward_return_column,
        n_quantiles=config.diagnostics.n_quantiles,
        min_observations=config.diagnostics.min_observations,
    )
    quantile_cumulative_returns = compute_quantile_cumulative_returns(
        signaled,
        signal_column=signal_column,
        forward_return_column=config.diagnostics.forward_return_column,
        n_quantiles=config.diagnostics.n_quantiles,
        min_observations=config.diagnostics.min_observations,
    )
    quantile_spread_series = compute_quantile_spread_series(
        signaled,
        signal_column=signal_column,
        forward_return_column=config.diagnostics.forward_return_column,
        n_quantiles=config.diagnostics.n_quantiles,
        min_observations=config.diagnostics.min_observations,
    )
    quantile_spread_stability = summarize_quantile_spread_stability(
        quantile_spread_series
    )
    coverage_summary = summarize_signal_coverage(
        signaled,
        signal_column=signal_column,
        forward_return_column=config.diagnostics.forward_return_column,
    )
    coverage_by_date = compute_signal_coverage_by_date(
        signaled,
        signal_column=signal_column,
        forward_return_column=config.diagnostics.forward_return_column,
    )
    grouped_coverage_by_date = _compute_grouped_coverage_by_date_from_config(
        signaled,
        signal_column=signal_column,
        config=config,
    )
    grouped_coverage_summary = _compute_grouped_coverage_summary_from_config(
        signaled,
        signal_column=signal_column,
        config=config,
    )

    return {
        "market_data": market_data,
        "benchmark_data": benchmark_data,
        "dataset": dataset,
        "signaled": signaled,
        "signal_column": signal_column,
        "weighted": weighted,
        "backtest": backtest,
        "portfolio_diversification_summary": portfolio_diversification_summary,
        "portfolio_group_exposure_summary": portfolio_group_exposure_summary,
        "portfolio_numeric_exposure_summary": portfolio_numeric_exposure_summary,
        "rolling_benchmark_risk": rolling_benchmark_risk,
        "performance_summary": performance_summary,
        "relative_performance_summary": relative_performance_summary,
        "risk_summary": risk_summary,
        "benchmark_risk_summary": benchmark_risk_summary,
        "ic_series": ic_series,
        "ic_summary": ic_summary,
        "rolling_ic_series": rolling_ic_series,
        "rolling_ic_summary": rolling_ic_summary,
        "ic_decay_summary": ic_decay_summary,
        "ic_decay_series": ic_decay_series,
        "grouped_ic_series": grouped_ic_series,
        "grouped_ic_summary": grouped_ic_summary,
        "quantile_summary": quantile_summary,
        "quantile_cumulative_returns": quantile_cumulative_returns,
        "quantile_spread_series": quantile_spread_series,
        "quantile_spread_stability": quantile_spread_stability,
        "coverage_summary": coverage_summary,
        "coverage_by_date": coverage_by_date,
        "grouped_coverage_by_date": grouped_coverage_by_date,
        "grouped_coverage_summary": grouped_coverage_summary,
    }


def validate_diagnostics_column(
    dataset: pd.DataFrame,
    config: AlphaForgeConfig,
) -> None:
    """Ensure the configured diagnostics label exists in the built dataset."""
    forward_return_column = config.diagnostics.forward_return_column
    if forward_return_column not in dataset.columns:
        raise WorkflowError(
            "The configured diagnostics.forward_return_column is not present in the built dataset: "
            f"{forward_return_column}."
        )


def diagnostics_forward_return_columns(config: AlphaForgeConfig) -> tuple[str, ...]:
    """Return configured label columns for IC decay summaries."""
    return tuple(
        f"forward_return_{horizon}d" for horizon in config.dataset.forward_horizons
    )


def _summarize_portfolio_group_exposure(
    weighted: pd.DataFrame,
    *,
    config: AlphaForgeConfig,
) -> pd.DataFrame:
    """Summarize target-weight group exposure when a group constraint is configured."""
    portfolio = config.portfolio
    if portfolio is None or portfolio.group_column is None:
        return pd.DataFrame()
    return summarize_group_exposure(
        weighted,
        group_column=portfolio.group_column,
        weight_column="portfolio_weight",
    )


def _summarize_portfolio_numeric_exposures(
    weighted: pd.DataFrame,
    *,
    config: AlphaForgeConfig,
) -> pd.DataFrame:
    """Summarize target-weight numeric exposures when explicitly configured."""
    if not config.diagnostics.exposure_columns:
        return pd.DataFrame()
    return summarize_numeric_exposures(
        weighted,
        exposure_columns=config.diagnostics.exposure_columns,
        weight_column="portfolio_weight",
    )


def _compute_grouped_ic_series_from_config(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    config: AlphaForgeConfig,
) -> pd.DataFrame:
    """Compute grouped IC diagnostics for explicitly configured group columns."""
    frames = [
        compute_grouped_ic_series(
            frame,
            signal_column=signal_column,
            forward_return_column=config.diagnostics.forward_return_column,
            group_column=group_column,
            method=config.diagnostics.ic_method,
            min_observations=config.diagnostics.min_observations,
        )
        for group_column in config.diagnostics.group_columns
    ]
    if not frames:
        return pd.DataFrame(
            columns=[
                "date",
                "group_column",
                "group_value",
                "ic",
                "observations",
                "method",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _compute_grouped_coverage_by_date_from_config(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    config: AlphaForgeConfig,
) -> pd.DataFrame:
    """Compute grouped coverage diagnostics for configured group columns."""
    frames = [
        compute_signal_coverage_by_date_and_group(
            frame,
            signal_column=signal_column,
            forward_return_column=config.diagnostics.forward_return_column,
            group_column=group_column,
        )
        for group_column in config.diagnostics.group_columns
    ]
    if not frames:
        return pd.DataFrame(
            columns=[
                "date",
                "group_column",
                "group_value",
                "total_rows",
                "signal_non_null_rows",
                "forward_return_non_null_rows",
                "usable_rows",
                "signal_coverage_ratio",
                "forward_return_coverage_ratio",
                "joint_coverage_ratio",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _compute_grouped_coverage_summary_from_config(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    config: AlphaForgeConfig,
) -> pd.DataFrame:
    """Summarize grouped coverage diagnostics for configured group columns."""
    frames = [
        summarize_signal_coverage_by_group(
            frame,
            signal_column=signal_column,
            forward_return_column=config.diagnostics.forward_return_column,
            group_column=group_column,
        )
        for group_column in config.diagnostics.group_columns
    ]
    if not frames:
        return pd.DataFrame(
            columns=[
                "group_column",
                "group_value",
                "dates",
                "total_rows",
                "signal_non_null_rows",
                "forward_return_non_null_rows",
                "usable_rows",
                "signal_coverage_ratio",
                "forward_return_coverage_ratio",
                "joint_coverage_ratio",
                "average_daily_usable_rows",
                "minimum_daily_usable_rows",
            ]
        )
    return pd.concat(frames, ignore_index=True)

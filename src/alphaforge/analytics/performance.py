"""Performance analytics for daily backtest output."""

from __future__ import annotations

import math

import pandas as pd

from alphaforge.common.errors import AlphaForgeError
from alphaforge.common.validation import (
    normalize_positive_int as _common_positive_int,
    parse_numeric_series as _common_numeric_series,
)


class AnalyticsError(AlphaForgeError):
    """Raised when performance analytics inputs or settings are invalid."""


def summarize_backtest(
    frame: pd.DataFrame,
    *,
    return_column: str = "net_return",
    turnover_column: str = "turnover",
    periods_per_year: int = 252,
) -> pd.Series:
    """Summarize daily backtest output into headline performance metrics."""
    periods_per_year = _normalize_positive_int(
        periods_per_year,
        parameter_name="periods_per_year",
    )
    dataset = _prepare_analytics_input(
        frame,
        return_column=return_column,
        turnover_column=turnover_column,
    )

    returns = dataset[return_column]
    turnover = dataset[turnover_column]
    periods = len(dataset)

    cumulative_return = (1.0 + returns).prod() - 1.0
    annualized_return = (1.0 + cumulative_return) ** (periods_per_year / periods) - 1.0

    volatility = returns.std(ddof=1)
    if pd.isna(volatility):
        annualized_volatility = math.nan
        sharpe_ratio = math.nan
    else:
        annualized_volatility = volatility * math.sqrt(periods_per_year)
        if annualized_volatility == 0.0:
            sharpe_ratio = math.nan
        else:
            sharpe_ratio = (
                returns.mean() / volatility * math.sqrt(periods_per_year)
            )

    drawdown = compute_drawdown_series(
        dataset,
        return_column=return_column,
    )["drawdown"]

    summary = pd.Series(
        {
            "periods": float(periods),
            "cumulative_return": cumulative_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": drawdown.min(),
            "average_turnover": turnover.mean(),
            "total_turnover": turnover.sum(),
            "hit_rate": returns.gt(0.0).mean(),
        },
        name="performance_summary",
    )
    return summary


def summarize_relative_performance(
    frame: pd.DataFrame,
    *,
    strategy_return_column: str = "net_return",
    benchmark_return_column: str = "benchmark_return",
    periods_per_year: int = 252,
) -> pd.Series:
    """Summarize benchmark-relative performance from aligned daily return series."""
    periods_per_year = _normalize_positive_int(
        periods_per_year,
        parameter_name="periods_per_year",
    )
    dataset = _prepare_relative_performance_input(
        frame,
        strategy_return_column=strategy_return_column,
        benchmark_return_column=benchmark_return_column,
    )

    strategy_returns = dataset[strategy_return_column]
    benchmark_returns = dataset[benchmark_return_column]
    active_returns = strategy_returns - benchmark_returns
    periods = len(dataset)

    benchmark_cumulative_return = (1.0 + benchmark_returns).prod() - 1.0
    benchmark_annualized_return = (
        (1.0 + benchmark_cumulative_return) ** (periods_per_year / periods) - 1.0
    )

    relative_return = (1.0 + strategy_returns).div(1.0 + benchmark_returns).sub(1.0)
    excess_cumulative_return = (1.0 + relative_return).prod() - 1.0
    annualized_excess_return = (
        (1.0 + excess_cumulative_return) ** (periods_per_year / periods) - 1.0
    )

    active_volatility = active_returns.std(ddof=1)
    if pd.isna(active_volatility):
        tracking_error = math.nan
        information_ratio = math.nan
    else:
        tracking_error = active_volatility * math.sqrt(periods_per_year)
        if tracking_error == 0.0:
            information_ratio = math.nan
        else:
            information_ratio = (
                active_returns.mean() / active_volatility * math.sqrt(periods_per_year)
            )

    return pd.Series(
        {
            "periods": float(periods),
            "benchmark_cumulative_return": benchmark_cumulative_return,
            "benchmark_annualized_return": benchmark_annualized_return,
            "excess_cumulative_return": excess_cumulative_return,
            "annualized_excess_return": annualized_excess_return,
            "average_daily_excess_return": active_returns.mean(),
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "excess_hit_rate": strategy_returns.gt(benchmark_returns).mean(),
        },
        name="relative_performance_summary",
    )


def compute_drawdown_series(
    frame: pd.DataFrame,
    *,
    return_column: str = "net_return",
) -> pd.DataFrame:
    """Compute cumulative NAV and drawdown from a backtest return series."""
    dataset = _prepare_analytics_input(
        frame,
        return_column=return_column,
        turnover_column=None,
    )

    nav = (1.0 + dataset[return_column]).cumprod()
    running_max_nav = nav.cummax()
    drawdown = nav.div(running_max_nav).sub(1.0)

    columns = {}
    if "date" in dataset.columns:
        columns["date"] = dataset["date"]
    columns["nav"] = nav
    columns["running_max_nav"] = running_max_nav
    columns["drawdown"] = drawdown
    return pd.DataFrame(columns)


def format_performance_summary(summary: pd.Series) -> str:
    """Format a headline performance summary as plain text."""
    required_keys = [
        "periods",
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "average_turnover",
        "total_turnover",
        "hit_rate",
    ]
    missing_keys = [key for key in required_keys if key not in summary.index]
    if missing_keys:
        missing_text = ", ".join(missing_keys)
        raise AnalyticsError(
            f"summary is missing required performance fields: {missing_text}."
        )

    lines = [
        "Performance Summary",
        f"Periods: {int(summary['periods'])}",
        f"Cumulative Return: {_format_percent(summary['cumulative_return'])}",
        f"Annualized Return: {_format_percent(summary['annualized_return'])}",
        f"Annualized Volatility: {_format_percent(summary['annualized_volatility'])}",
        f"Sharpe Ratio: {_format_number(summary['sharpe_ratio'])}",
        f"Max Drawdown: {_format_percent(summary['max_drawdown'])}",
        f"Average Turnover: {_format_number(summary['average_turnover'])}",
        f"Total Turnover: {_format_number(summary['total_turnover'])}",
        f"Hit Rate: {_format_percent(summary['hit_rate'])}",
    ]
    return "\n".join(lines)


def format_relative_performance_summary(summary: pd.Series) -> str:
    """Format benchmark-relative performance metrics as plain text."""
    required_keys = [
        "periods",
        "benchmark_cumulative_return",
        "benchmark_annualized_return",
        "excess_cumulative_return",
        "annualized_excess_return",
        "average_daily_excess_return",
        "tracking_error",
        "information_ratio",
        "excess_hit_rate",
    ]
    missing_keys = [key for key in required_keys if key not in summary.index]
    if missing_keys:
        missing_text = ", ".join(missing_keys)
        raise AnalyticsError(
            f"summary is missing required relative-performance fields: {missing_text}."
        )

    lines = [
        "Relative Performance Summary",
        f"Periods: {int(summary['periods'])}",
        "Benchmark Cumulative Return: "
        f"{_format_percent(summary['benchmark_cumulative_return'])}",
        "Benchmark Annualized Return: "
        f"{_format_percent(summary['benchmark_annualized_return'])}",
        f"Excess Cumulative Return: {_format_percent(summary['excess_cumulative_return'])}",
        f"Annualized Excess Return: {_format_percent(summary['annualized_excess_return'])}",
        "Average Daily Excess Return: "
        f"{_format_percent(summary['average_daily_excess_return'])}",
        f"Tracking Error: {_format_percent(summary['tracking_error'])}",
        f"Information Ratio: {_format_number(summary['information_ratio'])}",
        f"Excess Hit Rate: {_format_percent(summary['excess_hit_rate'])}",
    ]
    return "\n".join(lines)


def _prepare_analytics_input(
    frame: pd.DataFrame,
    *,
    return_column: str,
    turnover_column: str | None,
) -> pd.DataFrame:
    """Validate and parse daily analytics inputs."""
    if return_column not in frame.columns:
        raise AnalyticsError(f"backtest results are missing '{return_column}'.")

    dataset = frame.copy()
    dataset[return_column] = _parse_numeric_column(
        dataset[return_column],
        column_name=return_column,
    )

    if turnover_column is not None:
        if turnover_column not in dataset.columns:
            raise AnalyticsError(f"backtest results are missing '{turnover_column}'.")
        dataset[turnover_column] = _parse_numeric_column(
            dataset[turnover_column],
            column_name=turnover_column,
        )

    if dataset.empty:
        raise AnalyticsError("backtest results must contain at least one row.")

    return dataset.reset_index(drop=True)


def _prepare_relative_performance_input(
    frame: pd.DataFrame,
    *,
    strategy_return_column: str,
    benchmark_return_column: str,
) -> pd.DataFrame:
    """Validate benchmark-relative performance inputs without silent date coercion."""
    dataset = frame.copy()
    for column in [strategy_return_column, benchmark_return_column]:
        if column not in dataset.columns:
            raise AnalyticsError(f"backtest results are missing '{column}'.")
        dataset[column] = _parse_numeric_column(dataset[column], column_name=column)

    if "date" not in dataset.columns:
        raise AnalyticsError("backtest results are missing 'date'.")
    dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
    if dataset["date"].isna().any():
        raise AnalyticsError("backtest results contain invalid date values in 'date'.")
    if dataset["date"].duplicated().any():
        raise AnalyticsError("backtest results contain duplicate dates.")
    if dataset.empty:
        raise AnalyticsError("backtest results must contain at least one row.")

    return dataset.sort_values("date", kind="mergesort").reset_index(drop=True)


def _parse_numeric_column(values: pd.Series, *, column_name: str) -> pd.Series:
    """Parse numeric analytics columns without silent coercion."""
    return _common_numeric_series(
        values,
        column_name=column_name,
        source="backtest results",
        error_factory=AnalyticsError,
        missing_values_are_invalid=True,
        verb="contain",
    )


def _normalize_positive_int(value: int, *, parameter_name: str) -> int:
    """Validate positive integer parameters."""
    return _common_positive_int(
        value,
        parameter_name=parameter_name,
        error_factory=AnalyticsError,
    )


def _format_percent(value: float) -> str:
    """Format a decimal value as a percentage."""
    if pd.isna(value):
        return "NaN"
    return f"{value:.2%}"


def _format_number(value: float) -> str:
    """Format a scalar metric with two decimals."""
    if pd.isna(value):
        return "NaN"
    return f"{value:.2f}"

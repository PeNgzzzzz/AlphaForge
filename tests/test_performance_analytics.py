"""Tests for performance analytics on daily backtest output."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from alphaforge.analytics import (
    AnalyticsError,
    compute_drawdown_series,
    format_performance_summary,
    format_relative_performance_summary,
    summarize_backtest,
    summarize_relative_performance,
)


def test_summarize_backtest_computes_expected_metrics() -> None:
    """Summary metrics should match the backtest return and turnover series."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "net_return": [0.01, -0.02, 0.03, 0.00],
            "turnover": [0.50, 0.20, 0.10, 0.00],
        }
    )

    summary = summarize_backtest(frame, periods_per_year=252)

    returns = pd.Series([0.01, -0.02, 0.03, 0.00], dtype=float)
    cumulative_return = (1.0 + returns).prod() - 1.0
    annualized_return = (1.0 + cumulative_return) ** (252 / 4) - 1.0
    annualized_volatility = returns.std(ddof=1) * math.sqrt(252)
    sharpe_ratio = returns.mean() / returns.std(ddof=1) * math.sqrt(252)

    assert summary["periods"] == pytest.approx(4.0)
    assert summary["cumulative_return"] == pytest.approx(cumulative_return)
    assert summary["annualized_return"] == pytest.approx(annualized_return)
    assert summary["annualized_volatility"] == pytest.approx(annualized_volatility)
    assert summary["sharpe_ratio"] == pytest.approx(sharpe_ratio)
    assert summary["max_drawdown"] == pytest.approx(-0.02)
    assert summary["average_turnover"] == pytest.approx(0.20)
    assert summary["total_turnover"] == pytest.approx(0.80)
    assert summary["hit_rate"] == pytest.approx(0.50)


def test_compute_drawdown_series_tracks_nav_and_running_peak() -> None:
    """Drawdown output should track cumulative NAV against its running maximum."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "net_return": [0.10, -0.20, 0.05],
        }
    )

    drawdown = compute_drawdown_series(frame)

    assert drawdown["nav"].tolist() == pytest.approx([1.10, 0.88, 0.924])
    assert drawdown["running_max_nav"].tolist() == pytest.approx([1.10, 1.10, 1.10])
    assert drawdown["drawdown"].tolist() == pytest.approx([0.0, -0.20, -0.16])


def test_summarize_backtest_returns_nan_sharpe_for_zero_volatility() -> None:
    """A zero-volatility return stream should not produce infinite Sharpe."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "net_return": [0.0, 0.0, 0.0],
            "turnover": [0.0, 0.0, 0.0],
        }
    )

    summary = summarize_backtest(frame)

    assert summary["cumulative_return"] == pytest.approx(0.0)
    assert summary["annualized_volatility"] == pytest.approx(0.0)
    assert np.isnan(summary["sharpe_ratio"])
    assert summary["max_drawdown"] == pytest.approx(0.0)
    assert summary["hit_rate"] == pytest.approx(0.0)


def test_format_performance_summary_renders_key_metrics() -> None:
    """The formatted summary should surface the main performance fields."""
    summary = pd.Series(
        {
            "periods": 4.0,
            "cumulative_return": 0.10,
            "annualized_return": 0.20,
            "annualized_volatility": 0.15,
            "sharpe_ratio": 1.25,
            "max_drawdown": -0.05,
            "average_turnover": 0.30,
            "total_turnover": 1.20,
            "hit_rate": 0.50,
        }
    )

    formatted = format_performance_summary(summary)

    assert "Performance Summary" in formatted
    assert "Cumulative Return: 10.00%" in formatted
    assert "Sharpe Ratio: 1.25" in formatted
    assert "Hit Rate: 50.00%" in formatted


def test_summarize_relative_performance_computes_expected_metrics() -> None:
    """Relative metrics should match aligned strategy and benchmark returns."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "net_return": [0.02, -0.01, 0.03, 0.00],
            "benchmark_return": [0.01, -0.02, 0.01, 0.01],
            "turnover": [0.5, 0.2, 0.1, 0.0],
        }
    )

    summary = summarize_relative_performance(frame, periods_per_year=252)

    strategy_returns = pd.Series([0.02, -0.01, 0.03, 0.00], dtype=float)
    benchmark_returns = pd.Series([0.01, -0.02, 0.01, 0.01], dtype=float)
    active_returns = strategy_returns - benchmark_returns
    benchmark_cumulative_return = (1.0 + benchmark_returns).prod() - 1.0
    benchmark_annualized_return = (1.0 + benchmark_cumulative_return) ** (252 / 4) - 1.0
    relative_returns = (1.0 + strategy_returns).div(1.0 + benchmark_returns).sub(1.0)
    excess_cumulative_return = (1.0 + relative_returns).prod() - 1.0
    annualized_excess_return = (1.0 + excess_cumulative_return) ** (252 / 4) - 1.0
    tracking_error = active_returns.std(ddof=1) * math.sqrt(252)
    information_ratio = active_returns.mean() / active_returns.std(ddof=1) * math.sqrt(252)

    assert summary["periods"] == pytest.approx(4.0)
    assert summary["benchmark_cumulative_return"] == pytest.approx(
        benchmark_cumulative_return
    )
    assert summary["benchmark_annualized_return"] == pytest.approx(
        benchmark_annualized_return
    )
    assert summary["excess_cumulative_return"] == pytest.approx(
        excess_cumulative_return
    )
    assert summary["annualized_excess_return"] == pytest.approx(
        annualized_excess_return
    )
    assert summary["average_daily_excess_return"] == pytest.approx(active_returns.mean())
    assert summary["tracking_error"] == pytest.approx(tracking_error)
    assert summary["information_ratio"] == pytest.approx(information_ratio)
    assert summary["excess_hit_rate"] == pytest.approx(0.75)


def test_format_relative_performance_summary_renders_key_metrics() -> None:
    """Formatted relative output should expose the main active-return fields."""
    summary = pd.Series(
        {
            "periods": 4.0,
            "benchmark_cumulative_return": 0.08,
            "benchmark_annualized_return": 0.12,
            "excess_cumulative_return": 0.03,
            "annualized_excess_return": 0.05,
            "average_daily_excess_return": 0.001,
            "tracking_error": 0.07,
            "information_ratio": 0.9,
            "excess_hit_rate": 0.75,
        }
    )

    formatted = format_relative_performance_summary(summary)

    assert "Relative Performance Summary" in formatted
    assert "Benchmark Cumulative Return: 8.00%" in formatted
    assert "Tracking Error: 7.00%" in formatted
    assert "Information Ratio: 0.90" in formatted


def test_performance_analytics_validate_inputs() -> None:
    """Invalid analytics inputs should fail loudly."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "net_return": [0.01],
            "turnover": [0.1],
        }
    )

    with pytest.raises(AnalyticsError, match="periods_per_year"):
        summarize_backtest(frame, periods_per_year=0)

    with pytest.raises(AnalyticsError, match="missing 'net_return'"):
        summarize_backtest(frame.drop(columns=["net_return"]))

    with pytest.raises(AnalyticsError, match="missing 'turnover'"):
        summarize_backtest(frame.drop(columns=["turnover"]))

    bad_frame = frame.copy()
    bad_frame["net_return"] = bad_frame["net_return"].astype("object")
    bad_frame.loc[0, "net_return"] = "bad"
    with pytest.raises(AnalyticsError, match="invalid numeric values"):
        summarize_backtest(bad_frame)

    with pytest.raises(AnalyticsError, match="at least one row"):
        summarize_backtest(frame.iloc[0:0])

    with pytest.raises(AnalyticsError, match="required performance fields"):
        format_performance_summary(pd.Series({"periods": 1.0}))

    with pytest.raises(AnalyticsError, match="missing 'benchmark_return'"):
        summarize_relative_performance(frame.drop(columns=["benchmark_return"], errors="ignore"))

    duplicate_dates = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "net_return": [0.01, 0.02],
            "benchmark_return": [0.0, 0.01],
            "turnover": [0.1, 0.2],
        }
    )
    with pytest.raises(AnalyticsError, match="duplicate dates"):
        summarize_relative_performance(duplicate_dates)

    with pytest.raises(AnalyticsError, match="required relative-performance fields"):
        format_relative_performance_summary(pd.Series({"periods": 1.0}))

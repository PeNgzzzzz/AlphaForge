"""Tests for daily risk analytics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from alphaforge.risk import (
    RiskError,
    compute_rolling_benchmark_risk,
    format_benchmark_risk_summary,
    format_risk_summary,
    summarize_risk,
    summarize_rolling_benchmark_risk,
    summarize_weight_concentration,
)


def test_summarize_risk_computes_expected_metrics() -> None:
    """Risk summary should match the daily return and exposure series."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "net_return": [0.01, -0.02, 0.03, 0.0],
            "gross_exposure": [1.0, 0.8, 1.2, 0.0],
            "net_exposure": [1.0, 0.0, 1.0, 0.0],
        }
    )

    summary = summarize_risk(frame, periods_per_year=252, var_confidence=0.95)
    returns = pd.Series([0.01, -0.02, 0.03, 0.0], dtype=float)
    downside_returns = returns.clip(upper=0.0)

    assert summary["periods"] == pytest.approx(4.0)
    assert summary["realized_volatility"] == pytest.approx(
        returns.std(ddof=1) * math.sqrt(252)
    )
    assert summary["downside_volatility"] == pytest.approx(
        math.sqrt(downside_returns.pow(2).mean()) * math.sqrt(252)
    )
    assert summary["value_at_risk"] == pytest.approx(returns.quantile(0.05))
    assert summary["conditional_value_at_risk"] == pytest.approx(-0.02)
    assert summary["average_gross_exposure"] == pytest.approx(0.75)
    assert summary["average_net_exposure"] == pytest.approx(0.50)


def test_compute_rolling_benchmark_risk_matches_linear_relationship() -> None:
    """Rolling beta and correlation should reflect the benchmark relationship."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "net_return": [0.02, -0.04, 0.06, 0.08],
        }
    )
    benchmark = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "benchmark_return": [0.01, -0.02, 0.03, 0.04],
        }
    )

    rolling = compute_rolling_benchmark_risk(frame, benchmark, window=3)

    assert pd.isna(rolling.loc[0, "rolling_beta"])
    assert pd.isna(rolling.loc[1, "rolling_beta"])
    assert rolling.loc[2, "rolling_beta"] == pytest.approx(2.0)
    assert rolling.loc[3, "rolling_beta"] == pytest.approx(2.0)
    assert rolling.loc[2, "rolling_correlation"] == pytest.approx(1.0)
    assert rolling.loc[3, "rolling_correlation"] == pytest.approx(1.0)


def test_summarize_rolling_benchmark_risk_computes_expected_metrics() -> None:
    """Rolling benchmark risk summary should condense beta/correlation diagnostics."""
    rolling = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "rolling_beta": [math.nan, math.nan, 2.0, 2.0],
            "rolling_correlation": [math.nan, math.nan, 1.0, 1.0],
        }
    )

    summary = summarize_rolling_benchmark_risk(rolling)

    assert summary["periods"] == pytest.approx(4.0)
    assert summary["valid_periods"] == pytest.approx(2.0)
    assert summary["average_rolling_beta"] == pytest.approx(2.0)
    assert summary["latest_rolling_beta"] == pytest.approx(2.0)
    assert summary["average_rolling_correlation"] == pytest.approx(1.0)
    assert summary["latest_rolling_correlation"] == pytest.approx(1.0)


def test_summarize_weight_concentration_computes_expected_metrics() -> None:
    """Concentration summary should reflect gross exposure and weight dispersion."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                ]
            ),
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "effective_weight": [0.6, 0.4, 0.5, -0.5],
        }
    )

    summary = summarize_weight_concentration(frame)

    assert summary["periods"] == pytest.approx(2.0)
    assert summary["average_gross_exposure"] == pytest.approx(1.0)
    assert summary["max_gross_exposure"] == pytest.approx(1.0)
    assert summary["average_net_exposure"] == pytest.approx(0.5)
    assert summary["average_herfindahl_index"] == pytest.approx(0.51)
    assert summary["average_max_abs_weight"] == pytest.approx(0.55)


def test_format_risk_summary_renders_key_metrics() -> None:
    """Formatted risk output should expose the main headline fields."""
    summary = pd.Series(
        {
            "periods": 4.0,
            "realized_volatility": 0.15,
            "downside_volatility": 0.10,
            "value_at_risk": -0.02,
            "conditional_value_at_risk": -0.03,
            "var_confidence": 0.95,
            "average_gross_exposure": 0.80,
            "average_net_exposure": 0.10,
        }
    )

    formatted = format_risk_summary(summary)

    assert "Risk Summary" in formatted
    assert "Realized Volatility: 15.00%" in formatted
    assert "VaR (95%): -2.00%" in formatted
    assert "Average Gross Exposure: 0.80" in formatted


def test_format_benchmark_risk_summary_renders_key_metrics() -> None:
    """Formatted benchmark risk output should surface beta/correlation diagnostics."""
    summary = pd.Series(
        {
            "periods": 10.0,
            "valid_periods": 7.0,
            "average_rolling_beta": 1.1,
            "latest_rolling_beta": 0.9,
            "average_rolling_correlation": 0.4,
            "latest_rolling_correlation": 0.6,
        }
    )

    formatted = format_benchmark_risk_summary(summary, window=3)

    assert "Benchmark Risk Summary" in formatted
    assert "Rolling Window: 3" in formatted
    assert "Average Rolling Beta: 1.10" in formatted
    assert "Latest Rolling Correlation: 0.60" in formatted


def test_risk_metrics_validate_inputs() -> None:
    """Invalid risk inputs should fail loudly."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "net_return": [0.01, -0.02],
            "gross_exposure": [1.0, 1.0],
            "net_exposure": [0.5, 0.0],
        }
    )
    benchmark = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "benchmark_return": [0.01, -0.02],
        }
    )

    with pytest.raises(RiskError, match="periods_per_year"):
        summarize_risk(frame, periods_per_year=0)

    with pytest.raises(RiskError, match="var_confidence"):
        summarize_risk(frame, var_confidence=1.0)

    with pytest.raises(RiskError, match="missing 'gross_exposure'"):
        summarize_risk(frame.drop(columns=["gross_exposure"]))

    with pytest.raises(RiskError, match="align exactly to strategy dates"):
        compute_rolling_benchmark_risk(frame, benchmark.iloc[[0]], window=2)

    benchmark_with_extra_date = pd.concat(
        [
            benchmark,
            pd.DataFrame(
                {
                    "date": [pd.Timestamp("2024-01-04")],
                    "benchmark_return": [0.0],
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(RiskError, match="align exactly to strategy dates"):
        compute_rolling_benchmark_risk(frame, benchmark_with_extra_date, window=2)

    duplicate_benchmark = pd.concat([benchmark, benchmark.iloc[[0]]], ignore_index=True)
    with pytest.raises(RiskError, match="duplicate dates"):
        compute_rolling_benchmark_risk(frame, duplicate_benchmark, window=2)

    bad_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "symbol": ["AAPL"],
            "effective_weight": ["bad"],
        }
    )
    with pytest.raises(RiskError, match="invalid numeric values"):
        summarize_weight_concentration(bad_weights)

    with pytest.raises(RiskError, match="required risk fields"):
        format_risk_summary(pd.Series({"periods": 1.0}))

    with pytest.raises(RiskError, match="rolling benchmark risk frame is missing required columns"):
        summarize_rolling_benchmark_risk(pd.DataFrame({"date": pd.to_datetime(["2024-01-02"])}))

    invalid_rolling = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "rolling_beta": ["bad"],
            "rolling_correlation": [0.5],
        }
    )
    with pytest.raises(RiskError, match="invalid numeric values"):
        summarize_rolling_benchmark_risk(invalid_rolling)

    with pytest.raises(RiskError, match="required benchmark-risk fields"):
        format_benchmark_risk_summary(pd.Series({"periods": 1.0}), window=3)

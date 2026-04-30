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
    summarize_group_exposure,
    summarize_numeric_exposures,
    summarize_portfolio_diversification,
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


def test_summarize_portfolio_diversification_computes_expected_metrics() -> None:
    """Diversification summary should use absolute weights without netting sides."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-03",
                ]
            ),
            "symbol": [
                "AAPL",
                "MSFT",
                "NVDA",
                "GOOG",
                "AMZN",
                "META",
                "AAPL",
                "MSFT",
                "NVDA",
                "TSLA",
            ],
            "portfolio_weight": [
                0.30,
                0.25,
                0.15,
                0.10,
                0.10,
                0.10,
                0.40,
                -0.30,
                0.20,
                -0.10,
            ],
        }
    )

    summary = summarize_portfolio_diversification(frame)

    first_day_effective_positions = 1.0 / 0.205
    second_day_effective_positions = 1.0 / 0.30
    assert summary["periods"] == pytest.approx(2.0)
    assert summary["average_holdings_count"] == pytest.approx(5.0)
    assert summary["min_holdings_count"] == pytest.approx(4.0)
    assert summary["average_long_count"] == pytest.approx(4.0)
    assert summary["average_short_count"] == pytest.approx(1.0)
    assert summary["average_effective_number_of_positions"] == pytest.approx(
        (first_day_effective_positions + second_day_effective_positions) / 2.0
    )
    assert summary["min_effective_number_of_positions"] == pytest.approx(
        second_day_effective_positions
    )
    assert summary["average_effective_position_ratio"] == pytest.approx(
        ((first_day_effective_positions / 6.0) + (second_day_effective_positions / 4.0))
        / 2.0
    )
    assert summary["average_top_position_weight_share"] == pytest.approx(0.35)
    assert summary["max_top_position_weight_share"] == pytest.approx(0.40)
    assert summary["average_top_five_weight_share"] == pytest.approx(0.95)
    assert summary["max_top_five_weight_share"] == pytest.approx(1.0)


def test_summarize_group_exposure_computes_group_diagnostics() -> None:
    """Group exposure summary should surface gross, net, and missing-label exposure."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-03",
                ]
            ),
            "symbol": ["AAPL", "MSFT", "TSLA", "F", "AAPL", "MSFT", "TSLA", "F"],
            "portfolio_weight": [0.4, 0.1, -0.2, 0.0, 0.0, 0.3, -0.3, 0.2],
            "classification_sector": [
                "Technology",
                "Technology",
                "Consumer",
                None,
                "Technology",
                "Technology",
                "Consumer",
                " ",
            ],
        }
    )

    summary = summarize_group_exposure(
        frame,
        group_column="classification_sector",
        weight_column="portfolio_weight",
    )
    by_group = summary.set_index("group_value")

    assert by_group.loc["Technology", "group_column"] == "classification_sector"
    assert by_group.loc["Technology", "periods"] == pytest.approx(2.0)
    assert by_group.loc["Technology", "average_gross_exposure"] == pytest.approx(0.4)
    assert by_group.loc["Technology", "max_gross_exposure"] == pytest.approx(0.5)
    assert by_group.loc["Technology", "average_net_exposure"] == pytest.approx(0.4)
    assert by_group.loc["Technology", "max_abs_net_exposure"] == pytest.approx(0.5)
    assert by_group.loc["Technology", "average_holdings_count"] == pytest.approx(1.5)
    assert by_group.loc["Technology", "max_holdings_count"] == pytest.approx(2.0)

    assert by_group.loc["Consumer", "average_gross_exposure"] == pytest.approx(0.25)
    assert by_group.loc["Consumer", "average_net_exposure"] == pytest.approx(-0.25)
    assert bool(by_group.loc["<missing>", "is_missing_group"]) is True
    assert by_group.loc["<missing>", "average_gross_exposure"] == pytest.approx(0.1)
    assert by_group.loc["<missing>", "max_abs_net_exposure"] == pytest.approx(0.2)


def test_summarize_numeric_exposures_computes_target_weight_diagnostics() -> None:
    """Numeric exposure summary should surface weighted exposure and coverage."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-03",
                ]
            ),
            "symbol": ["AAPL", "MSFT", "TSLA", "AAPL", "MSFT", "TSLA"],
            "portfolio_weight": [0.50, -0.25, 0.0, 0.40, -0.30, 0.10],
            "style_beta": [1.2, 0.8, None, None, 1.1, 0.9],
        }
    )

    summary = summarize_numeric_exposures(
        frame,
        exposure_columns=["style_beta"],
        weight_column="portfolio_weight",
    )
    row = summary.set_index("exposure_column").loc["style_beta"]

    first_day_average = ((0.50 * 1.2) + (0.25 * 0.8)) / 0.75
    second_day_average = ((0.30 * 1.1) + (0.10 * 0.9)) / 0.40
    assert row["periods"] == pytest.approx(2.0)
    assert row["average_absolute_weighted_exposure"] == pytest.approx(
        (first_day_average + second_day_average) / 2.0
    )
    assert row["latest_absolute_weighted_exposure"] == pytest.approx(
        second_day_average
    )
    assert row["average_net_weighted_exposure"] == pytest.approx(
        (0.40 + -0.24) / 2.0
    )
    assert row["latest_net_weighted_exposure"] == pytest.approx(-0.24)
    assert row["average_gross_weight_with_exposure"] == pytest.approx(0.575)
    assert row["average_gross_weight_missing_exposure"] == pytest.approx(0.20)
    assert row["average_active_positions"] == pytest.approx(2.5)
    assert row["average_positions_with_exposure"] == pytest.approx(2.0)
    assert row["average_missing_exposure_positions"] == pytest.approx(0.5)


def test_summarize_numeric_exposures_rejects_missing_exposure_column() -> None:
    """Configured numeric exposure columns must exist in the weight panel."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "symbol": ["AAPL"],
            "portfolio_weight": [1.0],
        }
    )

    with pytest.raises(
        RiskError,
        match="weight panel is missing required columns: style_beta",
    ):
        summarize_numeric_exposures(frame, exposure_columns=["style_beta"])


def test_summarize_numeric_exposures_rejects_invalid_numeric_values() -> None:
    """Configured numeric exposure columns should fail fast on malformed values."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "symbol": ["AAPL"],
            "portfolio_weight": [1.0],
            "style_beta": ["not-a-number"],
        }
    )

    with pytest.raises(
        RiskError,
        match="risk inputs contain invalid numeric values in 'style_beta'",
    ):
        summarize_numeric_exposures(frame, exposure_columns=["style_beta"])


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

    with pytest.raises(RiskError, match="invalid numeric values"):
        summarize_portfolio_diversification(
            bad_weights,
            weight_column="effective_weight",
        )

    with pytest.raises(RiskError, match="group_column"):
        summarize_group_exposure(frame, group_column="")

    with pytest.raises(RiskError, match="required columns"):
        summarize_group_exposure(frame, group_column="classification_sector")

    bad_group_weights = frame.assign(
        symbol=["AAPL", "MSFT"],
        classification_sector="Technology",
        portfolio_weight=["bad", 0.1],
    )
    with pytest.raises(RiskError, match="invalid numeric values"):
        summarize_group_exposure(
            bad_group_weights,
            group_column="classification_sector",
            weight_column="portfolio_weight",
        )

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

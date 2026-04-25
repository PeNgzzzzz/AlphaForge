"""Tests for cross-sectional factor diagnostics."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from alphaforge.analytics import (
    FactorDiagnosticsError,
    compute_grouped_ic_series,
    compute_ic_decay_series,
    compute_ic_decay_summary,
    compute_ic_series,
    compute_quantile_bucket_returns,
    compute_quantile_spread_series,
    compute_rolling_ic_series,
    compute_signal_coverage_by_date,
    summarize_grouped_ic,
    summarize_ic,
    summarize_rolling_ic,
    summarize_signal_coverage,
)


def test_compute_ic_series_supports_pearson_and_spearman() -> None:
    """IC diagnostics should compute per-date cross-sectional correlations."""
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
            "momentum_signal_3d": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "forward_return_1d": [0.10, 0.20, 0.30, 0.30, 0.20, 0.10],
        }
    )

    pearson = compute_ic_series(
        frame,
        signal_column="momentum_signal_3d",
        forward_return_column="forward_return_1d",
        method="pearson",
    )
    spearman = compute_ic_series(
        frame,
        signal_column="momentum_signal_3d",
        forward_return_column="forward_return_1d",
        method="spearman",
    )

    assert pearson["ic"].tolist() == pytest.approx([1.0, -1.0])
    assert spearman["ic"].tolist() == pytest.approx([1.0, -1.0])
    assert pearson["observations"].tolist() == pytest.approx([3.0, 3.0])


def test_summarize_ic_reports_mean_spread_and_hit_rate() -> None:
    """IC summary should aggregate mean, dispersion, IR, and hit rate."""
    ic_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "ic": [1.0, -1.0],
            "observations": [3.0, 3.0],
        }
    )

    summary = summarize_ic(ic_frame)

    assert summary["periods"] == pytest.approx(2.0)
    assert summary["valid_periods"] == pytest.approx(2.0)
    assert summary["mean_ic"] == pytest.approx(0.0)
    assert summary["ic_std"] == pytest.approx(math.sqrt(2.0))
    assert summary["ic_ir"] == pytest.approx(0.0)
    assert summary["positive_ic_ratio"] == pytest.approx(0.5)
    assert summary["average_observations"] == pytest.approx(3.0)


def test_compute_rolling_ic_series_uses_trailing_valid_ic_observations() -> None:
    """Rolling IC should use only current and prior IC observations."""
    ic_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-08",
                ]
            ),
            "ic": [0.10, 0.20, -0.10, math.nan, 0.30],
            "observations": [5.0, 5.0, 5.0, 1.0, 5.0],
        }
    )

    rolling = compute_rolling_ic_series(ic_frame, window=3, min_periods=2)

    assert pd.isna(rolling.loc[0, "rolling_mean_ic"])
    assert rolling["rolling_mean_ic"].iloc[1:].tolist() == pytest.approx(
        [0.15, 0.0666666667, 0.05, 0.10]
    )
    assert rolling["rolling_positive_ic_ratio"].iloc[1:].tolist() == pytest.approx(
        [1.0, 2.0 / 3.0, 0.5, 0.5]
    )
    assert rolling["rolling_valid_periods"].iloc[1:].tolist() == pytest.approx(
        [2.0, 3.0, 2.0, 2.0]
    )


def test_summarize_rolling_ic_reports_latest_and_range() -> None:
    """Rolling IC summary should expose the latest trailing diagnostics."""
    rolling = compute_rolling_ic_series(
        pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-02", "2024-01-03", "2024-01-04"]
                ),
                "ic": [0.10, 0.20, -0.10],
            }
        ),
        window=2,
    )

    summary = summarize_rolling_ic(rolling)

    assert summary["periods"] == pytest.approx(3.0)
    assert summary["valid_periods"] == pytest.approx(2.0)
    assert summary["window"] == pytest.approx(2.0)
    assert summary["min_periods"] == pytest.approx(2.0)
    assert summary["latest_date"] == pd.Timestamp("2024-01-04")
    assert summary["latest_rolling_mean_ic"] == pytest.approx(0.05)
    assert summary["latest_rolling_positive_ic_ratio"] == pytest.approx(0.5)
    assert summary["minimum_rolling_mean_ic"] == pytest.approx(0.05)
    assert summary["maximum_rolling_mean_ic"] == pytest.approx(0.15)


def test_compute_ic_decay_summary_reports_configured_horizons() -> None:
    """IC decay should summarize each configured forward-return label."""
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
            "symbol": ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"],
            "signal": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "forward_return_1d": [0.01, 0.02, 0.03, 0.01, 0.03, 0.02],
            "forward_return_5d": [0.03, 0.02, 0.01, 0.03, 0.01, 0.02],
        }
    )

    decay = compute_ic_decay_summary(
        frame,
        signal_column="signal",
        forward_return_columns=["forward_return_1d", "forward_return_5d"],
        method="pearson",
        min_observations=2,
    )

    assert decay["forward_return_column"].tolist() == [
        "forward_return_1d",
        "forward_return_5d",
    ]
    assert decay["horizon"].tolist() == pytest.approx([1.0, 5.0])
    assert decay["valid_periods"].tolist() == pytest.approx([2.0, 2.0])
    assert decay["mean_ic"].tolist() == pytest.approx([0.75, -0.75])
    assert decay["positive_ic_ratio"].tolist() == pytest.approx([1.0, 0.0])


def test_compute_ic_decay_series_reports_per_date_horizon_ic() -> None:
    """IC decay series should expose each configured horizon through time."""
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
            "symbol": ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"],
            "signal": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "forward_return_1d": [0.01, 0.02, 0.03, 0.01, 0.03, 0.02],
            "forward_return_5d": [0.03, 0.02, 0.01, 0.03, 0.01, 0.02],
        }
    )

    decay_series = compute_ic_decay_series(
        frame,
        signal_column="signal",
        forward_return_columns=["forward_return_1d", "forward_return_5d"],
        method="pearson",
        min_observations=2,
    )

    assert decay_series["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-02",
        "2024-01-03",
        "2024-01-03",
    ]
    assert decay_series["forward_return_column"].tolist() == [
        "forward_return_1d",
        "forward_return_5d",
        "forward_return_1d",
        "forward_return_5d",
    ]
    assert decay_series["horizon"].tolist() == pytest.approx([1.0, 5.0, 1.0, 5.0])
    assert decay_series["order"].tolist() == pytest.approx([0.0, 1.0, 0.0, 1.0])
    assert decay_series["observations"].tolist() == pytest.approx([3.0, 3.0, 3.0, 3.0])
    assert decay_series["ic"].tolist() == pytest.approx([1.0, -1.0, 0.5, -0.5])
    assert decay_series["method"].tolist() == ["pearson"] * 4


def test_compute_grouped_ic_series_reports_per_date_group_ic() -> None:
    """Grouped IC should compute same-date cross-sectional IC inside each group."""
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
                    "2024-01-03",
                ]
            ),
            "symbol": ["A", "B", "C", "D", "A", "B", "C", "D", "E"],
            "sector": [
                "Tech",
                "Tech",
                "Energy",
                "Energy",
                "Tech",
                "Tech",
                "Energy",
                "Energy",
                None,
            ],
            "signal": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 9.0],
            "forward_return_1d": [
                0.01,
                0.02,
                0.02,
                0.01,
                0.01,
                0.03,
                0.03,
                0.01,
                0.99,
            ],
        }
    )

    grouped = compute_grouped_ic_series(
        frame,
        signal_column="signal",
        forward_return_column="forward_return_1d",
        group_column="sector",
        method="pearson",
        min_observations=2,
    )

    assert grouped["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-02",
        "2024-01-03",
        "2024-01-03",
    ]
    assert grouped["group_value"].tolist() == ["Energy", "Tech", "Energy", "Tech"]
    assert grouped["ic"].tolist() == pytest.approx([-1.0, 1.0, -1.0, 1.0])
    assert grouped["observations"].tolist() == pytest.approx([2.0, 2.0, 2.0, 2.0])
    assert grouped["group_value"].isna().sum() == 0


def test_summarize_grouped_ic_reports_group_level_summary() -> None:
    """Grouped IC summary should aggregate per-date grouped IC series."""
    grouped = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"]
            ),
            "group_column": ["sector", "sector", "sector", "sector"],
            "group_value": ["Energy", "Energy", "Tech", "Tech"],
            "ic": [-1.0, -0.5, 1.0, 0.5],
            "observations": [3.0, 2.0, 3.0, 2.0],
            "method": ["pearson", "pearson", "pearson", "pearson"],
        }
    )

    summary = summarize_grouped_ic(grouped)

    assert summary["group_value"].tolist() == ["Energy", "Tech"]
    assert summary["periods"].tolist() == pytest.approx([2.0, 2.0])
    assert summary["valid_periods"].tolist() == pytest.approx([2.0, 2.0])
    assert summary["mean_ic"].tolist() == pytest.approx([-0.75, 0.75])
    assert summary["average_observations"].tolist() == pytest.approx([2.5, 2.5])


def test_compute_quantile_bucket_returns_aggregates_daily_bucket_means() -> None:
    """Quantile analysis should aggregate per-date bucket returns across time."""
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
            "momentum_signal_3d": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            "forward_return_1d": [0.01, 0.02, 0.03, 0.04, 0.02, 0.01, 0.04, 0.03],
        }
    )

    summary = compute_quantile_bucket_returns(
        frame,
        signal_column="momentum_signal_3d",
        forward_return_column="forward_return_1d",
        n_quantiles=2,
    )

    assert summary["quantile"].tolist() == [1, 2]
    assert summary.loc[summary["quantile"] == 1, "mean_forward_return"].iloc[0] == pytest.approx(0.015)
    assert summary.loc[summary["quantile"] == 2, "mean_forward_return"].iloc[0] == pytest.approx(0.035)
    assert summary.loc[summary["quantile"] == 1, "mean_signal"].iloc[0] == pytest.approx(1.5)
    assert summary.loc[summary["quantile"] == 2, "mean_signal"].iloc[0] == pytest.approx(3.5)
    assert summary.loc[summary["quantile"] == 1, "average_count"].iloc[0] == pytest.approx(2.0)
    assert summary.loc[summary["quantile"] == 2, "periods"].iloc[0] == pytest.approx(2.0)


def test_summarize_signal_coverage_counts_jointly_usable_rows() -> None:
    """Coverage should reflect non-null signal and forward return overlap."""
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
            "momentum_signal_3d": [1.0, None, 2.0, 3.0],
            "forward_return_1d": [0.01, 0.02, None, 0.04],
        }
    )

    summary = summarize_signal_coverage(
        frame,
        signal_column="momentum_signal_3d",
        forward_return_column="forward_return_1d",
    )

    assert summary["dates"] == pytest.approx(2.0)
    assert summary["total_rows"] == pytest.approx(4.0)
    assert summary["signal_non_null_rows"] == pytest.approx(3.0)
    assert summary["forward_return_non_null_rows"] == pytest.approx(3.0)
    assert summary["usable_rows"] == pytest.approx(2.0)
    assert summary["signal_coverage_ratio"] == pytest.approx(0.75)
    assert summary["forward_return_coverage_ratio"] == pytest.approx(0.75)
    assert summary["joint_coverage_ratio"] == pytest.approx(0.50)
    assert summary["average_daily_usable_rows"] == pytest.approx(1.0)
    assert summary["minimum_daily_usable_rows"] == pytest.approx(1.0)


def test_compute_signal_coverage_by_date_reports_daily_ratios() -> None:
    """Daily coverage diagnostics should retain per-date counts and ratios."""
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
            "momentum_signal_3d": [1.0, None, 2.0, 3.0],
            "forward_return_1d": [0.01, 0.02, None, 0.04],
        }
    )

    coverage_by_date = compute_signal_coverage_by_date(
        frame,
        signal_column="momentum_signal_3d",
        forward_return_column="forward_return_1d",
    )

    assert coverage_by_date["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-03",
    ]
    assert coverage_by_date["usable_rows"].tolist() == pytest.approx([1.0, 1.0])
    assert coverage_by_date["joint_coverage_ratio"].tolist() == pytest.approx([0.5, 0.5])


def test_compute_quantile_spread_series_reports_top_bottom_spread() -> None:
    """Quantile spread diagnostics should keep one top-bottom spread per date."""
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
            "momentum_signal_3d": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            "forward_return_1d": [0.01, 0.02, 0.03, 0.04, 0.02, 0.01, 0.04, 0.03],
        }
    )

    spread_series = compute_quantile_spread_series(
        frame,
        signal_column="momentum_signal_3d",
        forward_return_column="forward_return_1d",
        n_quantiles=2,
    )

    assert spread_series["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-03",
    ]
    assert spread_series["top_bottom_spread"].tolist() == pytest.approx([0.02, 0.02])
    assert spread_series["bottom_quantile"].tolist() == pytest.approx([1.0, 1.0])
    assert spread_series["top_quantile"].tolist() == pytest.approx([2.0, 2.0])


def test_factor_diagnostics_validate_inputs() -> None:
    """Invalid diagnostics inputs should fail loudly."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "signal": [1.0, 2.0],
            "forward_return_1d": [0.01, 0.02],
        }
    )

    with pytest.raises(FactorDiagnosticsError, match="required columns"):
        compute_ic_series(
            frame.drop(columns=["signal"]),
            signal_column="signal",
            forward_return_column="forward_return_1d",
        )

    with pytest.raises(FactorDiagnosticsError, match="method"):
        compute_ic_series(
            frame,
            signal_column="signal",
            forward_return_column="forward_return_1d",
            method="kendall",
        )

    with pytest.raises(FactorDiagnosticsError, match="n_quantiles"):
        compute_quantile_bucket_returns(
            frame,
            signal_column="signal",
            forward_return_column="forward_return_1d",
            n_quantiles=1,
        )

    with pytest.raises(FactorDiagnosticsError, match="min_observations must be greater than or equal to n_quantiles"):
        compute_quantile_bucket_returns(
            frame,
            signal_column="signal",
            forward_return_column="forward_return_1d",
            n_quantiles=3,
            min_observations=2,
        )

    bad_frame = frame.copy()
    bad_frame["signal"] = bad_frame["signal"].astype("object")
    bad_frame.loc[0, "signal"] = "bad"
    with pytest.raises(FactorDiagnosticsError, match="invalid numeric values"):
        summarize_signal_coverage(
            bad_frame,
            signal_column="signal",
            forward_return_column="forward_return_1d",
        )

    with pytest.raises(FactorDiagnosticsError, match="required columns"):
        summarize_ic(pd.DataFrame({"ic": [1.0]}))

    with pytest.raises(FactorDiagnosticsError, match="window"):
        compute_rolling_ic_series(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                    "ic": [0.1, 0.2],
                }
            ),
            window=1,
        )

    with pytest.raises(FactorDiagnosticsError, match="min_periods"):
        compute_rolling_ic_series(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                    "ic": [0.1, 0.2],
                }
            ),
            window=2,
            min_periods=3,
        )

    with pytest.raises(FactorDiagnosticsError, match="forward_return_columns"):
        compute_ic_decay_summary(
            frame,
            signal_column="signal",
            forward_return_columns=[],
        )

    with pytest.raises(FactorDiagnosticsError, match="duplicate"):
        compute_ic_decay_summary(
            frame,
            signal_column="signal",
            forward_return_columns=["forward_return_1d", "forward_return_1d"],
        )

    with pytest.raises(FactorDiagnosticsError, match="required columns"):
        compute_ic_decay_summary(
            frame,
            signal_column="signal",
            forward_return_columns=["forward_return_5d"],
        )

    with pytest.raises(FactorDiagnosticsError, match="required columns"):
        compute_grouped_ic_series(
            frame,
            signal_column="signal",
            forward_return_column="forward_return_1d",
            group_column="sector",
        )

    with pytest.raises(FactorDiagnosticsError, match="non-empty string"):
        compute_grouped_ic_series(
            frame.assign(sector="Tech"),
            signal_column="signal",
            forward_return_column="forward_return_1d",
            group_column=" ",
        )

    with pytest.raises(FactorDiagnosticsError, match="distinct"):
        compute_grouped_ic_series(
            frame,
            signal_column="signal",
            forward_return_column="forward_return_1d",
            group_column="signal",
        )

    with pytest.raises(FactorDiagnosticsError, match="grouped_ic_frame"):
        summarize_grouped_ic(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-02"]),
                    "group_column": ["sector"],
                    "group_value": ["Tech"],
                    "ic": [0.5],
                }
            )
        )

"""Tests for static report chart rendering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from alphaforge.analytics import (
    VisualizationError,
    save_compare_summary_chart,
    save_coverage_summary_chart,
    save_coverage_timeseries_chart,
    save_drawdown_chart,
    save_exposure_turnover_chart,
    save_ic_cumulative_chart,
    save_ic_decay_chart,
    save_ic_series_chart,
    save_nav_overview_chart,
    save_quantile_bucket_chart,
    save_quantile_cumulative_chart,
    save_quantile_spread_chart,
    save_rolling_benchmark_risk_chart,
)


def test_save_nav_overview_chart_writes_png(tmp_path: Path) -> None:
    """The NAV chart should write a non-empty PNG file."""
    backtest = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "gross_nav": [1.00, 1.02, 1.03],
            "net_nav": [1.00, 1.01, 1.02],
            "benchmark_nav": [1.00, 1.00, 1.01],
            "relative_nav": [1.00, 1.01, 1.01],
        }
    )

    output_path = save_nav_overview_chart(backtest, tmp_path / "nav.png")

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_drawdown_and_exposure_turnover_charts_write_pngs(tmp_path: Path) -> None:
    """Drawdown and exposure/turnover charts should render from backtest output."""
    backtest = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "net_return": [0.00, -0.01, 0.02],
            "gross_exposure": [1.0, 0.8, 0.9],
            "net_exposure": [1.0, 0.8, 0.9],
            "turnover": [0.0, 0.2, 0.1],
            "target_turnover": [0.0, 0.3, 0.15],
        }
    )

    drawdown_path = save_drawdown_chart(backtest, tmp_path / "drawdown.png")
    exposure_turnover_path = save_exposure_turnover_chart(
        backtest,
        tmp_path / "exposure_turnover.png",
    )

    assert drawdown_path.exists()
    assert drawdown_path.stat().st_size > 0
    assert exposure_turnover_path.exists()
    assert exposure_turnover_path.stat().st_size > 0


def test_save_ic_quantile_and_benchmark_risk_charts_write_pngs(tmp_path: Path) -> None:
    """Factor and benchmark diagnostics charts should render to disk."""
    ic_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "ic": [0.5, -0.1, 0.3],
            "observations": [4.0, 4.0, 4.0],
        }
    )
    quantile_summary = pd.DataFrame(
        {
            "quantile": [1, 2],
            "mean_forward_return": [-0.01, 0.02],
        }
    )
    benchmark_risk = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "rolling_beta": [None, 0.3, 0.4],
            "rolling_correlation": [None, 0.2, 0.6],
        }
    )

    ic_path = save_ic_series_chart(ic_frame, tmp_path / "ic.png")
    quantile_path = save_quantile_bucket_chart(quantile_summary, tmp_path / "quantile.png")
    benchmark_risk_path = save_rolling_benchmark_risk_chart(
        benchmark_risk,
        tmp_path / "benchmark_risk.png",
    )
    ic_cumulative_path = save_ic_cumulative_chart(ic_frame, tmp_path / "ic_cumulative.png")
    ic_decay_path = save_ic_decay_chart(
        pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2024-01-02",
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-03",
                    ]
                ),
                "horizon": [1.0, 5.0, 1.0, 5.0],
                "ic": [0.5, 0.2, -0.1, 0.3],
                "observations": [4.0, 4.0, 4.0, 4.0],
            }
        ),
        tmp_path / "ic_decay.png",
    )
    quantile_spread_path = save_quantile_spread_chart(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "top_bottom_spread": [0.02, -0.01],
            }
        ),
        tmp_path / "quantile_spread.png",
    )
    quantile_cumulative_path = save_quantile_cumulative_chart(
        pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
                ),
                "quantile": [1, 2, 1, 2],
                "cumulative_forward_return": [0.01, 0.02, 0.03, 0.04],
            }
        ),
        tmp_path / "quantile_cumulative.png",
    )

    assert ic_path.exists()
    assert ic_path.stat().st_size > 0
    assert ic_cumulative_path.exists()
    assert ic_cumulative_path.stat().st_size > 0
    assert ic_decay_path.exists()
    assert ic_decay_path.stat().st_size > 0
    assert quantile_path.exists()
    assert quantile_path.stat().st_size > 0
    assert quantile_spread_path.exists()
    assert quantile_spread_path.stat().st_size > 0
    assert quantile_cumulative_path.exists()
    assert quantile_cumulative_path.stat().st_size > 0
    assert benchmark_risk_path.exists()
    assert benchmark_risk_path.stat().st_size > 0


def test_save_coverage_summary_chart_rejects_missing_fields(tmp_path: Path) -> None:
    """Coverage chart input should fail loudly when required fields are missing."""
    summary = pd.Series({"signal_coverage_ratio": 0.5, "joint_coverage_ratio": 0.4})

    with pytest.raises(VisualizationError, match="missing required fields"):
        save_coverage_summary_chart(summary, tmp_path / "coverage.png")


def test_save_ic_decay_chart_rejects_missing_fields(tmp_path: Path) -> None:
    """IC decay chart input should fail loudly when required fields are missing."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "horizon": [1.0],
            "ic": [0.5],
        }
    )

    with pytest.raises(VisualizationError, match="missing required columns"):
        save_ic_decay_chart(frame, tmp_path / "ic_decay.png")


def test_save_coverage_timeseries_and_compare_summary_charts_write_pngs(
    tmp_path: Path,
) -> None:
    """Coverage-through-time and compare summary charts should render to disk."""
    coverage_by_date = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "signal_coverage_ratio": [0.5, 1.0],
            "forward_return_coverage_ratio": [1.0, 0.5],
            "joint_coverage_ratio": [0.5, 0.5],
        }
    )
    compare_summary = pd.DataFrame(
        {
            "run_id": ["run-a", "run-b"],
            "command": ["sweep-signal", "walk-forward-signal"],
            "summary_cumulative_return": [0.10, 0.05],
            "summary_sharpe_ratio": [1.2, 0.8],
            "summary_mean_ic": [0.3, 0.2],
        }
    )

    coverage_path = save_coverage_timeseries_chart(
        coverage_by_date,
        tmp_path / "coverage_timeseries.png",
    )
    compare_path = save_compare_summary_chart(
        compare_summary,
        tmp_path / "compare_summary.png",
    )

    assert coverage_path.exists()
    assert coverage_path.stat().st_size > 0
    assert compare_path.exists()
    assert compare_path.stat().st_size > 0

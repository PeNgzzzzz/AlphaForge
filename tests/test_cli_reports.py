"""Tests for CLI report rendering helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alphaforge.cli.reports import (
    build_report_metadata,
    render_report_text,
    write_report_html_page,
)
from alphaforge.common import load_pipeline_config


def _minimal_report_context() -> dict[str, object]:
    """Build a compact report context for renderer and metadata tests."""
    market_data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": ["AAPL", "AAPL"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [1000.0, 1100.0],
        }
    )
    context = {
        "market_data": market_data,
        "benchmark_data": None,
        "dataset": market_data.copy(),
        "signal_column": "momentum_2d",
        "portfolio_diversification_summary": pd.Series(
            {"average_effective_holdings": 1.0}
        ),
        "portfolio_group_exposure_summary": pd.DataFrame(),
        "portfolio_numeric_exposure_summary": pd.DataFrame(),
        "backtest": pd.DataFrame(
            {
                "is_rebalance_date": [True, False],
                "turnover_limit_applied": [False, False],
                "target_turnover": [1.0, 0.0],
                "turnover": [1.0, 0.0],
                "gross_target_exposure": [1.0, 1.0],
                "gross_exposure": [1.0, 1.0],
                "target_holdings_count": [1.0, 1.0],
                "holdings_count": [1.0, 1.0],
                "target_effective_weight_gap": [0.0, 0.0],
                "commission_cost": [0.0005, 0.0],
                "slippage_cost": [0.0, 0.0],
                "transaction_cost": [0.0005, 0.0],
            }
        ),
        "performance_summary": pd.Series(
            {
                "periods": 2,
                "cumulative_return": 0.02,
                "annualized_return": 0.10,
                "annualized_volatility": 0.15,
                "sharpe_ratio": 0.67,
                "max_drawdown": -0.01,
                "average_turnover": 0.5,
                "total_turnover": 1.0,
                "hit_rate": 0.5,
            }
        ),
        "relative_performance_summary": None,
        "risk_summary": pd.Series(
            {
                "periods": 2,
                "realized_volatility": 0.15,
                "downside_volatility": 0.05,
                "value_at_risk": -0.02,
                "conditional_value_at_risk": -0.03,
                "var_confidence": 0.95,
                "average_gross_exposure": 1.0,
                "average_net_exposure": 1.0,
            }
        ),
        "benchmark_risk_summary": None,
        "ic_summary": pd.Series({"mean_ic": 0.10, "ic_ir": 0.25}),
        "rolling_ic_summary": pd.Series(
            {"latest_rolling_mean_ic": 0.12, "latest_rolling_ic_ir": 0.30}
        ),
        "ic_decay_summary": pd.DataFrame(
            {"forward_return_column": ["forward_return_1d"], "mean_ic": [0.10]}
        ),
        "ic_decay_series": pd.DataFrame(
            {"date": pd.to_datetime(["2024-01-02"]), "ic": [0.10]}
        ),
        "grouped_ic_summary": pd.DataFrame(),
        "grouped_ic_series": pd.DataFrame(),
        "grouped_coverage_summary": pd.DataFrame(),
        "grouped_coverage_by_date": pd.DataFrame(),
        "quantile_summary": pd.DataFrame(
            {"quantile": [1, 2], "mean_forward_return": [-0.01, 0.02]}
        ),
        "quantile_cumulative_returns": pd.DataFrame(),
        "quantile_spread_stability": pd.Series({"periods": 0}),
        "coverage_summary": pd.Series(
            {"joint_coverage_ratio": 1.0, "average_daily_usable_rows": 2.0}
        ),
    }
    return context


def test_render_report_text_renders_sections_from_precomputed_context() -> None:
    """Text reports should be assembled from an already computed workflow context."""
    config = load_pipeline_config(Path("configs/momentum_example.toml"))
    context = _minimal_report_context()

    report_text = render_report_text(context, config=config)

    assert "Research Workflow" in report_text
    assert "Data Summary" in report_text
    assert "Portfolio Constraints" in report_text
    assert "Execution Summary" in report_text
    assert "Performance Summary" in report_text
    assert "Risk Summary" in report_text
    assert "Diagnostics Overview" in report_text
    assert "Quantile Bucket Returns" in report_text


def test_build_report_metadata_uses_precomputed_snapshots_and_context() -> None:
    """Report metadata should use injected snapshots without rebuilding pipeline state."""
    config = load_pipeline_config(Path("configs/momentum_example.toml"))
    context = _minimal_report_context()
    workflow_configuration = {
        "data": {"path": "data.csv", "price_adjustment": "raw"},
        "diagnostics": {"rolling_ic_window": 3},
    }
    research_metadata = {
        "dataset_feature_metadata": [{"column": "forward_return_1d"}],
        "signal_pipeline_metadata": {"factor": {"name": "momentum"}},
        "feature_cache_metadata": {"materialization": "metadata_only"},
    }

    metadata = build_report_metadata(
        context,
        config=config,
        config_path="configs/momentum_example.toml",
        workflow_configuration=workflow_configuration,
        research_metadata=research_metadata,
    )

    assert metadata["command"] == "report"
    assert metadata["config"] == "configs/momentum_example.toml"
    assert metadata["workflow_configuration"] is workflow_configuration
    assert metadata["dataset_feature_metadata"] == [{"column": "forward_return_1d"}]
    assert "Research Workflow" in metadata["report_sections"]
    assert "Portfolio Diversification Summary" in metadata["report_sections"]
    assert "Benchmark Summary" not in metadata["report_sections"]
    assert metadata["data_summary"]["rows"] == 2
    assert metadata["diagnostics_overview"]["top_bottom_quantile_spread"] == 0.03
    assert metadata["ic_decay_series"]["rows"][0]["date"] == "2024-01-02T00:00:00"


def test_write_report_html_page_renders_cards_charts_and_escaped_report(
    tmp_path: Path,
) -> None:
    """HTML reports should render metadata cards, chart cards, and escaped text."""
    metadata = {
        "command": "report",
        "config": "configs/example.toml",
        "performance_summary": pd.Series(
            {
                "cumulative_return": 0.1234,
                "periods": 2,
            }
        ),
        "relative_performance_summary": None,
        "diagnostics_overview": {
            "joint_coverage_ratio": 0.75,
            "average_daily_usable_rows": 4.5,
        },
        "chart_bundle": {
            "charts": [
                {
                    "title": "NAV <Overview>",
                    "description": "Net & gross NAV.",
                    "filename": "nav_overview.png",
                }
            ],
        },
    }

    html_path = write_report_html_page(
        report_text="Research <Workflow> & assumptions",
        metadata=metadata,
        artifact_dir=tmp_path,
    )

    assert html_path == tmp_path / "index.html"
    html_text = html_path.read_text(encoding="utf-8")
    assert "AlphaForge Research Report" in html_text
    assert "Command: report" in html_text
    assert "Config: configs/example.toml" in html_text
    assert "cumulative_return" in html_text
    assert "12.34%" in html_text
    assert "periods" in html_text
    assert "2" in html_text
    assert "joint_coverage_ratio" in html_text
    assert "75.00%" in html_text
    assert "NAV &lt;Overview&gt;" in html_text
    assert "Net &amp; gross NAV." in html_text
    assert "charts/nav_overview.png" in html_text
    assert "Research &lt;Workflow&gt; &amp; assumptions" in html_text


def test_write_report_html_page_handles_missing_charts(tmp_path: Path) -> None:
    """Report pages should remain useful when no chart entries are provided."""
    html_path = write_report_html_page(
        report_text="Plain report",
        metadata={"command": "report", "config": ""},
        artifact_dir=tmp_path,
    )

    html_text = html_path.read_text(encoding="utf-8")
    assert "No charts were generated for this artifact." in html_text
    assert "Plain report" in html_text

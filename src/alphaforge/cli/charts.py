"""Chart bundle writing helpers for CLI workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from alphaforge.analytics import (
    save_compare_summary_chart,
    save_coverage_summary_chart,
    save_coverage_timeseries_chart,
    save_drawdown_chart,
    save_exposure_turnover_chart,
    save_grouped_coverage_summary_chart,
    save_grouped_coverage_timeseries_chart,
    save_grouped_ic_summary_chart,
    save_grouped_ic_timeseries_chart,
    save_ic_cumulative_chart,
    save_ic_decay_chart,
    save_ic_series_chart,
    save_nav_overview_chart,
    save_quantile_bucket_chart,
    save_quantile_cumulative_chart,
    save_quantile_spread_chart,
    save_rolling_benchmark_risk_chart,
)
from alphaforge.cli.artifacts import write_json
from alphaforge.cli.errors import WorkflowError

__all__ = [
    "write_report_chart_bundle_from_context",
    "write_compare_chart_bundle",
]


def write_report_chart_bundle_from_context(
    context: dict[str, Any],
    *,
    output_dir: str | Path,
    workflow_configuration: dict[str, Any],
    config_path: str | None,
    command_name: str,
) -> dict[str, Any]:
    """Write a stable set of static report charts plus a small manifest."""
    chart_dir = Path(output_dir)
    if chart_dir.exists() and not chart_dir.is_dir():
        raise WorkflowError(
            f"Chart output path must be a directory path, not an existing file: {chart_dir}"
        )

    try:
        chart_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise WorkflowError(f"Failed to create chart directory {chart_dir}: {exc}") from exc

    chart_entries: list[dict[str, Any]] = []
    backtest = context["backtest"]

    nav_path = save_nav_overview_chart(backtest, chart_dir / "nav_overview.png")
    chart_entries.append(
        _build_chart_manifest_entry(
            chart_id="nav_overview",
            title="NAV Overview",
            filename=nav_path.name,
            description="Strategy net/gross NAV plus benchmark-relative NAV when available.",
        )
    )

    drawdown_path = save_drawdown_chart(backtest, chart_dir / "drawdown.png")
    chart_entries.append(
        _build_chart_manifest_entry(
            chart_id="drawdown",
            title="Drawdown",
            filename=drawdown_path.name,
            description="Net-return drawdown path from the configured backtest.",
        )
    )

    exposure_turnover_path = save_exposure_turnover_chart(
        backtest,
        chart_dir / "exposure_turnover.png",
    )
    chart_entries.append(
        _build_chart_manifest_entry(
            chart_id="exposure_turnover",
            title="Exposure And Turnover",
            filename=exposure_turnover_path.name,
            description="Gross/net exposure plus target vs realized turnover.",
        )
    )

    ic_series_path = save_ic_series_chart(context["ic_series"], chart_dir / "ic_series.png")
    chart_entries.append(
        _build_chart_manifest_entry(
            chart_id="ic_series",
            title="IC Series",
            filename=ic_series_path.name,
            description="Per-date IC values and observation counts.",
        )
    )

    ic_cumulative_path = save_ic_cumulative_chart(
        context["ic_series"],
        chart_dir / "ic_cumulative.png",
    )
    chart_entries.append(
        _build_chart_manifest_entry(
            chart_id="ic_cumulative",
            title="Cumulative IC",
            filename=ic_cumulative_path.name,
            description="Cumulative sum of per-date IC values with missing periods treated as zero.",
        )
    )

    ic_decay_path = save_ic_decay_chart(
        context["ic_decay_series"],
        chart_dir / "ic_decay_series.png",
    )
    chart_entries.append(
        _build_chart_manifest_entry(
            chart_id="ic_decay_series",
            title="IC Decay Series",
            filename=ic_decay_path.name,
            description="Per-date IC across configured forward-return horizons.",
        )
    )

    coverage_summary_path = save_coverage_summary_chart(
        context["coverage_summary"],
        chart_dir / "coverage_summary.png",
    )
    chart_entries.append(
        _build_chart_manifest_entry(
            chart_id="coverage_summary",
            title="Coverage Summary",
            filename=coverage_summary_path.name,
            description="Signal, forward-return, and joint usable coverage ratios.",
        )
    )

    coverage_timeseries_path = save_coverage_timeseries_chart(
        context["coverage_by_date"],
        chart_dir / "coverage_timeseries.png",
    )
    chart_entries.append(
        _build_chart_manifest_entry(
            chart_id="coverage_timeseries",
            title="Coverage Through Time",
            filename=coverage_timeseries_path.name,
            description="Per-date signal, label, and joint usable coverage ratios.",
        )
    )

    grouped_ic_series = context["grouped_ic_series"]
    if not grouped_ic_series.empty:
        grouped_ic_timeseries_path = save_grouped_ic_timeseries_chart(
            grouped_ic_series,
            chart_dir / "grouped_ic_timeseries.png",
        )
        chart_entries.append(
            _build_chart_manifest_entry(
                chart_id="grouped_ic_timeseries",
                title="Grouped IC Through Time",
                filename=grouped_ic_timeseries_path.name,
                description="Per-date IC values and observation counts by configured group value.",
            )
        )

    grouped_ic_summary = context["grouped_ic_summary"]
    if not grouped_ic_summary.empty:
        grouped_ic_path = save_grouped_ic_summary_chart(
            grouped_ic_summary,
            chart_dir / "grouped_ic_summary.png",
        )
        chart_entries.append(
            _build_chart_manifest_entry(
                chart_id="grouped_ic_summary",
                title="Grouped IC Summary",
                filename=grouped_ic_path.name,
                description="Mean IC and valid periods by configured group value.",
            )
        )

    grouped_coverage_by_date = context["grouped_coverage_by_date"]
    if not grouped_coverage_by_date.empty:
        grouped_coverage_timeseries_path = save_grouped_coverage_timeseries_chart(
            grouped_coverage_by_date,
            chart_dir / "grouped_coverage_timeseries.png",
        )
        chart_entries.append(
            _build_chart_manifest_entry(
                chart_id="grouped_coverage_timeseries",
                title="Grouped Coverage Through Time",
                filename=grouped_coverage_timeseries_path.name,
                description=(
                    "Per-date joint usable coverage ratios and usable row counts "
                    "by configured group value."
                ),
            )
        )

    grouped_coverage_summary = context["grouped_coverage_summary"]
    if not grouped_coverage_summary.empty:
        grouped_coverage_path = save_grouped_coverage_summary_chart(
            grouped_coverage_summary,
            chart_dir / "grouped_coverage_summary.png",
        )
        chart_entries.append(
            _build_chart_manifest_entry(
                chart_id="grouped_coverage_summary",
                title="Grouped Coverage Summary",
                filename=grouped_coverage_path.name,
                description=(
                    "Signal, forward-return, and joint usable coverage ratios "
                    "by configured group value."
                ),
            )
        )

    quantile_summary = context["quantile_summary"]
    if not quantile_summary.empty:
        quantile_path = save_quantile_bucket_chart(
            quantile_summary,
            chart_dir / "quantile_bucket_returns.png",
        )
        chart_entries.append(
            _build_chart_manifest_entry(
                chart_id="quantile_bucket_returns",
                title="Quantile Bucket Returns",
                filename=quantile_path.name,
                description="Mean forward returns by within-date signal quantile.",
            )
        )

    quantile_cumulative_returns = context["quantile_cumulative_returns"]
    if not quantile_cumulative_returns.empty:
        quantile_cumulative_path = save_quantile_cumulative_chart(
            quantile_cumulative_returns,
            chart_dir / "quantile_cumulative_returns.png",
        )
        chart_entries.append(
            _build_chart_manifest_entry(
                chart_id="quantile_cumulative_returns",
                title="Cumulative Quantile Mean Forward Returns",
                filename=quantile_cumulative_path.name,
                description=(
                    "Diagnostic cumulative paths from per-date quantile mean "
                    "forward returns; not a portfolio backtest."
                ),
            )
        )

    quantile_spread_series = context["quantile_spread_series"]
    if not quantile_spread_series.empty:
        quantile_spread_path = save_quantile_spread_chart(
            quantile_spread_series,
            chart_dir / "quantile_spread_timeseries.png",
        )
        chart_entries.append(
            _build_chart_manifest_entry(
                chart_id="quantile_spread_timeseries",
                title="Top-Bottom Quantile Spread",
                filename=quantile_spread_path.name,
                description="Per-date top-minus-bottom quantile forward-return spread.",
            )
        )

    rolling_benchmark_risk = context["rolling_benchmark_risk"]
    if rolling_benchmark_risk is not None:
        benchmark_risk_path = save_rolling_benchmark_risk_chart(
            rolling_benchmark_risk,
            chart_dir / "benchmark_risk.png",
        )
        chart_entries.append(
            _build_chart_manifest_entry(
                chart_id="benchmark_risk",
                title="Benchmark Risk",
                filename=benchmark_risk_path.name,
                description="Rolling beta and rolling correlation versus the configured benchmark.",
            )
        )

    manifest = {
        "command": command_name,
        "config": config_path or "",
        "chart_count": len(chart_entries),
        "workflow_configuration": workflow_configuration,
        "charts": chart_entries,
    }
    manifest_path = write_json(manifest, chart_dir / "manifest.json")
    return {
        "chart_dir": chart_dir,
        "manifest_path": manifest_path,
        "chart_count": len(chart_entries),
        "charts": chart_entries,
    }


def write_compare_chart_bundle(
    results: pd.DataFrame,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write a static chart bundle for a compare-runs artifact."""
    chart_dir = Path(output_dir)
    if chart_dir.exists() and not chart_dir.is_dir():
        raise WorkflowError(
            f"Chart output path must be a directory path, not an existing file: {chart_dir}"
        )

    try:
        chart_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise WorkflowError(f"Failed to create chart directory {chart_dir}: {exc}") from exc

    compare_chart_path = save_compare_summary_chart(
        results,
        chart_dir / "compare_summary_metrics.png",
    )
    chart_entries = [
        _build_chart_manifest_entry(
            chart_id="compare_summary_metrics",
            title="Compare Summary Metrics",
            filename=compare_chart_path.name,
            description="Multi-run comparison across cumulative return, Sharpe ratio, and mean IC.",
        )
    ]
    manifest = {
        "command": "compare-runs",
        "chart_count": len(chart_entries),
        "charts": chart_entries,
    }
    manifest_path = write_json(manifest, chart_dir / "manifest.json")
    return {
        "chart_dir": chart_dir,
        "manifest_path": manifest_path,
        "chart_count": len(chart_entries),
        "charts": chart_entries,
    }


def _build_chart_manifest_entry(
    *,
    chart_id: str,
    title: str,
    filename: str,
    description: str,
) -> dict[str, Any]:
    """Build one JSON-safe manifest entry for a generated chart file."""
    return {
        "chart_id": chart_id,
        "title": title,
        "filename": filename,
        "path": filename,
        "description": description,
    }

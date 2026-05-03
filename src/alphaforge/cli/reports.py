"""Report rendering helpers for CLI workflows."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from alphaforge.analytics import (
    format_performance_summary,
    format_relative_performance_summary,
)
from alphaforge.cli.artifacts import write_text
from alphaforge.cli.errors import WorkflowError
from alphaforge.common import AlphaForgeConfig
from alphaforge.risk import (
    format_benchmark_risk_summary,
    format_risk_summary,
)

__all__ = [
    "describe_benchmark_configuration",
    "describe_benchmark_data",
    "describe_borrow_availability_configuration",
    "describe_borrow_availability_data",
    "describe_classifications_configuration",
    "describe_classifications_data",
    "describe_corporate_actions_configuration",
    "describe_corporate_actions_data",
    "describe_data_quality",
    "describe_diagnostics_overview",
    "describe_execution_configuration",
    "describe_execution_results",
    "describe_fundamentals_configuration",
    "describe_fundamentals_data",
    "describe_market_data",
    "describe_memberships_configuration",
    "describe_memberships_data",
    "describe_portfolio_constraints",
    "describe_research_workflow",
    "describe_shares_outstanding_configuration",
    "describe_shares_outstanding_data",
    "describe_symbol_metadata_configuration",
    "describe_symbol_metadata_data",
    "describe_trading_calendar_configuration",
    "describe_trading_calendar_data",
    "describe_trading_status_configuration",
    "describe_trading_status_data",
    "describe_universe_configuration",
    "describe_universe_eligibility",
    "build_report_metadata",
    "render_report_text",
    "write_report_html_page",
    "_format_number_or_nan",
    "_format_percent_or_nan",
    "_summarize_benchmark_data",
    "_summarize_borrow_availability_data",
    "_summarize_classifications_data",
    "_summarize_corporate_actions_data",
    "_summarize_data_quality",
    "_summarize_diagnostics_overview",
    "_summarize_execution_results",
    "_summarize_fundamentals_data",
    "_summarize_market_data",
    "_summarize_memberships_data",
    "_summarize_shares_outstanding_data",
    "_summarize_symbol_metadata_data",
    "_summarize_trading_calendar_data",
    "_summarize_trading_status_data",
    "_summarize_universe_eligibility",
]


def build_report_metadata(
    context: dict[str, Any],
    *,
    config: AlphaForgeConfig,
    config_path: str | None,
    workflow_configuration: dict[str, Any],
    research_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build report artifact metadata from a precomputed report context."""
    quantile_summary = context["quantile_summary"]
    diagnostics_overview = _summarize_diagnostics_overview(
        context["ic_summary"],
        context["rolling_ic_summary"],
        context["coverage_summary"],
        quantile_summary,
    )
    report_sections = [
        "Research Workflow",
        "Data Summary",
        "Data Quality Summary",
        "Benchmark Configuration" if config.benchmark is not None else None,
        "Benchmark Summary" if config.benchmark is not None else None,
        "Universe Rules" if config.universe is not None else None,
        "Universe Summary" if config.universe is not None else None,
        "Portfolio Constraints",
        "Portfolio Diversification Summary",
        (
            "Portfolio Group Exposure Summary"
            if not context["portfolio_group_exposure_summary"].empty
            else None
        ),
        (
            "Portfolio Numeric Exposure Summary"
            if not context["portfolio_numeric_exposure_summary"].empty
            else None
        ),
        "Execution Assumptions",
        "Execution Summary",
        "Performance Summary",
        (
            "Relative Performance Summary"
            if context["relative_performance_summary"] is not None
            else None
        ),
        "Risk Summary",
        (
            "Benchmark Risk Summary"
            if context["benchmark_risk_summary"] is not None
            else None
        ),
        "Diagnostics Overview",
        "IC Summary",
        "Rolling IC Summary",
        "IC Decay Summary",
        (
            "Grouped IC Summary"
            if not context["grouped_ic_summary"].empty
            else None
        ),
        (
            "Grouped Coverage Summary"
            if not context["grouped_coverage_summary"].empty
            else None
        ),
        "Quantile Bucket Returns",
        (
            "Cumulative Quantile Mean Forward Returns"
            if not context["quantile_cumulative_returns"].empty
            else None
        ),
        (
            "Quantile Spread Stability"
            if context["quantile_spread_stability"]["periods"] > 0
            else None
        ),
        "Coverage Summary",
    ]

    return {
        "command": "report",
        "config": config_path or "",
        "row_count": int(len(context["backtest"])),
        "report_sections": [section for section in report_sections if section],
        "workflow_configuration": workflow_configuration,
        **research_metadata,
        "data_summary": _summarize_market_data(context["market_data"]),
        "data_quality_summary": _summarize_data_quality(context["market_data"]),
        "benchmark_summary": (
            _summarize_benchmark_data(context["benchmark_data"])
            if context["benchmark_data"] is not None
            else None
        ),
        "universe_summary": _summarize_universe_eligibility(context["dataset"]),
        "performance_summary": _series_to_metadata_dict(context["performance_summary"]),
        "relative_performance_summary": (
            _series_to_metadata_dict(context["relative_performance_summary"])
            if context["relative_performance_summary"] is not None
            else None
        ),
        "risk_summary": _series_to_metadata_dict(context["risk_summary"]),
        "portfolio_diversification_summary": _series_to_metadata_dict(
            context["portfolio_diversification_summary"]
        ),
        "portfolio_group_exposure_summary": {
            "rows": _dataframe_records(context["portfolio_group_exposure_summary"]),
        },
        "portfolio_numeric_exposure_summary": {
            "rows": _dataframe_records(
                context["portfolio_numeric_exposure_summary"]
            ),
        },
        "benchmark_risk_summary": (
            _series_to_metadata_dict(context["benchmark_risk_summary"])
            if context["benchmark_risk_summary"] is not None
            else None
        ),
        "diagnostics_overview": diagnostics_overview,
        "ic_summary": _series_to_metadata_dict(context["ic_summary"]),
        "rolling_ic_summary": _series_to_metadata_dict(
            context["rolling_ic_summary"]
        ),
        "ic_decay_summary": {
            "rows": _dataframe_records(context["ic_decay_summary"]),
        },
        "ic_decay_series": {
            "rows": _dataframe_records(context["ic_decay_series"]),
        },
        "grouped_ic_summary": {
            "rows": _dataframe_records(context["grouped_ic_summary"]),
        },
        "grouped_ic_series": {
            "rows": _dataframe_records(context["grouped_ic_series"]),
        },
        "grouped_coverage_summary": {
            "rows": _dataframe_records(context["grouped_coverage_summary"]),
        },
        "grouped_coverage_by_date": {
            "rows": _dataframe_records(context["grouped_coverage_by_date"]),
        },
        "coverage_summary": _series_to_metadata_dict(context["coverage_summary"]),
        "quantile_bucket_summary": {
            "rows": _dataframe_records(quantile_summary),
            "top_bottom_spread": diagnostics_overview.get("top_bottom_quantile_spread"),
        },
        "quantile_cumulative_returns": {
            "rows": _dataframe_records(context["quantile_cumulative_returns"]),
        },
        "quantile_spread_stability": _series_to_metadata_dict(
            context["quantile_spread_stability"]
        ),
    }


def render_report_text(context: dict[str, Any], *, config: AlphaForgeConfig) -> str:
    """Render the final Stage 4 report text from a precomputed context."""
    quantile_summary = context["quantile_summary"]
    quantile_text = (
        quantile_summary.to_string(index=False)
        if not quantile_summary.empty
        else "No quantile buckets produced for the configured signal/label coverage."
    )
    quantile_cumulative_returns = context["quantile_cumulative_returns"]
    quantile_cumulative_text = (
        quantile_cumulative_returns.to_string(index=False)
        if not quantile_cumulative_returns.empty
        else ""
    )
    quantile_spread_stability = context["quantile_spread_stability"]
    quantile_spread_stability_text = (
        quantile_spread_stability.to_string()
        if quantile_spread_stability["periods"] > 0
        else ""
    )
    grouped_ic_summary = context["grouped_ic_summary"]
    grouped_ic_text = (
        grouped_ic_summary.to_string(index=False)
        if not grouped_ic_summary.empty
        else ""
    )
    grouped_coverage_summary = context["grouped_coverage_summary"]
    grouped_coverage_text = (
        grouped_coverage_summary.to_string(index=False)
        if not grouped_coverage_summary.empty
        else ""
    )
    portfolio_group_exposure_summary = context["portfolio_group_exposure_summary"]
    portfolio_group_exposure_text = (
        portfolio_group_exposure_summary.to_string(index=False)
        if not portfolio_group_exposure_summary.empty
        else ""
    )
    portfolio_numeric_exposure_summary = context["portfolio_numeric_exposure_summary"]
    portfolio_numeric_exposure_text = (
        portfolio_numeric_exposure_summary.to_string(index=False)
        if not portfolio_numeric_exposure_summary.empty
        else ""
    )
    portfolio_diversification_text = context[
        "portfolio_diversification_summary"
    ].to_string()

    sections = [
        describe_research_workflow(context, config=config),
        describe_market_data(context["market_data"]),
        describe_data_quality(context["market_data"]),
        describe_benchmark_configuration(config),
        describe_benchmark_data(context["benchmark_data"], config=config),
        describe_universe_configuration(config),
        describe_universe_eligibility(context["dataset"]),
        describe_portfolio_constraints(config),
        "Portfolio Diversification Summary\n" + portfolio_diversification_text,
        (
            "Portfolio Group Exposure Summary\n" + portfolio_group_exposure_text
            if portfolio_group_exposure_text
            else ""
        ),
        (
            "Portfolio Numeric Exposure Summary\n"
            + portfolio_numeric_exposure_text
            if portfolio_numeric_exposure_text
            else ""
        ),
        describe_execution_configuration(config),
        describe_execution_results(context["backtest"]),
        format_performance_summary(context["performance_summary"]),
        (
            format_relative_performance_summary(context["relative_performance_summary"])
            if context["relative_performance_summary"] is not None
            else ""
        ),
        format_risk_summary(context["risk_summary"]),
        (
            format_benchmark_risk_summary(
                context["benchmark_risk_summary"],
                window=config.benchmark.rolling_window,
            )
            if context["benchmark_risk_summary"] is not None
            else ""
        ),
        describe_diagnostics_overview(
            context["ic_summary"],
            context["rolling_ic_summary"],
            context["coverage_summary"],
            quantile_summary,
        ),
        "IC Summary\n" + context["ic_summary"].to_string(),
        "Rolling IC Summary\n" + context["rolling_ic_summary"].to_string(),
        "IC Decay Summary\n" + context["ic_decay_summary"].to_string(index=False),
        "Grouped IC Summary\n" + grouped_ic_text if grouped_ic_text else "",
        (
            "Grouped Coverage Summary\n" + grouped_coverage_text
            if grouped_coverage_text
            else ""
        ),
        "Quantile Bucket Returns\n" + quantile_text,
        (
            "Cumulative Quantile Mean Forward Returns\n" + quantile_cumulative_text
            if quantile_cumulative_text
            else ""
        ),
        (
            "Quantile Spread Stability\n" + quantile_spread_stability_text
            if quantile_spread_stability_text
            else ""
        ),
        "Coverage Summary\n" + context["coverage_summary"].to_string(),
    ]
    return "\n\n".join(section for section in sections if section)


def write_report_html_page(
    *,
    report_text: str,
    metadata: dict[str, Any],
    artifact_dir: Path,
) -> Path:
    """Write one self-contained HTML report page for a report artifact bundle."""
    chart_bundle = metadata.get("chart_bundle", {})
    chart_entries = chart_bundle.get("charts", []) if isinstance(chart_bundle, dict) else []
    performance_summary = metadata.get("performance_summary")
    relative_performance_summary = metadata.get("relative_performance_summary")
    diagnostics_overview = metadata.get("diagnostics_overview")
    sections = [
        _render_html_summary_block(
            "Performance Summary",
            performance_summary,
            formatter=_format_html_metric_value,
        ),
        _render_html_summary_block(
            "Relative Performance Summary",
            relative_performance_summary,
            formatter=_format_html_metric_value,
        ),
        _render_html_summary_block(
            "Diagnostics Overview",
            diagnostics_overview,
            formatter=_format_html_metric_value,
        ),
    ]
    cards_html = "".join(section for section in sections if section)
    charts_html = "".join(
        _render_report_chart_card(chart)
        for chart in chart_entries
        if isinstance(chart, dict)
    )
    title = "AlphaForge Research Report"
    config_path = escape(str(metadata.get("config", "")))
    command_name = escape(str(metadata.get("command", "report")))
    report_body = escape(report_text)

    html_text = "\n".join(
        [
            "<!DOCTYPE html>",
            "<html lang=\"en\">",
            "<head>",
            "  <meta charset=\"utf-8\">",
            f"  <title>{title}</title>",
            "  <style>",
            "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #F6F8FA; color: #102A43; }",
            "    main { max-width: 1200px; margin: 0 auto; padding: 32px 24px 56px; }",
            "    h1, h2, h3 { margin: 0 0 12px; }",
            "    p { margin: 0; line-height: 1.5; }",
            "    .hero { background: linear-gradient(135deg, #0B4F6C, #1F9D8B); color: white; border-radius: 18px; padding: 24px; margin-bottom: 24px; }",
            "    .meta { margin-top: 8px; opacity: 0.88; }",
            "    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 24px; }",
            "    .card { background: white; border-radius: 16px; padding: 18px; box-shadow: 0 8px 24px rgba(16, 42, 67, 0.08); }",
            "    .metrics { display: grid; grid-template-columns: 1fr auto; row-gap: 8px; column-gap: 16px; }",
            "    .charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; margin-bottom: 24px; }",
            "    .chart-card img { width: 100%; height: auto; border-radius: 12px; background: #FFF; }",
            "    pre { white-space: pre-wrap; word-break: break-word; background: #0F172A; color: #E2E8F0; border-radius: 16px; padding: 20px; overflow-x: auto; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <main>",
            "    <section class=\"hero\">",
            f"      <h1>{title}</h1>",
            f"      <p class=\"meta\">Command: {command_name}</p>",
            f"      <p class=\"meta\">Config: {config_path}</p>",
            "    </section>",
            (
                f"    <section class=\"grid\">{cards_html}</section>"
                if cards_html
                else ""
            ),
            "    <section>",
            "      <h2>Chart Gallery</h2>",
            (
                f"      <div class=\"charts\">{charts_html}</div>"
                if charts_html
                else "      <p>No charts were generated for this artifact.</p>"
            ),
            "    </section>",
            "    <section>",
            "      <h2>Text Report</h2>",
            f"      <pre>{report_body}</pre>",
            "    </section>",
            "  </main>",
            "</body>",
            "</html>",
        ]
    )
    return write_text(html_text, artifact_dir / "index.html")


def describe_market_data(frame: pd.DataFrame) -> str:
    """Render a concise market data summary."""
    summary = _summarize_market_data(frame)
    return "\n".join(
        [
            "Data Summary",
            f"Rows: {summary['rows']}",
            f"Symbols: {summary['symbols']}",
            f"Trading Dates: {summary['trading_dates']}",
            f"Date Range: {summary['start_date']} -> {summary['end_date']}",
        ]
    )


def describe_research_workflow(
    context: dict[str, Any],
    *,
    config: AlphaForgeConfig,
) -> str:
    """Render a concise research-workflow overview for interview-ready reports."""
    signal = _require_signal_config(config)
    portfolio = _require_portfolio_config(config)
    backtest = _require_backtest_config(config)

    if signal.name == "momentum":
        signal_text = f"momentum (lookback={signal.lookback})"
    elif signal.name == "mean_reversion":
        signal_text = f"mean_reversion (lookback={signal.lookback})"
    else:
        signal_text = (
            f"trend (short_window={signal.short_window}, long_window={signal.long_window})"
        )
    signal_transform_text = _describe_signal_transform(signal)

    portfolio_text = (
        f"{portfolio.construction}, top_n={portfolio.top_n}, weighting={portfolio.weighting}"
    )
    if portfolio.construction == "long_short":
        bottom_n = portfolio.bottom_n if portfolio.bottom_n is not None else portfolio.top_n
        portfolio_text += f", bottom_n={bottom_n}"
    if portfolio.max_group_weight is not None:
        portfolio_text += (
            f", group_column={portfolio.group_column}, "
            f"max_group_weight={portfolio.max_group_weight}"
        )
    if portfolio.position_cap_column is not None:
        portfolio_text += f", position_cap_column={portfolio.position_cap_column}"
    if portfolio.factor_exposure_bounds:
        columns = ", ".join(bound.column for bound in portfolio.factor_exposure_bounds)
        portfolio_text += f", factor_exposure_bounds=[{columns}]"

    benchmark_text = config.benchmark.name if config.benchmark is not None else "None"
    universe_text = "enabled" if config.universe is not None else "disabled"

    return "\n".join(
        [
            "Research Workflow",
            f"Market Data File: {config.data.path.name}",
            f"Price Adjustment: {config.data.price_adjustment}",
            f"Benchmark: {benchmark_text}",
            f"Signal: {signal_text}",
            f"Signal Transform: {signal_transform_text}",
            f"Signal Column: {context['signal_column']}",
            f"Label Column: {config.diagnostics.forward_return_column}",
            f"Universe Filters: {universe_text}",
            f"Portfolio: {portfolio_text}",
            f"Rebalance Frequency: {backtest.rebalance_frequency}",
        ]
    )


def describe_data_quality(frame: pd.DataFrame) -> str:
    """Render a concise market-data quality summary for reports and validation."""
    summary = _summarize_data_quality(frame)
    return "\n".join(
        [
            "Data Quality Summary",
            "Rows Per Symbol (min/avg/max): "
            f"{summary['rows_per_symbol_min']}/"
            f"{summary['rows_per_symbol_avg']:.2f}/"
            f"{summary['rows_per_symbol_max']}",
            "Symbols Per Date (min/avg/max): "
            f"{summary['symbols_per_date_min']}/"
            f"{summary['symbols_per_date_avg']:.2f}/"
            f"{summary['symbols_per_date_max']}",
            f"Close Range: {summary['close_min']:.2f} -> {summary['close_max']:.2f}",
            f"Volume Range: {summary['volume_min']:.2f} -> {summary['volume_max']:.2f}",
        ]
    )


def describe_diagnostics_overview(
    ic_summary: pd.Series,
    rolling_ic_summary: pd.Series,
    coverage_summary: pd.Series,
    quantile_summary: pd.DataFrame,
) -> str:
    """Render a compact diagnostics overview ahead of the raw tables."""
    summary = _summarize_diagnostics_overview(
        ic_summary,
        rolling_ic_summary,
        coverage_summary,
        quantile_summary,
    )
    lines = [
        "Diagnostics Overview",
        f"Mean IC: {summary['mean_ic']:.2f}" if summary["mean_ic"] is not None else "Mean IC: NaN",
        f"IC IR: {summary['ic_ir']:.2f}" if summary["ic_ir"] is not None else "IC IR: NaN",
        "Latest Rolling Mean IC: "
        f"{summary['latest_rolling_mean_ic']:.2f}"
        if summary["latest_rolling_mean_ic"] is not None
        else "Latest Rolling Mean IC: NaN",
        "Latest Rolling IC IR: "
        f"{summary['latest_rolling_ic_ir']:.2f}"
        if summary["latest_rolling_ic_ir"] is not None
        else "Latest Rolling IC IR: NaN",
        "Joint Coverage Ratio: "
        f"{summary['joint_coverage_ratio']:.2%}"
        if summary["joint_coverage_ratio"] is not None
        else "Joint Coverage Ratio: NaN",
        "Average Daily Usable Rows: "
        f"{summary['average_daily_usable_rows']:.2f}"
        if summary["average_daily_usable_rows"] is not None
        else "Average Daily Usable Rows: NaN",
    ]
    top_bottom_spread = summary["top_bottom_quantile_spread"]
    if top_bottom_spread is not None:
        lines.append(f"Top-Bottom Quantile Spread: {top_bottom_spread:.2%}")
    return "\n".join(lines)


def describe_universe_configuration(config: AlphaForgeConfig) -> str:
    """Render the configured tradability-aware universe rules."""
    if config.universe is None:
        return ""

    universe = config.universe
    lines = [
        "Universe Rules",
        f"Lag: {universe.lag} day(s)",
    ]
    if universe.min_price is not None:
        lines.append(f"Minimum Price: {universe.min_price}")
    if universe.min_average_volume is not None:
        average_volume_window = (
            universe.average_volume_window or config.dataset.average_volume_window
        )
        lines.append(
            f"Minimum Average Volume ({average_volume_window}d): {universe.min_average_volume}"
        )
    if universe.min_average_dollar_volume is not None:
        average_dollar_volume_window = (
            universe.average_dollar_volume_window
            or config.dataset.average_volume_window
        )
        lines.append(
            "Minimum Average Dollar Volume "
            f"({average_dollar_volume_window}d): {universe.min_average_dollar_volume}"
        )
    if universe.min_listing_history_days is not None:
        lines.append(
            f"Minimum Listing History: {universe.min_listing_history_days} day(s)"
        )
    if universe.required_membership_indexes:
        lines.append(
            "Required Membership Indexes: "
            + ", ".join(universe.required_membership_indexes)
        )
    if universe.require_tradable:
        lines.append("Require Tradable Status: true")
    return "\n".join(lines)


def describe_benchmark_configuration(config: AlphaForgeConfig) -> str:
    """Render the configured benchmark settings."""
    if config.benchmark is None:
        return ""

    benchmark = config.benchmark
    return "\n".join(
        [
            "Benchmark Configuration",
            f"Name: {benchmark.name}",
            f"Return Column: {benchmark.return_column}",
            f"Rolling Window: {benchmark.rolling_window}",
        ]
    )


def describe_benchmark_data(
    frame: pd.DataFrame | None,
    *,
    config: AlphaForgeConfig,
) -> str:
    """Render a concise benchmark return-series summary."""
    if frame is None or config.benchmark is None:
        return ""

    summary = _summarize_benchmark_data(frame)
    return "\n".join(
        [
            "Benchmark Summary",
            f"Rows: {summary['rows']}",
            f"Date Range: {summary['start_date']} -> {summary['end_date']}",
            f"Average Daily Return: {_format_percent_or_nan(summary['average_daily_return'])}",
            "Realized Volatility: "
            f"{_format_percent_or_nan(summary['realized_volatility'])}",
        ]
    )


def describe_symbol_metadata_configuration(config: AlphaForgeConfig) -> str:
    """Render the configured symbol metadata settings."""
    if config.symbol_metadata is None:
        return ""

    symbol_metadata = config.symbol_metadata
    return "\n".join(
        [
            "Symbol Metadata Configuration",
            f"Listing Date Column: {symbol_metadata.listing_date_column}",
            f"Delisting Date Column: {symbol_metadata.delisting_date_column}",
        ]
    )


def describe_symbol_metadata_data(
    frame: pd.DataFrame | None,
    *,
    config: AlphaForgeConfig,
) -> str:
    """Render a concise symbol metadata summary."""
    if frame is None or config.symbol_metadata is None:
        return ""

    summary = _summarize_symbol_metadata_data(frame)
    return "\n".join(
        [
            "Symbol Metadata Summary",
            f"Symbols: {summary['symbols']}",
            f"Listed Date Range: {summary['listing_start_date']} -> {summary['listing_end_date']}",
            f"Active Symbols: {summary['active_symbols']}",
            f"Delisted Symbols: {summary['delisted_symbols']}",
        ]
    )


def describe_corporate_actions_configuration(config: AlphaForgeConfig) -> str:
    """Render the configured corporate-actions settings."""
    if config.corporate_actions is None:
        return ""

    corporate_actions = config.corporate_actions
    return "\n".join(
        [
            "Corporate Actions Configuration",
            f"Ex-Date Column: {corporate_actions.ex_date_column}",
            f"Action Type Column: {corporate_actions.action_type_column}",
            f"Split Ratio Column: {corporate_actions.split_ratio_column}",
            f"Cash Amount Column: {corporate_actions.cash_amount_column}",
        ]
    )


def describe_corporate_actions_data(
    frame: pd.DataFrame | None,
    *,
    config: AlphaForgeConfig,
) -> str:
    """Render a concise corporate-actions summary."""
    if frame is None or config.corporate_actions is None:
        return ""

    summary = _summarize_corporate_actions_data(frame)
    return "\n".join(
        [
            "Corporate Actions Summary",
            f"Rows: {summary['rows']}",
            f"Symbols: {summary['symbols']}",
            f"Ex-Date Range: {summary['start_date']} -> {summary['end_date']}",
            f"Splits: {summary['split_actions']}",
            f"Cash Dividends: {summary['cash_dividend_actions']}",
        ]
    )


def describe_fundamentals_configuration(config: AlphaForgeConfig) -> str:
    """Render the configured fundamentals settings."""
    if config.fundamentals is None:
        return ""

    fundamentals = config.fundamentals
    return "\n".join(
        [
            "Fundamentals Configuration",
            f"Period End Column: {fundamentals.period_end_column}",
            f"Release Date Column: {fundamentals.release_date_column}",
            f"Metric Name Column: {fundamentals.metric_name_column}",
            f"Metric Value Column: {fundamentals.metric_value_column}",
        ]
    )


def describe_fundamentals_data(
    frame: pd.DataFrame | None,
    *,
    config: AlphaForgeConfig,
) -> str:
    """Render a concise fundamentals summary."""
    if frame is None or config.fundamentals is None:
        return ""

    summary = _summarize_fundamentals_data(frame)
    return "\n".join(
        [
            "Fundamentals Summary",
            f"Rows: {summary['rows']}",
            f"Symbols: {summary['symbols']}",
            f"Metrics: {summary['metrics']}",
            f"Period-End Range: {summary['period_start_date']} -> {summary['period_end_date']}",
            f"Release-Date Range: {summary['release_start_date']} -> {summary['release_end_date']}",
        ]
    )


def describe_shares_outstanding_configuration(config: AlphaForgeConfig) -> str:
    """Render the configured shares-outstanding settings."""
    if config.shares_outstanding is None:
        return ""

    shares_outstanding = config.shares_outstanding
    return "\n".join(
        [
            "Shares Outstanding Configuration",
            f"Effective Date Column: {shares_outstanding.effective_date_column}",
            f"Shares Outstanding Column: {shares_outstanding.shares_outstanding_column}",
        ]
    )


def describe_shares_outstanding_data(
    frame: pd.DataFrame | None,
    *,
    config: AlphaForgeConfig,
) -> str:
    """Render a concise shares-outstanding summary."""
    if frame is None or config.shares_outstanding is None:
        return ""

    summary = _summarize_shares_outstanding_data(frame)
    return "\n".join(
        [
            "Shares Outstanding Summary",
            f"Rows: {summary['rows']}",
            f"Symbols: {summary['symbols']}",
            f"Effective Date Range: {summary['start_date']} -> {summary['end_date']}",
            f"Latest Shares Total: {summary['latest_shares_total']:.0f}",
        ]
    )


def describe_classifications_configuration(config: AlphaForgeConfig) -> str:
    """Render the configured classifications settings."""
    if config.classifications is None:
        return ""

    classifications = config.classifications
    return "\n".join(
        [
            "Classifications Configuration",
            f"Effective Date Column: {classifications.effective_date_column}",
            f"Sector Column: {classifications.sector_column}",
            f"Industry Column: {classifications.industry_column}",
        ]
    )


def describe_classifications_data(
    frame: pd.DataFrame | None,
    *,
    config: AlphaForgeConfig,
) -> str:
    """Render a concise classifications summary."""
    if frame is None or config.classifications is None:
        return ""

    summary = _summarize_classifications_data(frame)
    return "\n".join(
        [
            "Classifications Summary",
            f"Rows: {summary['rows']}",
            f"Symbols: {summary['symbols']}",
            f"Effective Date Range: {summary['start_date']} -> {summary['end_date']}",
            f"Sectors: {summary['sectors']}",
            f"Industries: {summary['industries']}",
        ]
    )


def describe_memberships_configuration(config: AlphaForgeConfig) -> str:
    """Render the configured memberships settings."""
    if config.memberships is None:
        return ""

    memberships = config.memberships
    return "\n".join(
        [
            "Memberships Configuration",
            f"Effective Date Column: {memberships.effective_date_column}",
            f"Index Column: {memberships.index_column}",
            f"Is Member Column: {memberships.is_member_column}",
        ]
    )


def describe_memberships_data(
    frame: pd.DataFrame | None,
    *,
    config: AlphaForgeConfig,
) -> str:
    """Render a concise memberships summary."""
    if frame is None or config.memberships is None:
        return ""

    summary = _summarize_memberships_data(frame)
    return "\n".join(
        [
            "Memberships Summary",
            f"Rows: {summary['rows']}",
            f"Symbols: {summary['symbols']}",
            f"Indexes: {summary['indexes']}",
            f"Effective Date Range: {summary['start_date']} -> {summary['end_date']}",
            f"Member Rows: {summary['member_rows']}",
            f"Non-Member Rows: {summary['non_member_rows']}",
        ]
    )


def describe_borrow_availability_configuration(config: AlphaForgeConfig) -> str:
    """Render the configured borrow availability settings."""
    if config.borrow_availability is None:
        return ""

    borrow_availability = config.borrow_availability
    return "\n".join(
        [
            "Borrow Availability Configuration",
            f"Effective Date Column: {borrow_availability.effective_date_column}",
            f"Is Borrowable Column: {borrow_availability.is_borrowable_column}",
            f"Borrow Fee Bps Column: {borrow_availability.borrow_fee_bps_column}",
        ]
    )


def describe_borrow_availability_data(
    frame: pd.DataFrame | None,
    *,
    config: AlphaForgeConfig,
) -> str:
    """Render a concise borrow availability summary."""
    if frame is None or config.borrow_availability is None:
        return ""

    summary = _summarize_borrow_availability_data(frame)
    return "\n".join(
        [
            "Borrow Availability Summary",
            f"Rows: {summary['rows']}",
            f"Symbols: {summary['symbols']}",
            f"Effective Date Range: {summary['start_date']} -> {summary['end_date']}",
            f"Borrowable Rows: {summary['borrowable_rows']}",
            f"Not Borrowable Rows: {summary['not_borrowable_rows']}",
            f"Fee Observations: {summary['fee_observations']}",
        ]
    )


def describe_trading_status_configuration(config: AlphaForgeConfig) -> str:
    """Render the configured trading status settings."""
    if config.trading_status is None:
        return ""

    trading_status = config.trading_status
    return "\n".join(
        [
            "Trading Status Configuration",
            f"Effective Date Column: {trading_status.effective_date_column}",
            f"Is Tradable Column: {trading_status.is_tradable_column}",
            f"Status Reason Column: {trading_status.status_reason_column}",
        ]
    )


def describe_trading_status_data(
    frame: pd.DataFrame | None,
    *,
    config: AlphaForgeConfig,
) -> str:
    """Render a concise trading status summary."""
    if frame is None or config.trading_status is None:
        return ""

    summary = _summarize_trading_status_data(frame)
    return "\n".join(
        [
            "Trading Status Summary",
            f"Rows: {summary['rows']}",
            f"Symbols: {summary['symbols']}",
            f"Effective Date Range: {summary['start_date']} -> {summary['end_date']}",
            f"Tradable Rows: {summary['tradable_rows']}",
            f"Not Tradable Rows: {summary['not_tradable_rows']}",
            f"Reason Observations: {summary['reason_observations']}",
        ]
    )


def describe_trading_calendar_configuration(config: AlphaForgeConfig) -> str:
    """Render the configured trading calendar settings."""
    if config.calendar is None:
        return ""

    calendar = config.calendar
    return "\n".join(
        [
            "Trading Calendar Configuration",
            f"Name: {calendar.name}",
            f"Date Column: {calendar.date_column}",
        ]
    )


def describe_trading_calendar_data(
    frame: pd.DataFrame | None,
    *,
    config: AlphaForgeConfig,
) -> str:
    """Render a concise trading calendar summary."""
    if frame is None or config.calendar is None:
        return ""

    summary = _summarize_trading_calendar_data(frame)
    return "\n".join(
        [
            "Trading Calendar Summary",
            f"Sessions: {summary['sessions']}",
            f"Date Range: {summary['start_date']} -> {summary['end_date']}",
        ]
    )


def describe_universe_eligibility(frame: pd.DataFrame) -> str:
    """Render a concise summary of lagged universe eligibility."""
    summary = _summarize_universe_eligibility(frame)
    if summary is None:
        return ""

    lines = [
        "Universe Summary",
        f"Eligible Rows: {summary['eligible_rows']}/{summary['total_rows']}",
        f"Excluded Rows: {summary['excluded_rows']}/{summary['total_rows']}",
        "Eligible Symbols Ever: "
        f"{summary['eligible_symbols']}/{summary['total_symbols']}",
        "Eligible Symbols Per Date (min/avg/max): "
        f"{summary['eligible_symbols_per_date_min']}/"
        f"{summary['eligible_symbols_per_date_avg']:.2f}/"
        f"{summary['eligible_symbols_per_date_max']}",
        f"First Eligible Date: {summary['first_eligible_date']}",
    ]

    reason_counts = summary["exclusion_reasons"]
    if reason_counts:
        lines.append(
            "Exclusion Reasons: "
            + ", ".join(f"{reason}={count}" for reason, count in reason_counts.items())
        )
    return "\n".join(lines)


def describe_portfolio_constraints(config: AlphaForgeConfig) -> str:
    """Render the configured portfolio construction rules and simple constraints."""
    if config.portfolio is None:
        return ""

    portfolio = config.portfolio
    lines = [
        "Portfolio Constraints",
        f"Construction: {portfolio.construction}",
        f"Top N: {portfolio.top_n}",
        f"Weighting: {portfolio.weighting}",
    ]
    if portfolio.construction == "long_only":
        lines.append(f"Target Exposure: {portfolio.exposure}")
    else:
        bottom_n = portfolio.bottom_n if portfolio.bottom_n is not None else portfolio.top_n
        lines.extend(
            [
                f"Bottom N: {bottom_n}",
                f"Long Exposure Target: {portfolio.long_exposure}",
                f"Short Exposure Target: {portfolio.short_exposure}",
            ]
        )
    if portfolio.max_position_weight is not None:
        lines.append(f"Max Position Weight: {portfolio.max_position_weight}")
    if portfolio.position_cap_column is not None:
        lines.append(f"Position Cap Column: {portfolio.position_cap_column}")
    if portfolio.max_group_weight is not None:
        lines.extend(
            [
                f"Group Column: {portfolio.group_column}",
                f"Max Group Weight: {portfolio.max_group_weight}",
            ]
        )
    if portfolio.factor_exposure_bounds:
        for bound in portfolio.factor_exposure_bounds:
            lines.append(
                "Factor Exposure Bound: "
                f"{bound.column} min={bound.min_exposure} max={bound.max_exposure}"
            )
    return "\n".join(lines)


def describe_execution_configuration(config: AlphaForgeConfig) -> str:
    """Render the configured backtest execution assumptions."""
    if config.backtest is None:
        return ""

    backtest = config.backtest
    lines = [
        "Execution Assumptions",
        f"Signal Delay: {backtest.signal_delay} day(s)",
        f"Fill Timing: {backtest.fill_timing}",
        f"Rebalance Frequency: {backtest.rebalance_frequency}",
        f"Initial NAV: {backtest.initial_nav}",
    ]
    if backtest.rebalance_stagger_column is not None:
        lines.append(
            "Rebalance Stagger: "
            f"{backtest.rebalance_stagger_column} "
            f"across {backtest.rebalance_stagger_count} buckets"
        )
    if backtest.transaction_cost_bps is not None:
        lines.append(
            "Transaction Cost Model: "
            f"legacy total {backtest.transaction_cost_bps} bps"
        )
    else:
        if backtest.commission_bps_column is not None:
            lines.append(f"Commission Bps Column: {backtest.commission_bps_column}")
        else:
            lines.append(f"Commission: {backtest.commission_bps} bps")
        if backtest.slippage_bps_column is not None:
            lines.append(f"Slippage Bps Column: {backtest.slippage_bps_column}")
        elif backtest.liquidity_bucket_column is not None:
            bucket_text = ", ".join(
                f"{entry.bucket}: {entry.slippage_bps} bps"
                for entry in backtest.slippage_bps_by_liquidity_bucket
            )
            lines.append(
                "Liquidity Bucket Slippage: "
                f"{backtest.liquidity_bucket_column} ({bucket_text})"
            )
        else:
            lines.append(f"Slippage: {backtest.slippage_bps} bps")
        if backtest.market_impact_bps_per_turnover > 0.0:
            lines.append(
                "Market Impact: "
                f"{backtest.market_impact_bps_per_turnover} bps per unit turnover"
            )
    if backtest.borrow_fee_bps_column is not None:
        lines.append(f"Borrow Fee Bps Column: {backtest.borrow_fee_bps_column}")
    if backtest.shortable_column is not None:
        lines.append(f"Shortable Column: {backtest.shortable_column}")
    if backtest.tradable_column is not None:
        lines.append(f"Tradable Column: {backtest.tradable_column}")
    if backtest.can_buy_column is not None:
        lines.append(f"Can Buy Column: {backtest.can_buy_column}")
    if backtest.can_sell_column is not None:
        lines.append(f"Can Sell Column: {backtest.can_sell_column}")
    if backtest.max_trade_weight_column is not None:
        lines.append(f"Max Trade Weight Column: {backtest.max_trade_weight_column}")
    if backtest.max_participation_rate is not None:
        lines.extend(
            [
                f"Max Participation Rate: {backtest.max_participation_rate}",
                f"Participation Notional: {backtest.participation_notional}",
            ]
        )
    if backtest.min_trade_weight is not None:
        lines.append(f"Minimum Trade Weight: {backtest.min_trade_weight}")
    if backtest.max_turnover is not None:
        lines.append(f"Max Turnover Per Rebalance: {backtest.max_turnover}")
    return "\n".join(lines)


def describe_execution_results(backtest: pd.DataFrame) -> str:
    """Render a concise summary of realized execution constraints and costs."""
    if backtest.empty:
        return ""

    summary = _summarize_execution_results(backtest)
    lines = [
        "Execution Summary",
        f"Rebalance Dates: {summary['rebalance_dates']}/{summary['periods']}",
        "Base Rebalance Dates: "
        f"{summary['base_rebalance_dates']}/{summary['periods']}",
        "Staggered Rebalance Skipped Dates: "
        f"{summary['rebalance_stagger_skipped_dates']}/{summary['periods']}",
        "Turnover Limit Applied Dates: "
        f"{summary['turnover_limit_dates']}/{summary['periods']}",
        "Short Availability Limit Applied Dates: "
        f"{summary['short_availability_limit_dates']}/{summary['periods']}",
        "Tradability Limit Applied Dates: "
        f"{summary['tradability_limit_dates']}/{summary['periods']}",
        "Buy Limit Applied Dates: "
        f"{summary['buy_limit_dates']}/{summary['periods']}",
        "Sell Limit Applied Dates: "
        f"{summary['sell_limit_dates']}/{summary['periods']}",
        "Participation Limit Applied Dates: "
        f"{summary['participation_limit_dates']}/{summary['periods']}",
        "Trade Limit Applied Dates: "
        f"{summary['trade_limit_dates']}/{summary['periods']}",
        "Trade Clip Applied Dates: "
        f"{summary['trade_clip_dates']}/{summary['periods']}",
        f"Average Target Turnover: {summary['average_target_turnover']:.2f}",
        f"Average Realized Turnover: {summary['average_realized_turnover']:.2f}",
        "Average Gross Target Exposure: "
        f"{summary['average_gross_target_exposure']:.2f}",
        f"Average Gross Exposure: {summary['average_gross_exposure']:.2f}",
        f"Average Target Holdings: {summary['average_target_holdings']:.2f}",
        f"Average Realized Holdings: {summary['average_realized_holdings']:.2f}",
        "Average Target-Effective Weight Gap: "
        f"{summary['average_target_effective_weight_gap']:.2f}",
        f"Total Commission Cost: {_format_percent_or_nan(summary['total_commission_cost'])}",
        f"Total Slippage Cost: {_format_percent_or_nan(summary['total_slippage_cost'])}",
        "Total Market Impact Cost: "
        f"{_format_percent_or_nan(summary['total_market_impact_cost'])}",
        f"Total Borrow Cost: {_format_percent_or_nan(summary['total_borrow_cost'])}",
        "Total Transaction Cost: "
        f"{_format_percent_or_nan(summary['total_transaction_cost'])}",
    ]
    return "\n".join(lines)


def _summarize_market_data(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize core market-data coverage."""
    unique_dates = frame["date"].drop_duplicates().sort_values(kind="mergesort")
    return {
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "trading_dates": int(unique_dates.nunique()),
        "start_date": frame["date"].min().date().isoformat(),
        "end_date": frame["date"].max().date().isoformat(),
    }


def _summarize_data_quality(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize simple coverage and range diagnostics for daily OHLCV data."""
    rows_per_symbol = frame.groupby("symbol", sort=True).size()
    symbols_per_date = frame.groupby("date", sort=True)["symbol"].nunique()
    return {
        "rows_per_symbol_min": int(rows_per_symbol.min()),
        "rows_per_symbol_avg": float(rows_per_symbol.mean()),
        "rows_per_symbol_max": int(rows_per_symbol.max()),
        "symbols_per_date_min": int(symbols_per_date.min()),
        "symbols_per_date_avg": float(symbols_per_date.mean()),
        "symbols_per_date_max": int(symbols_per_date.max()),
        "close_min": float(frame["close"].min()),
        "close_max": float(frame["close"].max()),
        "volume_min": float(frame["volume"].min()),
        "volume_max": float(frame["volume"].max()),
    }


def _summarize_benchmark_data(frame: pd.DataFrame | None) -> dict[str, Any] | None:
    """Summarize a benchmark return series for reporting and metadata."""
    if frame is None:
        return None

    realized_volatility = frame["benchmark_return"].std(ddof=1)
    return {
        "rows": int(len(frame)),
        "start_date": frame["date"].min().date().isoformat(),
        "end_date": frame["date"].max().date().isoformat(),
        "average_daily_return": float(frame["benchmark_return"].mean()),
        "realized_volatility": (
            float(realized_volatility * (252.0**0.5))
            if pd.notna(realized_volatility)
            else None
        ),
    }


def _summarize_trading_calendar_data(
    frame: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """Summarize trading calendar coverage for reporting and validation."""
    if frame is None:
        return None

    return {
        "sessions": int(len(frame)),
        "start_date": frame["date"].min().date().isoformat(),
        "end_date": frame["date"].max().date().isoformat(),
    }


def _summarize_symbol_metadata_data(
    frame: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """Summarize symbol metadata coverage for reporting and validation."""
    if frame is None:
        return None

    delisted_symbols = int(frame["delisting_date"].notna().sum())
    return {
        "symbols": int(frame["symbol"].nunique()),
        "listing_start_date": frame["listing_date"].min().date().isoformat(),
        "listing_end_date": frame["listing_date"].max().date().isoformat(),
        "active_symbols": int(len(frame) - delisted_symbols),
        "delisted_symbols": delisted_symbols,
    }


def _summarize_corporate_actions_data(
    frame: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """Summarize corporate-action coverage for validation output."""
    if frame is None:
        return None

    return {
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "start_date": frame["ex_date"].min().date().isoformat(),
        "end_date": frame["ex_date"].max().date().isoformat(),
        "split_actions": int(frame["action_type"].eq("split").sum()),
        "cash_dividend_actions": int(
            frame["action_type"].eq("cash_dividend").sum()
        ),
    }


def _summarize_fundamentals_data(
    frame: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """Summarize long-form fundamentals coverage for validation output."""
    if frame is None:
        return None

    return {
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "metrics": int(frame["metric_name"].nunique()),
        "period_start_date": frame["period_end_date"].min().date().isoformat(),
        "period_end_date": frame["period_end_date"].max().date().isoformat(),
        "release_start_date": frame["release_date"].min().date().isoformat(),
        "release_end_date": frame["release_date"].max().date().isoformat(),
    }


def _summarize_shares_outstanding_data(
    frame: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """Summarize shares-outstanding coverage for validation output."""
    if frame is None:
        return None

    latest_rows = frame.sort_values(
        ["symbol", "effective_date"],
        kind="mergesort",
    ).drop_duplicates(subset=["symbol"], keep="last")
    return {
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "start_date": frame["effective_date"].min().date().isoformat(),
        "end_date": frame["effective_date"].max().date().isoformat(),
        "latest_shares_total": float(latest_rows["shares_outstanding"].sum()),
    }


def _summarize_classifications_data(
    frame: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """Summarize sector/industry classifications coverage."""
    if frame is None:
        return None

    return {
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "start_date": frame["effective_date"].min().date().isoformat(),
        "end_date": frame["effective_date"].max().date().isoformat(),
        "sectors": int(frame["sector"].nunique()),
        "industries": int(frame["industry"].nunique()),
    }


def _summarize_memberships_data(
    frame: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """Summarize index membership coverage."""
    if frame is None:
        return None

    return {
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "indexes": int(frame["index_name"].nunique()),
        "start_date": frame["effective_date"].min().date().isoformat(),
        "end_date": frame["effective_date"].max().date().isoformat(),
        "member_rows": int(frame["is_member"].sum()),
        "non_member_rows": int((~frame["is_member"]).sum()),
    }


def _summarize_borrow_availability_data(
    frame: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """Summarize borrow availability coverage."""
    if frame is None:
        return None

    return {
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "start_date": frame["effective_date"].min().date().isoformat(),
        "end_date": frame["effective_date"].max().date().isoformat(),
        "borrowable_rows": int(frame["is_borrowable"].sum()),
        "not_borrowable_rows": int((~frame["is_borrowable"]).sum()),
        "fee_observations": int(frame["borrow_fee_bps"].notna().sum()),
    }


def _summarize_trading_status_data(
    frame: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """Summarize trading status coverage."""
    if frame is None:
        return None

    return {
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "start_date": frame["effective_date"].min().date().isoformat(),
        "end_date": frame["effective_date"].max().date().isoformat(),
        "tradable_rows": int(frame["is_tradable"].sum()),
        "not_tradable_rows": int((~frame["is_tradable"]).sum()),
        "reason_observations": int(frame["status_reason"].notna().sum()),
    }


def _summarize_universe_eligibility(frame: pd.DataFrame | None) -> dict[str, Any] | None:
    """Summarize lagged universe eligibility in a metadata-friendly form."""
    if frame is None or "is_universe_eligible" not in frame.columns:
        return None

    eligible = frame["is_universe_eligible"].fillna(False).astype(bool)
    excluded = ~eligible
    total_rows = int(len(frame))
    eligible_rows = int(eligible.sum())
    per_date_eligible_symbols = (
        frame.loc[eligible]
        .groupby("date", sort=True)["symbol"]
        .nunique()
        .reindex(frame["date"].drop_duplicates().sort_values(), fill_value=0)
    )
    exclusion_reasons: dict[str, int] = {}
    if excluded.any() and "universe_exclusion_reason" in frame.columns:
        reason_counts = (
            frame.loc[excluded, "universe_exclusion_reason"]
            .dropna()
            .astype(str)
            .str.split(";")
            .explode()
        )
        reason_counts = reason_counts.loc[reason_counts != ""].value_counts()
        exclusion_reasons = {str(reason): int(count) for reason, count in reason_counts.items()}

    return {
        "total_rows": total_rows,
        "eligible_rows": eligible_rows,
        "excluded_rows": total_rows - eligible_rows,
        "eligible_symbols": int(frame.loc[eligible, "symbol"].nunique()),
        "total_symbols": int(frame["symbol"].nunique()),
        "eligible_symbols_per_date_min": int(per_date_eligible_symbols.min()),
        "eligible_symbols_per_date_avg": float(per_date_eligible_symbols.mean()),
        "eligible_symbols_per_date_max": int(per_date_eligible_symbols.max()),
        "first_eligible_date": (
            frame.loc[eligible, "date"].min().date().isoformat()
            if eligible_rows > 0
            else "None"
        ),
        "exclusion_reasons": exclusion_reasons,
    }


def _summarize_execution_results(backtest: pd.DataFrame) -> dict[str, Any]:
    """Summarize realized execution behavior for reports and metadata."""
    return {
        "periods": int(len(backtest)),
        "base_rebalance_dates": int(
            backtest.get(
                "is_base_rebalance_date",
                backtest["is_rebalance_date"],
            )
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "rebalance_dates": int(
            backtest["is_rebalance_date"].fillna(False).astype(bool).sum()
        ),
        "rebalance_stagger_skipped_dates": int(
            backtest.get(
                "rebalance_stagger_skipped",
                pd.Series(False, index=backtest.index),
            )
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "turnover_limit_dates": int(
            backtest["turnover_limit_applied"].fillna(False).astype(bool).sum()
        ),
        "short_availability_limit_dates": int(
            backtest.get(
                "short_availability_limit_applied",
                pd.Series(False, index=backtest.index),
            )
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "tradability_limit_dates": int(
            backtest.get(
                "tradability_limit_applied",
                pd.Series(False, index=backtest.index),
            )
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "buy_limit_dates": int(
            backtest.get("buy_limit_applied", pd.Series(False, index=backtest.index))
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "sell_limit_dates": int(
            backtest.get("sell_limit_applied", pd.Series(False, index=backtest.index))
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "participation_limit_dates": int(
            backtest.get(
                "participation_limit_applied",
                pd.Series(False, index=backtest.index),
            )
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "trade_limit_dates": int(
            backtest.get("trade_limit_applied", pd.Series(False, index=backtest.index))
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "trade_clip_dates": int(
            backtest.get("trade_clip_applied", pd.Series(False, index=backtest.index))
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "average_target_turnover": float(backtest["target_turnover"].mean()),
        "average_realized_turnover": float(backtest["turnover"].mean()),
        "average_gross_target_exposure": float(backtest["gross_target_exposure"].mean()),
        "average_gross_exposure": float(backtest["gross_exposure"].mean()),
        "average_target_holdings": float(backtest["target_holdings_count"].mean()),
        "average_realized_holdings": float(backtest["holdings_count"].mean()),
        "average_target_effective_weight_gap": float(
            backtest["target_effective_weight_gap"].mean()
        ),
        "total_commission_cost": float(backtest["commission_cost"].sum()),
        "total_slippage_cost": float(backtest["slippage_cost"].sum()),
        "total_market_impact_cost": float(
            backtest.get(
                "market_impact_cost",
                pd.Series(0.0, index=backtest.index),
            ).sum()
        ),
        "total_borrow_cost": float(
            backtest.get("borrow_cost", pd.Series(0.0, index=backtest.index)).sum()
        ),
        "total_transaction_cost": float(backtest["transaction_cost"].sum()),
    }


def _summarize_diagnostics_overview(
    ic_summary: pd.Series,
    rolling_ic_summary: pd.Series,
    coverage_summary: pd.Series,
    quantile_summary: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize the key diagnostics headline numbers."""
    top_bottom_quantile_spread = None
    if not quantile_summary.empty and "quantile" in quantile_summary.columns:
        sorted_quantiles = quantile_summary.sort_values("quantile", kind="mergesort")
        if len(sorted_quantiles) >= 2:
            top_bottom_quantile_spread = float(
                sorted_quantiles.iloc[-1]["mean_forward_return"]
                - sorted_quantiles.iloc[0]["mean_forward_return"]
            )

    return {
        "mean_ic": _scalar_or_none(ic_summary.get("mean_ic")),
        "ic_ir": _scalar_or_none(ic_summary.get("ic_ir")),
        "latest_rolling_mean_ic": _scalar_or_none(
            rolling_ic_summary.get("latest_rolling_mean_ic")
        ),
        "latest_rolling_ic_ir": _scalar_or_none(
            rolling_ic_summary.get("latest_rolling_ic_ir")
        ),
        "joint_coverage_ratio": _scalar_or_none(
            coverage_summary.get("joint_coverage_ratio")
        ),
        "average_daily_usable_rows": _scalar_or_none(
            coverage_summary.get("average_daily_usable_rows")
        ),
        "top_bottom_quantile_spread": top_bottom_quantile_spread,
    }


def _describe_signal_transform(signal: Any) -> str:
    """Render the configured signal-transform pipeline succinctly."""
    parts: list[str] = []
    if signal.winsorize_quantile is not None:
        parts.append(f"winsorize_quantile={signal.winsorize_quantile}")
    if signal.clip_lower_bound is not None or signal.clip_upper_bound is not None:
        parts.append(
            "clip_bounds="
            f"[{signal.clip_lower_bound}, {signal.clip_upper_bound}]"
        )
    if signal.cross_sectional_residualize_columns:
        columns = ", ".join(signal.cross_sectional_residualize_columns)
        parts.append(f"cross_sectional_residualize_columns=[{columns}]")
    if signal.cross_sectional_neutralize_group_column is not None:
        parts.append(
            "cross_sectional_neutralize_group_column="
            f"{signal.cross_sectional_neutralize_group_column}"
        )
    if signal.cross_sectional_normalization != "none":
        parts.append(
            "cross_sectional_normalization="
            f"{signal.cross_sectional_normalization}"
        )
    if signal.cross_sectional_group_column is not None:
        parts.append(
            "cross_sectional_group_column="
            f"{signal.cross_sectional_group_column}"
        )
    if not parts:
        return "none"
    return ", ".join(parts)


def _format_percent_or_nan(value: float) -> str:
    """Format a decimal value as a percentage, preserving missing values."""
    if pd.isna(value):
        return "NaN"
    return f"{value:.2%}"


def _format_number_or_nan(value: float) -> str:
    """Format a scalar numeric value, preserving missing values."""
    if pd.isna(value):
        return "NaN"
    return f"{value:.2f}"


def _series_to_metadata_dict(summary: pd.Series) -> dict[str, Any]:
    """Convert a summary series into JSON-safe scalar metadata."""
    return {
        str(key): _scalar_or_none(value)
        for key, value in summary.items()
    }


def _dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a small DataFrame into JSON-safe record dictionaries."""
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        records.append({str(key): _scalar_or_none(value) for key, value in row.items()})
    return records


def _require_signal_config(config: AlphaForgeConfig):
    """Require a signal section for signal-dependent report rendering."""
    if config.signal is None:
        raise WorkflowError("The config must include a [signal] section for this command.")
    return config.signal


def _require_portfolio_config(config: AlphaForgeConfig):
    """Require a portfolio section for portfolio-dependent report rendering."""
    if config.portfolio is None:
        raise WorkflowError(
            "The config must include a [portfolio] section for this command."
        )
    return config.portfolio


def _require_backtest_config(config: AlphaForgeConfig):
    """Require a backtest section for backtest-dependent report rendering."""
    if config.backtest is None:
        raise WorkflowError(
            "The config must include a [backtest] section for this command."
        )
    return config.backtest


def _render_html_summary_block(
    title: str,
    summary: Any,
    *,
    formatter: Any,
) -> str:
    """Render one metric summary dict/series into an HTML card."""
    if summary is None:
        return ""
    items: list[tuple[str, Any]]
    if isinstance(summary, pd.Series):
        items = list(summary.items())
    elif isinstance(summary, dict):
        items = list(summary.items())
    else:
        return ""

    metrics_html = "".join(
        [
            f"<div>{escape(str(key))}</div><div>{escape(formatter(key, value))}</div>"
            for key, value in items
        ]
    )
    return (
        f"<article class=\"card\">"
        f"<h3>{escape(title)}</h3>"
        f"<div class=\"metrics\">{metrics_html}</div>"
        f"</article>"
    )


def _render_report_chart_card(chart: dict[str, Any]) -> str:
    """Render one chart card for the HTML report page."""
    title = escape(str(chart.get("title", "")))
    description = escape(str(chart.get("description", "")))
    path = escape(str(Path("charts") / str(chart.get("filename", ""))))
    return (
        "<article class=\"card chart-card\">"
        f"<h3>{title}</h3>"
        f"<p>{description}</p>"
        f"<img src=\"{path}\" alt=\"{title}\">"
        "</article>"
    )


def _format_html_metric_value(key: Any, value: Any) -> str:
    """Format one metric value for compact HTML summary cards."""
    scalar = _scalar_or_none(value)
    if scalar is None:
        return "NaN"
    if isinstance(scalar, bool):
        return "true" if scalar else "false"
    if isinstance(scalar, int):
        return str(scalar)
    if isinstance(scalar, float):
        key_text = str(key)
        if any(
            token in key_text
            for token in [
                "return",
                "volatility",
                "drawdown",
                "coverage_ratio",
                "hit_rate",
                "value_at_risk",
                "conditional_value_at_risk",
            ]
        ):
            return f"{scalar:.2%}"
        return f"{scalar:.2f}"
    return str(scalar)


def _scalar_or_none(value: Any) -> Any:
    """Convert pandas/numpy scalars into HTML-format-friendly Python values."""
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

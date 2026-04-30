"""Report rendering helpers for CLI workflows."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from alphaforge.cli.artifacts import write_text

__all__ = [
    "write_report_html_page",
]


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

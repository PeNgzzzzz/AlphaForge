"""Tests for CLI report rendering helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alphaforge.cli.reports import write_report_html_page


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

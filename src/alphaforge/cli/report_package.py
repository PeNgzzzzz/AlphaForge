"""Report package orchestration helpers for CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from alphaforge.cli.artifacts import write_artifact_bundle, write_json
from alphaforge.cli.charts import write_report_chart_bundle_from_context
from alphaforge.cli.report_context import build_report_context
from alphaforge.cli.reports import (
    build_report_metadata,
    render_report_text,
    write_report_html_page,
)
from alphaforge.cli.research_metadata import (
    build_config_snapshot,
    build_research_metadata_from_config,
)
from alphaforge.common import AlphaForgeConfig

__all__ = [
    "build_report_text",
    "build_report_package",
    "write_report_artifact_bundle",
    "write_report_chart_bundle",
]


def build_report_text(config: AlphaForgeConfig) -> str:
    """Run the full configured pipeline and render a text report."""
    _, report_text, _ = build_report_package(config)
    return report_text


def build_report_package(
    config: AlphaForgeConfig,
    *,
    config_path: str | None = None,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """Build the backtest, text report, and metadata for one configured research run."""
    context = build_report_context(config)
    report_text = render_report_text(context, config=config)
    workflow_configuration = build_config_snapshot(config)
    research_metadata = build_research_metadata_from_config(config)
    metadata = build_report_metadata(
        context,
        config=config,
        config_path=config_path,
        workflow_configuration=workflow_configuration,
        research_metadata=research_metadata,
    )
    return context["backtest"], report_text, metadata


def write_report_artifact_bundle(
    config: AlphaForgeConfig,
    artifact_dir: str | Path,
    *,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Write one full report artifact bundle, including static chart outputs."""
    context = build_report_context(config)
    report_text = render_report_text(context, config=config)
    workflow_configuration = build_config_snapshot(config)
    research_metadata = build_research_metadata_from_config(config)
    metadata = build_report_metadata(
        context,
        config=config,
        config_path=config_path,
        workflow_configuration=workflow_configuration,
        research_metadata=research_metadata,
    )
    artifact_paths = write_artifact_bundle(
        artifact_dir,
        results=context["backtest"],
        report_text=report_text,
        metadata=metadata,
    )
    chart_bundle = write_report_chart_bundle_from_context(
        context,
        output_dir=Path(artifact_paths["artifact_dir"]) / "charts",
        workflow_configuration=workflow_configuration,
        config_path=config_path,
        command_name="report",
    )
    enriched_metadata = dict(metadata)
    enriched_metadata["chart_bundle"] = {
        "chart_dir": "charts",
        "manifest_path": str(Path("charts") / "manifest.json"),
        "chart_count": chart_bundle["chart_count"],
        "charts": chart_bundle["charts"],
    }
    html_path = write_report_html_page(
        report_text=report_text,
        metadata=enriched_metadata,
        artifact_dir=artifact_paths["artifact_dir"],
    )
    enriched_metadata["html_report_path"] = html_path.name
    write_json(enriched_metadata, artifact_paths["metadata_path"])
    artifact_paths["chart_dir"] = chart_bundle["chart_dir"]
    artifact_paths["chart_manifest_path"] = chart_bundle["manifest_path"]
    artifact_paths["html_path"] = html_path
    return artifact_paths


def write_report_chart_bundle(
    config: AlphaForgeConfig,
    output_dir: str | Path,
    *,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Write one static chart bundle from the configured report context."""
    context = build_report_context(config)
    return write_report_chart_bundle_from_context(
        context,
        output_dir=output_dir,
        workflow_configuration=build_config_snapshot(config),
        config_path=config_path,
        command_name="plot-report",
    )

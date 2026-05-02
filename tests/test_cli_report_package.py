"""Tests for CLI report package orchestration helpers."""

from __future__ import annotations

import json
from pathlib import Path

from alphaforge.cli.report_package import (
    build_report_package,
    build_report_text,
    write_report_artifact_bundle,
    write_report_chart_bundle,
)
from alphaforge.common import load_pipeline_config


def test_build_report_package_returns_report_outputs() -> None:
    """Report packaging should return backtest rows, report text, and metadata."""
    config_path = Path("configs/momentum_example.toml")
    config = load_pipeline_config(config_path)

    backtest, report_text, metadata = build_report_package(
        config,
        config_path=str(config_path),
    )

    assert not backtest.empty
    assert "Research Workflow" in report_text
    assert "Data Quality Summary" in report_text
    assert metadata["command"] == "report"
    assert metadata["config"] == str(config_path)
    assert "workflow_configuration" in metadata
    assert metadata["signal_pipeline_metadata"]["factor"]["name"] == "momentum"
    assert "Research Workflow" in build_report_text(config)


def test_write_report_artifact_bundle_writes_report_package(
    tmp_path: Path,
) -> None:
    """Full report artifacts should include results, metadata, charts, and HTML."""
    config_path = Path("configs/momentum_example.toml")
    config = load_pipeline_config(config_path)

    paths = write_report_artifact_bundle(
        config,
        tmp_path / "report_artifact",
        config_path=str(config_path),
    )

    metadata = json.loads(Path(paths["metadata_path"]).read_text(encoding="utf-8"))

    assert Path(paths["results_path"]).exists()
    assert Path(paths["report_path"]).exists()
    assert Path(paths["metadata_path"]).exists()
    assert Path(paths["chart_manifest_path"]).exists()
    assert Path(paths["html_path"]).exists()
    assert metadata["command"] == "report"
    assert metadata["html_report_path"] == "index.html"
    assert metadata["chart_bundle"]["chart_count"] > 0


def test_write_report_chart_bundle_marks_plot_report_command(
    tmp_path: Path,
) -> None:
    """Standalone chart bundles should retain the plot-report command metadata."""
    config_path = Path("configs/momentum_example.toml")
    config = load_pipeline_config(config_path)

    paths = write_report_chart_bundle(
        config,
        tmp_path / "charts",
        config_path=str(config_path),
    )
    manifest = json.loads(Path(paths["manifest_path"]).read_text(encoding="utf-8"))

    assert Path(paths["chart_dir"]).exists()
    assert manifest["command"] == "plot-report"
    assert manifest["config"] == str(config_path)
    assert manifest["chart_count"] > 0

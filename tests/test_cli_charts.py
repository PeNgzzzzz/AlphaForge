"""Tests for CLI chart bundle helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alphaforge.cli.charts import write_compare_chart_bundle


def test_write_compare_chart_bundle_writes_manifest_and_png(tmp_path: Path) -> None:
    """Compare-runs chart bundles should write a PNG and stable manifest."""
    results = pd.DataFrame(
        {
            "run_id": ["run-a", "run-b"],
            "summary_cumulative_return": [0.12, -0.03],
            "summary_sharpe_ratio": [1.2, -0.4],
            "summary_mean_ic": [0.04, -0.01],
        }
    )

    bundle = write_compare_chart_bundle(results, output_dir=tmp_path / "charts")

    assert bundle["chart_dir"] == tmp_path / "charts"
    assert bundle["manifest_path"].exists()
    assert bundle["chart_count"] == 1
    assert (tmp_path / "charts" / "compare_summary_metrics.png").exists()

    manifest = json.loads(bundle["manifest_path"].read_text(encoding="utf-8"))
    assert manifest["command"] == "compare-runs"
    assert manifest["chart_count"] == 1
    assert manifest["charts"][0] == {
        "chart_id": "compare_summary_metrics",
        "title": "Compare Summary Metrics",
        "filename": "compare_summary_metrics.png",
        "path": "compare_summary_metrics.png",
        "description": (
            "Multi-run comparison across cumulative return, Sharpe ratio, and mean IC."
        ),
    }

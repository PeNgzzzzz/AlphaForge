"""Tests for CLI artifact writing helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from alphaforge.cli.artifacts import load_run_index, write_indexed_artifact_bundle


def test_write_indexed_artifact_bundle_writes_bundle_and_run_index(
    tmp_path: Path,
) -> None:
    """Indexed artifacts should write JSON-safe metadata and a valid runs.csv."""
    experiment_root = tmp_path / "experiment"
    results = pd.DataFrame({"parameter_value": [5], "cumulative_return": [0.12]})
    metadata = {
        "config": "configs/example.toml",
        "parameter": "lookback",
        "values": [5, 10],
        "selection_metric": "mean_ic",
        "row_count": 1,
        "overall_summary": {"cumulative_return": 0.12},
        "timestamp": pd.Timestamp("2024-01-02"),
        "missing": pd.NA,
        "tuple_values": (1, pd.NA),
    }

    paths = write_indexed_artifact_bundle(
        experiment_root,
        command_name="sweep-signal",
        results=results,
        report_text="report body",
        metadata=metadata,
    )

    assert paths["results_path"].exists()
    assert paths["report_path"].read_text(encoding="utf-8") == "report body"
    metadata_json = json.loads(paths["metadata_path"].read_text(encoding="utf-8"))
    assert metadata_json["timestamp"] == "2024-01-02T00:00:00"
    assert metadata_json["missing"] is None
    assert metadata_json["tuple_values"] == [1, None]

    run_index = load_run_index(experiment_root)
    assert len(run_index) == 1
    row = run_index.iloc[0]
    assert row["command"] == "sweep-signal"
    assert row["parameter"] == "lookback"
    assert row["values"] == "5,10"
    assert row["overall_cumulative_return"] == pytest.approx(0.12)
    assert paths["index_path"] == experiment_root / "runs.csv"

"""Artifact writing helpers for CLI workflows."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from alphaforge.cli.errors import WorkflowError

__all__ = [
    "write_dataframe",
    "write_text",
    "write_json",
    "write_artifact_bundle",
    "write_indexed_artifact_bundle",
    "load_run_index",
]


def write_dataframe(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to CSV, creating parent directories as needed."""
    output_path = Path(path)
    if output_path.exists() and output_path.is_dir():
        raise WorkflowError(
            f"Output path must be a file path, not an existing directory: {output_path}"
        )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
    except OSError as exc:
        raise WorkflowError(f"Failed to write CSV output to {output_path}: {exc}") from exc
    return output_path


def write_text(text: str, path: str | Path) -> Path:
    """Write plain text to disk, creating parent directories as needed."""
    output_path = Path(path)
    if output_path.exists() and output_path.is_dir():
        raise WorkflowError(
            f"Output path must be a file path, not an existing directory: {output_path}"
        )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise WorkflowError(f"Failed to write text output to {output_path}: {exc}") from exc
    return output_path


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    """Write JSON to disk, creating parent directories as needed."""
    output_path = Path(path)
    if output_path.exists() and output_path.is_dir():
        raise WorkflowError(
            f"Output path must be a file path, not an existing directory: {output_path}"
        )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                _make_json_safe(data),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise WorkflowError(f"Failed to write JSON output to {output_path}: {exc}") from exc
    return output_path


def write_artifact_bundle(
    artifact_dir: str | Path,
    *,
    results: pd.DataFrame,
    report_text: str,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    """Write a small reproducible artifact bundle for a CLI experiment run."""
    bundle_dir = Path(artifact_dir)
    if bundle_dir.exists() and not bundle_dir.is_dir():
        raise WorkflowError(
            f"Artifact path must be a directory path, not an existing file: {bundle_dir}"
        )

    try:
        bundle_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise WorkflowError(f"Failed to create artifact directory {bundle_dir}: {exc}") from exc

    results_path = write_dataframe(results, bundle_dir / "results.csv")
    report_path = write_text(report_text, bundle_dir / "report.txt")
    metadata_path = write_json(metadata, bundle_dir / "metadata.json")
    return {
        "artifact_dir": bundle_dir,
        "results_path": results_path,
        "report_path": report_path,
        "metadata_path": metadata_path,
    }


def write_indexed_artifact_bundle(
    experiment_root: str | Path,
    *,
    command_name: str,
    results: pd.DataFrame,
    report_text: str,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    """Write an artifact bundle under an experiment root and append a run index."""
    root_dir = Path(experiment_root)
    if root_dir.exists() and not root_dir.is_dir():
        raise WorkflowError(
            f"Experiment root must be a directory path, not an existing file: {root_dir}"
        )

    try:
        root_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise WorkflowError(f"Failed to create experiment root {root_dir}: {exc}") from exc

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = _build_run_id(command_name=command_name, metadata=metadata)
    bundle_dir = root_dir / run_id

    enriched_metadata = dict(metadata)
    enriched_metadata["run_id"] = run_id
    enriched_metadata["created_at"] = created_at
    enriched_metadata["artifact_dir"] = str(bundle_dir)

    artifact_paths = write_artifact_bundle(
        bundle_dir,
        results=results,
        report_text=report_text,
        metadata=enriched_metadata,
    )
    index_path = _append_run_index(
        root_dir / "runs.csv",
        _build_run_index_row(
            command_name=command_name,
            metadata=enriched_metadata,
            artifact_paths=artifact_paths,
        ),
    )
    artifact_paths["index_path"] = index_path
    return artifact_paths


def load_run_index(experiment_root: str | Path) -> pd.DataFrame:
    """Load and validate a lightweight experiment run index."""
    root_dir = Path(experiment_root)
    if not root_dir.exists():
        raise WorkflowError(f"Experiment root does not exist: {root_dir}")
    if not root_dir.is_dir():
        raise WorkflowError(f"Experiment root must be a directory: {root_dir}")

    index_path = root_dir / "runs.csv"
    if not index_path.exists():
        raise WorkflowError(f"Experiment root does not contain runs.csv: {index_path}")
    if not index_path.is_file():
        raise WorkflowError(f"Run index path must be a file: {index_path}")

    dataset = pd.read_csv(index_path)
    required_columns = [
        "run_id",
        "created_at",
        "command",
        "config",
        "parameter",
        "values",
        "row_count",
        "overall_cumulative_return",
        "artifact_dir",
        "results_path",
        "report_path",
        "metadata_path",
    ]
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise WorkflowError(f"runs.csv is missing required columns: {missing_text}.")
    if dataset.empty:
        raise WorkflowError(f"runs.csv does not contain any experiment runs: {index_path}")

    return dataset


def _make_json_safe(value: Any) -> Any:
    """Recursively convert pandas/NaN-containing values into JSON-safe objects."""
    if isinstance(value, dict):
        return {str(key): _make_json_safe(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_make_json_safe(item) for item in value]
    return _scalar_or_none(value)


def _build_run_id(*, command_name: str, metadata: dict[str, Any]) -> str:
    """Build a timestamped run identifier suitable for directory names."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    parameter = str(metadata.get("parameter", "run"))
    slug = _slugify(f"{command_name}-{parameter}")
    return f"{timestamp}-{slug}"


def _build_run_index_row(
    *,
    command_name: str,
    metadata: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    """Build one flattened row for the experiment run index."""
    overall_summary = metadata.get("overall_summary")
    overall_cumulative_return = None
    if isinstance(overall_summary, dict):
        overall_cumulative_return = overall_summary.get("cumulative_return")

    values = metadata.get("values")
    values_text = ""
    if isinstance(values, list):
        values_text = ",".join(str(value) for value in values)

    return {
        "run_id": metadata.get("run_id", ""),
        "created_at": metadata.get("created_at", ""),
        "command": command_name,
        "config": metadata.get("config", ""),
        "parameter": metadata.get("parameter", ""),
        "values": values_text,
        "selection_metric": metadata.get("selection_metric", ""),
        "train_periods": metadata.get("train_periods", ""),
        "test_periods": metadata.get("test_periods", ""),
        "row_count": metadata.get("row_count", ""),
        "overall_cumulative_return": overall_cumulative_return,
        "artifact_dir": str(artifact_paths["artifact_dir"]),
        "results_path": str(artifact_paths["results_path"]),
        "report_path": str(artifact_paths["report_path"]),
        "metadata_path": str(artifact_paths["metadata_path"]),
    }


def _append_run_index(path: Path, row: dict[str, Any]) -> Path:
    """Append one row to an experiment run index CSV."""
    if path.exists() and path.is_dir():
        raise WorkflowError(f"Run index path must be a file path, not a directory: {path}")

    new_row = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        all_columns = existing.columns.union(new_row.columns, sort=False)
        updated = existing.reindex(columns=all_columns).copy()
        updated.loc[len(updated)] = new_row.reindex(columns=all_columns).iloc[0]
    else:
        updated = new_row

    write_dataframe(updated, path)
    return path


def _slugify(value: str) -> str:
    """Convert free-form text into a conservative filesystem slug."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return slug or "run"


def _scalar_or_none(value: Any) -> Any:
    """Convert pandas/numpy scalars into JSON-safe Python values."""
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

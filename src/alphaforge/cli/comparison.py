"""Run index listing and comparison helpers for CLI workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from alphaforge.analytics import (
    validate_parameter_sweep_results,
    validate_walk_forward_results,
)
from alphaforge.cli.artifacts import (
    load_run_index,
    write_artifact_bundle,
    write_json,
)
from alphaforge.cli.charts import write_compare_chart_bundle
from alphaforge.cli.errors import WorkflowError
from alphaforge.cli.reports import (
    _format_number_or_nan,
    _format_percent_or_nan,
)
from alphaforge.cli.validation import normalize_positive_int

__all__ = [
    "build_compare_artifact_metadata",
    "build_compare_runs_report",
    "compare_indexed_runs",
    "format_run_index_table",
    "list_indexed_runs",
    "rank_compare_runs",
    "write_compare_artifact_bundle",
]


def write_compare_artifact_bundle(
    artifact_dir: str | Path,
    *,
    results: pd.DataFrame,
    report_text: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Write one compare-runs artifact bundle plus static multi-run charts."""
    artifact_paths = write_artifact_bundle(
        artifact_dir,
        results=results,
        report_text=report_text,
        metadata=metadata,
    )
    chart_bundle = write_compare_chart_bundle(
        results,
        output_dir=Path(artifact_paths["artifact_dir"]) / "charts",
    )
    enriched_metadata = dict(metadata)
    enriched_metadata["chart_bundle"] = {
        "chart_dir": "charts",
        "manifest_path": str(Path("charts") / "manifest.json"),
        "chart_count": chart_bundle["chart_count"],
        "charts": chart_bundle["charts"],
    }
    write_json(enriched_metadata, artifact_paths["metadata_path"])
    artifact_paths["chart_dir"] = chart_bundle["chart_dir"]
    artifact_paths["chart_manifest_path"] = chart_bundle["manifest_path"]
    return artifact_paths


def build_compare_artifact_metadata(
    *,
    experiment_root: str,
    run_ids: list[str],
    selection_mode: str,
    command_name_filter: str | None,
    parameter_filter: str | None,
    rank_by: list[str] | None,
    rank_weight: list[str] | None,
    sort_by: str | None,
    ascending: bool | None,
    limit: int | None,
    results: pd.DataFrame,
) -> dict[str, Any]:
    """Build enriched Stage 4 metadata for a compare-runs artifact bundle."""
    return {
        "command": "compare-runs",
        "experiment_root": experiment_root,
        "run_ids": run_ids,
        "selection_mode": selection_mode,
        "command_name_filter": command_name_filter,
        "parameter_filter": parameter_filter,
        "rank_by": rank_by,
        "rank_weight": rank_weight,
        "sort_by": sort_by,
        "ascending": ascending,
        "limit": limit,
        "row_count": int(len(results)),
        "comparison_summary": _summarize_compare_artifact_results(results),
    }


def list_indexed_runs(
    experiment_root: str | Path,
    *,
    command_name: str | None = None,
    parameter_name: str | None = None,
    sort_by: str = "created_at",
    ascending: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    """List indexed runs with optional filtering and sorting."""
    dataset = load_run_index(experiment_root).copy()

    if command_name is not None:
        dataset = dataset.loc[dataset["command"] == command_name].copy()
    if parameter_name is not None:
        dataset = dataset.loc[dataset["parameter"] == parameter_name].copy()
    if dataset.empty:
        raise WorkflowError("No indexed runs match the requested filters.")

    sort_by = _normalize_run_index_sort_key(sort_by)
    if sort_by in {"row_count", "overall_cumulative_return"}:
        dataset[sort_by] = pd.to_numeric(dataset[sort_by], errors="coerce")
    dataset = dataset.sort_values(sort_by, ascending=ascending, kind="mergesort")

    if limit is not None:
        limit = normalize_positive_int(limit, parameter_name="limit")
        dataset = dataset.head(limit)

    columns = [
        "run_id",
        "created_at",
        "command",
        "parameter",
        "values",
        "selection_metric",
        "row_count",
        "overall_cumulative_return",
        "artifact_dir",
    ]
    return dataset.loc[:, columns].reset_index(drop=True)


def rank_compare_runs(
    experiment_root: str | Path,
    *,
    command_name: str | None = None,
    parameter_name: str | None = None,
    rank_by: list[str],
    rank_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Rank filtered runs by one or more enriched comparison metrics."""
    normalized_rank_by = _normalize_compare_rank_by(rank_by)
    normalized_rank_weights = _normalize_compare_rank_weights(
        normalized_rank_by,
        rank_weights=rank_weights,
    )
    dataset = load_run_index(experiment_root).copy()

    if command_name is not None:
        dataset = dataset.loc[dataset["command"] == command_name].copy()
    if parameter_name is not None:
        dataset = dataset.loc[dataset["parameter"] == parameter_name].copy()
    if dataset.empty:
        raise WorkflowError("No indexed runs match the requested filters.")

    compared = compare_indexed_runs(
        experiment_root,
        run_ids=dataset["run_id"].astype(str).tolist(),
    ).copy()
    rank_columns = []
    for metric in normalized_rank_by:
        compared[metric] = pd.to_numeric(compared[metric], errors="coerce")
        rank_column = f"rank_{metric}"
        compared[rank_column] = compared[metric].rank(
            method="min",
            ascending=False,
            na_option="bottom",
        )
        rank_columns.append(rank_column)

    compared["average_rank"] = compared.loc[:, rank_columns].mean(axis=1)
    sort_columns = ["average_rank", *normalized_rank_by, "created_at"]
    ascending = [True, *([False] * len(normalized_rank_by)), False]
    weight_columns: list[str] = []
    if normalized_rank_weights is not None:
        weighted_score = 0.0
        for metric in normalized_rank_by:
            weight_column = f"weight_{metric}"
            compared[weight_column] = normalized_rank_weights[metric]
            weight_columns.append(weight_column)
            weighted_score += compared[f"rank_{metric}"] * normalized_rank_weights[metric]
        compared["weighted_rank_score"] = weighted_score
        sort_columns = ["weighted_rank_score", *normalized_rank_by, "created_at"]
        ascending = [True, *([False] * len(normalized_rank_by)), False]

    compared = compared.sort_values(
        sort_columns,
        ascending=ascending,
        kind="mergesort",
    ).reset_index(drop=True)

    columns = [
        "run_id",
        "created_at",
        "command",
        "parameter",
        "summary_scope",
        *normalized_rank_by,
        *rank_columns,
        "average_rank",
        *weight_columns,
        "weighted_rank_score" if normalized_rank_weights is not None else None,
        "artifact_dir",
    ]
    return compared.loc[:, [column for column in columns if column is not None]]


def compare_indexed_runs(
    experiment_root: str | Path,
    *,
    run_ids: list[str],
) -> pd.DataFrame:
    """Select specific indexed runs and enrich them with bundle-level summary metrics."""
    subset = _select_indexed_runs_for_comparison(
        experiment_root,
        run_ids=run_ids,
    )

    rows = [
        _build_compared_run_row(row)
        for _, row in subset.iterrows()
    ]
    return pd.DataFrame(rows)


def build_compare_runs_report(
    experiment_root: str | Path,
    *,
    run_ids: list[str],
    sweep_top_k: int = 3,
) -> str:
    """Render a compare-runs report with one summary table plus command-specific details."""
    subset = _select_indexed_runs_for_comparison(
        experiment_root,
        run_ids=run_ids,
    )
    sweep_top_k = normalize_positive_int(
        sweep_top_k,
        parameter_name="sweep_top_k",
    )

    summary_rows: list[dict[str, Any]] = []
    detail_sections: list[str] = []
    for _, row in subset.iterrows():
        metadata, results = _load_compared_run_bundle(row)
        summary_rows.append(
            _build_compared_run_row_from_bundle(
                row,
                metadata=metadata,
                results=results,
            )
        )
        detail_sections.append(
            _build_compare_run_detail_section(
                row,
                results=results,
                sweep_top_k=sweep_top_k,
            )
        )

    sections = [format_run_index_table(pd.DataFrame(summary_rows), title="Compared Runs")]
    sections.extend(detail_sections)
    return "\n\n".join(sections)


def format_run_index_table(frame: pd.DataFrame, *, title: str) -> str:
    """Format a run index query result as plain text."""
    dataset = frame.copy()
    if "row_count" in dataset.columns:
        dataset["row_count"] = pd.to_numeric(dataset["row_count"], errors="coerce").map(
            lambda value: "NaN" if pd.isna(value) else str(int(value))
        )
    if "overall_cumulative_return" in dataset.columns:
        dataset["overall_cumulative_return"] = pd.to_numeric(
            dataset["overall_cumulative_return"],
            errors="coerce",
        ).map(_format_percent_or_nan)
    if "summary_cumulative_return" in dataset.columns:
        dataset["summary_cumulative_return"] = pd.to_numeric(
            dataset["summary_cumulative_return"],
            errors="coerce",
        ).map(_format_percent_or_nan)
    for column in ("summary_sharpe_ratio", "summary_mean_ic"):
        if column in dataset.columns:
            dataset[column] = pd.to_numeric(dataset[column], errors="coerce").map(
                _format_number_or_nan
            )
    numeric_rank_columns = [
        column
        for column in dataset.columns
        if column == "average_rank"
        or column == "weighted_rank_score"
        or column.startswith("rank_")
        or column.startswith("weight_")
    ]
    for column in numeric_rank_columns:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce").map(
            _format_number_or_nan
        )
    return title + "\n" + dataset.to_string(index=False)


def _summarize_compare_artifact_results(results: pd.DataFrame) -> dict[str, Any]:
    """Summarize the headline outputs from a compare-runs result table."""
    if results.empty:
        return {
            "commands": [],
            "summary_scopes": [],
            "best_run_by_cumulative_return": None,
            "best_run_by_mean_ic": None,
        }

    summary: dict[str, Any] = {
        "commands": sorted(results["command"].dropna().astype(str).unique().tolist())
        if "command" in results.columns
        else [],
        "summary_scopes": (
            sorted(results["summary_scope"].dropna().astype(str).unique().tolist())
            if "summary_scope" in results.columns
            else []
        ),
        "best_run_by_cumulative_return": None,
        "best_run_by_mean_ic": None,
    }

    if "summary_cumulative_return" in results.columns:
        sorted_by_return = results.assign(
            summary_cumulative_return=pd.to_numeric(
                results["summary_cumulative_return"],
                errors="coerce",
            )
        ).sort_values(
            ["summary_cumulative_return", "created_at"],
            ascending=[False, False],
            kind="mergesort",
        )
        sorted_by_return = sorted_by_return.loc[
            sorted_by_return["summary_cumulative_return"].notna()
        ]
        if not sorted_by_return.empty:
            summary["best_run_by_cumulative_return"] = _dataframe_records(
                sorted_by_return.loc[
                    :,
                    [
                        column
                        for column in [
                            "run_id",
                            "command",
                            "summary_scope",
                            "summary_cumulative_return",
                            "summary_sharpe_ratio",
                            "summary_mean_ic",
                        ]
                        if column in sorted_by_return.columns
                    ],
                ].head(1)
            )[0]

    if "summary_mean_ic" in results.columns:
        sorted_by_ic = results.assign(
            summary_mean_ic=pd.to_numeric(results["summary_mean_ic"], errors="coerce")
        ).sort_values(
            ["summary_mean_ic", "created_at"],
            ascending=[False, False],
            kind="mergesort",
        )
        sorted_by_ic = sorted_by_ic.loc[sorted_by_ic["summary_mean_ic"].notna()]
        if not sorted_by_ic.empty:
            summary["best_run_by_mean_ic"] = _dataframe_records(
                sorted_by_ic.loc[
                    :,
                    [
                        column
                        for column in [
                            "run_id",
                            "command",
                            "summary_scope",
                            "summary_cumulative_return",
                            "summary_sharpe_ratio",
                            "summary_mean_ic",
                        ]
                        if column in sorted_by_ic.columns
                    ],
                ].head(1)
            )[0]

    return summary


def _select_indexed_runs_for_comparison(
    experiment_root: str | Path,
    *,
    run_ids: list[str],
) -> pd.DataFrame:
    """Load and order the requested run ids for comparison."""
    if not run_ids:
        raise WorkflowError("compare-runs requires at least one run_id.")

    normalized_run_ids = []
    seen: set[str] = set()
    for run_id in run_ids:
        if not isinstance(run_id, str) or not run_id.strip():
            raise WorkflowError("run_id values must be non-empty strings.")
        normalized_run_id = run_id.strip()
        if normalized_run_id in seen:
            raise WorkflowError(
                f"run_id values must be unique; received duplicate {normalized_run_id}."
            )
        normalized_run_ids.append(normalized_run_id)
        seen.add(normalized_run_id)

    dataset = load_run_index(experiment_root).copy()
    subset = dataset.loc[dataset["run_id"].isin(normalized_run_ids)].copy()

    missing_run_ids = [
        run_id for run_id in normalized_run_ids if run_id not in set(subset["run_id"])
    ]
    if missing_run_ids:
        missing_text = ", ".join(missing_run_ids)
        raise WorkflowError(
            f"Requested run_id values were not found in runs.csv: {missing_text}."
        )

    order = {run_id: index for index, run_id in enumerate(normalized_run_ids)}
    subset["run_order"] = subset["run_id"].map(order)
    return subset.sort_values("run_order", kind="mergesort").reset_index(drop=True)


def _normalize_compare_rank_by(rank_by: list[str]) -> list[str]:
    """Validate supported multi-metric compare ranking inputs."""
    if not rank_by:
        raise WorkflowError("rank_by must contain at least one metric.")

    allowed = {
        "summary_cumulative_return",
        "summary_sharpe_ratio",
        "summary_mean_ic",
    }
    normalized: list[str] = []
    seen: set[str] = set()
    for metric in rank_by:
        if metric not in allowed:
            allowed_text = ", ".join(sorted(allowed))
            raise WorkflowError(f"rank_by must be one of {{{allowed_text}}}.")
        if metric in seen:
            raise WorkflowError(f"rank_by metrics must be unique; received duplicate {metric}.")
        normalized.append(metric)
        seen.add(metric)
    return normalized


def _normalize_compare_rank_weights(
    rank_by: list[str],
    *,
    rank_weights: dict[str, float] | None,
) -> dict[str, float] | None:
    """Validate and normalize compare-runs rank weights."""
    if rank_weights is None:
        return None

    rank_by_set = set(rank_by)
    weight_keys = set(rank_weights)
    if weight_keys != rank_by_set:
        missing = sorted(rank_by_set - weight_keys)
        extra = sorted(weight_keys - rank_by_set)
        problems: list[str] = []
        if missing:
            problems.append("missing " + ", ".join(missing))
        if extra:
            problems.append("unexpected " + ", ".join(extra))
        raise WorkflowError(
            "rank_weights must match rank_by exactly: " + "; ".join(problems) + "."
        )

    normalized: dict[str, float] = {}
    total_weight = 0.0
    for metric in rank_by:
        weight = rank_weights[metric]
        if pd.isna(weight) or weight <= 0:
            raise WorkflowError("rank_weights values must be positive finite numbers.")
        normalized[metric] = float(weight)
        total_weight += float(weight)

    if total_weight <= 0:
        raise WorkflowError("rank_weights values must sum to a positive number.")

    return {
        metric: weight / total_weight
        for metric, weight in normalized.items()
    }


def _build_compared_run_row(index_row: pd.Series) -> dict[str, Any]:
    """Build a richer comparison row by reading a run's bundle files."""
    metadata, results = _load_compared_run_bundle(index_row)
    return _build_compared_run_row_from_bundle(
        index_row,
        metadata=metadata,
        results=results,
    )


def _build_compared_run_row_from_bundle(
    index_row: pd.Series,
    *,
    metadata: dict[str, Any],
    results: pd.DataFrame,
) -> dict[str, Any]:
    """Build one summary row from already-loaded run bundle artifacts."""
    base_row = {
        "run_id": str(index_row["run_id"]),
        "created_at": str(index_row["created_at"]),
        "command": str(index_row["command"]),
        "config": str(index_row["config"]),
        "parameter": str(index_row["parameter"]),
        "candidate_values": _stringify_candidate_values(metadata.get("values")),
        "selection_metric": _stringify_optional_text(index_row.get("selection_metric")),
        "row_count": index_row.get("row_count"),
        "artifact_dir": str(index_row["artifact_dir"]),
    }

    command_name = str(index_row["command"])
    if command_name == "sweep-signal":
        return {
            **base_row,
            **_summarize_sweep_run(results),
        }
    if command_name == "walk-forward-signal":
        return {
            **base_row,
            **_summarize_walk_forward_run(results, metadata=metadata),
        }

    raise WorkflowError(f"Unsupported command in runs.csv: {command_name}")


def _load_compared_run_bundle(index_row: pd.Series) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load one run's metadata and results artifacts from its index row."""
    metadata = _load_json_file(
        Path(str(index_row["metadata_path"])),
        description="run metadata",
    )
    results = _load_csv_file(
        Path(str(index_row["results_path"])),
        description="run results",
    )
    return metadata, results


def _build_compare_run_detail_section(
    index_row: pd.Series,
    *,
    results: pd.DataFrame,
    sweep_top_k: int,
) -> str:
    """Build one command-specific detail section for compare-runs."""
    run_id = str(index_row["run_id"])
    parameter = str(index_row["parameter"])
    command_name = str(index_row["command"])

    if command_name == "sweep-signal":
        return _format_sweep_top_candidates_section(
            run_id=run_id,
            parameter=parameter,
            results=results,
            top_k=sweep_top_k,
        )
    if command_name == "walk-forward-signal":
        return _format_walk_forward_folds_section(
            run_id=run_id,
            parameter=parameter,
            results=results,
        )

    raise WorkflowError(f"Unsupported command in runs.csv: {command_name}")


def _summarize_sweep_run(results: pd.DataFrame) -> dict[str, Any]:
    """Summarize one sweep run from its results table."""
    required_columns = [
        "parameter_value",
        "cumulative_return",
        "sharpe_ratio",
        "mean_ic",
    ]
    _ensure_required_columns(results, required_columns, description="sweep results")

    dataset = results.copy()
    for column in required_columns:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
    best_index = dataset["cumulative_return"].idxmax()
    best_row = dataset.loc[best_index]
    return {
        "summary_parameter_values": _format_compact_numeric(best_row["parameter_value"]),
        "summary_scope": "best_in_sample_candidate",
        "summary_cumulative_return": best_row["cumulative_return"],
        "summary_sharpe_ratio": best_row["sharpe_ratio"],
        "summary_mean_ic": best_row["mean_ic"],
    }


def _summarize_walk_forward_run(
    results: pd.DataFrame,
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Summarize one walk-forward run from its results table and metadata."""
    required_columns = [
        "selected_parameter_value",
        "test_mean_ic",
    ]
    _ensure_required_columns(results, required_columns, description="walk-forward results")

    dataset = results.copy()
    dataset["selected_parameter_value"] = pd.to_numeric(
        dataset["selected_parameter_value"],
        errors="coerce",
    )
    dataset["test_mean_ic"] = pd.to_numeric(dataset["test_mean_ic"], errors="coerce")

    overall_summary = metadata.get("overall_summary")
    if not isinstance(overall_summary, dict):
        raise WorkflowError("walk-forward metadata.json is missing overall_summary.")

    summary_parameter_values = ",".join(
        _format_compact_numeric(value)
        for value in sorted(dataset["selected_parameter_value"].dropna().unique())
    )
    return {
        "summary_parameter_values": summary_parameter_values,
        "summary_scope": "overall_out_of_sample",
        "summary_cumulative_return": overall_summary.get("cumulative_return"),
        "summary_sharpe_ratio": overall_summary.get("sharpe_ratio"),
        "summary_mean_ic": dataset["test_mean_ic"].mean(),
    }


def _format_sweep_top_candidates_section(
    *,
    run_id: str,
    parameter: str,
    results: pd.DataFrame,
    top_k: int,
) -> str:
    """Format the top-k in-sample sweep candidates for one compared run."""
    dataset = validate_parameter_sweep_results(results)
    top_candidates = (
        dataset.sort_values(
            ["cumulative_return", "sharpe_ratio", "mean_ic", "parameter_value"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
        .head(top_k)
        .reset_index(drop=True)
    )
    formatted = top_candidates.loc[
        :,
        [
            "parameter_value",
            "cumulative_return",
            "max_drawdown",
            "sharpe_ratio",
            "mean_ic",
            "joint_coverage_ratio",
        ],
    ].copy()
    formatted["parameter_value"] = formatted["parameter_value"].map(
        _format_compact_numeric
    )
    for column in ("cumulative_return", "max_drawdown", "joint_coverage_ratio"):
        formatted[column] = formatted[column].map(_format_percent_or_nan)
    for column in ("sharpe_ratio", "mean_ic"):
        formatted[column] = formatted[column].map(_format_number_or_nan)

    return "\n".join(
        [
            f"Sweep Top Candidates: {run_id}",
            f"Parameter: {parameter}",
            f"Candidates Shown: {len(formatted)}/{len(dataset)}",
            formatted.to_string(index=False),
        ]
    )


def _format_walk_forward_folds_section(
    *,
    run_id: str,
    parameter: str,
    results: pd.DataFrame,
) -> str:
    """Format fold-level walk-forward diagnostics for one compared run."""
    dataset = validate_walk_forward_results(results)
    formatted = dataset.loc[
        :,
        [
            "fold_index",
            "selected_parameter_value",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "train_selection_score",
            "test_cumulative_return",
            "test_sharpe_ratio",
            "test_mean_ic",
            "test_joint_coverage_ratio",
        ],
    ].copy()
    formatted["fold_index"] = formatted["fold_index"].map(_format_compact_numeric)
    formatted["selected_parameter_value"] = formatted["selected_parameter_value"].map(
        _format_compact_numeric
    )

    selection_metric = str(dataset.loc[0, "selection_metric"])
    if selection_metric == "cumulative_return":
        formatted["train_selection_score"] = formatted["train_selection_score"].map(
            _format_percent_or_nan
        )
    else:
        formatted["train_selection_score"] = formatted["train_selection_score"].map(
            _format_number_or_nan
        )

    for column in ("test_cumulative_return", "test_joint_coverage_ratio"):
        formatted[column] = formatted[column].map(_format_percent_or_nan)
    for column in ("test_sharpe_ratio", "test_mean_ic"):
        formatted[column] = formatted[column].map(_format_number_or_nan)

    return "\n".join(
        [
            f"Walk-Forward Folds: {run_id}",
            f"Parameter: {parameter}",
            f"Selection Metric: {selection_metric}",
            formatted.to_string(index=False),
        ]
    )


def _load_json_file(path: Path, *, description: str) -> dict[str, Any]:
    """Load one JSON document from disk."""
    if not path.exists() or not path.is_file():
        raise WorkflowError(f"{description} file does not exist: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"Failed to read {description} file: {path}") from exc


def _load_csv_file(path: Path, *, description: str) -> pd.DataFrame:
    """Load one CSV document from disk."""
    if not path.exists() or not path.is_file():
        raise WorkflowError(f"{description} file does not exist: {path}")
    try:
        dataset = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise WorkflowError(f"Failed to read {description} file: {path}") from exc
    if dataset.empty:
        raise WorkflowError(f"{description} file does not contain any rows: {path}")
    return dataset


def _ensure_required_columns(
    frame: pd.DataFrame,
    required_columns: list[str],
    *,
    description: str,
) -> None:
    """Ensure a CSV-backed comparison frame has the required columns."""
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise WorkflowError(f"{description} are missing required columns: {missing_text}.")


def _stringify_candidate_values(values: Any) -> str:
    """Render candidate parameter values into a compact string."""
    if isinstance(values, list):
        return ",".join(str(value) for value in values)
    if pd.isna(values):
        return ""
    return str(values)


def _stringify_optional_text(value: Any) -> str:
    """Render optional textual values while preserving missing entries as empty strings."""
    if pd.isna(value):
        return ""
    return str(value)


def _normalize_run_index_sort_key(sort_by: str) -> str:
    """Validate supported run index sort keys."""
    if sort_by not in {"created_at", "command", "parameter", "row_count", "overall_cumulative_return"}:
        raise WorkflowError(
            "sort_by must be one of {'created_at', 'command', 'parameter', 'row_count', 'overall_cumulative_return'}."
        )
    return sort_by


def _format_compact_numeric(value: float) -> str:
    """Render numeric parameter values without unnecessary decimals."""
    if pd.isna(value):
        return "NaN"
    return str(int(value))


def _dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a small DataFrame into JSON-safe record dictionaries."""
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        records.append({str(key): _scalar_or_none(value) for key, value in row.items()})
    return records


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

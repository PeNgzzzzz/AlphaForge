"""Config-driven CLI workflows for AlphaForge."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from alphaforge.analytics import (
    summarize_backtest,
)
from alphaforge.cli.artifacts import (
    write_artifact_bundle,
    write_dataframe,
    write_indexed_artifact_bundle,
    write_json,
)
from alphaforge.cli.charts import (
    write_compare_chart_bundle,
    write_report_chart_bundle_from_context,
)
from alphaforge.cli.comparison import (
    build_compare_artifact_metadata,
    build_compare_runs_report,
    compare_indexed_runs,
    format_run_index_table,
    list_indexed_runs,
    rank_compare_runs,
    write_compare_artifact_bundle,
)
from alphaforge.cli.data_loading import (
    dataset_requires_benchmark_returns as _dataset_requires_benchmark_returns,
    dataset_requires_fundamentals as _dataset_requires_fundamentals,
    dataset_requires_memberships as _dataset_requires_memberships,
    dataset_requires_shares_outstanding as _dataset_requires_shares_outstanding,
    dataset_requires_trading_status as _dataset_requires_trading_status,
    load_benchmark_returns_from_config,
    load_borrow_availability_from_config,
    load_classifications_from_config,
    load_corporate_actions_from_config,
    load_fundamentals_from_config,
    load_market_data_from_config,
    load_memberships_from_config,
    load_shares_outstanding_from_config,
    load_symbol_metadata_from_config,
    load_trading_calendar_from_config,
    load_trading_status_from_config,
)
from alphaforge.cli.errors import WorkflowError
from alphaforge.cli.pipeline import (
    add_signal_from_config,
    build_dataset_from_config,
    build_dataset_from_market_data,
    build_weights_from_config,
    require_backtest_config,
    require_signal_config,
    run_backtest_from_config,
    run_backtest_with_config as _run_backtest_with_config,
)
from alphaforge.cli.report_context import (
    build_report_context,
    build_report_context as _build_report_context,
    validate_diagnostics_column as _validate_diagnostics_column,
)
from alphaforge.cli.research_metadata import (
    build_config_snapshot as _build_config_snapshot,
    build_dataset_feature_metadata_from_config as _build_dataset_feature_metadata_from_config,
    build_research_context_metadata,
    build_research_metadata_from_config as _build_research_metadata_from_config,
    build_signal_pipeline_metadata_from_config as _build_signal_pipeline_metadata_from_config,
    dataframe_records as _dataframe_records,
    scalar_or_none as _scalar_or_none,
    series_to_metadata_dict as _series_to_metadata_dict,
)
from alphaforge.cli.parameter_sweep import (
    build_sweep_artifact_metadata,
    normalize_sweep_values as _normalize_sweep_values,
    replace_signal_parameter as _replace_signal_parameter,
    run_signal_parameter_sweep,
)
from alphaforge.cli.reports import (
    _describe_signal_transform,
    _format_number_or_nan,
    _format_percent_or_nan,
    describe_benchmark_configuration,
    describe_benchmark_data,
    describe_borrow_availability_configuration,
    describe_borrow_availability_data,
    describe_classifications_configuration,
    describe_classifications_data,
    describe_corporate_actions_configuration,
    describe_corporate_actions_data,
    describe_data_quality,
    describe_diagnostics_overview,
    describe_execution_configuration,
    describe_execution_results,
    describe_fundamentals_configuration,
    describe_fundamentals_data,
    describe_market_data,
    describe_memberships_configuration,
    describe_memberships_data,
    describe_portfolio_constraints,
    describe_research_workflow,
    describe_shares_outstanding_configuration,
    describe_shares_outstanding_data,
    describe_symbol_metadata_configuration,
    describe_symbol_metadata_data,
    describe_trading_calendar_configuration,
    describe_trading_calendar_data,
    describe_trading_status_configuration,
    describe_trading_status_data,
    describe_universe_configuration,
    describe_universe_eligibility,
    build_report_metadata,
    render_report_text,
    write_report_html_page,
)
from alphaforge.cli.validation import normalize_positive_int
from alphaforge.cli.validation_report import build_validate_data_text
from alphaforge.cli.walk_forward import (
    build_walk_forward_artifact_metadata as _build_walk_forward_artifact_metadata,
    build_walk_forward_folds,
    evaluate_walk_forward_slice,
    extract_unique_dates,
    extract_walk_forward_selection_score,
    normalize_walk_forward_selection_metric,
)
from alphaforge.common import AlphaForgeConfig


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
    context = _build_report_context(config)
    report_text = render_report_text(context, config=config)
    workflow_configuration = _build_config_snapshot(config)
    research_metadata = _build_research_metadata_from_config(config)
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
    context = _build_report_context(config)
    report_text = render_report_text(context, config=config)
    workflow_configuration = _build_config_snapshot(config)
    research_metadata = _build_research_metadata_from_config(config)
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
    context = _build_report_context(config)
    return write_report_chart_bundle_from_context(
        context,
        output_dir=output_dir,
        workflow_configuration=_build_config_snapshot(config),
        config_path=config_path,
        command_name="plot-report",
    )


def run_walk_forward_parameter_selection(
    config: AlphaForgeConfig,
    *,
    parameter_name: str,
    values: list[int],
    train_periods: int,
    test_periods: int,
    selection_metric: str = "cumulative_return",
) -> tuple[pd.DataFrame, pd.Series]:
    """Run a rolling walk-forward evaluation with in-sample parameter selection."""
    require_backtest_config(config)
    candidate_values = _normalize_sweep_values(values)
    train_periods = normalize_positive_int(
        train_periods,
        parameter_name="train_periods",
    )
    test_periods = normalize_positive_int(
        test_periods,
        parameter_name="test_periods",
    )
    selection_metric = normalize_walk_forward_selection_metric(selection_metric)

    dataset = build_dataset_from_config(config)
    unique_dates = extract_unique_dates(dataset)
    folds = build_walk_forward_folds(
        unique_dates,
        train_periods=train_periods,
        test_periods=test_periods,
    )

    candidates = []
    for parameter_value in candidate_values:
        candidate_config = _replace_signal_parameter(
            config,
            parameter_name=parameter_name,
            parameter_value=parameter_value,
        )
        signaled, signal_column = add_signal_from_config(dataset, candidate_config)
        _validate_diagnostics_column(signaled, candidate_config)
        weighted = build_weights_from_config(
            signaled,
            score_column=signal_column,
            config=candidate_config,
        )
        candidates.append(
            {
                "parameter_value": parameter_value,
                "signal_column": signal_column,
                "signaled": signaled,
                "weighted": weighted,
            }
        )

    fold_rows = []
    oos_backtests: list[pd.DataFrame] = []
    for fold_index, fold in enumerate(folds, start=1):
        best_candidate = None
        best_score = float("-inf")
        best_train_evaluation = None

        for candidate in candidates:
            train_evaluation = evaluate_walk_forward_slice(
                signaled=candidate["signaled"],
                weighted=candidate["weighted"],
                signal_column=candidate["signal_column"],
                config=config,
                evaluation_dates=fold["train_dates"],
            )
            candidate_score = extract_walk_forward_selection_score(
                train_evaluation,
                selection_metric=selection_metric,
            )
            if candidate_score > best_score:
                best_candidate = candidate
                best_score = candidate_score
                best_train_evaluation = train_evaluation

        if best_candidate is None or best_train_evaluation is None:
            raise WorkflowError("walk-forward selection could not choose a valid candidate.")

        test_evaluation = evaluate_walk_forward_slice(
            signaled=best_candidate["signaled"],
            weighted=best_candidate["weighted"],
            signal_column=best_candidate["signal_column"],
            config=config,
            evaluation_dates=fold["test_dates"],
        )
        oos_backtests.append(test_evaluation["backtest"])

        fold_rows.append(
            {
                "fold_index": float(fold_index),
                "parameter_name": parameter_name,
                "selected_parameter_value": float(best_candidate["parameter_value"]),
                "signal_column": best_candidate["signal_column"],
                "selection_metric": selection_metric,
                "train_start": fold["train_dates"][0].date().isoformat(),
                "train_end": fold["train_dates"][-1].date().isoformat(),
                "test_start": fold["test_dates"][0].date().isoformat(),
                "test_end": fold["test_dates"][-1].date().isoformat(),
                "train_selection_score": best_score,
                "train_cumulative_return": best_train_evaluation["performance_summary"][
                    "cumulative_return"
                ],
                "train_mean_ic": best_train_evaluation["ic_summary"]["mean_ic"],
                "test_cumulative_return": test_evaluation["performance_summary"][
                    "cumulative_return"
                ],
                "test_max_drawdown": test_evaluation["performance_summary"][
                    "max_drawdown"
                ],
                "test_sharpe_ratio": test_evaluation["performance_summary"][
                    "sharpe_ratio"
                ],
                "test_mean_ic": test_evaluation["ic_summary"]["mean_ic"],
                "test_joint_coverage_ratio": test_evaluation["coverage_summary"][
                    "joint_coverage_ratio"
                ],
            }
        )

    combined_oos_backtest = pd.concat(oos_backtests, ignore_index=True).sort_values(
        "date",
        kind="mergesort",
    )
    overall_summary = summarize_backtest(combined_oos_backtest.reset_index(drop=True))
    return pd.DataFrame(fold_rows), overall_summary


def build_walk_forward_artifact_metadata(
    config: AlphaForgeConfig,
    *,
    config_path: str,
    parameter_name: str,
    values: list[int],
    train_periods: int,
    test_periods: int,
    selection_metric: str,
    fold_results: pd.DataFrame,
    overall_summary: pd.Series,
) -> dict[str, Any]:
    """Build enriched Stage 4 metadata for a walk-forward artifact bundle."""
    return _build_walk_forward_artifact_metadata(
        config_path=config_path,
        parameter_name=parameter_name,
        values=values,
        train_periods=train_periods,
        test_periods=test_periods,
        selection_metric=selection_metric,
        fold_results=fold_results,
        overall_summary=overall_summary,
        research_context=build_research_context_metadata(config),
    )

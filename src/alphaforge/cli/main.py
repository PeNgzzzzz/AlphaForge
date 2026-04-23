"""Command-line interface for AlphaForge."""

from __future__ import annotations

import argparse
import math
import sys
from typing import Optional, Sequence

from alphaforge import __version__
from alphaforge.analytics import (
    AnalyticsError,
    FactorDiagnosticsError,
    ParameterSweepError,
    VisualizationError,
    format_parameter_sweep_results,
    format_performance_summary,
    WalkForwardError,
    format_walk_forward_report,
)
from alphaforge.backtest import BacktestError
from alphaforge.common import ConfigError, load_pipeline_config
from alphaforge.cli.workflows import (
    WorkflowError,
    build_compare_artifact_metadata,
    build_report_package,
    build_validate_data_text,
    build_compare_runs_report,
    compare_indexed_runs,
    build_dataset_from_config,
    build_sweep_artifact_metadata,
    build_walk_forward_artifact_metadata,
    describe_market_data,
    format_run_index_table,
    list_indexed_runs,
    load_market_data_from_config,
    rank_compare_runs,
    run_backtest_from_config,
    run_signal_parameter_sweep,
    run_walk_forward_parameter_selection,
    write_report_artifact_bundle,
    write_artifact_bundle,
    write_compare_artifact_bundle,
    write_report_chart_bundle,
    write_indexed_artifact_bundle,
    write_dataframe,
)
from alphaforge.data import DataValidationError
from alphaforge.portfolio import PortfolioConstructionError
from alphaforge.risk import RiskError


def _handle_validate_data(args: argparse.Namespace) -> int:
    """Run the validate-data CLI command."""
    config = load_pipeline_config(args.config)
    print(build_validate_data_text(config))
    return 0


def _handle_build_dataset(args: argparse.Namespace) -> int:
    """Run the build-dataset CLI command."""
    config = load_pipeline_config(args.config)
    dataset = build_dataset_from_config(config)

    if args.output is not None:
        output_path = write_dataframe(dataset, args.output)
        print(f"Saved dataset to {output_path}")
    else:
        print(dataset.to_string(index=False))
    return 0


def _handle_run_backtest(args: argparse.Namespace) -> int:
    """Run the run-backtest CLI command."""
    config = load_pipeline_config(args.config)
    backtest = run_backtest_from_config(config)

    if args.output is not None:
        output_path = write_dataframe(backtest, args.output)
        print(f"Saved backtest results to {output_path}")
    else:
        print(backtest.to_string(index=False))
    return 0


def _handle_report(args: argparse.Namespace) -> int:
    """Run the report CLI command."""
    config = load_pipeline_config(args.config)
    if args.artifact_dir is not None:
        artifact_paths = write_report_artifact_bundle(
            config,
            args.artifact_dir,
            config_path=args.config,
        )
        print(f"Saved report artifacts to {artifact_paths['artifact_dir']}")
        print(f"Saved report charts to {artifact_paths['chart_dir']}")
    else:
        _, report_text, _ = build_report_package(config, config_path=args.config)
        print(report_text)
    return 0


def _handle_plot_report(args: argparse.Namespace) -> int:
    """Run the plot-report CLI command."""
    config = load_pipeline_config(args.config)
    chart_paths = write_report_chart_bundle(
        config,
        args.output_dir,
        config_path=args.config,
    )
    print(f"Saved report charts to {chart_paths['chart_dir']}")
    print(f"Saved chart manifest to {chart_paths['manifest_path']}")
    return 0


def _handle_sweep_signal(args: argparse.Namespace) -> int:
    """Run the sweep-signal CLI command."""
    config = load_pipeline_config(args.config)
    results = run_signal_parameter_sweep(
        config,
        parameter_name=args.parameter,
        values=args.values,
    )
    report_text = format_parameter_sweep_results(results)

    if args.experiment_root is not None:
        metadata = build_sweep_artifact_metadata(
            config,
            config_path=args.config,
            parameter_name=args.parameter,
            values=args.values,
            results=results,
        )
        artifact_paths = write_indexed_artifact_bundle(
            args.experiment_root,
            command_name="sweep-signal",
            results=results,
            report_text=report_text,
            metadata=metadata,
        )
        print(f"Saved sweep experiment run to {artifact_paths['artifact_dir']}")
        print(f"Updated run index at {artifact_paths['index_path']}")
    elif args.artifact_dir is not None:
        metadata = build_sweep_artifact_metadata(
            config,
            config_path=args.config,
            parameter_name=args.parameter,
            values=args.values,
            results=results,
        )
        artifact_paths = write_artifact_bundle(
            args.artifact_dir,
            results=results,
            report_text=report_text,
            metadata=metadata,
        )
        print(f"Saved sweep artifacts to {artifact_paths['artifact_dir']}")
    elif args.output is not None:
        output_path = write_dataframe(results, args.output)
        print(f"Saved sweep results to {output_path}")
    else:
        print(report_text)
    return 0


def _handle_walk_forward_signal(args: argparse.Namespace) -> int:
    """Run the walk-forward-signal CLI command."""
    config = load_pipeline_config(args.config)
    fold_results, overall_summary = run_walk_forward_parameter_selection(
        config,
        parameter_name=args.parameter,
        values=args.values,
        train_periods=args.train_periods,
        test_periods=args.test_periods,
        selection_metric=args.selection_metric,
    )
    report_text = format_walk_forward_report(fold_results, overall_summary)

    if args.experiment_root is not None:
        metadata = build_walk_forward_artifact_metadata(
            config,
            config_path=args.config,
            parameter_name=args.parameter,
            values=args.values,
            train_periods=args.train_periods,
            test_periods=args.test_periods,
            selection_metric=args.selection_metric,
            fold_results=fold_results,
            overall_summary=overall_summary,
        )
        artifact_paths = write_indexed_artifact_bundle(
            args.experiment_root,
            command_name="walk-forward-signal",
            results=fold_results,
            report_text=report_text,
            metadata=metadata,
        )
        print(f"Saved walk-forward experiment run to {artifact_paths['artifact_dir']}")
        print(f"Updated run index at {artifact_paths['index_path']}")
    elif args.artifact_dir is not None:
        metadata = build_walk_forward_artifact_metadata(
            config,
            config_path=args.config,
            parameter_name=args.parameter,
            values=args.values,
            train_periods=args.train_periods,
            test_periods=args.test_periods,
            selection_metric=args.selection_metric,
            fold_results=fold_results,
            overall_summary=overall_summary,
        )
        artifact_paths = write_artifact_bundle(
            args.artifact_dir,
            results=fold_results,
            report_text=report_text,
            metadata=metadata,
        )
        print(f"Saved walk-forward artifacts to {artifact_paths['artifact_dir']}")
    elif args.output is not None:
        output_path = write_dataframe(fold_results, args.output)
        print(format_performance_summary(overall_summary))
        print(f"Saved walk-forward results to {output_path}")
    else:
        print(report_text)
    return 0


def _handle_list_runs(args: argparse.Namespace) -> int:
    """Run the list-runs CLI command."""
    runs = list_indexed_runs(
        args.experiment_root,
        command_name=args.command_name,
        parameter_name=args.parameter,
        sort_by=args.sort_by,
        ascending=args.ascending,
        limit=args.limit,
    )
    print(format_run_index_table(runs, title="Indexed Runs"))
    return 0


def _handle_compare_runs(args: argparse.Namespace) -> int:
    """Run the compare-runs CLI command."""
    run_ids, selection_text = _resolve_compare_run_ids(args)
    results = compare_indexed_runs(
        args.experiment_root,
        run_ids=run_ids,
    )
    report_text = build_compare_runs_report(
        args.experiment_root,
        run_ids=run_ids,
    )
    if selection_text is not None:
        report_text = selection_text + "\n\n" + report_text

    if args.artifact_dir is not None:
        selection_mode = (
            "explicit_run_ids"
            if args.run_id is not None
            else (
                "ranked_filtered_index_query"
                if args.rank_by is not None
                else "filtered_index_query"
            )
        )
        metadata = build_compare_artifact_metadata(
            experiment_root=args.experiment_root,
            run_ids=run_ids,
            selection_mode=selection_mode,
            command_name_filter=args.command_name,
            parameter_filter=args.parameter,
            rank_by=list(args.rank_by) if args.rank_by is not None else None,
            rank_weight=list(args.rank_weight) if args.rank_weight is not None else None,
            sort_by=args.sort_by if args.run_id is None else None,
            ascending=bool(args.ascending) if args.run_id is None else None,
            limit=args.limit if args.run_id is None else None,
            results=results,
        )
        artifact_paths = write_compare_artifact_bundle(
            args.artifact_dir,
            results=results,
            report_text=report_text,
            metadata=metadata,
        )
        print(f"Saved compare artifacts to {artifact_paths['artifact_dir']}")
        print(f"Saved compare charts to {artifact_paths['chart_dir']}")
    else:
        print(report_text)
    return 0


def _resolve_compare_run_ids(args: argparse.Namespace) -> tuple[list[str], Optional[str]]:
    """Resolve compare-runs inputs into ordered run ids and an optional selection summary."""
    if args.run_id is not None:
        if (
            args.command_name is not None
            or args.parameter is not None
            or args.rank_by is not None
            or args.rank_weight is not None
            or args.limit is not None
            or args.ascending
            or args.sort_by != "created_at"
        ):
            raise WorkflowError(
                "compare-runs cannot combine explicit --run-id values with automatic selection options."
            )
        return list(args.run_id), None

    if args.limit is None:
        raise WorkflowError(
            "compare-runs without --run-id requires --limit to keep automatic selection explicit."
        )

    rank_weights = _parse_compare_rank_weights(
        args.rank_weight,
        rank_by=args.rank_by,
    )
    if args.rank_by is not None:
        if args.sort_by != "created_at" or args.ascending:
            raise WorkflowError(
                "compare-runs cannot combine --rank-by with --sort-by or --ascending."
            )
        ranked_runs = rank_compare_runs(
            args.experiment_root,
            command_name=args.command_name,
            parameter_name=args.parameter,
            rank_by=list(args.rank_by),
            rank_weights=rank_weights,
        )
        ranked_runs = ranked_runs.head(args.limit).reset_index(drop=True)
        selection_text = format_run_index_table(
            ranked_runs,
            title="Ranked Compare Selection",
        )
        return ranked_runs["run_id"].astype(str).tolist(), selection_text

    selected_runs = list_indexed_runs(
        args.experiment_root,
        command_name=args.command_name,
        parameter_name=args.parameter,
        sort_by=args.sort_by,
        ascending=args.ascending,
        limit=args.limit,
    )
    return selected_runs["run_id"].astype(str).tolist(), None


def _parse_compare_rank_weights(
    rank_weight_args: list[str] | None,
    *,
    rank_by: list[str] | None,
) -> Optional[dict[str, float]]:
    """Parse CLI compare rank weights from `metric=value` strings."""
    if rank_weight_args is None:
        return None
    if rank_by is None:
        raise WorkflowError("compare-runs --rank-weight requires --rank-by.")

    parsed: dict[str, float] = {}
    for item in rank_weight_args:
        if "=" not in item:
            raise WorkflowError(
                "compare-runs --rank-weight values must use the form metric=value."
            )
        metric, value_text = item.split("=", 1)
        metric = metric.strip()
        value_text = value_text.strip()
        if metric == "" or value_text == "":
            raise WorkflowError(
                "compare-runs --rank-weight values must use the form metric=value."
            )
        if metric in parsed:
            raise WorkflowError(
                f"compare-runs --rank-weight received duplicate metric {metric}."
            )
        try:
            value = float(value_text)
        except ValueError as exc:
            raise WorkflowError(
                f"compare-runs --rank-weight value for {metric} must be numeric."
            ) from exc
        if not math.isfinite(value) or value <= 0:
            raise WorkflowError(
                "compare-runs --rank-weight values must be positive finite numbers."
            )
        parsed[metric] = value

    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="alphaforge",
        description="AlphaForge is a modular quant research and development workbench.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    validate_parser = subparsers.add_parser(
        "validate-data",
        help="Load and validate market data from a pipeline config.",
    )
    validate_parser.add_argument("--config", required=True, help="Path to a TOML config.")
    validate_parser.set_defaults(handler=_handle_validate_data)

    dataset_parser = subparsers.add_parser(
        "build-dataset",
        help="Build a research dataset from a pipeline config.",
    )
    dataset_parser.add_argument("--config", required=True, help="Path to a TOML config.")
    dataset_parser.add_argument(
        "--output",
        help="Optional CSV output path for the built dataset.",
    )
    dataset_parser.set_defaults(handler=_handle_build_dataset)

    backtest_parser = subparsers.add_parser(
        "run-backtest",
        help="Run the configured backtest pipeline.",
    )
    backtest_parser.add_argument("--config", required=True, help="Path to a TOML config.")
    backtest_parser.add_argument(
        "--output",
        help="Optional CSV output path for daily backtest results.",
    )
    backtest_parser.set_defaults(handler=_handle_run_backtest)

    report_parser = subparsers.add_parser(
        "report",
        help="Run the full configured pipeline and print a text report.",
    )
    report_parser.add_argument("--config", required=True, help="Path to a TOML config.")
    report_parser.add_argument(
        "--artifact-dir",
        help="Optional artifact directory for results.csv, report.txt, and metadata.json.",
    )
    report_parser.set_defaults(handler=_handle_report)

    plot_report_parser = subparsers.add_parser(
        "plot-report",
        help="Run the full configured pipeline and write static report charts.",
    )
    plot_report_parser.add_argument(
        "--config",
        required=True,
        help="Path to a TOML config.",
    )
    plot_report_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory path for PNG charts and manifest.json.",
    )
    plot_report_parser.set_defaults(handler=_handle_plot_report)

    sweep_parser = subparsers.add_parser(
        "sweep-signal",
        help="Run a simple signal parameter sweep from a pipeline config.",
    )
    sweep_parser.add_argument("--config", required=True, help="Path to a TOML config.")
    sweep_parser.add_argument(
        "--parameter",
        required=True,
        choices=["lookback", "short_window", "long_window"],
        help="Signal parameter to sweep.",
    )
    sweep_parser.add_argument(
        "--values",
        required=True,
        nargs="+",
        type=int,
        help="One or more positive integer candidate values.",
    )
    sweep_output_group = sweep_parser.add_mutually_exclusive_group()
    sweep_output_group.add_argument(
        "--output",
        help="Optional CSV output path for sweep results.",
    )
    sweep_output_group.add_argument(
        "--artifact-dir",
        help="Optional artifact directory for results.csv, report.txt, and metadata.json.",
    )
    sweep_output_group.add_argument(
        "--experiment-root",
        help="Optional experiment root that creates timestamped run directories and updates runs.csv.",
    )
    sweep_parser.set_defaults(handler=_handle_sweep_signal)

    walk_forward_parser = subparsers.add_parser(
        "walk-forward-signal",
        help="Run a conservative walk-forward signal parameter selection workflow.",
    )
    walk_forward_parser.add_argument(
        "--config",
        required=True,
        help="Path to a TOML config.",
    )
    walk_forward_parser.add_argument(
        "--parameter",
        required=True,
        choices=["lookback", "short_window", "long_window"],
        help="Signal parameter to select walk-forward candidates from.",
    )
    walk_forward_parser.add_argument(
        "--values",
        required=True,
        nargs="+",
        type=int,
        help="One or more positive integer candidate values.",
    )
    walk_forward_parser.add_argument(
        "--train-periods",
        required=True,
        type=int,
        help="Number of unique dates in each train fold.",
    )
    walk_forward_parser.add_argument(
        "--test-periods",
        required=True,
        type=int,
        help="Number of unique dates in each test fold.",
    )
    walk_forward_parser.add_argument(
        "--selection-metric",
        default="cumulative_return",
        choices=["cumulative_return", "sharpe_ratio", "mean_ic"],
        help="Train-fold metric used to select the winning candidate.",
    )
    walk_forward_output_group = walk_forward_parser.add_mutually_exclusive_group()
    walk_forward_output_group.add_argument(
        "--output",
        help="Optional CSV output path for walk-forward fold results.",
    )
    walk_forward_output_group.add_argument(
        "--artifact-dir",
        help="Optional artifact directory for results.csv, report.txt, and metadata.json.",
    )
    walk_forward_output_group.add_argument(
        "--experiment-root",
        help="Optional experiment root that creates timestamped run directories and updates runs.csv.",
    )
    walk_forward_parser.set_defaults(handler=_handle_walk_forward_signal)

    list_runs_parser = subparsers.add_parser(
        "list-runs",
        help="List indexed experiment runs from an experiment root.",
    )
    list_runs_parser.add_argument(
        "--experiment-root",
        required=True,
        help="Path to an experiment root containing runs.csv.",
    )
    list_runs_parser.add_argument(
        "--command-name",
        choices=["sweep-signal", "walk-forward-signal"],
        help="Optional command filter.",
    )
    list_runs_parser.add_argument(
        "--parameter",
        choices=["lookback", "short_window", "long_window"],
        help="Optional parameter filter.",
    )
    list_runs_parser.add_argument(
        "--sort-by",
        default="created_at",
        choices=["created_at", "command", "parameter", "row_count", "overall_cumulative_return"],
        help="Run index column used for sorting.",
    )
    list_runs_parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort in ascending order instead of descending.",
    )
    list_runs_parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of runs to display.",
    )
    list_runs_parser.set_defaults(handler=_handle_list_runs)

    compare_runs_parser = subparsers.add_parser(
        "compare-runs",
        help="Compare specific indexed experiment runs from an experiment root.",
    )
    compare_runs_parser.add_argument(
        "--experiment-root",
        required=True,
        help="Path to an experiment root containing runs.csv.",
    )
    compare_runs_parser.add_argument(
        "--run-id",
        nargs="+",
        help="One or more explicit run_id values from runs.csv.",
    )
    compare_runs_parser.add_argument(
        "--command-name",
        choices=["sweep-signal", "walk-forward-signal"],
        help="Optional command filter for automatic run selection.",
    )
    compare_runs_parser.add_argument(
        "--parameter",
        choices=["lookback", "short_window", "long_window"],
        help="Optional parameter filter for automatic run selection.",
    )
    compare_runs_parser.add_argument(
        "--rank-by",
        nargs="+",
        choices=[
            "summary_cumulative_return",
            "summary_sharpe_ratio",
            "summary_mean_ic",
        ],
        help="Optional enriched metrics used for average-rank compare selection.",
    )
    compare_runs_parser.add_argument(
        "--rank-weight",
        nargs="+",
        help="Optional metric=value weights used together with --rank-by.",
    )
    compare_runs_parser.add_argument(
        "--sort-by",
        default="created_at",
        choices=["created_at", "command", "parameter", "row_count", "overall_cumulative_return"],
        help="Run index column used when automatically selecting runs.",
    )
    compare_runs_parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort in ascending order when automatically selecting runs.",
    )
    compare_runs_parser.add_argument(
        "--limit",
        type=int,
        help="Required when omitting --run-id; selects the top N filtered runs.",
    )
    compare_runs_parser.add_argument(
        "--artifact-dir",
        help="Optional directory path for a compare-runs artifact bundle.",
    )
    compare_runs_parser.set_defaults(handler=_handle_compare_runs)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the AlphaForge CLI."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not hasattr(args, "handler"):
        parser.print_help()
        return 0

    try:
        return int(args.handler(args))
    except (
        AnalyticsError,
        BacktestError,
        ConfigError,
        DataValidationError,
        FactorDiagnosticsError,
        ParameterSweepError,
        PortfolioConstructionError,
        RiskError,
        VisualizationError,
        WalkForwardError,
        WorkflowError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

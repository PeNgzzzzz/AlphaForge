"""Walk-forward evaluation helpers for CLI workflows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from alphaforge.analytics import (
    compute_ic_series,
    summarize_backtest,
    summarize_ic,
    summarize_signal_coverage,
)
from alphaforge.backtest import run_daily_backtest
from alphaforge.cli.errors import WorkflowError
from alphaforge.common import AlphaForgeConfig

__all__ = [
    "build_walk_forward_artifact_metadata",
    "build_walk_forward_folds",
    "evaluate_walk_forward_slice",
    "extract_unique_dates",
    "extract_walk_forward_selection_score",
    "normalize_walk_forward_selection_metric",
    "select_augmented_dates",
]


def build_walk_forward_artifact_metadata(
    *,
    config_path: str,
    parameter_name: str,
    values: list[int],
    train_periods: int,
    test_periods: int,
    selection_metric: str,
    fold_results: pd.DataFrame,
    overall_summary: pd.Series,
    research_context: dict[str, Any],
) -> dict[str, Any]:
    """Build enriched metadata for a walk-forward artifact bundle."""
    selected_values = pd.Series(dtype=float)
    if "selected_parameter_value" in fold_results.columns:
        selected_values = pd.to_numeric(
            fold_results["selected_parameter_value"],
            errors="coerce",
        ).dropna()

    selection_distribution: dict[str, int] = {}
    if not selected_values.empty:
        value_counts = selected_values.value_counts(sort=False).sort_index()
        selection_distribution = {
            _format_compact_numeric(float(value)): int(count)
            for value, count in value_counts.items()
        }

    return {
        "command": "walk-forward-signal",
        "config": config_path,
        "parameter": parameter_name,
        "values": values,
        "train_periods": train_periods,
        "test_periods": test_periods,
        "selection_metric": selection_metric,
        "row_count": int(len(fold_results)),
        "overall_summary": _series_to_metadata_dict(overall_summary),
        "research_context": research_context,
        "fold_count": int(len(fold_results)),
        "selected_parameter_values": [
            _scalar_or_none(value)
            for value in sorted(selected_values.unique().tolist())
        ],
        "selection_distribution": selection_distribution,
        "test_period_start": (
            str(fold_results["test_start"].min())
            if "test_start" in fold_results.columns and not fold_results.empty
            else None
        ),
        "test_period_end": (
            str(fold_results["test_end"].max())
            if "test_end" in fold_results.columns and not fold_results.empty
            else None
        ),
    }


def evaluate_walk_forward_slice(
    *,
    signaled: pd.DataFrame,
    weighted: pd.DataFrame,
    signal_column: str,
    config: AlphaForgeConfig,
    evaluation_dates: list[pd.Timestamp],
) -> dict[str, object]:
    """Evaluate one date slice from a precomputed candidate panel."""
    if not evaluation_dates:
        raise WorkflowError("walk-forward evaluation dates must not be empty.")

    backtest_config = _require_backtest_config(config)

    history_periods: int | None = backtest_config.signal_delay + 1
    if (
        backtest_config.rebalance_frequency != "daily"
        or backtest_config.max_turnover is not None
    ):
        history_periods = None

    augmented_dates = select_augmented_dates(
        weighted,
        evaluation_dates=evaluation_dates,
        history_periods=history_periods,
    )
    weighted_slice = weighted.loc[weighted["date"].isin(augmented_dates)].copy()
    backtest = _run_backtest_with_config(weighted_slice, config=config)
    filtered_backtest = (
        backtest.loc[backtest["date"].isin(evaluation_dates)]
        .sort_values("date", kind="mergesort")
        .reset_index(drop=True)
    )
    if filtered_backtest.empty:
        raise WorkflowError("walk-forward backtest evaluation produced no rows.")

    diagnostics_slice = (
        signaled.loc[signaled["date"].isin(evaluation_dates)]
        .sort_values(["date", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
    ic_series = compute_ic_series(
        diagnostics_slice,
        signal_column=signal_column,
        forward_return_column=config.diagnostics.forward_return_column,
        method=config.diagnostics.ic_method,
        min_observations=config.diagnostics.min_observations,
    )

    return {
        "backtest": filtered_backtest,
        "performance_summary": summarize_backtest(filtered_backtest),
        "ic_summary": summarize_ic(ic_series),
        "coverage_summary": summarize_signal_coverage(
            diagnostics_slice,
            signal_column=signal_column,
            forward_return_column=config.diagnostics.forward_return_column,
        ),
    }


def extract_walk_forward_selection_score(
    evaluation: dict[str, object],
    *,
    selection_metric: str,
) -> float:
    """Extract one train-slice selection metric from a fold evaluation."""
    performance_summary = evaluation["performance_summary"]
    ic_summary = evaluation["ic_summary"]

    if not isinstance(performance_summary, pd.Series) or not isinstance(
        ic_summary, pd.Series
    ):
        raise WorkflowError("walk-forward evaluation summaries must be pandas Series.")

    if selection_metric == "cumulative_return":
        score = performance_summary["cumulative_return"]
    elif selection_metric == "sharpe_ratio":
        score = performance_summary["sharpe_ratio"]
    else:
        score = ic_summary["mean_ic"]

    if pd.isna(score):
        return float("-inf")
    return float(score)


def normalize_walk_forward_selection_metric(selection_metric: str) -> str:
    """Validate the train-slice metric used for walk-forward selection."""
    if selection_metric not in {"cumulative_return", "sharpe_ratio", "mean_ic"}:
        raise WorkflowError(
            "selection_metric must be one of {'cumulative_return', 'sharpe_ratio', 'mean_ic'}."
        )
    return selection_metric


def build_walk_forward_folds(
    unique_dates: list[pd.Timestamp],
    *,
    train_periods: int,
    test_periods: int,
) -> list[dict[str, list[pd.Timestamp]]]:
    """Create rolling fixed-length train/test folds over unique dates."""
    if len(unique_dates) < train_periods + test_periods:
        raise WorkflowError(
            "walk-forward requires at least train_periods + test_periods unique dates."
        )

    folds: list[dict[str, list[pd.Timestamp]]] = []
    split_index = train_periods
    while split_index + test_periods <= len(unique_dates):
        folds.append(
            {
                "train_dates": unique_dates[split_index - train_periods : split_index],
                "test_dates": unique_dates[split_index : split_index + test_periods],
            }
        )
        split_index += test_periods

    if not folds:
        raise WorkflowError("walk-forward configuration produced no valid folds.")

    return folds


def extract_unique_dates(frame: pd.DataFrame) -> list[pd.Timestamp]:
    """Extract sorted unique dates from a dated panel."""
    if "date" not in frame.columns:
        raise WorkflowError("dated workflow inputs must contain a 'date' column.")

    unique_dates = (
        pd.to_datetime(frame["date"], errors="coerce")
        .drop_duplicates()
        .sort_values(kind="mergesort")
        .tolist()
    )
    if not unique_dates:
        raise WorkflowError("dated workflow inputs must contain at least one date.")
    return unique_dates


def select_augmented_dates(
    frame: pd.DataFrame,
    *,
    evaluation_dates: list[pd.Timestamp],
    history_periods: int | None,
) -> list[pd.Timestamp]:
    """Select evaluation dates plus enough prior history for conservative backtests."""
    all_dates = extract_unique_dates(frame)
    start_date = evaluation_dates[0]
    end_date = evaluation_dates[-1]
    try:
        start_index = all_dates.index(start_date)
        end_index = all_dates.index(end_date)
    except ValueError as exc:
        raise WorkflowError(
            "walk-forward evaluation dates must exist in the input panel."
        ) from exc

    if history_periods is None:
        augmented_start = 0
    else:
        augmented_start = max(0, start_index - history_periods)
    return all_dates[augmented_start : end_index + 1]


def _run_backtest_with_config(
    frame: pd.DataFrame,
    *,
    config: AlphaForgeConfig,
) -> pd.DataFrame:
    """Run the shared backtest workflow using config-driven execution settings."""
    backtest_config = _require_backtest_config(config)
    return run_daily_backtest(
        frame,
        signal_delay=backtest_config.signal_delay,
        rebalance_frequency=backtest_config.rebalance_frequency,
        transaction_cost_bps=backtest_config.transaction_cost_bps,
        commission_bps=backtest_config.commission_bps,
        slippage_bps=backtest_config.slippage_bps,
        max_turnover=backtest_config.max_turnover,
        initial_nav=backtest_config.initial_nav,
    )


def _require_backtest_config(config: AlphaForgeConfig):
    """Require a backtest section for backtest-dependent helpers."""
    if config.backtest is None:
        raise WorkflowError(
            "The config must include a [backtest] section for this command."
        )
    return config.backtest


def _series_to_metadata_dict(summary: pd.Series) -> dict[str, Any]:
    """Convert a summary series into JSON-safe scalar metadata."""
    return {
        str(key): _scalar_or_none(value)
        for key, value in summary.items()
    }


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


def _format_compact_numeric(value: float) -> str:
    """Render numeric parameter values without unnecessary decimals."""
    if pd.isna(value):
        return "NaN"
    return str(int(value))

"""Signal parameter sweep workflow helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from alphaforge.analytics import (
    compute_ic_series,
    summarize_backtest,
    summarize_ic,
    summarize_signal_coverage,
)
from alphaforge.cli.errors import WorkflowError
from alphaforge.cli.pipeline import (
    add_signal_from_config,
    build_dataset_from_config,
    build_weights_from_config,
    require_portfolio_config,
    require_signal_config,
    run_backtest_with_config,
)
from alphaforge.cli.report_context import validate_diagnostics_column
from alphaforge.cli.research_metadata import (
    build_research_context_metadata,
    dataframe_records,
)
from alphaforge.cli.validation import normalize_positive_int
from alphaforge.common import AlphaForgeConfig


def run_signal_parameter_sweep(
    config: AlphaForgeConfig,
    *,
    parameter_name: str,
    values: list[int],
) -> pd.DataFrame:
    """Run a simple signal parameter sweep against a fixed pipeline config."""
    portfolio_config = require_portfolio_config(config)
    candidate_values = normalize_sweep_values(values)
    dataset = build_dataset_from_config(config)

    rows = []
    for parameter_value in candidate_values:
        candidate_config = replace_signal_parameter(
            config,
            parameter_name=parameter_name,
            parameter_value=parameter_value,
        )
        signaled, signal_column = add_signal_from_config(dataset, candidate_config)
        validate_diagnostics_column(signaled, candidate_config)
        weighted = build_weights_from_config(
            signaled,
            score_column=signal_column,
            config=candidate_config,
        )
        backtest = run_backtest_with_config(weighted, config=config)
        performance_summary = summarize_backtest(backtest)
        ic_series = compute_ic_series(
            signaled,
            signal_column=signal_column,
            forward_return_column=config.diagnostics.forward_return_column,
            method=config.diagnostics.ic_method,
            min_observations=config.diagnostics.min_observations,
        )
        ic_summary = summarize_ic(ic_series)
        coverage_summary = summarize_signal_coverage(
            signaled,
            signal_column=signal_column,
            forward_return_column=config.diagnostics.forward_return_column,
        )
        rows.append(
            {
                "parameter_name": parameter_name,
                "parameter_value": float(parameter_value),
                "signal_column": signal_column,
                "cumulative_return": performance_summary["cumulative_return"],
                "max_drawdown": performance_summary["max_drawdown"],
                "sharpe_ratio": performance_summary["sharpe_ratio"],
                "average_turnover": performance_summary["average_turnover"],
                "hit_rate": performance_summary["hit_rate"],
                "mean_ic": ic_summary["mean_ic"],
                "ic_ir": ic_summary["ic_ir"],
                "joint_coverage_ratio": coverage_summary["joint_coverage_ratio"],
                "construction": portfolio_config.construction,
            }
        )

    return pd.DataFrame(rows)


def build_sweep_artifact_metadata(
    config: AlphaForgeConfig,
    *,
    config_path: str,
    parameter_name: str,
    values: list[int],
    results: pd.DataFrame,
) -> dict[str, Any]:
    """Build enriched Stage 4 metadata for a sweep artifact bundle."""
    ranked_results = results.sort_values(
        ["cumulative_return", "sharpe_ratio", "mean_ic", "parameter_value"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return {
        "command": "sweep-signal",
        "config": config_path,
        "parameter": parameter_name,
        "values": values,
        "row_count": int(len(results)),
        "research_context": build_research_context_metadata(config),
        "best_candidate": (
            dataframe_records(ranked_results.head(1))[0]
            if not ranked_results.empty
            else None
        ),
        "top_candidates": dataframe_records(ranked_results.head(3)),
    }


def replace_signal_parameter(
    config: AlphaForgeConfig,
    *,
    parameter_name: str,
    parameter_value: int,
) -> AlphaForgeConfig:
    """Return a copy of config with one supported signal parameter replaced."""
    signal_config = require_signal_config(config)
    parameter_value = normalize_positive_int(
        parameter_value,
        parameter_name=f"sweep value for {parameter_name}",
    )

    if signal_config.name in {"momentum", "mean_reversion"}:
        if parameter_name != "lookback":
            raise WorkflowError(
                f"signal '{signal_config.name}' only supports sweeping the 'lookback' parameter."
            )
        return replace(
            config,
            signal=replace(signal_config, lookback=parameter_value),
        )

    if parameter_name == "short_window":
        long_window = signal_config.long_window or 60
        if parameter_value >= long_window:
            raise WorkflowError(
                "sweep short_window values must stay smaller than the configured long_window."
            )
        return replace(
            config,
            signal=replace(signal_config, short_window=parameter_value),
        )

    if parameter_name == "long_window":
        short_window = signal_config.short_window or 20
        if parameter_value <= short_window:
            raise WorkflowError(
                "sweep long_window values must stay larger than the configured short_window."
            )
        return replace(
            config,
            signal=replace(signal_config, long_window=parameter_value),
        )

    raise WorkflowError(
        "trend signals only support sweeping 'short_window' or 'long_window'."
    )


def normalize_sweep_values(values: list[int]) -> list[int]:
    """Validate and normalize an ordered list of sweep candidates."""
    if not values:
        raise WorkflowError("sweep values must contain at least one positive integer.")

    normalized: list[int] = []
    seen: set[int] = set()
    for value in values:
        numeric_value = normalize_positive_int(value, parameter_name="sweep value")
        if numeric_value in seen:
            raise WorkflowError(
                f"sweep values must be unique; received duplicate value {numeric_value}."
            )
        normalized.append(numeric_value)
        seen.add(numeric_value)
    return normalized


__all__ = [
    "build_sweep_artifact_metadata",
    "normalize_sweep_values",
    "replace_signal_parameter",
    "run_signal_parameter_sweep",
]

"""Growth-style features derived from PIT fundamentals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from alphaforge.data import DataValidationError, validate_fundamentals, validate_ohlcv
from alphaforge.features.fundamentals_join import (
    attach_fundamentals_asof,
    fundamental_column_name,
)


def attach_fundamental_growth_rates(
    frame: pd.DataFrame,
    fundamentals: pd.DataFrame,
    *,
    trading_calendar: pd.DataFrame | None = None,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Attach period-over-period growth rates from PIT fundamentals.

    Each output is named ``growth_<metric>`` and computes
    ``current_metric_value / prior_metric_value - 1`` across adjacent
    ``period_end_date`` observations for the same symbol and metric.

    Timing convention:
    - each feature row is anchored at the close of ``date``
    - the current fundamental period becomes usable only on the next market
      session after its date-only ``release_date``
    - the prior period must have a release date no later than the current
      period release date
    - nonpositive prior values are treated as unavailable instead of producing
      hard-to-interpret growth rates
    """
    dataset = validate_ohlcv(frame, source="growth feature input").copy()
    validated_fundamentals = validate_fundamentals(
        fundamentals,
        source="fundamentals input",
    )
    selected_metrics = normalize_growth_metrics(
        metrics,
        available_metrics=validated_fundamentals["metric_name"],
    )
    output_columns = {
        metric_name: growth_column_name(metric_name)
        for metric_name in selected_metrics
    }
    if len(set(output_columns.values())) != len(output_columns):
        raise ValueError(
            "growth_metrics contains metric names that normalize to the same "
            "growth output column."
        )

    conflicting_columns = [
        column_name
        for column_name in output_columns.values()
        if column_name in dataset.columns
    ]
    if conflicting_columns:
        conflict_text = ", ".join(conflicting_columns)
        raise ValueError(
            "research dataset already contains growth output columns: "
            f"{conflict_text}."
        )

    for column_name in output_columns.values():
        dataset[column_name] = np.nan

    event_metric_names = {
        metric_name: _growth_event_metric_name(metric_name)
        for metric_name in selected_metrics
    }
    growth_events = _build_growth_events(
        validated_fundamentals,
        selected_metrics=selected_metrics,
        event_metric_names=event_metric_names,
    )
    if growth_events.empty:
        return dataset

    available_event_metrics = tuple(
        pd.Index(growth_events["metric_name"]).drop_duplicates().tolist()
    )
    attached = attach_fundamentals_asof(
        dataset,
        growth_events,
        trading_calendar=trading_calendar,
        metrics=available_event_metrics,
    )
    for metric_name, output_column in output_columns.items():
        event_metric_name = event_metric_names[metric_name]
        if event_metric_name not in available_event_metrics:
            continue
        temp_column = fundamental_column_name(event_metric_name)
        dataset[output_column] = attached[temp_column]

    return dataset


def normalize_growth_metrics(
    metrics: Sequence[str],
    *,
    available_metrics: pd.Series | None = None,
) -> tuple[str, ...]:
    """Validate and normalize growth metric selections."""
    if isinstance(metrics, str):
        raw_metrics = (metrics,)
    else:
        raw_metrics = tuple(metrics)

    if not raw_metrics:
        raise ValueError("growth_metrics must contain at least one metric name.")

    normalized_metrics = tuple(
        _normalize_growth_metric_name(metric_name) for metric_name in raw_metrics
    )
    if len(set(normalized_metrics)) != len(normalized_metrics):
        raise ValueError("growth_metrics must not contain duplicate metric names.")

    output_columns = [growth_column_name(metric_name) for metric_name in normalized_metrics]
    if len(set(output_columns)) != len(output_columns):
        raise ValueError(
            "growth_metrics contains metric names that normalize to the same "
            "growth output column."
        )

    if available_metrics is not None:
        available = set(available_metrics.tolist())
        missing_metrics = sorted(
            metric_name
            for metric_name in normalized_metrics
            if metric_name not in available
        )
        if missing_metrics:
            missing_text = ", ".join(missing_metrics)
            raise DataValidationError(
                "fundamentals input is missing configured metric_name values: "
                f"{missing_text}."
            )

    return normalized_metrics


def growth_column_name(metric_name: str) -> str:
    """Build the growth output column name for one metric."""
    return f"growth_{_metric_slug(metric_name)}"


def _build_growth_events(
    fundamentals: pd.DataFrame,
    *,
    selected_metrics: tuple[str, ...],
    event_metric_names: dict[str, str],
) -> pd.DataFrame:
    """Convert raw fundamental period rows into growth-rate release events."""
    filtered = fundamentals.loc[
        fundamentals["metric_name"].isin(selected_metrics)
    ].copy()
    _validate_no_restatement_lineage(filtered)

    event_frames: list[pd.DataFrame] = []
    for (_, metric_name), metric_rows in filtered.groupby(
        ["symbol", "metric_name"],
        sort=False,
    ):
        ordered = metric_rows.sort_values(
            ["period_end_date", "release_date"],
            kind="mergesort",
        ).copy()
        prior_value = ordered["metric_value"].shift(1)
        prior_period_end = ordered["period_end_date"].shift(1)
        prior_release_date = ordered["release_date"].shift(1)
        growth = ordered["metric_value"].div(prior_value).sub(1.0)
        valid_growth = (
            prior_period_end.notna()
            & prior_period_end.lt(ordered["period_end_date"])
            & prior_release_date.le(ordered["release_date"])
            & prior_value.gt(0.0)
            & np.isfinite(growth)
        )
        if not bool(valid_growth.any()):
            continue

        events = ordered.loc[
            valid_growth,
            ["symbol", "period_end_date", "release_date"],
        ].copy()
        events["metric_name"] = event_metric_names[metric_name]
        events["metric_value"] = growth.loc[valid_growth].to_numpy(dtype=float)
        event_frames.append(events)

    if not event_frames:
        return pd.DataFrame(
            columns=[
                "symbol",
                "period_end_date",
                "release_date",
                "metric_name",
                "metric_value",
            ]
        )

    return pd.concat(event_frames, ignore_index=True).sort_values(
        ["symbol", "metric_name", "period_end_date", "release_date"],
        kind="mergesort",
    )


def _validate_no_restatement_lineage(fundamentals: pd.DataFrame) -> None:
    """Reject duplicate period rows because restatement lineage is not modeled."""
    duplicated = fundamentals.duplicated(
        subset=["symbol", "metric_name", "period_end_date"],
        keep=False,
    )
    if not bool(duplicated.any()):
        return

    conflicting_rows = (
        fundamentals.loc[
            duplicated,
            ["symbol", "metric_name", "period_end_date"],
        ]
        .drop_duplicates()
        .sort_values(["symbol", "metric_name", "period_end_date"])
    )
    sample = ", ".join(
        (
            f"({row.symbol}, {row.metric_name}, "
            f"{row.period_end_date.date().isoformat()})"
        )
        for row in conflicting_rows.itertuples(index=False)
    )
    raise DataValidationError(
        "growth_metrics does not model restatement lineage; fundamentals input "
        "contains multiple rows for the same symbol/metric/period_end_date: "
        f"{sample}."
    )


def _normalize_growth_metric_name(metric_name: object) -> str:
    """Validate one selected growth metric name."""
    if not isinstance(metric_name, str):
        raise ValueError("growth_metrics must contain only non-empty strings.")
    normalized = metric_name.strip()
    if normalized == "":
        raise ValueError("growth_metrics must contain only non-empty strings.")
    return normalized


def _growth_event_metric_name(metric_name: str) -> str:
    """Build an internal metric name used for reusing fundamentals asof joins."""
    return f"alphaforge_growth_feature_{_metric_slug(metric_name)}"


def _metric_slug(metric_name: str) -> str:
    """Convert a fundamental metric name into its dataset-column slug."""
    return fundamental_column_name(metric_name).removeprefix("fundamental_")

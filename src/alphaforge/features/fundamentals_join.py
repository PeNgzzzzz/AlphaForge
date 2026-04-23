"""Point-in-time-safe fundamentals joins for research datasets."""

from __future__ import annotations

from collections.abc import Sequence
import re

import numpy as np
import pandas as pd

from alphaforge.data import DataValidationError, validate_fundamentals

_NON_IDENTIFIER_PATTERN = re.compile(r"[^0-9A-Za-z]+")


def attach_fundamentals_asof(
    dataset: pd.DataFrame,
    fundamentals: pd.DataFrame,
    *,
    trading_calendar: pd.DataFrame | None = None,
    metrics: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Attach the latest next-session-available fundamentals to each symbol/date row.

    Timing convention:
    - ``release_date`` is date-only, so same-day availability is ambiguous
    - to stay conservative, a release becomes usable on the next market session
    - when a trading calendar is configured, the next session comes from that calendar
    - otherwise, the next observed market date in the dataset is used
    """
    validated_fundamentals = validate_fundamentals(
        fundamentals,
        source="fundamentals input",
    )
    selected_metrics = _normalize_selected_metrics(
        metrics,
        available_metrics=validated_fundamentals["metric_name"],
    )
    output_columns = {
        metric_name: _fundamental_column_name(metric_name)
        for metric_name in selected_metrics
    }
    if len(set(output_columns.values())) != len(output_columns):
        raise DataValidationError(
            "fundamentals input contains metric_name values that normalize to the "
            "same dataset column name."
        )
    conflicting_columns = [
        column_name
        for column_name in output_columns.values()
        if column_name in dataset.columns
    ]
    if conflicting_columns:
        conflict_text = ", ".join(conflicting_columns)
        raise DataValidationError(
            "research dataset already contains fundamentals output columns: "
            f"{conflict_text}."
        )

    prepared_fundamentals = _prepare_fundamentals_for_join(
        validated_fundamentals,
        dataset_dates=dataset["date"],
        trading_calendar=trading_calendar,
        selected_metrics=selected_metrics,
    )
    attached = dataset.copy()
    for column_name in output_columns.values():
        attached[column_name] = np.nan

    grouped_fundamentals = {
        symbol: symbol_frame
        for symbol, symbol_frame in prepared_fundamentals.groupby("symbol", sort=False)
    }
    for symbol, row_indexes in attached.groupby("symbol", sort=False).groups.items():
        symbol_fundamentals = grouped_fundamentals.get(symbol)
        if symbol_fundamentals is None or symbol_fundamentals.empty:
            continue

        row_index = pd.Index(row_indexes)
        symbol_dates = attached.loc[row_index, "date"].to_numpy()
        for metric_name, column_name in output_columns.items():
            metric_rows = symbol_fundamentals.loc[
                symbol_fundamentals["metric_name"] == metric_name,
                ["availability_date", "metric_value"],
            ]
            if metric_rows.empty:
                continue

            availability_dates = metric_rows["availability_date"].to_numpy()
            metric_values = metric_rows["metric_value"].to_numpy(dtype=float)
            positions = availability_dates.searchsorted(symbol_dates, side="right") - 1
            joined_values = np.full(len(row_index), np.nan, dtype=float)
            valid_positions = positions >= 0
            joined_values[valid_positions] = metric_values[positions[valid_positions]]
            attached.loc[row_index, column_name] = joined_values

    return attached


def _prepare_fundamentals_for_join(
    fundamentals: pd.DataFrame,
    *,
    dataset_dates: pd.Series,
    trading_calendar: pd.DataFrame | None,
    selected_metrics: tuple[str, ...],
) -> pd.DataFrame:
    """Filter fundamentals rows and assign next-session availability dates."""
    filtered = fundamentals.loc[
        fundamentals["metric_name"].isin(selected_metrics)
    ].copy()
    reference_dates = _resolve_reference_dates(
        dataset_dates,
        trading_calendar=trading_calendar,
    )
    availability_positions = reference_dates.searchsorted(
        filtered["release_date"],
        side="right",
    )
    has_availability_date = availability_positions < len(reference_dates)
    filtered["availability_date"] = pd.NaT
    if bool(has_availability_date.any()):
        filtered.loc[has_availability_date, "availability_date"] = (
            reference_dates.take(availability_positions[has_availability_date])
            .to_numpy(dtype="datetime64[ns]")
        )

    usable = filtered.loc[filtered["availability_date"].notna()].copy()
    duplicated = usable.duplicated(
        subset=["symbol", "metric_name", "availability_date"],
        keep=False,
    )
    if duplicated.any():
        conflicting_rows = (
            usable.loc[
                duplicated,
                ["symbol", "metric_name", "availability_date"],
            ]
            .drop_duplicates()
            .sort_values(["symbol", "metric_name", "availability_date"])
        )
        sample = ", ".join(
            (
                f"({row.symbol}, {row.metric_name}, "
                f"{row.availability_date.date().isoformat()})"
            )
            for row in conflicting_rows.itertuples(index=False)
        )
        raise DataValidationError(
            "fundamentals input contains multiple rows that become available on "
            "the same next-session date for one symbol/metric: "
            f"{sample}."
        )

    return filtered.sort_values(
        ["symbol", "metric_name", "availability_date", "period_end_date"],
        kind="mergesort",
    ).reset_index(drop=True)


def _resolve_reference_dates(
    dataset_dates: pd.Series,
    *,
    trading_calendar: pd.DataFrame | None,
) -> pd.Index:
    """Resolve the ordered session dates used for next-session availability."""
    if trading_calendar is None:
        return pd.Index(dataset_dates.drop_duplicates().sort_values())
    return pd.Index(trading_calendar["date"])


def _normalize_selected_metrics(
    metrics: Sequence[str] | None,
    *,
    available_metrics: pd.Series,
) -> tuple[str, ...]:
    """Normalize the requested metric list and validate availability."""
    if metrics is None:
        normalized_metrics = tuple(
            pd.Index(available_metrics)
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
    else:
        normalized_metrics = tuple(
            _normalize_metric_name(metric_name) for metric_name in metrics
        )

    if not normalized_metrics:
        raise ValueError("fundamental_metrics must contain at least one metric name.")
    if len(set(normalized_metrics)) != len(normalized_metrics):
        raise ValueError("fundamental_metrics must not contain duplicate metric names.")

    available = set(available_metrics.tolist())
    missing_metrics = sorted(
        metric_name for metric_name in normalized_metrics if metric_name not in available
    )
    if missing_metrics:
        missing_text = ", ".join(missing_metrics)
        raise DataValidationError(
            "fundamentals input is missing configured metric_name values: "
            f"{missing_text}."
        )
    return normalized_metrics


def _normalize_metric_name(metric_name: str) -> str:
    """Normalize one selected metric name."""
    if not isinstance(metric_name, str):
        raise ValueError("fundamental_metrics must contain only non-empty strings.")
    normalized = metric_name.strip()
    if normalized == "":
        raise ValueError("fundamental_metrics must contain only non-empty strings.")
    return normalized


def _fundamental_column_name(metric_name: str) -> str:
    """Normalize a metric name into a deterministic dataset column name."""
    normalized = _NON_IDENTIFIER_PATTERN.sub("_", metric_name.strip().lower()).strip("_")
    if normalized == "":
        raise DataValidationError(
            "fundamentals input contains a metric_name that cannot be converted "
            "into a dataset column name."
        )
    return f"fundamental_{normalized}"

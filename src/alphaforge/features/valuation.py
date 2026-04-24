"""Valuation-style features derived from PIT fundamentals and close prices."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from alphaforge.data import validate_ohlcv
from alphaforge.features.fundamentals_join import fundamental_column_name


def attach_fundamental_price_ratios(
    frame: pd.DataFrame,
    *,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Attach conservative fundamental-to-price valuation ratios.

    Each output is ``fundamental_<metric> / close`` and is named
    ``valuation_<metric>_to_price``. The function assumes the referenced
    fundamental columns were already attached with point-in-time-safe
    availability semantics.

    Timing convention:
    - each feature row is anchored at the close of ``date``
    - the numerator uses the latest PIT-available fundamental metric
    - the denominator uses same-day ``close``, known by that close
    """
    dataset = validate_ohlcv(frame, source="valuation feature input").copy()
    selected_metrics = _normalize_valuation_metrics(metrics)
    output_columns = {
        metric_name: _valuation_column_name(metric_name)
        for metric_name in selected_metrics
    }
    if len(set(output_columns.values())) != len(output_columns):
        raise ValueError(
            "valuation_metrics contains metric names that normalize to the same "
            "valuation output column."
        )

    conflicting_columns = [
        column_name
        for column_name in output_columns.values()
        if column_name in dataset.columns
    ]
    if conflicting_columns:
        conflict_text = ", ".join(conflicting_columns)
        raise ValueError(
            "research dataset already contains valuation output columns: "
            f"{conflict_text}."
        )

    missing_columns = [
        fundamental_column_name(metric_name)
        for metric_name in selected_metrics
        if fundamental_column_name(metric_name) not in dataset.columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(
            "valuation feature input is missing required fundamental columns: "
            f"{missing_text}."
        )

    for metric_name, output_column in output_columns.items():
        fundamental_column = fundamental_column_name(metric_name)
        values = pd.to_numeric(dataset[fundamental_column], errors="coerce")
        invalid_values = dataset[fundamental_column].notna() & values.isna()
        if invalid_values.any():
            raise ValueError(
                "valuation feature input contains invalid numeric values in "
                f"{fundamental_column!r}."
            )
        dataset[output_column] = values.div(dataset["close"])
        dataset[output_column] = dataset[output_column].mask(
            ~np.isfinite(dataset[output_column])
        )

    return dataset


def _normalize_valuation_metrics(metrics: Sequence[str]) -> tuple[str, ...]:
    """Validate valuation metric selection."""
    if isinstance(metrics, str):
        raw_metrics = (metrics,)
    else:
        raw_metrics = tuple(metrics)

    if not raw_metrics:
        raise ValueError("valuation_metrics must contain at least one metric name.")

    normalized_metrics: list[str] = []
    for metric_name in raw_metrics:
        if not isinstance(metric_name, str):
            raise ValueError(
                "valuation_metrics must contain only non-empty strings."
            )
        normalized = metric_name.strip()
        if normalized == "":
            raise ValueError(
                "valuation_metrics must contain only non-empty strings."
            )
        normalized_metrics.append(normalized)

    if len(set(normalized_metrics)) != len(normalized_metrics):
        raise ValueError("valuation_metrics must not contain duplicate metric names.")
    return tuple(normalized_metrics)


def _valuation_column_name(metric_name: str) -> str:
    """Build the valuation output column name for one metric."""
    fundamental_column = fundamental_column_name(metric_name)
    normalized_metric = fundamental_column.removeprefix("fundamental_")
    return f"valuation_{normalized_metric}_to_price"

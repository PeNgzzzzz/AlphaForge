"""Shared helpers for PIT fundamental ratio features."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from alphaforge.common.validation import (
    normalize_unique_non_empty_string_pair_sequence as _common_string_pair_sequence,
)
from alphaforge.data import validate_ohlcv
from alphaforge.features.fundamentals_join import fundamental_column_name

FundamentalRatioMetric = tuple[str, str]


def attach_fundamental_ratio_features(
    frame: pd.DataFrame,
    *,
    metrics: Sequence[Sequence[str]],
    output_prefix: str,
    metrics_name: str,
    source: str,
    feature_name: str,
) -> pd.DataFrame:
    """Attach ratios from already PIT-available fundamental columns."""
    dataset = validate_ohlcv(frame, source=source).copy()
    selected_metrics = normalize_fundamental_ratio_metrics(
        metrics,
        metrics_name=metrics_name,
    )
    output_columns = {
        metric_pair: fundamental_ratio_column_name(output_prefix, *metric_pair)
        for metric_pair in selected_metrics
    }
    if len(set(output_columns.values())) != len(output_columns):
        raise ValueError(
            f"{metrics_name} contains metric pairs that normalize to the same "
            f"{feature_name} output column."
        )

    conflicting_columns = [
        column_name
        for column_name in output_columns.values()
        if column_name in dataset.columns
    ]
    if conflicting_columns:
        conflict_text = ", ".join(conflicting_columns)
        raise ValueError(
            f"research dataset already contains {feature_name} output columns: "
            f"{conflict_text}."
        )

    required_columns = []
    for numerator_metric, denominator_metric in selected_metrics:
        required_columns.extend(
            [
                fundamental_column_name(numerator_metric),
                fundamental_column_name(denominator_metric),
            ]
        )
    missing_columns = [
        column_name
        for column_name in dict.fromkeys(required_columns)
        if column_name not in dataset.columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(
            f"{feature_name} feature input is missing required fundamental columns: "
            f"{missing_text}."
        )

    for metric_pair, output_column in output_columns.items():
        numerator_metric, denominator_metric = metric_pair
        numerator_column = fundamental_column_name(numerator_metric)
        denominator_column = fundamental_column_name(denominator_metric)

        numerator = _coerce_numeric_feature(
            dataset,
            numerator_column,
            feature_name=feature_name,
        )
        denominator = _coerce_numeric_feature(
            dataset,
            denominator_column,
            feature_name=feature_name,
        ).where(lambda values: values > 0.0)
        ratio = numerator.div(denominator)
        dataset[output_column] = ratio.mask(~np.isfinite(ratio))

    return dataset


def normalize_fundamental_ratio_metrics(
    metrics: Sequence[Sequence[str]],
    *,
    metrics_name: str,
) -> tuple[FundamentalRatioMetric, ...]:
    """Validate and normalize numerator/denominator metric pairs."""
    if isinstance(metrics, str):
        raise ValueError(
            f"{metrics_name} must contain [numerator, denominator] metric pairs."
        )

    raw_pairs = tuple(metrics)
    if not raw_pairs:
        raise ValueError(f"{metrics_name} must contain at least one metric pair.")

    pair_error_message = (
        f"{metrics_name} must contain [numerator, denominator] metric pairs."
    )
    normalized_pairs = _common_string_pair_sequence(
        raw_pairs,
        parameter_name=metrics_name,
        pair_error_message=pair_error_message,
        item_error_message=(
            f"{metrics_name} must contain only non-empty string metric names."
        ),
        duplicate_error_message=f"{metrics_name} must not contain duplicate metric pairs.",
        equal_items_error_message=(
            f"{metrics_name} numerator and denominator must produce different "
            "fundamental columns."
        ),
    )

    fundamental_column_pairs: list[tuple[str, str]] = []
    for numerator_metric, denominator_metric in normalized_pairs:
        numerator_column = fundamental_column_name(numerator_metric)
        denominator_column = fundamental_column_name(denominator_metric)
        if numerator_column == denominator_column:
            raise ValueError(
                f"{metrics_name} numerator and denominator must produce different "
                "fundamental columns."
            )

        fundamental_column_pairs.append((numerator_column, denominator_column))

    if len(set(fundamental_column_pairs)) != len(fundamental_column_pairs):
        raise ValueError(f"{metrics_name} must not contain duplicate metric pairs.")
    return tuple(normalized_pairs)


def fundamental_ratio_column_name(
    output_prefix: str,
    numerator_metric: str,
    denominator_metric: str,
) -> str:
    """Build the ratio output column name for one metric pair."""
    numerator_slug = _metric_slug(numerator_metric)
    denominator_slug = _metric_slug(denominator_metric)
    return f"{output_prefix}_{numerator_slug}_to_{denominator_slug}"


def _coerce_numeric_feature(
    dataset: pd.DataFrame,
    column_name: str,
    *,
    feature_name: str,
) -> pd.Series:
    """Parse a feature column as numeric and fail on non-numeric non-null values."""
    values = pd.to_numeric(dataset[column_name], errors="coerce")
    invalid_values = dataset[column_name].notna() & values.isna()
    if invalid_values.any():
        raise ValueError(
            f"{feature_name} feature input contains invalid numeric values in "
            f"{column_name!r}."
        )
    return values


def _metric_slug(metric_name: str) -> str:
    """Convert a fundamental metric name into its dataset-column slug."""
    return fundamental_column_name(metric_name).removeprefix("fundamental_")

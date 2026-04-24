"""Quality-style features derived from PIT fundamentals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from alphaforge.data import validate_ohlcv
from alphaforge.features.fundamentals_join import fundamental_column_name

QualityRatioMetric = tuple[str, str]


def attach_quality_ratios(
    frame: pd.DataFrame,
    *,
    metrics: Sequence[Sequence[str]],
) -> pd.DataFrame:
    """Attach conservative quality ratios from PIT-available fundamentals.

    Each output is named ``quality_<numerator>_to_<denominator>`` and computes
    ``fundamental_<numerator> / fundamental_<denominator>``. The function
    assumes the referenced fundamental columns were already attached with
    point-in-time-safe availability semantics.

    Timing convention:
    - each feature row is anchored at the close of ``date``
    - numerator and denominator use the latest PIT-available fundamental values
    - nonpositive denominators are treated as unavailable instead of inverted
      into potentially misleading ratios
    """
    dataset = validate_ohlcv(frame, source="quality feature input").copy()
    selected_metrics = normalize_quality_ratio_metrics(metrics)
    output_columns = {
        metric_pair: quality_ratio_column_name(*metric_pair)
        for metric_pair in selected_metrics
    }
    if len(set(output_columns.values())) != len(output_columns):
        raise ValueError(
            "quality_ratio_metrics contains metric pairs that normalize to the "
            "same quality output column."
        )

    conflicting_columns = [
        column_name
        for column_name in output_columns.values()
        if column_name in dataset.columns
    ]
    if conflicting_columns:
        conflict_text = ", ".join(conflicting_columns)
        raise ValueError(
            "research dataset already contains quality output columns: "
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
            "quality feature input is missing required fundamental columns: "
            f"{missing_text}."
        )

    for metric_pair, output_column in output_columns.items():
        numerator_metric, denominator_metric = metric_pair
        numerator_column = fundamental_column_name(numerator_metric)
        denominator_column = fundamental_column_name(denominator_metric)

        numerator = _coerce_numeric_feature(dataset, numerator_column)
        denominator = _coerce_numeric_feature(dataset, denominator_column).where(
            lambda values: values > 0.0
        )
        ratio = numerator.div(denominator)
        dataset[output_column] = ratio.mask(~np.isfinite(ratio))

    return dataset


def normalize_quality_ratio_metrics(
    metrics: Sequence[Sequence[str]],
) -> tuple[QualityRatioMetric, ...]:
    """Validate and normalize quality ratio metric pairs."""
    if isinstance(metrics, str):
        raise ValueError(
            "quality_ratio_metrics must contain [numerator, denominator] metric pairs."
        )

    raw_pairs = tuple(metrics)
    if not raw_pairs:
        raise ValueError("quality_ratio_metrics must contain at least one metric pair.")

    normalized_pairs: list[QualityRatioMetric] = []
    fundamental_column_pairs: list[tuple[str, str]] = []
    for metric_pair in raw_pairs:
        if isinstance(metric_pair, str):
            raise ValueError(
                "quality_ratio_metrics must contain [numerator, denominator] "
                "metric pairs."
            )
        pair_values = tuple(metric_pair)
        if len(pair_values) != 2:
            raise ValueError(
                "quality_ratio_metrics must contain [numerator, denominator] "
                "metric pairs."
            )

        numerator_metric = _normalize_quality_metric_name(pair_values[0])
        denominator_metric = _normalize_quality_metric_name(pair_values[1])
        numerator_column = fundamental_column_name(numerator_metric)
        denominator_column = fundamental_column_name(denominator_metric)
        if numerator_column == denominator_column:
            raise ValueError(
                "quality_ratio_metrics numerator and denominator must produce "
                "different fundamental columns."
            )

        normalized_pairs.append((numerator_metric, denominator_metric))
        fundamental_column_pairs.append((numerator_column, denominator_column))

    if len(set(fundamental_column_pairs)) != len(fundamental_column_pairs):
        raise ValueError(
            "quality_ratio_metrics must not contain duplicate metric pairs."
        )
    return tuple(normalized_pairs)


def quality_ratio_column_name(
    numerator_metric: str,
    denominator_metric: str,
) -> str:
    """Build the quality-ratio output column name for one metric pair."""
    numerator_slug = _metric_slug(numerator_metric)
    denominator_slug = _metric_slug(denominator_metric)
    return f"quality_{numerator_slug}_to_{denominator_slug}"


def _coerce_numeric_feature(dataset: pd.DataFrame, column_name: str) -> pd.Series:
    """Parse a feature column as numeric and fail on non-numeric non-null values."""
    values = pd.to_numeric(dataset[column_name], errors="coerce")
    invalid_values = dataset[column_name].notna() & values.isna()
    if invalid_values.any():
        raise ValueError(
            "quality feature input contains invalid numeric values in "
            f"{column_name!r}."
        )
    return values


def _normalize_quality_metric_name(metric_name: object) -> str:
    """Validate one selected metric name."""
    if not isinstance(metric_name, str):
        raise ValueError(
            "quality_ratio_metrics must contain only non-empty string metric names."
        )
    normalized = metric_name.strip()
    if normalized == "":
        raise ValueError(
            "quality_ratio_metrics must contain only non-empty string metric names."
        )
    return normalized


def _metric_slug(metric_name: str) -> str:
    """Convert a fundamental metric name into its dataset-column slug."""
    return fundamental_column_name(metric_name).removeprefix("fundamental_")

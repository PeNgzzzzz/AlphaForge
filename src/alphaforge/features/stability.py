"""Balance-sheet stability-style features derived from PIT fundamentals."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from alphaforge.features.ratio_features import (
    FundamentalRatioMetric,
    attach_fundamental_ratio_features,
    fundamental_ratio_column_name,
    normalize_fundamental_ratio_metrics,
)

StabilityRatioMetric = FundamentalRatioMetric


def attach_stability_ratios(
    frame: pd.DataFrame,
    *,
    metrics: Sequence[Sequence[str]],
) -> pd.DataFrame:
    """Attach conservative balance-sheet stability ratios from PIT fundamentals.

    Each output is named ``stability_<numerator>_to_<denominator>`` and computes
    ``fundamental_<numerator> / fundamental_<denominator>``. The function
    assumes the referenced fundamental columns were already attached with
    point-in-time-safe availability semantics.

    Timing convention:
    - each feature row is anchored at the close of ``date``
    - numerator and denominator use the latest PIT-available fundamental values
    - nonpositive denominators are treated as unavailable instead of inverted
      into potentially misleading ratios
    """
    return attach_fundamental_ratio_features(
        frame,
        metrics=metrics,
        output_prefix="stability",
        metrics_name="stability_ratio_metrics",
        source="stability feature input",
        feature_name="stability",
    )


def normalize_stability_ratio_metrics(
    metrics: Sequence[Sequence[str]],
) -> tuple[StabilityRatioMetric, ...]:
    """Validate and normalize stability ratio metric pairs."""
    return normalize_fundamental_ratio_metrics(
        metrics,
        metrics_name="stability_ratio_metrics",
    )


def stability_ratio_column_name(
    numerator_metric: str,
    denominator_metric: str,
) -> str:
    """Build the stability-ratio output column name for one metric pair."""
    return fundamental_ratio_column_name(
        "stability",
        numerator_metric,
        denominator_metric,
    )

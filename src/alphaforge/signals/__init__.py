"""Signal generation utilities for AlphaForge."""

from alphaforge.signals.cross_sectional import (
    SignalTransformDefinition,
    apply_cross_sectional_signal_transform,
    apply_signal_transform_pipeline,
    clip_signal_by_date,
    get_signal_transform_definition,
    list_signal_transform_definitions,
    rank_normalize_signal_by_date,
    robust_zscore_signal_by_date,
    winsorize_signal_by_date,
    zscore_signal_by_date,
)
from alphaforge.signals.definitions import (
    FactorDefinition,
    build_factor_signal,
    get_factor_definition,
    list_factor_definitions,
)
from alphaforge.signals.metadata import build_signal_pipeline_metadata
from alphaforge.signals.price_signals import (
    add_mean_reversion_signal,
    add_momentum_signal,
    add_trend_signal,
)

__all__ = [
    "apply_cross_sectional_signal_transform",
    "add_mean_reversion_signal",
    "add_momentum_signal",
    "add_trend_signal",
    "build_factor_signal",
    "build_signal_pipeline_metadata",
    "clip_signal_by_date",
    "FactorDefinition",
    "get_factor_definition",
    "get_signal_transform_definition",
    "list_signal_transform_definitions",
    "list_factor_definitions",
    "apply_signal_transform_pipeline",
    "rank_normalize_signal_by_date",
    "robust_zscore_signal_by_date",
    "SignalTransformDefinition",
    "winsorize_signal_by_date",
    "zscore_signal_by_date",
]

"""Research dataset feature provenance metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
import hashlib
import json
import re

from alphaforge.common.validation import (
    normalize_positive_int as _common_positive_int,
    normalize_unique_non_empty_string_sequence as _common_string_sequence,
    normalize_unique_non_empty_string_pair_sequence as _common_string_pair_sequence,
)
from alphaforge.features.fundamentals_join import fundamental_column_name
from alphaforge.features.growth import growth_column_name
from alphaforge.features.quality import quality_ratio_column_name
from alphaforge.features.stability import stability_ratio_column_name

_NON_IDENTIFIER_PATTERN = re.compile(r"[^0-9A-Za-z]+")

_CLOSE_ANCHORED_TIMING = (
    "close-anchored; inputs are observable by the row's market close"
)
_NEXT_SESSION_FUNDAMENTAL_TIMING = (
    "date-only fundamentals release_date is usable on the next market session"
)
_EFFECTIVE_DATE_TIMING = (
    "date-only effective_date is active on the first market session not earlier "
    "than that date"
)


@dataclass(frozen=True)
class FeatureMetadata:
    """Serializable provenance metadata for one research dataset column."""

    column: str
    role: str
    family: str
    source: str
    inputs: tuple[str, ...]
    timing: str
    missing_policy: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a JSON-friendly dictionary."""
        return {
            "column": self.column,
            "role": self.role,
            "family": self.family,
            "source": self.source,
            "inputs": list(self.inputs),
            "timing": self.timing,
            "missing_policy": self.missing_policy,
            "parameters": dict(self.parameters),
        }


def build_research_dataset_feature_metadata(
    *,
    forward_horizons: Sequence[int] = (1,),
    volatility_window: int = 20,
    average_volume_window: int = 20,
    average_true_range_window: int | None = None,
    normalized_average_true_range_window: int | None = None,
    amihud_illiquidity_window: int | None = None,
    dollar_volume_shock_window: int | None = None,
    dollar_volume_zscore_window: int | None = None,
    volume_shock_window: int | None = None,
    relative_volume_window: int | None = None,
    relative_dollar_volume_window: int | None = None,
    garman_klass_volatility_window: int | None = None,
    parkinson_volatility_window: int | None = None,
    rogers_satchell_volatility_window: int | None = None,
    yang_zhang_volatility_window: int | None = None,
    realized_volatility_window: int | None = None,
    higher_moments_window: int | None = None,
    benchmark_residual_return_window: int | None = None,
    benchmark_rolling_window: int | None = None,
    fundamental_metrics: Sequence[str] = (),
    valuation_metrics: Sequence[str] = (),
    quality_ratio_metrics: Sequence[Sequence[str]] = (),
    growth_metrics: Sequence[str] = (),
    stability_ratio_metrics: Sequence[Sequence[str]] = (),
    classification_fields: Sequence[str] = (),
    membership_indexes: Sequence[str] = (),
    borrow_fields: Sequence[str] = (),
    include_market_cap: bool = False,
    market_cap_bucket_count: int | None = None,
    universe_enabled: bool = False,
    universe_lag: int | None = None,
    universe_average_volume_window: int | None = None,
    universe_average_dollar_volume_window: int | None = None,
    universe_required_membership_indexes: Sequence[str] = (),
    universe_require_tradable: bool = False,
) -> list[dict[str, Any]]:
    """Build JSON-friendly provenance metadata for configured dataset columns."""
    horizons = _normalize_positive_int_sequence(
        forward_horizons,
        field_name="forward_horizons",
    )
    volatility_window = _normalize_positive_int(
        volatility_window,
        field_name="volatility_window",
    )
    average_volume_window = _normalize_positive_int(
        average_volume_window,
        field_name="average_volume_window",
    )
    normalized_fundamental_metrics = _normalize_string_sequence(
        fundamental_metrics,
        field_name="fundamental_metrics",
    )
    normalized_valuation_metrics = _normalize_string_sequence(
        valuation_metrics,
        field_name="valuation_metrics",
    )
    normalized_quality_ratios = _normalize_metric_pair_sequence(
        quality_ratio_metrics,
        field_name="quality_ratio_metrics",
    )
    normalized_growth_metrics = _normalize_string_sequence(
        growth_metrics,
        field_name="growth_metrics",
    )
    normalized_stability_ratios = _normalize_metric_pair_sequence(
        stability_ratio_metrics,
        field_name="stability_ratio_metrics",
    )
    normalized_membership_indexes = _normalize_string_sequence(
        membership_indexes,
        field_name="membership_indexes",
    )
    normalized_universe_required_membership_indexes = _normalize_string_sequence(
        universe_required_membership_indexes,
        field_name="universe_required_membership_indexes",
    )
    metadata_membership_indexes = _merge_string_sequences(
        normalized_membership_indexes,
        normalized_universe_required_membership_indexes,
    )
    if not isinstance(universe_require_tradable, bool):
        raise ValueError("universe_require_tradable must be a boolean.")
    if not isinstance(include_market_cap, bool):
        raise ValueError("include_market_cap must be a boolean.")
    normalized_market_cap_bucket_count = None
    if market_cap_bucket_count is not None:
        normalized_market_cap_bucket_count = _normalize_positive_int(
            market_cap_bucket_count,
            field_name="market_cap_bucket_count",
        )
        if normalized_market_cap_bucket_count < 2:
            raise ValueError("market_cap_bucket_count must be at least 2.")
        if not include_market_cap:
            raise ValueError(
                "market_cap_bucket_count requires include_market_cap=True."
            )

    entries: list[FeatureMetadata] = []

    def add(
        column: str,
        *,
        role: str,
        family: str,
        source: str,
        inputs: Sequence[str],
        timing: str,
        missing_policy: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        entries.append(
            FeatureMetadata(
                column=column,
                role=role,
                family=family,
                source=source,
                inputs=tuple(inputs),
                timing=timing,
                missing_policy=missing_policy,
                parameters={} if parameters is None else parameters,
            )
        )

    add(
        "daily_return",
        role="feature",
        family="return",
        source="ohlcv",
        inputs=("close", "previous close"),
        timing=_CLOSE_ANCHORED_TIMING,
        missing_policy="first observation per symbol is missing",
    )
    add(
        "log_return",
        role="feature",
        family="return",
        source="ohlcv",
        inputs=("daily_return",),
        timing=_CLOSE_ANCHORED_TIMING,
        missing_policy="missing when daily_return is missing",
    )
    add(
        f"rolling_volatility_{volatility_window}d",
        role="feature",
        family="rolling_return_statistic",
        source="ohlcv",
        inputs=("daily_return",),
        timing=_CLOSE_ANCHORED_TIMING,
        missing_policy="requires a full trailing window per symbol",
        parameters={"window": volatility_window},
    )
    add(
        f"rolling_average_volume_{average_volume_window}d",
        role="feature",
        family="rolling_liquidity_statistic",
        source="ohlcv",
        inputs=("volume",),
        timing=_CLOSE_ANCHORED_TIMING,
        missing_policy="requires a full trailing window per symbol",
        parameters={"window": average_volume_window},
    )
    for horizon in horizons:
        add(
            f"forward_return_{horizon}d",
            role="label",
            family="forward_return",
            source="ohlcv",
            inputs=("close", f"close + {horizon} sessions"),
            timing="future label; not valid as a same-row feature input",
            missing_policy="tail observations per symbol are missing",
            parameters={"horizon": horizon},
        )

    _append_optional_window_features(
        add,
        average_true_range_window=average_true_range_window,
        normalized_average_true_range_window=normalized_average_true_range_window,
        amihud_illiquidity_window=amihud_illiquidity_window,
        dollar_volume_shock_window=dollar_volume_shock_window,
        dollar_volume_zscore_window=dollar_volume_zscore_window,
        volume_shock_window=volume_shock_window,
        relative_volume_window=relative_volume_window,
        relative_dollar_volume_window=relative_dollar_volume_window,
        garman_klass_volatility_window=garman_klass_volatility_window,
        parkinson_volatility_window=parkinson_volatility_window,
        rogers_satchell_volatility_window=rogers_satchell_volatility_window,
        yang_zhang_volatility_window=yang_zhang_volatility_window,
        realized_volatility_window=realized_volatility_window,
        higher_moments_window=higher_moments_window,
        benchmark_residual_return_window=benchmark_residual_return_window,
        benchmark_rolling_window=benchmark_rolling_window,
    )

    for metric_name in _merge_fundamental_metric_names(
        normalized_fundamental_metrics,
        normalized_valuation_metrics,
        normalized_quality_ratios,
        normalized_growth_metrics,
        normalized_stability_ratios,
    ):
        add(
            fundamental_column_name(metric_name),
            role="feature",
            family="fundamental",
            source="fundamentals",
            inputs=(metric_name,),
            timing=_NEXT_SESSION_FUNDAMENTAL_TIMING,
            missing_policy="missing before first available release or when metric is unavailable",
        )

    for metric_name in normalized_valuation_metrics:
        add(
            _valuation_column_name(metric_name),
            role="feature",
            family="valuation",
            source="fundamentals+ohlcv",
            inputs=(fundamental_column_name(metric_name), "close"),
            timing=(
                "next-session-safe fundamental numerator divided by same-day close"
            ),
            missing_policy="missing when the fundamental input is unavailable",
        )
    for numerator_metric, denominator_metric in normalized_quality_ratios:
        add(
            quality_ratio_column_name(numerator_metric, denominator_metric),
            role="feature",
            family="quality_ratio",
            source="fundamentals",
            inputs=(
                fundamental_column_name(numerator_metric),
                fundamental_column_name(denominator_metric),
            ),
            timing="ratio of next-session-safe fundamental inputs",
            missing_policy="missing when inputs are unavailable or denominator is nonpositive",
        )
    for metric_name in normalized_growth_metrics:
        add(
            growth_column_name(metric_name),
            role="feature",
            family="growth",
            source="fundamentals",
            inputs=(metric_name, "prior adjacent period value"),
            timing=_NEXT_SESSION_FUNDAMENTAL_TIMING,
            missing_policy=(
                "missing without a prior adjacent period, before release, or when "
                "the prior value is nonpositive"
            ),
        )
    for numerator_metric, denominator_metric in normalized_stability_ratios:
        add(
            stability_ratio_column_name(numerator_metric, denominator_metric),
            role="feature",
            family="stability_ratio",
            source="fundamentals",
            inputs=(
                fundamental_column_name(numerator_metric),
                fundamental_column_name(denominator_metric),
            ),
            timing="ratio of next-session-safe fundamental inputs",
            missing_policy="missing when inputs are unavailable or denominator is nonpositive",
        )

    for field_name in _normalize_string_sequence(
        classification_fields,
        field_name="classification_fields",
    ):
        add(
            f"classification_{field_name.strip().lower()}",
            role="descriptor",
            family="classification",
            source="classifications",
            inputs=(field_name,),
            timing=_EFFECTIVE_DATE_TIMING,
            missing_policy="missing before first effective record",
        )
    for index_name in metadata_membership_indexes:
        add(
            f"membership_{_slug(index_name)}",
            role="descriptor",
            family="membership",
            source="memberships",
            inputs=(index_name,),
            timing=_EFFECTIVE_DATE_TIMING,
            missing_policy="missing before first effective membership record",
        )
    for field_name in _normalize_string_sequence(
        borrow_fields,
        field_name="borrow_fields",
    ):
        add(
            _borrow_column_name(field_name),
            role="descriptor",
            family="borrow_availability",
            source="borrow_availability",
            inputs=(field_name,),
            timing=_EFFECTIVE_DATE_TIMING,
            missing_policy="missing before first effective borrow record",
        )
    if universe_require_tradable:
        add(
            "trading_is_tradable",
            role="descriptor",
            family="trading_status",
            source="trading_status",
            inputs=("is_tradable",),
            timing=_EFFECTIVE_DATE_TIMING,
            missing_policy="missing before first effective trading status record",
        )

    if include_market_cap:
        add(
            "shares_outstanding",
            role="descriptor",
            family="shares_outstanding",
            source="shares_outstanding",
            inputs=("shares_outstanding",),
            timing=_EFFECTIVE_DATE_TIMING,
            missing_policy="missing before first effective shares-outstanding record",
        )
        add(
            "market_cap",
            role="feature",
            family="size",
            source="shares_outstanding+ohlcv",
            inputs=("shares_outstanding", "close"),
            timing=(
                "effective-date-safe shares outstanding multiplied by same-day close"
            ),
            missing_policy="missing when shares_outstanding is unavailable",
        )
        if normalized_market_cap_bucket_count is not None:
            add(
                "market_cap_bucket",
                role="descriptor",
                family="size_bucket",
                source="market_cap",
                inputs=("market_cap",),
                timing=(
                    "same-date cross-sectional bucket based on "
                    "effective-date-safe market_cap"
                ),
                missing_policy=(
                    "missing when market_cap is unavailable or a date has too "
                    "few usable names for stable buckets"
                ),
                parameters={"n_buckets": normalized_market_cap_bucket_count},
            )

    if universe_enabled:
        universe_parameters: dict[str, Any] = {}
        if universe_lag is not None:
            universe_parameters["lag"] = _normalize_positive_int(
                universe_lag,
                field_name="universe_lag",
            )
        if universe_average_volume_window is not None:
            universe_parameters["average_volume_window"] = _normalize_positive_int(
                universe_average_volume_window,
                field_name="universe_average_volume_window",
            )
        if universe_average_dollar_volume_window is not None:
            universe_parameters["average_dollar_volume_window"] = _normalize_positive_int(
                universe_average_dollar_volume_window,
                field_name="universe_average_dollar_volume_window",
            )
        universe_source = "ohlcv+symbol_metadata"
        universe_inputs = ["lagged price/liquidity/listing-history observations"]
        if normalized_universe_required_membership_indexes:
            universe_parameters["required_membership_indexes"] = list(
                normalized_universe_required_membership_indexes
            )
            universe_source += "+memberships"
            universe_inputs.append("lagged effective-date-safe membership status")
        if universe_require_tradable:
            universe_parameters["require_tradable"] = True
            universe_source += "+trading_status"
            universe_inputs.append("lagged effective-date-safe trading status")
        add(
            "is_universe_eligible",
            role="filter",
            family="universe_eligibility",
            source=universe_source,
            inputs=tuple(universe_inputs),
            timing="eligibility uses lagged per-symbol observations",
            missing_policy="rows without enough lagged history are ineligible",
            parameters=universe_parameters,
        )

    return [entry.to_dict() for entry in entries]


def build_research_feature_cache_metadata(
    *,
    dataset_feature_metadata: Sequence[Mapping[str, Any]],
    signal_pipeline_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a stable metadata-only cache identity for a research feature plan.

    This does not materialize or reuse feature values. It records an auditable
    cache key that future storage layers can use without mixing feature columns
    with future-return labels.
    """
    normalized_features = _normalize_feature_metadata_entries(
        dataset_feature_metadata
    )
    normalized_signal = (
        None
        if signal_pipeline_metadata is None
        else _normalize_json_mapping(
            signal_pipeline_metadata,
            field_name="signal_pipeline_metadata",
        )
    )

    feature_entries = tuple(
        entry for entry in normalized_features if entry["role"] != "label"
    )
    label_entries = tuple(
        entry for entry in normalized_features if entry["role"] == "label"
    )
    payload = {
        "dataset_feature_metadata": normalized_features,
        "signal_pipeline_metadata": normalized_signal,
    }

    signal_columns: dict[str, str] = {}
    if normalized_signal is not None:
        raw_signal_column = normalized_signal.get("raw_signal_column")
        final_signal_column = normalized_signal.get("final_signal_column")
        if isinstance(raw_signal_column, str):
            signal_columns["raw_signal_column"] = raw_signal_column
        if isinstance(final_signal_column, str):
            signal_columns["final_signal_column"] = final_signal_column

    return {
        "schema_version": 1,
        "cache_key": _metadata_digest(payload),
        "cache_key_algorithm": "sha256 over canonical JSON metadata payload",
        "materialization": "metadata_only",
        "timing": (
            "feature cache identity is derived from point-in-time metadata; "
            "future-return labels are tracked separately and are not reusable "
            "feature inputs"
        ),
        "feature_columns": [entry["column"] for entry in feature_entries],
        "label_columns": [entry["column"] for entry in label_entries],
        "signal_columns": signal_columns,
        "fingerprints": {
            "feature_plan": _metadata_digest(feature_entries),
            "label_plan": _metadata_digest(label_entries),
            "signal_pipeline": _metadata_digest(normalized_signal),
        },
    }


def _append_optional_window_features(add: Any, **windows: int | None) -> None:
    """Append metadata for optional rolling/range/liquidity features."""
    window_specs = {
        "average_true_range_window": (
            "average_true_range_{window}d",
            "range",
            "ohlcv",
            ("high", "low", "previous close"),
            _CLOSE_ANCHORED_TIMING,
            "requires a full trailing true-range window",
        ),
        "normalized_average_true_range_window": (
            "normalized_average_true_range_{window}d",
            "range",
            "ohlcv",
            ("high", "low", "previous close", "close"),
            _CLOSE_ANCHORED_TIMING,
            "requires a full trailing true-range window",
        ),
        "amihud_illiquidity_window": (
            "amihud_illiquidity_{window}d",
            "liquidity",
            "ohlcv",
            ("daily_return", "close", "volume"),
            _CLOSE_ANCHORED_TIMING,
            "requires a full trailing window and positive dollar volume",
        ),
        "dollar_volume_shock_window": (
            "dollar_volume_shock_{window}d",
            "liquidity",
            "ohlcv",
            ("close", "volume", "prior log dollar volume"),
            "same-day dollar volume compared with a prior rolling baseline",
            "missing without a full prior rolling baseline or positive dollar volume",
        ),
        "dollar_volume_zscore_window": (
            "dollar_volume_zscore_{window}d",
            "liquidity",
            "ohlcv",
            ("close", "volume", "prior log dollar volume"),
            "same-day dollar volume compared with a prior rolling baseline",
            "missing without a full prior rolling baseline or positive sample standard deviation",
        ),
        "volume_shock_window": (
            "volume_shock_{window}d",
            "liquidity",
            "ohlcv",
            ("volume", "prior log volume"),
            "same-day volume compared with a prior rolling baseline",
            "missing without a full prior rolling baseline or positive volume",
        ),
        "relative_volume_window": (
            "relative_volume_{window}d",
            "liquidity",
            "ohlcv",
            ("volume", "prior average volume"),
            "same-day volume divided by a prior rolling baseline",
            "missing without a full prior rolling baseline or positive baseline",
        ),
        "relative_dollar_volume_window": (
            "relative_dollar_volume_{window}d",
            "liquidity",
            "ohlcv",
            ("close", "volume", "prior average dollar volume"),
            "same-day dollar volume divided by a prior rolling baseline",
            "missing without a full prior rolling baseline or positive baseline",
        ),
        "garman_klass_volatility_window": (
            "garman_klass_volatility_{window}d",
            "range_volatility",
            "ohlcv",
            ("open", "high", "low", "close"),
            _CLOSE_ANCHORED_TIMING,
            "requires a full trailing window and positive variance proxy",
        ),
        "parkinson_volatility_window": (
            "parkinson_volatility_{window}d",
            "range_volatility",
            "ohlcv",
            ("high", "low"),
            _CLOSE_ANCHORED_TIMING,
            "requires a full trailing window",
        ),
        "rogers_satchell_volatility_window": (
            "rogers_satchell_volatility_{window}d",
            "range_volatility",
            "ohlcv",
            ("open", "high", "low", "close"),
            _CLOSE_ANCHORED_TIMING,
            "requires a full trailing window",
        ),
        "yang_zhang_volatility_window": (
            "yang_zhang_volatility_{window}d",
            "range_volatility",
            "ohlcv",
            ("open", "high", "low", "close", "previous close"),
            _CLOSE_ANCHORED_TIMING,
            "requires a full trailing window",
        ),
        "benchmark_residual_return_window": (
            "benchmark_residual_return_{window}d",
            "benchmark_residual",
            "ohlcv+benchmark",
            ("daily_return", "benchmark_return", "prior rolling alpha/beta"),
            "same-day return residualized with prior rolling market-model estimates",
            "missing without a full prior estimation window",
        ),
    }
    for window_name, spec in window_specs.items():
        window = windows[window_name]
        if window is None:
            continue
        normalized_window = _normalize_positive_int(window, field_name=window_name)
        column_template, family, source, inputs, timing, missing_policy = spec
        add(
            column_template.format(window=normalized_window),
            role="feature",
            family=family,
            source=source,
            inputs=inputs,
            timing=timing,
            missing_policy=missing_policy,
            parameters={"window": normalized_window},
        )

    realized_window = windows["realized_volatility_window"]
    if realized_window is not None:
        normalized_window = _normalize_positive_int(
            realized_window,
            field_name="realized_volatility_window",
        )
        for column_name in (
            f"realized_volatility_{normalized_window}d",
            f"downside_realized_volatility_{normalized_window}d",
            f"upside_realized_volatility_{normalized_window}d",
        ):
            add(
                column_name,
                role="feature",
                family="realized_volatility",
                source="ohlcv",
                inputs=("daily_return",),
                timing=_CLOSE_ANCHORED_TIMING,
                missing_policy="requires a full trailing return window",
                parameters={"window": normalized_window},
            )

    higher_moments_window = windows["higher_moments_window"]
    if higher_moments_window is not None:
        normalized_window = _normalize_positive_int(
            higher_moments_window,
            field_name="higher_moments_window",
        )
        for column_name in (
            f"rolling_skew_{normalized_window}d",
            f"rolling_kurtosis_{normalized_window}d",
        ):
            add(
                column_name,
                role="feature",
                family="higher_moment",
                source="ohlcv",
                inputs=("daily_return",),
                timing=_CLOSE_ANCHORED_TIMING,
                missing_policy="requires a full trailing return window",
                parameters={"window": normalized_window},
            )

    benchmark_window = windows["benchmark_rolling_window"]
    if benchmark_window is not None:
        normalized_window = _normalize_positive_int(
            benchmark_window,
            field_name="benchmark_rolling_window",
        )
        for column_name in (
            f"rolling_benchmark_beta_{normalized_window}d",
            f"rolling_benchmark_correlation_{normalized_window}d",
        ):
            add(
                column_name,
                role="feature",
                family="benchmark_statistic",
                source="ohlcv+benchmark",
                inputs=("daily_return", "benchmark_return"),
                timing=_CLOSE_ANCHORED_TIMING,
                missing_policy="requires exact benchmark alignment and a full trailing window",
                parameters={"window": normalized_window},
            )


def _merge_fundamental_metric_names(
    fundamental_metrics: Sequence[str],
    valuation_metrics: Sequence[str],
    quality_ratio_metrics: Sequence[tuple[str, str]],
    growth_metrics: Sequence[str],
    stability_ratio_metrics: Sequence[tuple[str, str]],
) -> tuple[str, ...]:
    """Merge fundamental-derived inputs without duplicate metadata entries."""
    merged: list[str] = []
    seen: set[str] = set()

    def append(metric_name: str) -> None:
        if metric_name in seen:
            return
        merged.append(metric_name)
        seen.add(metric_name)

    for metric_name in fundamental_metrics:
        append(metric_name)
    for metric_name in valuation_metrics:
        append(metric_name)
    for numerator_metric, denominator_metric in quality_ratio_metrics:
        append(numerator_metric)
        append(denominator_metric)
    for metric_name in growth_metrics:
        append(metric_name)
    for numerator_metric, denominator_metric in stability_ratio_metrics:
        append(numerator_metric)
        append(denominator_metric)
    return tuple(merged)


def _merge_string_sequences(
    first: Sequence[str],
    second: Sequence[str],
) -> tuple[str, ...]:
    """Merge two string sequences while preserving first-seen order."""
    merged: list[str] = []
    seen_values: set[str] = set()
    for sequence in (first, second):
        for value in sequence:
            if value in seen_values:
                continue
            merged.append(value)
            seen_values.add(value)
    return tuple(merged)


def _normalize_positive_int_sequence(
    values: Sequence[int],
    *,
    field_name: str,
) -> tuple[int, ...]:
    """Normalize a non-empty sequence of positive integers."""
    if isinstance(values, int) and not isinstance(values, bool):
        raw_values = (values,)
    else:
        raw_values = tuple(values)
    if not raw_values:
        raise ValueError(f"{field_name} must contain at least one positive integer.")
    return tuple(
        _normalize_positive_int(value, field_name=field_name)
        for value in raw_values
    )


def _normalize_positive_int(value: object, *, field_name: str) -> int:
    """Validate a positive integer metadata parameter."""
    return _common_positive_int(value, parameter_name=field_name)


def _normalize_string_sequence(
    values: Sequence[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    """Normalize a sequence of unique non-empty strings."""
    return _common_string_sequence(
        values,
        parameter_name=field_name,
        allow_scalar=True,
        item_error_message=f"{field_name} must contain only non-empty strings.",
        duplicate_error_message=f"{field_name} must not contain duplicates.",
    )


def _normalize_metric_pair_sequence(
    values: Sequence[Sequence[str]],
    *,
    field_name: str,
) -> tuple[tuple[str, str], ...]:
    """Normalize a sequence of unique numerator/denominator metric pairs."""
    if isinstance(values, str):
        raise ValueError(
            f"{field_name} must contain [numerator, denominator] metric pairs."
        )
    pair_error_message = (
        f"{field_name} must contain [numerator, denominator] metric pairs."
    )
    return _common_string_pair_sequence(
        tuple(values),
        parameter_name=field_name,
        pair_error_message=pair_error_message,
        item_error_message=f"{field_name} must contain only non-empty strings.",
        duplicate_error_message=f"{field_name} must not contain duplicate metric pairs.",
        allow_equal_items=True,
    )


def _valuation_column_name(metric_name: str) -> str:
    """Build the valuation output column name for one metric."""
    return f"valuation_{_metric_slug(metric_name)}_to_price"


def _borrow_column_name(field_name: str) -> str:
    """Build the borrow availability output column name for one field."""
    normalized = field_name.strip().lower()
    if normalized == "is_borrowable":
        return "borrow_is_borrowable"
    if normalized == "borrow_fee_bps":
        return "borrow_fee_bps"
    raise ValueError(
        "borrow_fields must contain only 'is_borrowable' or 'borrow_fee_bps'."
    )


def _metric_slug(metric_name: str) -> str:
    """Convert a fundamental metric name into its dataset-column slug."""
    return fundamental_column_name(metric_name).removeprefix("fundamental_")


def _slug(value: str) -> str:
    """Normalize a descriptor value into a deterministic column-name slug."""
    normalized = _NON_IDENTIFIER_PATTERN.sub("_", value.strip().lower()).strip("_")
    if normalized == "":
        raise ValueError("metadata values must produce a non-empty column slug.")
    return normalized


def _normalize_feature_metadata_entries(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Normalize feature metadata entries for stable cache-key generation."""
    normalized_entries: list[dict[str, Any]] = []
    seen_columns: set[str] = set()
    required_fields = {
        "column",
        "role",
        "family",
        "source",
        "inputs",
        "timing",
        "missing_policy",
        "parameters",
    }
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("dataset_feature_metadata entries must be mappings.")
        missing_fields = sorted(required_fields.difference(entry))
        if missing_fields:
            raise ValueError(
                "dataset_feature_metadata entries are missing required fields: "
                + ", ".join(missing_fields)
                + "."
            )
        normalized_entry = _normalize_json_mapping(
            entry,
            field_name="dataset_feature_metadata",
        )
        column = normalized_entry["column"]
        if not isinstance(column, str) or column.strip() == "":
            raise ValueError("dataset_feature_metadata column must be non-empty.")
        if column in seen_columns:
            raise ValueError(
                f"dataset_feature_metadata contains duplicate column {column!r}."
            )
        seen_columns.add(column)
        role = normalized_entry["role"]
        if not isinstance(role, str) or role.strip() == "":
            raise ValueError("dataset_feature_metadata role must be non-empty.")
        normalized_entries.append(normalized_entry)
    return tuple(sorted(normalized_entries, key=lambda entry: str(entry["column"])))


def _normalize_json_mapping(
    value: Mapping[str, Any],
    *,
    field_name: str,
) -> dict[str, Any]:
    """Normalize a mapping to JSON-compatible values with string keys."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    return {
        str(key): _normalize_json_value(
            nested_value,
            field_name=f"{field_name}.{key}",
        )
        for key, nested_value in value.items()
    }


def _normalize_json_value(value: Any, *, field_name: str) -> Any:
    """Normalize values used in metadata cache payloads."""
    if isinstance(value, Mapping):
        return _normalize_json_mapping(value, field_name=field_name)
    if isinstance(value, tuple):
        return [
            _normalize_json_value(item, field_name=field_name)
            for item in value
        ]
    if isinstance(value, list):
        return [
            _normalize_json_value(item, field_name=field_name)
            for item in value
        ]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise ValueError(f"{field_name} must contain JSON-compatible metadata values.")


def _metadata_digest(payload: Any) -> str:
    """Hash a JSON-compatible metadata payload deterministically."""
    encoded = json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

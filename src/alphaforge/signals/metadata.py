"""Metadata helpers for configured signal pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from alphaforge.signals.cross_sectional import get_signal_transform_definition
from alphaforge.signals.definitions import get_factor_definition


def build_signal_pipeline_metadata(
    *,
    factor_name: str,
    factor_parameters: Mapping[str, Any] | None = None,
    winsorize_quantile: float | None = None,
    clip_lower_bound: float | None = None,
    clip_upper_bound: float | None = None,
    normalization: str = "none",
    normalization_group_column: str | None = None,
    raw_signal_column: str | None = None,
) -> dict[str, Any]:
    """Build JSON-friendly metadata for one configured signal pipeline."""
    factor_definition = get_factor_definition(factor_name)
    normalized_factor_parameters = factor_definition.normalize_parameters(
        factor_parameters
    )
    raw_column = raw_signal_column or factor_definition.output_column(
        normalized_factor_parameters
    )

    transform_pipeline = []
    final_column = raw_column
    if winsorize_quantile is not None:
        winsorize_definition = get_signal_transform_definition("winsorize")
        winsorize_parameters = winsorize_definition.normalize_parameters(
            {"quantile": winsorize_quantile}
        )
        output_column = winsorize_definition.output_column(final_column)
        transform_pipeline.append(
            _build_transform_step_metadata(
                winsorize_definition.to_metadata(),
                parameters=winsorize_parameters,
                input_column=final_column,
                output_column=output_column,
            )
        )
        final_column = output_column

    if clip_lower_bound is not None or clip_upper_bound is not None:
        clip_definition = get_signal_transform_definition("clip")
        clip_parameters = clip_definition.normalize_parameters(
            {
                "lower_bound": clip_lower_bound,
                "upper_bound": clip_upper_bound,
            }
        )
        output_column = clip_definition.output_column(final_column)
        transform_pipeline.append(
            _build_transform_step_metadata(
                clip_definition.to_metadata(),
                parameters=clip_parameters,
                input_column=final_column,
                output_column=output_column,
            )
        )
        final_column = output_column

    normalization_name = _normalize_normalization_name(normalization)
    group_column = _normalize_optional_group_column_name(
        normalization_group_column,
    )
    if group_column is not None and normalization_name == "none":
        raise ValueError(
            "normalization_group_column requires a non-'none' normalization."
        )
    if normalization_name != "none":
        transform_definition = get_signal_transform_definition(normalization_name)
        output_column = transform_definition.output_column(final_column)
        extra_metadata = {}
        if group_column is not None:
            extra_metadata["group_column"] = group_column
            extra_metadata["group_scope"] = "date_and_group"
        transform_pipeline.append(
            _build_transform_step_metadata(
                transform_definition.to_metadata(),
                parameters=transform_definition.normalize_parameters({}),
                input_column=final_column,
                output_column=output_column,
                extra_metadata=extra_metadata,
            )
        )
        final_column = output_column

    factor_metadata = factor_definition.to_metadata()
    factor_metadata["parameters"] = normalized_factor_parameters
    factor_metadata["output_column"] = raw_column

    return {
        "factor": factor_metadata,
        "raw_signal_column": raw_column,
        "final_signal_column": final_column,
        "transform_pipeline": transform_pipeline,
        "timing": (
            "factor is close-anchored; transforms are same-date only; "
            "execution delay is applied downstream"
        ),
    }


def _build_transform_step_metadata(
    definition_metadata: dict[str, Any],
    *,
    parameters: Mapping[str, Any],
    input_column: str,
    output_column: str,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach configured column lineage to transform definition metadata."""
    step_metadata = dict(definition_metadata)
    step_metadata["parameters"] = dict(parameters)
    step_metadata["input_column"] = input_column
    step_metadata["output_column"] = output_column
    if extra_metadata is not None:
        step_metadata.update(extra_metadata)
    return step_metadata


def _normalize_normalization_name(value: str) -> str:
    """Normalize configured cross-sectional normalization for metadata output."""
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(
            "normalization must be one of "
            "{'none', 'rank', 'robust_zscore', 'zscore'}."
        )
    normalized = value.strip().lower()
    if normalized not in {"none", "rank", "robust_zscore", "zscore"}:
        raise ValueError(
            "normalization must be one of "
            "{'none', 'rank', 'robust_zscore', 'zscore'}."
        )
    return normalized


def _normalize_optional_group_column_name(value: str | None) -> str | None:
    """Normalize optional configured normalization group column metadata."""
    if value is None:
        return None
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError("normalization_group_column must be a non-empty string.")
    return value.strip()

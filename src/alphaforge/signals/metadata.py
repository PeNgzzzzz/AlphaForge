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
    normalization: str = "none",
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

    normalization_name = _normalize_normalization_name(normalization)
    if normalization_name != "none":
        transform_definition = get_signal_transform_definition(normalization_name)
        output_column = transform_definition.output_column(final_column)
        transform_pipeline.append(
            _build_transform_step_metadata(
                transform_definition.to_metadata(),
                parameters=transform_definition.normalize_parameters({}),
                input_column=final_column,
                output_column=output_column,
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
) -> dict[str, Any]:
    """Attach configured column lineage to transform definition metadata."""
    step_metadata = dict(definition_metadata)
    step_metadata["parameters"] = dict(parameters)
    step_metadata["input_column"] = input_column
    step_metadata["output_column"] = output_column
    return step_metadata


def _normalize_normalization_name(value: str) -> str:
    """Normalize configured cross-sectional normalization for metadata output."""
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError("normalization must be one of {'none', 'rank', 'zscore'}.")
    normalized = value.strip().lower()
    if normalized not in {"none", "rank", "zscore"}:
        raise ValueError("normalization must be one of {'none', 'rank', 'zscore'}.")
    return normalized

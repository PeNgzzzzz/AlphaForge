"""Shared validation helpers for AlphaForge."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
import math

ExceptionFactory = Callable[[str], Exception]

__all__ = [
    "normalize_finite_float",
    "normalize_non_negative_float",
    "normalize_non_empty_string",
    "normalize_positive_float",
    "normalize_positive_int",
    "require_columns",
]


def normalize_finite_float(
    value: object,
    *,
    parameter_name: str,
    error_factory: ExceptionFactory = ValueError,
) -> float:
    """Validate finite float parameters without changing caller error types."""
    numeric_value = _coerce_float(
        value,
        parameter_name=parameter_name,
        expected_description="finite float",
        error_factory=error_factory,
    )
    if not math.isfinite(numeric_value):
        raise error_factory(f"{parameter_name} must be a finite float.")
    return numeric_value


def normalize_non_negative_float(
    value: object,
    *,
    parameter_name: str,
    error_factory: ExceptionFactory = ValueError,
) -> float:
    """Validate non-negative float parameters without changing caller error types."""
    numeric_value = _coerce_float(
        value,
        parameter_name=parameter_name,
        expected_description="non-negative float",
        error_factory=error_factory,
    )
    if math.isnan(numeric_value) or numeric_value < 0.0:
        raise error_factory(f"{parameter_name} must be a non-negative float.")
    return numeric_value


def normalize_positive_float(
    value: object,
    *,
    parameter_name: str,
    error_factory: ExceptionFactory = ValueError,
) -> float:
    """Validate positive float parameters without changing caller error types."""
    numeric_value = _coerce_float(
        value,
        parameter_name=parameter_name,
        expected_description="positive float",
        error_factory=error_factory,
    )
    if math.isnan(numeric_value) or numeric_value <= 0.0:
        raise error_factory(f"{parameter_name} must be a positive float.")
    return numeric_value


def normalize_positive_int(
    value: object,
    *,
    parameter_name: str,
    error_factory: ExceptionFactory = ValueError,
) -> int:
    """Validate positive integer parameters without changing caller error types."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise error_factory(f"{parameter_name} must be a positive integer.")
    return value


def normalize_non_empty_string(
    value: object,
    *,
    parameter_name: str,
    error_factory: ExceptionFactory = ValueError,
) -> str:
    """Validate non-empty string parameters without changing caller error types."""
    if not isinstance(value, str) or not value.strip():
        raise error_factory(f"{parameter_name} must be a non-empty string.")
    return value.strip()


def require_columns(
    available_columns: Collection[str],
    required_columns: Sequence[str],
    *,
    source: str,
    error_factory: ExceptionFactory = ValueError,
    verb: str = "is",
) -> None:
    """Fail fast when required columns are missing while preserving error types."""
    missing_columns = [
        column for column in required_columns if column not in available_columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise error_factory(
            f"{source} {verb} missing required columns: {missing_text}."
        )


def _coerce_float(
    value: object,
    *,
    parameter_name: str,
    expected_description: str,
    error_factory: ExceptionFactory,
) -> float:
    """Parse float-like values while rejecting bools and preserving error types."""
    if isinstance(value, bool):
        raise error_factory(f"{parameter_name} must be a {expected_description}.")

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise error_factory(
            f"{parameter_name} must be a {expected_description}."
        ) from exc

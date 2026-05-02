"""Shared validation helpers for AlphaForge."""

from __future__ import annotations

from collections.abc import Callable
import math

ExceptionFactory = Callable[[str], Exception]

__all__ = ["normalize_finite_float", "normalize_positive_int"]


def normalize_finite_float(
    value: object,
    *,
    parameter_name: str,
    error_factory: ExceptionFactory = ValueError,
) -> float:
    """Validate finite float parameters without changing caller error types."""
    if isinstance(value, bool):
        raise error_factory(f"{parameter_name} must be a finite float.")

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise error_factory(f"{parameter_name} must be a finite float.") from exc

    if not math.isfinite(numeric_value):
        raise error_factory(f"{parameter_name} must be a finite float.")
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

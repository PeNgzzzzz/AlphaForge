"""Shared validation helpers for AlphaForge."""

from __future__ import annotations

from collections.abc import Callable

ExceptionFactory = Callable[[str], Exception]

__all__ = ["normalize_positive_int"]


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

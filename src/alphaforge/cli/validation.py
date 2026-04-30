"""Shared validation helpers for CLI workflows."""

from __future__ import annotations

from alphaforge.cli.errors import WorkflowError

__all__ = ["normalize_positive_int"]


def normalize_positive_int(value: int, *, parameter_name: str) -> int:
    """Validate positive integer CLI parameters."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise WorkflowError(f"{parameter_name} must be a positive integer.")
    return value

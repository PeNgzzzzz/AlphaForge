"""Shared validation helpers for CLI workflows."""

from __future__ import annotations

from alphaforge.cli.errors import WorkflowError
from alphaforge.common.validation import normalize_positive_int as _common_positive_int

__all__ = ["normalize_positive_int"]


def normalize_positive_int(value: int, *, parameter_name: str) -> int:
    """Validate positive integer CLI parameters."""
    return _common_positive_int(
        value,
        parameter_name=parameter_name,
        error_factory=WorkflowError,
    )

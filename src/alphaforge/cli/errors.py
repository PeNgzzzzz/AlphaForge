"""Shared CLI workflow exceptions."""

from __future__ import annotations


class WorkflowError(ValueError):
    """Raised when a CLI workflow cannot run from the provided config."""

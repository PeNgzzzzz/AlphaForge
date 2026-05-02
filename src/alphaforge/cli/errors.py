"""Shared CLI workflow exceptions."""

from __future__ import annotations

from alphaforge.common.errors import AlphaForgeError


class WorkflowError(AlphaForgeError):
    """Raised when a CLI workflow cannot run from the provided config."""

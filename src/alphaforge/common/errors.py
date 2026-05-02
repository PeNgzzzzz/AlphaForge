"""Shared exception types for AlphaForge."""

from __future__ import annotations

__all__ = ["AlphaForgeError"]


class AlphaForgeError(ValueError):
    """Base class for AlphaForge validation and workflow errors."""

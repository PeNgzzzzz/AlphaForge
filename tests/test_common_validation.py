"""Tests for shared validation helpers."""

from __future__ import annotations

import pytest

from alphaforge.common.validation import normalize_finite_float, normalize_positive_int


class CustomValidationError(ValueError):
    """Raised by test-only validation wrappers."""


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), "x", True, False],
)
def test_normalize_finite_float_rejects_invalid_values(value: object) -> None:
    """Finite float validation should reject invalid finite-float inputs."""
    with pytest.raises(ValueError, match="clip_lower_bound must be a finite float"):
        normalize_finite_float(value, parameter_name="clip_lower_bound")


def test_normalize_finite_float_preserves_custom_error_type() -> None:
    """Callers should preserve package-specific finite-float exception types."""
    with pytest.raises(
        CustomValidationError,
        match="factor_exposure_bounds.min_exposure must be a finite float",
    ):
        normalize_finite_float(
            float("inf"),
            parameter_name="factor_exposure_bounds.min_exposure",
            error_factory=CustomValidationError,
        )


def test_normalize_finite_float_returns_valid_float() -> None:
    """Valid finite float-like values should normalize to floats."""
    assert normalize_finite_float("1.5", parameter_name="clip_upper_bound") == 1.5


@pytest.mark.parametrize("value", [0, -1, 1.0, "1", True, False])
def test_normalize_positive_int_rejects_invalid_values(value: object) -> None:
    """Positive integer validation should reject bools, non-ints, and non-positive ints."""
    with pytest.raises(ValueError, match="window must be a positive integer"):
        normalize_positive_int(value, parameter_name="window")


def test_normalize_positive_int_preserves_custom_error_type() -> None:
    """Callers should preserve package-specific exception types through the helper."""
    with pytest.raises(CustomValidationError, match="lookback must be a positive integer"):
        normalize_positive_int(
            0,
            parameter_name="lookback",
            error_factory=CustomValidationError,
        )


def test_normalize_positive_int_returns_valid_int() -> None:
    """Valid positive integers should round-trip unchanged."""
    assert normalize_positive_int(3, parameter_name="top_n") == 3

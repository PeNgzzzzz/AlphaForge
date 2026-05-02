"""Tests for shared validation helpers."""

from __future__ import annotations

import pytest

from alphaforge.common.validation import (
    normalize_finite_float,
    normalize_non_negative_float,
    normalize_non_empty_string,
    normalize_positive_float,
    normalize_positive_int,
    require_columns,
)


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


@pytest.mark.parametrize("value", [float("nan"), -0.1, "x", True, False])
def test_normalize_non_negative_float_rejects_invalid_values(value: object) -> None:
    """Non-negative float validation should reject invalid runtime inputs."""
    with pytest.raises(ValueError, match="exposure must be a non-negative float"):
        normalize_non_negative_float(value, parameter_name="exposure")


def test_normalize_non_negative_float_preserves_custom_error_type() -> None:
    """Callers should preserve package-specific non-negative float exception types."""
    with pytest.raises(
        CustomValidationError,
        match="transaction_cost_bps must be a non-negative float",
    ):
        normalize_non_negative_float(
            float("nan"),
            parameter_name="transaction_cost_bps",
            error_factory=CustomValidationError,
        )


def test_normalize_non_negative_float_returns_valid_float() -> None:
    """Valid non-negative float-like values should normalize to floats."""
    assert normalize_non_negative_float("0.5", parameter_name="exposure") == 0.5
    assert normalize_non_negative_float(
        float("inf"),
        parameter_name="exposure",
    ) == float("inf")


@pytest.mark.parametrize("value", [float("nan"), 0.0, -0.1, "x", True, False])
def test_normalize_positive_float_rejects_invalid_values(value: object) -> None:
    """Positive float validation should reject invalid runtime inputs."""
    with pytest.raises(ValueError, match="initial_nav must be a positive float"):
        normalize_positive_float(value, parameter_name="initial_nav")


def test_normalize_positive_float_preserves_custom_error_type() -> None:
    """Callers should preserve package-specific positive float exception types."""
    with pytest.raises(CustomValidationError, match="initial_nav must be a positive float"):
        normalize_positive_float(
            0.0,
            parameter_name="initial_nav",
            error_factory=CustomValidationError,
        )


def test_normalize_positive_float_returns_valid_float() -> None:
    """Valid positive float-like values should normalize to floats."""
    assert normalize_positive_float("1.5", parameter_name="initial_nav") == 1.5
    assert normalize_positive_float(
        float("inf"),
        parameter_name="initial_nav",
    ) == float("inf")


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


@pytest.mark.parametrize("value", ["", " ", 3, None, True, False])
def test_normalize_non_empty_string_rejects_invalid_values(value: object) -> None:
    """Non-empty string validation should reject blank and non-string inputs."""
    with pytest.raises(ValueError, match="group_column must be a non-empty string"):
        normalize_non_empty_string(value, parameter_name="group_column")


def test_normalize_non_empty_string_preserves_custom_error_type() -> None:
    """Callers should preserve package-specific non-empty string exception types."""
    with pytest.raises(
        CustomValidationError,
        match="exposure_columns must be a non-empty string",
    ):
        normalize_non_empty_string(
            "",
            parameter_name="exposure_columns",
            error_factory=CustomValidationError,
        )


def test_normalize_non_empty_string_returns_stripped_string() -> None:
    """Valid non-empty strings should normalize by trimming surrounding whitespace."""
    assert (
        normalize_non_empty_string("  sector  ", parameter_name="group_column")
        == "sector"
    )


def test_require_columns_rejects_missing_columns_with_custom_error_type() -> None:
    """Required-column checks should preserve source text and caller error type."""
    with pytest.raises(
        CustomValidationError,
        match="weight panel is missing required columns: symbol, portfolio_weight",
    ):
        require_columns(
            {"date", "close"},
            ("date", "symbol", "portfolio_weight"),
            source="weight panel",
            error_factory=CustomValidationError,
        )


def test_require_columns_supports_are_verb_for_plural_sources() -> None:
    """Required-column checks should preserve existing plural-source messages."""
    with pytest.raises(
        ValueError,
        match="comparison results are missing required columns: run_id",
    ):
        require_columns(
            {"created_at"},
            ("run_id", "created_at"),
            source="comparison results",
            verb="are",
        )


def test_require_columns_allows_complete_column_sets() -> None:
    """Complete required-column sets should pass without returning a value."""
    assert (
        require_columns(
            {"date", "symbol", "portfolio_weight"},
            ("date", "symbol"),
            source="weight panel",
        )
        is None
    )

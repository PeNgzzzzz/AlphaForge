"""Tests for shared validation helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.common.validation import (
    normalize_finite_float,
    normalize_non_negative_float,
    normalize_non_empty_string,
    normalize_non_empty_string_series,
    normalize_optional_finite_float,
    normalize_optional_non_negative_float,
    normalize_optional_positive_float,
    normalize_positive_float,
    normalize_positive_int,
    normalize_unique_non_empty_string_sequence,
    normalize_unique_non_empty_string_pair_sequence,
    parse_numeric_series,
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


def test_optional_float_helpers_return_none_for_missing_values() -> None:
    """Optional float validation helpers should pass through None."""
    assert (
        normalize_optional_finite_float(
            None,
            parameter_name="factor_exposure_bounds.min_exposure",
        )
        is None
    )
    assert (
        normalize_optional_non_negative_float(
            None,
            parameter_name="max_turnover",
        )
        is None
    )
    assert (
        normalize_optional_positive_float(
            None,
            parameter_name="max_position_weight",
        )
        is None
    )


def test_optional_float_helpers_preserve_custom_error_types() -> None:
    """Optional float validation should preserve package-specific errors."""
    with pytest.raises(
        CustomValidationError,
        match="factor_exposure_bounds.max_exposure must be a finite float",
    ):
        normalize_optional_finite_float(
            float("nan"),
            parameter_name="factor_exposure_bounds.max_exposure",
            error_factory=CustomValidationError,
        )

    with pytest.raises(
        CustomValidationError,
        match="max_turnover must be a non-negative float",
    ):
        normalize_optional_non_negative_float(
            -0.1,
            parameter_name="max_turnover",
            error_factory=CustomValidationError,
        )

    with pytest.raises(
        CustomValidationError,
        match="max_position_weight must be a positive float",
    ):
        normalize_optional_positive_float(
            0.0,
            parameter_name="max_position_weight",
            error_factory=CustomValidationError,
        )


def test_optional_float_helpers_return_normalized_floats() -> None:
    """Valid optional float-like values should normalize to floats."""
    assert normalize_optional_finite_float(
        "1.5",
        parameter_name="factor_exposure_bounds.min_exposure",
    ) == pytest.approx(1.5)
    assert normalize_optional_non_negative_float(
        "0.5",
        parameter_name="max_turnover",
    ) == pytest.approx(0.5)
    assert normalize_optional_positive_float(
        "0.5",
        parameter_name="max_position_weight",
    ) == pytest.approx(0.5)


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


def test_normalize_non_empty_string_series_rejects_empty_values_with_custom_error() -> None:
    """String-series validation should preserve caller source text and error type."""
    values = pd.Series(["momentum", " "])

    with pytest.raises(
        CustomValidationError,
        match="parameter sweep results contain missing or empty values in 'signal_column'",
    ):
        normalize_non_empty_string_series(
            values,
            column_name="signal_column",
            source="parameter sweep results",
            error_factory=CustomValidationError,
            verb="contain",
        )


def test_normalize_non_empty_string_series_returns_stripped_values() -> None:
    """Valid string series should normalize by trimming whitespace."""
    normalized = normalize_non_empty_string_series(
        pd.Series(["  lookback  ", "short_window"]),
        column_name="parameter_name",
        source="parameter sweep results",
        verb="contain",
    )

    assert normalized.tolist() == ["lookback", "short_window"]


def test_normalize_unique_non_empty_string_sequence_returns_stripped_values() -> None:
    """Unique string sequences should normalize by trimming surrounding whitespace."""
    assert normalize_unique_non_empty_string_sequence(
        ["  sector  ", "industry"],
        parameter_name="classification_fields",
    ) == ("sector", "industry")


def test_normalize_unique_non_empty_string_sequence_can_accept_scalar() -> None:
    """Callers can opt into treating one string as a one-item sequence."""
    assert normalize_unique_non_empty_string_sequence(
        "  revenue  ",
        parameter_name="fundamental_metrics",
        allow_scalar=True,
    ) == ("revenue",)


@pytest.mark.parametrize("value", ["sector", 3, None])
def test_normalize_unique_non_empty_string_sequence_rejects_non_sequences(
    value: object,
) -> None:
    """String-sequence validation should reject scalars unless explicitly allowed."""
    with pytest.raises(ValueError, match="feature_columns must be a sequence"):
        normalize_unique_non_empty_string_sequence(
            value,
            parameter_name="feature_columns",
        )


@pytest.mark.parametrize("value", [["sector", " "], ["sector", 3]])
def test_normalize_unique_non_empty_string_sequence_rejects_invalid_items(
    value: object,
) -> None:
    """String-sequence validation should reject blank and non-string members."""
    with pytest.raises(ValueError, match="classification_fields.*non-empty strings"):
        normalize_unique_non_empty_string_sequence(
            value,
            parameter_name="classification_fields",
        )


def test_normalize_unique_non_empty_string_sequence_rejects_duplicates() -> None:
    """Duplicate normalized values should fail after whitespace normalization."""
    with pytest.raises(
        ValueError,
        match="membership_indexes must not contain duplicates",
    ):
        normalize_unique_non_empty_string_sequence(
            ["S&P 500", " S&P 500 "],
            parameter_name="membership_indexes",
        )


def test_normalize_unique_string_sequence_preserves_custom_error_type() -> None:
    """Callers should preserve package-specific sequence exception types."""
    with pytest.raises(
        CustomValidationError,
        match="feature_columns must be unique strings",
    ):
        normalize_unique_non_empty_string_sequence(
            ["daily_return", "daily_return"],
            parameter_name="feature_columns",
            error_factory=CustomValidationError,
            duplicate_error_message="feature_columns must be unique strings.",
        )


def test_normalize_unique_string_pair_sequence_returns_stripped_pairs() -> None:
    """String-pair sequences should normalize pair members by trimming whitespace."""
    assert normalize_unique_non_empty_string_pair_sequence(
        [[" net_income ", " total_assets "]],
        parameter_name="quality_ratio_metrics",
    ) == (("net_income", "total_assets"),)


@pytest.mark.parametrize(
    "value",
    ["net_income", 3, None, ["net_income"], [["net_income"]]],
)
def test_normalize_unique_string_pair_sequence_rejects_invalid_shapes(
    value: object,
) -> None:
    """String-pair validation should reject non-pair structures."""
    with pytest.raises(ValueError, match="quality_ratio_metrics"):
        normalize_unique_non_empty_string_pair_sequence(
            value,
            parameter_name="quality_ratio_metrics",
        )


@pytest.mark.parametrize(
    "value",
    [[["net_income", " "]], [["net_income", 3]]],
)
def test_normalize_unique_string_pair_sequence_rejects_invalid_items(
    value: object,
) -> None:
    """String-pair validation should reject blank and non-string members."""
    with pytest.raises(ValueError, match="quality_ratio_metrics.*non-empty"):
        normalize_unique_non_empty_string_pair_sequence(
            value,
            parameter_name="quality_ratio_metrics",
        )


def test_normalize_unique_string_pair_sequence_rejects_equal_items() -> None:
    """Metric-pair style validation can reject identical normalized members."""
    with pytest.raises(ValueError, match="quality_ratio_metrics.*different"):
        normalize_unique_non_empty_string_pair_sequence(
            [["total_assets", " total_assets "]],
            parameter_name="quality_ratio_metrics",
        )


def test_normalize_unique_string_pair_sequence_can_allow_equal_items() -> None:
    """Callers can opt out when equal pair members are valid metadata inputs."""
    assert normalize_unique_non_empty_string_pair_sequence(
        [["total_assets", " total_assets "]],
        parameter_name="quality_ratio_metrics",
        allow_equal_items=True,
    ) == (("total_assets", "total_assets"),)


def test_normalize_unique_string_pair_sequence_rejects_duplicates() -> None:
    """Duplicate normalized pairs should fail after whitespace normalization."""
    with pytest.raises(ValueError, match="quality_ratio_metrics.*duplicate"):
        normalize_unique_non_empty_string_pair_sequence(
            [["net_income", "total_assets"], [" net_income ", " total_assets "]],
            parameter_name="quality_ratio_metrics",
        )


def test_normalize_unique_string_pair_sequence_preserves_custom_error_type() -> None:
    """Callers should preserve package-specific pair exception types."""
    with pytest.raises(
        CustomValidationError,
        match="quality_ratio_metrics must be explicit metric pairs",
    ):
        normalize_unique_non_empty_string_pair_sequence(
            [["net_income"]],
            parameter_name="quality_ratio_metrics",
            error_factory=CustomValidationError,
            pair_error_message=(
                "quality_ratio_metrics must be explicit metric pairs."
            ),
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


def test_parse_numeric_series_rejects_invalid_values_with_custom_error_type() -> None:
    """Shared numeric parsing should preserve caller source text and error type."""
    values = pd.Series([1.0, "bad", None])

    with pytest.raises(
        CustomValidationError,
        match="factor frame contains invalid numeric values in 'mean_ic'",
    ):
        parse_numeric_series(
            values,
            column_name="mean_ic",
            source="factor frame",
            error_factory=CustomValidationError,
            allow_missing=True,
        )


def test_parse_numeric_series_rejects_missing_values_by_default() -> None:
    """Required numeric columns should distinguish missing values by default."""
    values = pd.Series([1.0, None])

    with pytest.raises(
        ValueError,
        match="chart input contains missing numeric values in 'net_nav'",
    ):
        parse_numeric_series(
            values,
            column_name="net_nav",
            source="chart input",
        )


def test_parse_numeric_series_can_treat_missing_values_as_invalid() -> None:
    """Legacy callers can keep combined missing-or-invalid error text."""
    values = pd.Series([1.0, None])

    with pytest.raises(
        ValueError,
        match="backtest results contain invalid numeric values in 'net_return'",
    ):
        parse_numeric_series(
            values,
            column_name="net_return",
            source="backtest results",
            missing_values_are_invalid=True,
            verb="contain",
        )


def test_parse_numeric_series_can_allow_missing_values() -> None:
    """Optional numeric columns should preserve genuine missing values."""
    parsed = parse_numeric_series(
        pd.Series(["1.5", None]),
        column_name="style_beta",
        source="risk inputs",
        allow_missing=True,
    )

    assert parsed.iloc[0] == pytest.approx(1.5)
    assert pd.isna(parsed.iloc[1])


def test_parse_numeric_series_can_reject_non_finite_values() -> None:
    """Callers that require finite values should reject inf without rejecting NA."""
    values = pd.Series([1.0, float("inf"), None])

    with pytest.raises(
        CustomValidationError,
        match="risk inputs contain non-finite numeric values in 'style_beta'",
    ):
        parse_numeric_series(
            values,
            column_name="style_beta",
            source="risk inputs",
            error_factory=CustomValidationError,
            allow_missing=True,
            require_finite=True,
            verb="contain",
        )

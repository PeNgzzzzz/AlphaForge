"""Tests for fundamental feature metric-name selection validation."""

from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.data import DataValidationError
from alphaforge.features.fundamentals_join import attach_fundamentals_asof
from alphaforge.features.growth import normalize_growth_metrics
from alphaforge.features.valuation import attach_fundamental_price_ratios


def _market_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": ["AAPL", "AAPL"],
            "open": [99.0, 100.0],
            "high": [101.0, 102.0],
            "low": [98.0, 99.0],
            "close": [100.0, 101.0],
            "volume": [1_000_000, 1_100_000],
        }
    )


def _fundamentals_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "period_end_date": pd.to_datetime(["2023-12-31"]),
            "release_date": pd.to_datetime(["2024-01-02"]),
            "metric_name": ["revenue"],
            "metric_value": [10.0],
        }
    )


def test_attach_fundamentals_asof_accepts_scalar_metric_selection() -> None:
    """Fundamental selections should share scalar string handling with feature helpers."""
    result = attach_fundamentals_asof(
        _market_frame(),
        _fundamentals_frame(),
        metrics=" revenue ",
    )

    assert pd.isna(result.loc[0, "fundamental_revenue"])
    assert result.loc[1, "fundamental_revenue"] == pytest.approx(10.0)


def test_attach_fundamentals_asof_rejects_duplicate_metric_selection() -> None:
    """Duplicate selected fundamentals should fail after whitespace normalization."""
    with pytest.raises(
        ValueError,
        match="fundamental_metrics must not contain duplicate metric names",
    ):
        attach_fundamentals_asof(
            _market_frame(),
            _fundamentals_frame(),
            metrics=("revenue", " revenue "),
        )


def test_attach_fundamental_price_ratios_rejects_invalid_metric_names() -> None:
    """Valuation metric selections should reject blank and non-string members."""
    frame = _market_frame().assign(fundamental_eps=[1.0, 2.0])

    with pytest.raises(
        ValueError,
        match="valuation_metrics must contain only non-empty strings",
    ):
        attach_fundamental_price_ratios(
            frame,
            metrics=("eps", " "),
        )


def test_attach_fundamental_price_ratios_rejects_duplicate_output_columns() -> None:
    """Metric names that slug to one valuation column should remain explicit errors."""
    frame = _market_frame().assign(
        fundamental_book_value=[10.0, 11.0],
        fundamental_book_value_per_share=[5.0, 6.0],
    )

    with pytest.raises(
        ValueError,
        match="valuation_metrics contains metric names that normalize to the same",
    ):
        attach_fundamental_price_ratios(
            frame,
            metrics=("book value", "book-value"),
        )


def test_normalize_growth_metrics_uses_shared_string_selection_validation() -> None:
    """Growth metric selections should trim, deduplicate, and check availability."""
    assert normalize_growth_metrics(
        " revenue ",
        available_metrics=pd.Series(["revenue"]),
    ) == ("revenue",)

    with pytest.raises(
        ValueError,
        match="growth_metrics must not contain duplicate metric names",
    ):
        normalize_growth_metrics(("revenue", " revenue "))

    with pytest.raises(
        ValueError,
        match="growth_metrics contains metric names that normalize to the same",
    ):
        normalize_growth_metrics(("gross profit", "gross-profit"))

    with pytest.raises(
        DataValidationError,
        match="fundamentals input is missing configured metric_name values: margin",
    ):
        normalize_growth_metrics(
            ("margin",),
            available_metrics=pd.Series(["revenue"]),
        )

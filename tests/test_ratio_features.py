"""Tests for shared fundamental ratio feature helpers."""

from __future__ import annotations

import pytest

from alphaforge.features.quality import normalize_quality_ratio_metrics
from alphaforge.features.stability import normalize_stability_ratio_metrics


def test_normalize_quality_ratio_metrics_returns_stripped_pairs() -> None:
    """Quality metric-pair selections should trim each metric name."""
    assert normalize_quality_ratio_metrics(
        ((" net_income ", " total_assets "),),
    ) == (("net_income", "total_assets"),)


def test_normalize_quality_ratio_metrics_rejects_empty_selection() -> None:
    """Ratio features require at least one configured numerator/denominator pair."""
    with pytest.raises(
        ValueError,
        match="quality_ratio_metrics must contain at least one metric pair",
    ):
        normalize_quality_ratio_metrics(())


def test_normalize_quality_ratio_metrics_rejects_invalid_pair_shape() -> None:
    """Metric-pair selections should fail fast on non-pair structures."""
    with pytest.raises(
        ValueError,
        match="quality_ratio_metrics must contain \\[numerator, denominator\\]",
    ):
        normalize_quality_ratio_metrics(("net_income",))  # type: ignore[arg-type]


def test_normalize_quality_ratio_metrics_rejects_same_fundamental_column() -> None:
    """Different raw names that normalize to one fundamental column should fail."""
    with pytest.raises(
        ValueError,
        match=(
            "quality_ratio_metrics numerator and denominator must produce "
            "different fundamental columns"
        ),
    ):
        normalize_quality_ratio_metrics((("total_assets", "total-assets"),))


def test_normalize_stability_ratio_metrics_rejects_duplicate_fundamental_pairs() -> None:
    """Duplicate ratio pairs should be rejected after fundamental column normalization."""
    with pytest.raises(
        ValueError,
        match="stability_ratio_metrics must not contain duplicate metric pairs",
    ):
        normalize_stability_ratio_metrics(
            (
                ("total debt", "total assets"),
                ("total_debt", "total_assets"),
            ),
        )

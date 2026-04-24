"""Tests for research dataset feature provenance metadata."""

from __future__ import annotations

import pytest

from alphaforge.features import build_research_dataset_feature_metadata


def test_build_research_dataset_feature_metadata_describes_configured_features() -> None:
    """Feature metadata should expose timing, inputs, and missing-data policy."""
    metadata = build_research_dataset_feature_metadata(
        forward_horizons=(1, 5),
        volatility_window=3,
        average_volume_window=4,
        average_true_range_window=2,
        benchmark_residual_return_window=3,
        fundamental_metrics=("revenue",),
        valuation_metrics=("eps",),
        quality_ratio_metrics=(("net_income", "total_assets"),),
        growth_metrics=("revenue",),
        stability_ratio_metrics=(("total_debt", "total_assets"),),
        classification_fields=("sector",),
        membership_indexes=("S&P 500",),
        borrow_fields=("is_borrowable",),
        universe_enabled=True,
        universe_lag=1,
        universe_average_volume_window=2,
    )

    by_column = {entry["column"]: entry for entry in metadata}

    assert by_column["forward_return_5d"]["role"] == "label"
    assert by_column["forward_return_5d"]["timing"].startswith("future label")
    assert by_column["rolling_volatility_3d"]["parameters"] == {"window": 3}
    assert "next market session" in by_column["fundamental_revenue"]["timing"]
    assert by_column["valuation_eps_to_price"]["inputs"] == [
        "fundamental_eps",
        "close",
    ]
    assert by_column["quality_net_income_to_total_assets"]["family"] == (
        "quality_ratio"
    )
    assert by_column["growth_revenue"]["missing_policy"].startswith(
        "missing without a prior adjacent period"
    )
    assert by_column["stability_total_debt_to_total_assets"]["missing_policy"] == (
        "missing when inputs are unavailable or denominator is nonpositive"
    )
    assert by_column["membership_s_p_500"]["timing"].startswith("date-only")
    assert by_column["is_universe_eligible"]["role"] == "filter"
    assert by_column["is_universe_eligible"]["parameters"]["lag"] == 1


def test_build_research_dataset_feature_metadata_rejects_invalid_values() -> None:
    """Feature metadata inputs should fail fast on invalid config-like values."""
    with pytest.raises(ValueError, match="forward_horizons"):
        build_research_dataset_feature_metadata(forward_horizons=(0,))

    with pytest.raises(ValueError, match="borrow_fields"):
        build_research_dataset_feature_metadata(borrow_fields=("shortable",))

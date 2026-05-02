"""Tests for research dataset feature provenance metadata."""

from __future__ import annotations

import pytest

from alphaforge.features import (
    build_research_dataset_feature_metadata,
    build_research_feature_cache_metadata,
)
from alphaforge.signals import build_signal_pipeline_metadata


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
        include_market_cap=True,
        market_cap_bucket_count=3,
        universe_enabled=True,
        universe_lag=1,
        universe_average_volume_window=2,
        universe_required_membership_indexes=("NASDAQ 100",),
        universe_require_tradable=True,
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
    assert by_column["membership_nasdaq_100"]["source"] == "memberships"
    assert by_column["trading_is_tradable"]["source"] == "trading_status"
    assert by_column["shares_outstanding"]["family"] == "shares_outstanding"
    assert by_column["market_cap"]["inputs"] == ["shares_outstanding", "close"]
    assert by_column["market_cap_bucket"]["family"] == "size_bucket"
    assert by_column["market_cap_bucket"]["parameters"] == {"n_buckets": 3}
    assert by_column["is_universe_eligible"]["role"] == "filter"
    assert by_column["is_universe_eligible"]["parameters"]["lag"] == 1
    assert by_column["is_universe_eligible"]["parameters"][
        "required_membership_indexes"
    ] == ["NASDAQ 100"]
    assert by_column["is_universe_eligible"]["parameters"]["require_tradable"] is True
    assert "memberships" in by_column["is_universe_eligible"]["source"]
    assert "trading_status" in by_column["is_universe_eligible"]["source"]


def test_build_research_dataset_feature_metadata_rejects_invalid_values() -> None:
    """Feature metadata inputs should fail fast on invalid config-like values."""
    with pytest.raises(ValueError, match="forward_horizons"):
        build_research_dataset_feature_metadata(forward_horizons=(0,))

    with pytest.raises(ValueError, match="borrow_fields"):
        build_research_dataset_feature_metadata(borrow_fields=("shortable",))

    with pytest.raises(ValueError, match="fundamental_metrics.*duplicates"):
        build_research_dataset_feature_metadata(
            fundamental_metrics=("revenue", " revenue "),
        )

    with pytest.raises(ValueError, match="include_market_cap"):
        build_research_dataset_feature_metadata(
            include_market_cap="true",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="universe_require_tradable"):
        build_research_dataset_feature_metadata(
            universe_require_tradable="true",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="market_cap_bucket_count"):
        build_research_dataset_feature_metadata(
            include_market_cap=True,
            market_cap_bucket_count=1,
        )

    with pytest.raises(ValueError, match="requires include_market_cap"):
        build_research_dataset_feature_metadata(
            market_cap_bucket_count=3,
        )

    with pytest.raises(
        ValueError,
        match="quality_ratio_metrics must contain \\[numerator, denominator\\]",
    ):
        build_research_dataset_feature_metadata(
            quality_ratio_metrics=("net_income",),  # type: ignore[arg-type]
        )

    with pytest.raises(
        ValueError,
        match="quality_ratio_metrics must not contain duplicate metric pairs",
    ):
        build_research_dataset_feature_metadata(
            quality_ratio_metrics=(
                ("net_income", "total_assets"),
                (" net_income ", " total_assets "),
            ),
        )


def test_build_research_dataset_feature_metadata_preserves_equal_ratio_pair_inputs() -> None:
    """Metadata should preserve its existing tolerance for equal metric-pair items."""
    metadata = build_research_dataset_feature_metadata(
        quality_ratio_metrics=(("total_assets", " total_assets "),),
    )
    by_column = {entry["column"]: entry for entry in metadata}

    assert by_column["quality_total_assets_to_total_assets"]["inputs"] == [
        "fundamental_total_assets",
        "fundamental_total_assets",
    ]


def test_build_research_feature_cache_metadata_builds_stable_cache_identity() -> None:
    """Cache metadata should fingerprint the feature and signal plan only."""
    dataset_metadata = build_research_dataset_feature_metadata(
        forward_horizons=(1,),
        volatility_window=3,
        average_volume_window=4,
        classification_fields=("sector",),
    )
    signal_metadata = build_signal_pipeline_metadata(
        factor_name="momentum",
        factor_parameters={"lookback": 20},
        normalization="rank",
    )

    cache_metadata = build_research_feature_cache_metadata(
        dataset_feature_metadata=dataset_metadata,
        signal_pipeline_metadata=signal_metadata,
    )
    reordered_cache_metadata = build_research_feature_cache_metadata(
        dataset_feature_metadata=tuple(reversed(dataset_metadata)),
        signal_pipeline_metadata=dict(reversed(tuple(signal_metadata.items()))),
    )
    changed_cache_metadata = build_research_feature_cache_metadata(
        dataset_feature_metadata=dataset_metadata,
        signal_pipeline_metadata=build_signal_pipeline_metadata(
            factor_name="momentum",
            factor_parameters={"lookback": 10},
        ),
    )

    assert cache_metadata["schema_version"] == 1
    assert len(cache_metadata["cache_key"]) == 64
    assert cache_metadata["materialization"] == "metadata_only"
    assert cache_metadata["cache_key"] == reordered_cache_metadata["cache_key"]
    assert cache_metadata["cache_key"] != changed_cache_metadata["cache_key"]
    assert "daily_return" in cache_metadata["feature_columns"]
    assert "classification_sector" in cache_metadata["feature_columns"]
    assert "forward_return_1d" in cache_metadata["label_columns"]
    assert "forward_return_1d" not in cache_metadata["feature_columns"]
    assert cache_metadata["signal_columns"] == {
        "raw_signal_column": "momentum_signal_20d",
        "final_signal_column": "momentum_signal_20d_rank",
    }
    for fingerprint in cache_metadata["fingerprints"].values():
        assert len(fingerprint) == 64


def test_build_research_feature_cache_metadata_rejects_invalid_entries() -> None:
    """Cache metadata should fail fast on malformed provenance records."""
    dataset_metadata = build_research_dataset_feature_metadata()

    with pytest.raises(ValueError, match="duplicate column"):
        build_research_feature_cache_metadata(
            dataset_feature_metadata=(*dataset_metadata, dataset_metadata[0]),
        )

    with pytest.raises(ValueError, match="missing required fields"):
        build_research_feature_cache_metadata(
            dataset_feature_metadata=({"column": "daily_return"},),
        )

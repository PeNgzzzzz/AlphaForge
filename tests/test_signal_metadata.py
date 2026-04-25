"""Tests for configured signal pipeline metadata."""

from __future__ import annotations

import pytest

from alphaforge.signals import build_signal_pipeline_metadata


def test_signal_pipeline_metadata_records_factor_and_transform_lineage() -> None:
    """Configured metadata should align factor output and transform columns."""
    metadata = build_signal_pipeline_metadata(
        factor_name="momentum",
        factor_parameters={"lookback": 20},
        winsorize_quantile=0.1,
        clip_lower_bound=-2.0,
        clip_upper_bound=2.0,
        normalization="rank",
    )

    assert metadata["factor"]["name"] == "momentum"
    assert metadata["factor"]["parameters"] == {"lookback": 20}
    assert metadata["factor"]["output_column"] == "momentum_signal_20d"
    assert metadata["raw_signal_column"] == "momentum_signal_20d"
    assert metadata["final_signal_column"] == (
        "momentum_signal_20d_winsorized_clipped_rank"
    )
    assert "execution delay" in metadata["timing"]
    assert [
        step["name"] for step in metadata["transform_pipeline"]
    ] == ["winsorize", "clip", "rank"]
    assert metadata["transform_pipeline"][0]["parameters"] == {"quantile": 0.1}
    assert metadata["transform_pipeline"][0]["input_column"] == "momentum_signal_20d"
    assert metadata["transform_pipeline"][0]["output_column"] == (
        "momentum_signal_20d_winsorized"
    )
    assert metadata["transform_pipeline"][1]["input_column"] == (
        "momentum_signal_20d_winsorized"
    )
    assert metadata["transform_pipeline"][1]["output_column"] == (
        "momentum_signal_20d_winsorized_clipped"
    )
    assert metadata["transform_pipeline"][1]["parameters"] == {
        "lower_bound": -2.0,
        "upper_bound": 2.0,
    }
    assert metadata["transform_pipeline"][2]["input_column"] == (
        "momentum_signal_20d_winsorized_clipped"
    )
    assert metadata["transform_pipeline"][2]["output_column"] == (
        "momentum_signal_20d_winsorized_clipped_rank"
    )


def test_signal_pipeline_metadata_supports_no_transform_pipeline() -> None:
    """Pipelines without transforms should keep raw and final signal columns equal."""
    metadata = build_signal_pipeline_metadata(
        factor_name="mean_reversion",
        factor_parameters={"lookback": 5},
    )

    assert metadata["raw_signal_column"] == "mean_reversion_signal_5d"
    assert metadata["final_signal_column"] == "mean_reversion_signal_5d"
    assert metadata["transform_pipeline"] == []


def test_signal_pipeline_metadata_supports_robust_zscore_normalization() -> None:
    """Robust z-score should be represented as a same-date transform step."""
    metadata = build_signal_pipeline_metadata(
        factor_name="momentum",
        factor_parameters={"lookback": 10},
        normalization="robust_zscore",
    )

    assert metadata["raw_signal_column"] == "momentum_signal_10d"
    assert metadata["final_signal_column"] == (
        "momentum_signal_10d_robust_zscore"
    )
    assert [
        step["name"] for step in metadata["transform_pipeline"]
    ] == ["robust_zscore"]
    assert metadata["transform_pipeline"][0]["parameters"] == {}
    assert metadata["transform_pipeline"][0]["input_column"] == (
        "momentum_signal_10d"
    )
    assert metadata["transform_pipeline"][0]["output_column"] == (
        "momentum_signal_10d_robust_zscore"
    )


def test_signal_pipeline_metadata_records_grouped_normalization_scope() -> None:
    """Grouped normalization should record the explicit dataset group column."""
    metadata = build_signal_pipeline_metadata(
        factor_name="momentum",
        factor_parameters={"lookback": 10},
        normalization="zscore",
        normalization_group_column="classification_sector",
    )

    assert metadata["final_signal_column"] == "momentum_signal_10d_zscore"
    step = metadata["transform_pipeline"][0]
    assert step["name"] == "zscore"
    assert step["group_column"] == "classification_sector"
    assert step["group_scope"] == "date_and_group"


def test_signal_pipeline_metadata_records_grouped_neutralization_scope() -> None:
    """Grouped de-meaning should record its explicit dataset group column."""
    metadata = build_signal_pipeline_metadata(
        factor_name="momentum",
        factor_parameters={"lookback": 10},
        neutralize_group_column="classification_sector",
        normalization="zscore",
    )

    assert metadata["final_signal_column"] == (
        "momentum_signal_10d_demeaned_zscore"
    )
    assert [
        step["name"] for step in metadata["transform_pipeline"]
    ] == ["demean", "zscore"]
    step = metadata["transform_pipeline"][0]
    assert step["group_column"] == "classification_sector"
    assert step["group_scope"] == "date_and_group"
    assert step["neutralization"] == "group_demean"


def test_signal_pipeline_metadata_fails_fast_on_invalid_inputs() -> None:
    """Metadata construction should reuse definition-level validation semantics."""
    with pytest.raises(ValueError, match="factor name"):
        build_signal_pipeline_metadata(factor_name="carry")

    with pytest.raises(ValueError, match="normalization"):
        build_signal_pipeline_metadata(
            factor_name="momentum",
            normalization="robust",
        )

    with pytest.raises(ValueError, match="normalization_group_column"):
        build_signal_pipeline_metadata(
            factor_name="momentum",
            normalization_group_column="classification_sector",
        )

    with pytest.raises(ValueError, match="neutralize_group_column"):
        build_signal_pipeline_metadata(
            factor_name="momentum",
            neutralize_group_column=" ",
        )

    with pytest.raises(ValueError, match="winsorize_quantile"):
        build_signal_pipeline_metadata(
            factor_name="momentum",
            winsorize_quantile=0.5,
        )

    with pytest.raises(ValueError, match="clip_lower_bound"):
        build_signal_pipeline_metadata(
            factor_name="momentum",
            clip_lower_bound=1.0,
            clip_upper_bound=1.0,
        )

    with pytest.raises(ValueError, match="clip_upper_bound"):
        build_signal_pipeline_metadata(
            factor_name="momentum",
            clip_lower_bound=-1.0,
        )

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
        normalization="rank",
    )

    assert metadata["factor"]["name"] == "momentum"
    assert metadata["factor"]["parameters"] == {"lookback": 20}
    assert metadata["factor"]["output_column"] == "momentum_signal_20d"
    assert metadata["raw_signal_column"] == "momentum_signal_20d"
    assert metadata["final_signal_column"] == "momentum_signal_20d_winsorized_rank"
    assert "execution delay" in metadata["timing"]
    assert [
        step["name"] for step in metadata["transform_pipeline"]
    ] == ["winsorize", "rank"]
    assert metadata["transform_pipeline"][0]["parameters"] == {"quantile": 0.1}
    assert metadata["transform_pipeline"][0]["input_column"] == "momentum_signal_20d"
    assert metadata["transform_pipeline"][0]["output_column"] == (
        "momentum_signal_20d_winsorized"
    )
    assert metadata["transform_pipeline"][1]["input_column"] == (
        "momentum_signal_20d_winsorized"
    )
    assert metadata["transform_pipeline"][1]["output_column"] == (
        "momentum_signal_20d_winsorized_rank"
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


def test_signal_pipeline_metadata_fails_fast_on_invalid_inputs() -> None:
    """Metadata construction should reuse definition-level validation semantics."""
    with pytest.raises(ValueError, match="factor name"):
        build_signal_pipeline_metadata(factor_name="carry")

    with pytest.raises(ValueError, match="normalization"):
        build_signal_pipeline_metadata(
            factor_name="momentum",
            normalization="robust",
        )

    with pytest.raises(ValueError, match="winsorize_quantile"):
        build_signal_pipeline_metadata(
            factor_name="momentum",
            winsorize_quantile=0.5,
        )

"""Tests for materialized research feature caches."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from alphaforge.features import (
    FeatureCacheError,
    build_research_dataset_feature_metadata,
    build_research_feature_cache_metadata,
    load_research_feature_cache,
    load_research_feature_cache_manifest,
    write_research_feature_cache,
)
from alphaforge.signals import build_signal_pipeline_metadata


def test_write_and_load_research_feature_cache_excludes_future_labels(
    tmp_path: Path,
) -> None:
    """Materialized caches should only include reusable features and signals."""
    cache_metadata = _build_cache_metadata()
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": ["AAPL", "AAPL"],
            "daily_return": [0.01, -0.02],
            "log_return": [0.00995, -0.0202],
            "rolling_volatility_2d": [None, 0.03],
            "rolling_average_volume_3d": [None, 1050.0],
            "momentum_signal_2d": [0.02, 0.01],
            "forward_return_1d": [0.04, None],
            "unused_column": [1, 2],
        }
    )

    paths = write_research_feature_cache(
        frame,
        tmp_path,
        cache_metadata=cache_metadata,
    )
    loaded = load_research_feature_cache(
        tmp_path,
        expected_cache_key=cache_metadata["cache_key"],
    )
    manifest = load_research_feature_cache_manifest(tmp_path)

    assert paths["data_path"] == tmp_path / "features.parquet"
    assert paths["manifest_path"] == tmp_path / "manifest.json"
    assert loaded.columns.tolist() == [
        "date",
        "symbol",
        "daily_return",
        "log_return",
        "rolling_average_volume_3d",
        "rolling_volatility_2d",
        "momentum_signal_2d",
    ]
    assert "forward_return_1d" not in loaded.columns
    assert "unused_column" not in loaded.columns
    assert manifest["cache_key"] == cache_metadata["cache_key"]
    assert manifest["materialization"] == "parquet_feature_frame"
    assert manifest["excluded_label_columns"] == ["forward_return_1d"]
    assert manifest["row_count"] == 2


def test_write_research_feature_cache_rejects_label_overlap(tmp_path: Path) -> None:
    """Cache metadata must not list future labels as reusable feature columns."""
    cache_metadata = dict(_build_cache_metadata())
    cache_metadata["feature_columns"] = [
        *cache_metadata["feature_columns"],
        "forward_return_1d",
    ]
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "symbol": ["AAPL"],
            "daily_return": [0.01],
            "forward_return_1d": [0.02],
        }
    )

    with pytest.raises(FeatureCacheError, match="label columns"):
        write_research_feature_cache(
            frame,
            tmp_path,
            cache_metadata=cache_metadata,
        )


def test_write_research_feature_cache_rejects_missing_columns(
    tmp_path: Path,
) -> None:
    """Missing feature or signal columns should fail before writing artifacts."""
    cache_metadata = _build_cache_metadata()
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "symbol": ["AAPL"],
            "daily_return": [0.01],
            "log_return": [0.00995],
        }
    )

    with pytest.raises(FeatureCacheError, match="rolling_volatility_2d"):
        write_research_feature_cache(
            frame,
            tmp_path,
            cache_metadata=cache_metadata,
        )


def test_load_research_feature_cache_rejects_cache_key_mismatch(
    tmp_path: Path,
) -> None:
    """Readers should fail fast when the manifest cache key is not expected."""
    cache_metadata = _build_cache_metadata()
    frame = _cacheable_frame()
    write_research_feature_cache(frame, tmp_path, cache_metadata=cache_metadata)

    with pytest.raises(FeatureCacheError, match="cache key mismatch"):
        load_research_feature_cache(tmp_path, expected_cache_key="0" * 64)


def test_load_research_feature_cache_rejects_manifest_column_tampering(
    tmp_path: Path,
) -> None:
    """Manifest cached columns must match feature plus signal columns."""
    cache_metadata = _build_cache_metadata()
    write_research_feature_cache(
        _cacheable_frame(),
        tmp_path,
        cache_metadata=cache_metadata,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cached_columns"].append("forward_return_1d")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FeatureCacheError, match="cached_columns"):
        load_research_feature_cache_manifest(tmp_path)


def _build_cache_metadata() -> dict[str, object]:
    """Build a cache metadata fixture with one close-anchored signal column."""
    dataset_metadata = build_research_dataset_feature_metadata(
        forward_horizons=(1,),
        volatility_window=2,
        average_volume_window=3,
    )
    signal_metadata = build_signal_pipeline_metadata(
        factor_name="momentum",
        factor_parameters={"lookback": 2},
    )
    return build_research_feature_cache_metadata(
        dataset_feature_metadata=dataset_metadata,
        signal_pipeline_metadata=signal_metadata,
    )


def _cacheable_frame() -> pd.DataFrame:
    """Build a frame containing all cacheable fixture columns."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "symbol": ["AAPL"],
            "daily_return": [0.01],
            "log_return": [0.00995],
            "rolling_volatility_2d": [0.02],
            "rolling_average_volume_3d": [1000.0],
            "momentum_signal_2d": [0.03],
            "forward_return_1d": [0.04],
        }
    )

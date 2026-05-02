"""Tests for CLI research metadata assembly helpers."""

from __future__ import annotations

import math

import pandas as pd

from alphaforge.cli.research_metadata import (
    build_config_snapshot,
    build_research_context_metadata,
    build_research_metadata_from_config,
    dataframe_records,
    scalar_or_none,
    series_to_metadata_dict,
)
from alphaforge.common import load_pipeline_config


def test_build_config_snapshot_records_research_assumptions() -> None:
    """Workflow snapshots should preserve configured research assumptions."""
    config = load_pipeline_config("configs/stage4_flagship_example.toml")

    snapshot = build_config_snapshot(config)

    assert snapshot["data"]["price_adjustment"] == "raw"
    assert snapshot["benchmark"]["name"] == "Synthetic Benchmark"
    assert snapshot["benchmark"]["rolling_window"] == 3
    assert snapshot["dataset"]["forward_horizons"] == [1]
    assert snapshot["signal"]["name"] == "momentum"
    assert snapshot["portfolio"]["max_position_weight"] == 0.6
    assert snapshot["backtest"]["signal_delay"] == 1
    assert snapshot["backtest"]["fill_timing"] == "close"
    assert snapshot["backtest"]["commission_bps_column"] is None
    assert snapshot["backtest"]["slippage_bps_column"] is None
    assert snapshot["backtest"]["liquidity_bucket_column"] is None
    assert snapshot["backtest"]["slippage_bps_by_liquidity_bucket"] == {}
    assert snapshot["backtest"]["borrow_fee_bps_column"] is None
    assert snapshot["backtest"]["shortable_column"] is None
    assert snapshot["backtest"]["max_trade_weight_column"] is None
    assert snapshot["backtest"]["max_participation_rate"] is None
    assert snapshot["backtest"]["participation_notional"] is None
    assert snapshot["backtest"]["min_trade_weight"] is None
    assert snapshot["universe"]["lag"] == 1


def test_build_research_metadata_from_config_records_feature_cache_plan() -> None:
    """Research metadata should expose feature, signal, and cache provenance."""
    config = load_pipeline_config("configs/stage3_benchmark_example.toml")

    metadata = build_research_metadata_from_config(config)

    feature_columns = {
        entry["column"] for entry in metadata["dataset_feature_metadata"]
    }
    assert "forward_return_1d" in feature_columns
    assert metadata["signal_pipeline_metadata"]["factor"]["name"] == "momentum"
    assert metadata["feature_cache_metadata"]["materialization"] == "metadata_only"
    assert len(metadata["feature_cache_metadata"]["cache_key"]) == 64
    assert "forward_return_1d" in metadata["feature_cache_metadata"]["label_columns"]
    assert "forward_return_1d" not in metadata["feature_cache_metadata"]["feature_columns"]


def test_build_research_context_metadata_includes_data_benchmark_and_universe() -> None:
    """Research context metadata should include lightweight input summaries."""
    config = load_pipeline_config("configs/stage4_flagship_example.toml")

    metadata = build_research_context_metadata(config)

    assert metadata["data_summary"]["rows"] == 40
    assert metadata["benchmark_summary"]["rows"] == 10
    assert metadata["universe_summary"]["eligible_rows"] > 0
    assert metadata["workflow_configuration"]["benchmark"]["name"] == (
        "Synthetic Benchmark"
    )
    assert metadata["signal_pipeline_metadata"]["factor"]["name"] == "momentum"


def test_metadata_scalar_helpers_convert_json_safe_values() -> None:
    """Metadata helper outputs should avoid pandas/numpy scalar leakage."""
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")],
            "metric": [1.25],
            "missing": [math.nan],
        }
    )

    records = dataframe_records(frame)
    series_metadata = series_to_metadata_dict(frame.iloc[0])

    assert records == [
        {
            "date": "2024-01-02T00:00:00",
            "metric": 1.25,
            "missing": None,
        }
    ]
    assert series_metadata["date"] == "2024-01-02T00:00:00"
    assert series_metadata["missing"] is None
    assert scalar_or_none(pd.NA) is None

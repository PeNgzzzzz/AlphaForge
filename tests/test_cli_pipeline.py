"""Tests for config-driven CLI pipeline assembly helpers."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from alphaforge.cli.errors import WorkflowError
from alphaforge.cli.pipeline import (
    align_benchmark_to_backtest,
    build_dataset_from_config,
    run_backtest_from_config,
    signal_parameters_from_config,
)
from alphaforge.common import load_pipeline_config


def test_run_backtest_from_config_attaches_strict_benchmark_columns() -> None:
    """Pipeline backtests should preserve exact-date benchmark-relative diagnostics."""
    config = load_pipeline_config("configs/stage3_benchmark_example.toml")

    backtest = run_backtest_from_config(config)

    assert {
        "benchmark_return",
        "excess_return",
        "benchmark_nav",
        "relative_return",
        "relative_nav",
    }.issubset(backtest.columns)
    assert backtest["benchmark_return"].notna().all()


def test_align_benchmark_to_backtest_rejects_misaligned_dates() -> None:
    """Benchmark attachment should fail loudly instead of silently reindexing."""
    backtest = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "net_return": [0.01, -0.02],
        }
    )
    benchmark = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-04"]),
            "benchmark_return": [0.005, -0.01],
        }
    )

    with pytest.raises(WorkflowError, match="benchmark returns must align exactly"):
        align_benchmark_to_backtest(backtest, benchmark)


def test_build_dataset_from_config_supports_benchmark_features() -> None:
    """Dataset assembly should continue loading required benchmark inputs on demand."""
    base_config = load_pipeline_config("configs/stage3_benchmark_example.toml")
    config = replace(
        base_config,
        dataset=replace(base_config.dataset, benchmark_rolling_window=3),
    )

    dataset = build_dataset_from_config(config)

    assert "rolling_benchmark_beta_3d" in dataset.columns
    assert "rolling_benchmark_correlation_3d" in dataset.columns


def test_signal_parameters_from_config_keeps_only_explicit_factor_parameters() -> None:
    """Signal metadata should receive only configured factor parameters."""
    config = load_pipeline_config("configs/trend_example.toml")

    assert config.signal is not None
    assert signal_parameters_from_config(config.signal) == {
        "short_window": 2,
        "long_window": 4,
    }

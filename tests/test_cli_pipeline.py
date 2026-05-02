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
    run_backtest_with_config,
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


def test_run_backtest_with_config_applies_fill_timing() -> None:
    """Config-driven backtests should preserve explicit fill timing assumptions."""
    base_config = load_pipeline_config("configs/sample_pipeline.toml")
    assert base_config.backtest is not None
    config = replace(
        base_config,
        backtest=replace(base_config.backtest, fill_timing="next_close"),
    )
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
            ("2024-01-04", "AAPL", 121.0, 1.0),
            ("2024-01-05", "AAPL", 133.1, 1.0),
        ]
    )

    backtest = run_backtest_with_config(frame, config=config)

    third_day = backtest.loc[backtest["date"] == pd.Timestamp("2024-01-04")].iloc[0]
    fourth_day = backtest.loc[backtest["date"] == pd.Timestamp("2024-01-05")].iloc[0]
    assert third_day["gross_return"] == pytest.approx(0.0)
    assert third_day["gross_exposure"] == pytest.approx(0.0)
    assert fourth_day["gross_return"] == pytest.approx(0.10)
    assert fourth_day["gross_exposure"] == pytest.approx(1.0)


def test_run_backtest_with_config_applies_row_level_cost_bps() -> None:
    """Config-driven backtests should pass row-level cost columns to the engine."""
    base_config = load_pipeline_config("configs/sample_pipeline.toml")
    assert base_config.backtest is not None
    config = replace(
        base_config,
        backtest=replace(
            base_config.backtest,
            transaction_cost_bps=None,
            commission_bps=0.0,
            slippage_bps=0.0,
            commission_bps_column="row_commission_bps",
            slippage_bps_column="row_slippage_bps",
        ),
    )
    frame = _panel_with_weights(
        [
            ("2024-01-02", "AAPL", 100.0, 0.0),
            ("2024-01-03", "AAPL", 110.0, 1.0),
            ("2024-01-04", "AAPL", 121.0, 0.0),
        ]
    )
    frame["row_commission_bps"] = [0.0, 0.0, 4.0]
    frame["row_slippage_bps"] = [0.0, 0.0, 6.0]

    backtest = run_backtest_with_config(frame, config=config)

    third_day = backtest.loc[backtest["date"] == pd.Timestamp("2024-01-04")].iloc[0]
    assert third_day["turnover"] == pytest.approx(1.0)
    assert third_day["commission_cost"] == pytest.approx(0.0004)
    assert third_day["slippage_cost"] == pytest.approx(0.0006)
    assert third_day["transaction_cost"] == pytest.approx(0.001)


def test_signal_parameters_from_config_keeps_only_explicit_factor_parameters() -> None:
    """Signal metadata should receive only configured factor parameters."""
    config = load_pipeline_config("configs/trend_example.toml")

    assert config.signal is not None
    assert signal_parameters_from_config(config.signal) == {
        "short_window": 2,
        "long_window": 4,
    }


def _panel_with_weights(
    rows: list[tuple[str, str, float, float | None]],
) -> pd.DataFrame:
    """Build a minimal OHLCV panel with an attached target weight column."""
    records = []
    for date, symbol, close, weight in rows:
        records.append(
            {
                "date": date,
                "symbol": symbol,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1_000,
                "portfolio_weight": weight,
            }
        )
    return pd.DataFrame(records)

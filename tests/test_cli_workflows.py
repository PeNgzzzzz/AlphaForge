"""Tests for config-driven CLI workflows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from alphaforge.cli.main import main
from alphaforge.cli.workflows import (
    compare_indexed_runs,
    load_benchmark_returns_from_config,
    load_corporate_actions_from_config,
    load_fundamentals_from_config,
    load_market_data_from_config,
    load_symbol_metadata_from_config,
    load_trading_calendar_from_config,
)
from alphaforge.common import ConfigError, load_pipeline_config


def test_load_pipeline_config_resolves_relative_paths(tmp_path: Path) -> None:
    """Config loading should resolve relative data paths from the config location."""
    data_path = tmp_path / "sample.csv"
    data_path.write_text(
        "date,symbol,open,high,low,close,volume\n"
        "2024-01-02,AAPL,100,101,99,100,1000\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(
        """
[data]
path = "sample.csv"

[signal]
name = "momentum"
lookback = 1
""".strip(),
        encoding="utf-8",
    )

    config = load_pipeline_config(config_path)

    assert config.data.path == data_path.resolve()
    assert config.signal is not None
    assert config.signal.name == "momentum"


def test_load_pipeline_config_parses_universe_section(tmp_path: Path) -> None:
    """Optional universe settings should parse into the top-level config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        universe_overrides={
            "min_price": "96.0",
            "min_average_volume": "950.0",
            "min_average_dollar_volume": "100000.0",
            "min_listing_history_days": "2",
            "lag": "1",
            "average_volume_window": "2",
            "average_dollar_volume_window": "2",
        },
    )

    config = load_pipeline_config(config_path)

    assert config.universe is not None
    assert config.universe.min_price == pytest.approx(96.0)
    assert config.universe.min_average_volume == pytest.approx(950.0)
    assert config.universe.min_average_dollar_volume == pytest.approx(100000.0)
    assert config.universe.min_listing_history_days == 2
    assert config.universe.lag == 1
    assert config.universe.average_volume_window == 2
    assert config.universe.average_dollar_volume_window == 2


def test_load_pipeline_config_parses_data_price_adjustment(tmp_path: Path) -> None:
    """Optional data.price_adjustment should parse into the top-level config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        data_overrides={
            "price_adjustment": '"split_adjusted"',
        },
        corporate_actions_rows=[
            ("AAPL", "2024-01-05", "split", "2.0", ""),
        ],
    )

    config = load_pipeline_config(config_path)

    assert config.data.price_adjustment == "split_adjusted"


def test_load_pipeline_config_parses_stage2_execution_settings(tmp_path: Path) -> None:
    """Optional Stage 2 portfolio and backtest settings should parse cleanly."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        portfolio_overrides={
            "top_n": "2",
            "max_position_weight": "0.55",
        },
        backtest_overrides={
            "transaction_cost_bps": None,
            "rebalance_frequency": '"weekly"',
            "commission_bps": "2.0",
            "slippage_bps": "3.0",
            "max_turnover": "0.5",
        },
    )

    config = load_pipeline_config(config_path)

    assert config.portfolio is not None
    assert config.portfolio.max_position_weight == pytest.approx(0.55)
    assert config.backtest is not None
    assert config.backtest.rebalance_frequency == "weekly"
    assert config.backtest.transaction_cost_bps is None
    assert config.backtest.commission_bps == pytest.approx(2.0)
    assert config.backtest.slippage_bps == pytest.approx(3.0)
    assert config.backtest.max_turnover == pytest.approx(0.5)


def test_load_pipeline_config_parses_benchmark_section(tmp_path: Path) -> None:
    """Optional benchmark settings should parse into the top-level config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        benchmark_overrides={
            "name": '"Synthetic Benchmark"',
            "return_column": '"benchmark_return"',
            "rolling_window": "3",
        },
    )

    config = load_pipeline_config(config_path)

    assert config.benchmark is not None
    assert config.benchmark.name == "Synthetic Benchmark"
    assert config.benchmark.return_column == "benchmark_return"
    assert config.benchmark.rolling_window == 3


def test_load_pipeline_config_parses_symbol_metadata_section(tmp_path: Path) -> None:
    """Optional symbol metadata settings should parse into the top-level config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        symbol_metadata_overrides={
            "listing_date_column": '"ipo_date"',
            "delisting_date_column": '"delisted_on"',
        },
        symbol_metadata_rows=[
            ("AAPL", "1980-12-12", ""),
            ("MSFT", "1986-03-13", ""),
        ],
    )

    config = load_pipeline_config(config_path)

    assert config.symbol_metadata is not None
    assert config.symbol_metadata.listing_date_column == "ipo_date"
    assert config.symbol_metadata.delisting_date_column == "delisted_on"


def test_load_pipeline_config_parses_corporate_actions_section(tmp_path: Path) -> None:
    """Optional corporate-actions settings should parse into the top-level config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        corporate_actions_overrides={
            "ex_date_column": '"event_date"',
            "action_type_column": '"event_type"',
            "split_ratio_column": '"ratio"',
            "cash_amount_column": '"cash"',
        },
        corporate_actions_rows=[
            ("AAPL", "2024-01-04", "split", "2.0", ""),
            ("MSFT", "2024-01-05", "cash_dividend", "", "0.62"),
        ],
    )

    config = load_pipeline_config(config_path)

    assert config.corporate_actions is not None
    assert config.corporate_actions.ex_date_column == "event_date"
    assert config.corporate_actions.action_type_column == "event_type"
    assert config.corporate_actions.split_ratio_column == "ratio"
    assert config.corporate_actions.cash_amount_column == "cash"


def test_load_pipeline_config_parses_fundamentals_section(tmp_path: Path) -> None:
    """Optional fundamentals settings should parse into the top-level config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        fundamentals_overrides={
            "period_end_column": '"period_end"',
            "release_date_column": '"released_on"',
            "metric_name_column": '"metric"',
            "metric_value_column": '"value"',
        },
        fundamentals_rows=[
            ("AAPL", "2023-12-31", "2024-01-30", "revenue", "119000"),
            ("MSFT", "2023-12-31", "2024-01-25", "revenue", "62000"),
        ],
    )

    config = load_pipeline_config(config_path)

    assert config.fundamentals is not None
    assert config.fundamentals.period_end_column == "period_end"
    assert config.fundamentals.release_date_column == "released_on"
    assert config.fundamentals.metric_name_column == "metric"
    assert config.fundamentals.metric_value_column == "value"


def test_load_pipeline_config_parses_calendar_section(tmp_path: Path) -> None:
    """Optional calendar settings should parse into the top-level config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        calendar_overrides={
            "name": '"NYSE Calendar"',
            "date_column": '"session_date"',
        },
        calendar_rows=["2024-01-02", "2024-01-03"],
    )

    config = load_pipeline_config(config_path)

    assert config.calendar is not None
    assert config.calendar.name == "NYSE Calendar"
    assert config.calendar.date_column == "session_date"


def test_load_benchmark_returns_from_config_normalizes_date_dtype(tmp_path: Path) -> None:
    """Loaded benchmark dates should use a stable nanosecond datetime dtype."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        benchmark_overrides={
            "name": '"Synthetic Benchmark"',
            "return_column": '"benchmark_return"',
            "rolling_window": "3",
        },
    )

    config = load_pipeline_config(config_path)
    benchmark = load_benchmark_returns_from_config(config)

    assert pd.api.types.is_datetime64_ns_dtype(benchmark["date"])


def test_load_symbol_metadata_from_config_normalizes_date_dtypes(
    tmp_path: Path,
) -> None:
    """Loaded symbol metadata dates should use stable nanosecond dtypes."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        symbol_metadata_rows=[
            ("AAPL", "1980-12-12", ""),
            ("MSFT", "1986-03-13", ""),
        ],
    )

    config = load_pipeline_config(config_path)
    symbol_metadata = load_symbol_metadata_from_config(config)

    assert pd.api.types.is_datetime64_ns_dtype(symbol_metadata["listing_date"])
    assert pd.api.types.is_datetime64_ns_dtype(symbol_metadata["delisting_date"])


def test_load_corporate_actions_from_config_normalizes_date_dtype(
    tmp_path: Path,
) -> None:
    """Loaded corporate-action dates should use a stable nanosecond dtype."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        corporate_actions_rows=[
            ("AAPL", "2024-01-04", "split", "2.0", ""),
            ("MSFT", "2024-01-05", "cash_dividend", "", "0.62"),
        ],
    )

    config = load_pipeline_config(config_path)
    corporate_actions = load_corporate_actions_from_config(config)

    assert pd.api.types.is_datetime64_ns_dtype(corporate_actions["ex_date"])


def test_load_fundamentals_from_config_normalizes_date_dtypes(
    tmp_path: Path,
) -> None:
    """Loaded fundamentals dates should use stable nanosecond dtypes."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        fundamentals_rows=[
            ("AAPL", "2023-12-31", "2024-01-30", "revenue", "119000"),
            ("MSFT", "2023-12-31", "2024-01-25", "revenue", "62000"),
        ],
    )

    config = load_pipeline_config(config_path)
    fundamentals = load_fundamentals_from_config(config)

    assert pd.api.types.is_datetime64_ns_dtype(fundamentals["period_end_date"])
    assert pd.api.types.is_datetime64_ns_dtype(fundamentals["release_date"])


def test_load_trading_calendar_from_config_normalizes_date_dtype(
    tmp_path: Path,
) -> None:
    """Loaded calendar dates should use a stable nanosecond datetime dtype."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        calendar_rows=["2024-01-02", "2024-01-03"],
    )

    config = load_pipeline_config(config_path)
    trading_calendar = load_trading_calendar_from_config(config)

    assert pd.api.types.is_datetime64_ns_dtype(trading_calendar["date"])


def test_load_market_data_from_config_applies_split_adjustments(
    tmp_path: Path,
) -> None:
    """Configured split-adjusted market data should rescale pre-split OHLCV rows."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        data_overrides={
            "price_adjustment": '"split_adjusted"',
        },
        corporate_actions_rows=[
            ("AAPL", "2024-01-05", "split", "2.0", ""),
        ],
    )
    data_path = tmp_path / "sample.csv"
    data_path.write_text(
        "\n".join(
            [
                "date,symbol,open,high,low,close,volume",
                "2024-01-04,AAPL,100,102,99,101,1000",
                "2024-01-05,AAPL,50,51,49,50.5,2000",
                "2024-01-08,AAPL,55,56,54,55.5,2100",
            ]
        ),
        encoding="utf-8",
    )

    config = load_pipeline_config(config_path)
    market_data = load_market_data_from_config(config)

    assert market_data["open"].tolist() == pytest.approx([50.0, 50.0, 55.0])
    assert market_data["close"].tolist() == pytest.approx([50.5, 50.5, 55.5])
    assert market_data["volume"].tolist() == pytest.approx([2000.0, 2000.0, 2100.0])
    assert market_data["price_adjustment_factor"].tolist() == pytest.approx(
        [0.5, 1.0, 1.0]
    )


def test_load_benchmark_returns_from_config_canonicalizes_custom_return_column(
    tmp_path: Path,
) -> None:
    """Configured benchmark return columns should map into the canonical output schema."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        benchmark_overrides={
            "name": '"Synthetic Benchmark"',
            "return_column": '"spx_return"',
            "rolling_window": "3",
        },
    )
    benchmark_path = tmp_path / "benchmark.csv"
    benchmark_path.write_text(
        "\n".join(
            [
                "date,spx_return,source_name",
                "2024-01-03,0.01,SPX",
                "2024-01-02,0.00,SPX",
                "2024-01-04,0.01,SPX",
                "2024-01-05,0.00,SPX",
                "2024-01-08,0.01,SPX",
                "2024-01-09,0.01,SPX",
            ]
        ),
        encoding="utf-8",
    )

    config = load_pipeline_config(config_path)
    benchmark = load_benchmark_returns_from_config(config)

    assert list(benchmark.columns) == ["date", "benchmark_return", "source_name"]
    assert benchmark["benchmark_return"].tolist() == pytest.approx(
        [0.0, 0.01, 0.01, 0.0, 0.01, 0.01]
    )


def test_load_corporate_actions_from_config_canonicalizes_custom_columns(
    tmp_path: Path,
) -> None:
    """Configured corporate-action columns should map into the canonical schema."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        corporate_actions_overrides={
            "ex_date_column": '"event_date"',
            "action_type_column": '"event_type"',
            "split_ratio_column": '"ratio"',
            "cash_amount_column": '"cash"',
        },
        corporate_actions_rows=[
            ("AAPL", "2024-01-04", "split", "2.0", ""),
            ("MSFT", "2024-01-05", "cash_dividend", "", "0.62"),
        ],
    )
    corporate_actions_path = tmp_path / "corporate_actions.csv"
    corporate_actions_path.write_text(
        "\n".join(
            [
                "symbol,event_date,event_type,ratio,cash,source_name",
                "MSFT,2024-01-05,cash_dividend,,0.62,vendor",
                "AAPL,2024-01-04,split,2.0,,vendor",
            ]
        ),
        encoding="utf-8",
    )

    config = load_pipeline_config(config_path)
    corporate_actions = load_corporate_actions_from_config(config)

    assert list(corporate_actions.columns) == [
        "symbol",
        "ex_date",
        "action_type",
        "split_ratio",
        "cash_amount",
        "source_name",
    ]
    assert corporate_actions["action_type"].tolist() == ["split", "cash_dividend"]


def test_load_fundamentals_from_config_canonicalizes_custom_columns(
    tmp_path: Path,
) -> None:
    """Configured fundamentals columns should map into the canonical schema."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        fundamentals_overrides={
            "period_end_column": '"period_end"',
            "release_date_column": '"released_on"',
            "metric_name_column": '"metric"',
            "metric_value_column": '"value"',
        },
        fundamentals_rows=[
            ("AAPL", "2023-12-31", "2024-01-30", "revenue", "119000"),
            ("MSFT", "2023-12-31", "2024-01-25", "revenue", "62000"),
        ],
    )
    fundamentals_path = tmp_path / "fundamentals.csv"
    fundamentals_path.write_text(
        "\n".join(
            [
                "symbol,period_end,released_on,metric,value,currency",
                "MSFT,2023-12-31,2024-01-25,revenue,62000,USD",
                "AAPL,2023-12-31,2024-01-30,revenue,119000,USD",
            ]
        ),
        encoding="utf-8",
    )

    config = load_pipeline_config(config_path)
    fundamentals = load_fundamentals_from_config(config)

    assert list(fundamentals.columns) == [
        "symbol",
        "period_end_date",
        "release_date",
        "metric_name",
        "metric_value",
        "currency",
    ]
    assert fundamentals["metric_name"].tolist() == ["revenue", "revenue"]
    assert fundamentals["metric_value"].tolist() == pytest.approx([119000.0, 62000.0])


def test_validate_data_command_fails_cleanly_on_intraday_benchmark_dates(
    tmp_path: Path, capsys
) -> None:
    """Benchmark timestamps must remain date-only under CLI validation."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        benchmark_overrides={
            "name": '"Synthetic Benchmark"',
            "rolling_window": "3",
        },
    )
    benchmark_path = tmp_path / "benchmark.csv"
    benchmark_path.write_text(
        "\n".join(
            [
                "date,benchmark_return",
                "2024-01-02 16:00:00,0.00",
                "2024-01-03,0.01",
                "2024-01-04,0.01",
                "2024-01-05,0.00",
                "2024-01-08,0.01",
                "2024-01-09,0.01",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["validate-data", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "intraday" in captured.err


def test_validate_data_command_prints_fundamentals_summary(
    tmp_path: Path, capsys
) -> None:
    """validate-data should summarize configured fundamentals coverage."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        fundamentals_rows=[
            ("AAPL", "2023-12-31", "2024-01-30", "revenue", "119000"),
            ("MSFT", "2023-12-31", "2024-01-25", "revenue", "62000"),
            ("AAPL", "2023-12-31", "2024-01-30", "net_income", "34000"),
        ],
    )

    exit_code = main(["validate-data", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Fundamentals Configuration" in captured.out
    assert "Fundamentals Summary" in captured.out
    assert "Metrics: 2" in captured.out


def test_load_pipeline_config_rejects_split_adjusted_without_corporate_actions(
    tmp_path: Path,
) -> None:
    """Split-adjusted data requires a corporate-actions input section."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        data_overrides={
            "price_adjustment": '"split_adjusted"',
        },
    )

    with pytest.raises(
        ConfigError,
        match="requires a \\[corporate_actions\\] section",
    ):
        load_pipeline_config(config_path)


def test_validate_data_command_prints_corporate_actions_summary(
    tmp_path: Path, capsys
) -> None:
    """validate-data should summarize configured corporate actions."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        corporate_actions_rows=[
            ("AAPL", "2024-01-04", "split", "2.0", ""),
            ("MSFT", "2024-01-05", "cash_dividend", "", "0.62"),
        ],
    )

    exit_code = main(["validate-data", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Corporate Actions Configuration" in captured.out
    assert "Corporate Actions Summary" in captured.out
    assert "Cash Dividends: 1" in captured.out


def test_validate_data_command_fails_cleanly_on_off_calendar_corporate_action_dates(
    tmp_path: Path, capsys
) -> None:
    """Corporate-action ex-dates should respect the configured trading calendar."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        calendar_rows=[
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ],
        corporate_actions_rows=[
            ("AAPL", "2024-01-04", "split", "2.0", ""),
            ("MSFT", "2024-01-10", "cash_dividend", "", "0.62"),
        ],
    )

    exit_code = main(["validate-data", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "corporate actions" in captured.err
    assert "configured trading calendar" in captured.err


def test_load_pipeline_config_rejects_empty_universe_section(tmp_path: Path) -> None:
    """A universe section must define at least one filtering threshold."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        universe_overrides={
            "lag": "1",
        },
    )

    with pytest.raises(ConfigError, match="at least one filtering threshold"):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_mixed_legacy_and_split_costs(
    tmp_path: Path,
) -> None:
    """Legacy and split transaction cost fields should not be accepted together."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        backtest_overrides={
            "commission_bps": "1.0",
        },
    )

    with pytest.raises(ConfigError, match="cannot be combined"):
        load_pipeline_config(config_path)


def test_validate_data_command_prints_market_summary(tmp_path: Path, capsys) -> None:
    """The validate-data command should load config-driven market data."""
    config_path = _write_pipeline_fixture(tmp_path)

    exit_code = main(["validate-data", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Data Summary" in captured.out
    assert "Data Quality Summary" in captured.out
    assert "Symbols: 2" in captured.out


def test_validate_data_command_prints_benchmark_preview_when_configured(
    tmp_path: Path, capsys
) -> None:
    """validate-data should preview configured benchmark settings and date coverage."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        benchmark_overrides={
            "name": '"Synthetic Benchmark"',
            "rolling_window": "3",
        },
    )

    exit_code = main(["validate-data", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Benchmark Configuration" in captured.out
    assert "Benchmark Summary" in captured.out
    assert "Rolling Window: 3" in captured.out


def test_validate_data_command_prints_symbol_metadata_preview_when_configured(
    tmp_path: Path, capsys
) -> None:
    """validate-data should preview configured symbol metadata coverage."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        symbol_metadata_rows=[
            ("AAPL", "1980-12-12", ""),
            ("MSFT", "1986-03-13", ""),
        ],
    )

    exit_code = main(["validate-data", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Symbol Metadata Configuration" in captured.out
    assert "Symbol Metadata Summary" in captured.out
    assert "Active Symbols: 2" in captured.out


def test_validate_data_command_prints_trading_calendar_preview_when_configured(
    tmp_path: Path, capsys
) -> None:
    """validate-data should preview configured trading calendar coverage."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        calendar_rows=[
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ],
    )

    exit_code = main(["validate-data", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Trading Calendar Configuration" in captured.out
    assert "Trading Calendar Summary" in captured.out
    assert "Sessions: 6" in captured.out


def test_validate_data_command_fails_cleanly_when_market_data_breaks_calendar(
    tmp_path: Path, capsys
) -> None:
    """validate-data should fail when market-data dates are off-calendar."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        calendar_rows=["2024-01-02", "2024-01-03"],
    )

    exit_code = main(["validate-data", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "configured trading calendar" in captured.err


def test_build_dataset_command_uses_trading_calendar_for_listing_history(
    tmp_path: Path, capsys
) -> None:
    """build-dataset should route trading calendars into listing-history filters."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        universe_overrides={
            "min_listing_history_days": "3",
            "lag": "1",
        },
        symbol_metadata_rows=[
            ("AAPL", "2024-01-02", ""),
        ],
        calendar_rows=["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
    )
    data_path = tmp_path / "sample.csv"
    data_path.write_text(
        "\n".join(
            [
                "date,symbol,open,high,low,close,volume",
                "2024-01-02,AAPL,100,101,99,100,1000",
                "2024-01-04,AAPL,110,111,109,110,1100",
                "2024-01-05,AAPL,121,122,120,121,1200",
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "dataset.csv"

    exit_code = main(
        ["build-dataset", "--config", str(config_path), "--output", str(output_path)]
    )
    _ = capsys.readouterr()

    assert exit_code == 0
    dataset = pd.read_csv(output_path)
    last_row = dataset.loc[dataset["date"] == "2024-01-05"].iloc[0]
    assert float(last_row["universe_lagged_listing_history_days"]) == pytest.approx(3.0)


def test_build_dataset_command_uses_symbol_metadata_for_listing_history(
    tmp_path: Path, capsys
) -> None:
    """build-dataset should route symbol metadata into listing-history filters."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        universe_overrides={
            "min_listing_history_days": "3",
            "lag": "1",
        },
        symbol_metadata_rows=[
            ("AAPL", "2024-01-02", ""),
            ("MSFT", "2024-01-02", ""),
        ],
    )
    data_path = tmp_path / "sample.csv"
    data_path.write_text(
        "\n".join(
            [
                "date,symbol,open,high,low,close,volume",
                "2024-01-02,AAPL,100,101,99,100,1000",
                "2024-01-04,AAPL,110,111,109,110,1100",
                "2024-01-05,AAPL,121,122,120,121,1200",
                "2024-01-02,MSFT,200,201,199,200,2000",
                "2024-01-03,MSFT,210,211,209,210,2100",
                "2024-01-04,MSFT,220,221,219,220,2200",
                "2024-01-05,MSFT,230,231,229,230,2300",
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "dataset.csv"

    exit_code = main(
        ["build-dataset", "--config", str(config_path), "--output", str(output_path)]
    )
    _ = capsys.readouterr()

    assert exit_code == 0
    dataset = pd.read_csv(output_path)
    aapl_last_row = dataset.loc[
        (dataset["symbol"] == "AAPL") & (dataset["date"] == "2024-01-05")
    ].iloc[0]
    assert float(aapl_last_row["universe_lagged_listing_history_days"]) == pytest.approx(
        3.0
    )


def test_validate_data_command_fails_cleanly_when_config_is_missing(
    tmp_path: Path, capsys
) -> None:
    """A missing config path should return a stable CLI error."""
    missing_path = tmp_path / "missing.toml"

    exit_code = main(["validate-data", "--config", str(missing_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Config file does not exist" in captured.err


def test_validate_data_command_fails_cleanly_on_invalid_toml(
    tmp_path: Path, capsys
) -> None:
    """Invalid TOML should return a stable CLI error."""
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        """
[data
path = "sample.csv"
""".strip(),
        encoding="utf-8",
    )

    exit_code = main(["validate-data", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Invalid TOML in config file" in captured.err


def test_validate_data_command_fails_cleanly_when_required_section_is_missing(
    tmp_path: Path, capsys
) -> None:
    """Missing required config sections should return a stable CLI error."""
    config_path = tmp_path / "missing-data-section.toml"
    config_path.write_text(
        """
[signal]
name = "momentum"
lookback = 1
""".strip(),
        encoding="utf-8",
    )

    exit_code = main(["validate-data", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "missing required section [data]" in captured.err


def test_build_dataset_command_writes_output_csv(tmp_path: Path, capsys) -> None:
    """The build-dataset command should persist the configured research dataset."""
    config_path = _write_pipeline_fixture(tmp_path)
    output_path = tmp_path / "dataset.csv"

    exit_code = main(
        ["build-dataset", "--config", str(config_path), "--output", str(output_path)]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_path.exists()
    written = pd.read_csv(output_path)
    assert "forward_return_1d" in written.columns
    assert "Saved dataset" in captured.out


def test_build_dataset_command_writes_universe_columns_when_configured(
    tmp_path: Path, capsys
) -> None:
    """The build-dataset command should expose lagged universe eligibility columns."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        universe_overrides={
            "min_price": "96.0",
            "min_average_volume": "950.0",
            "min_average_dollar_volume": "100000.0",
            "min_listing_history_days": "2",
            "lag": "1",
            "average_volume_window": "2",
            "average_dollar_volume_window": "2",
        },
    )
    output_path = tmp_path / "dataset_with_universe.csv"

    exit_code = main(
        ["build-dataset", "--config", str(config_path), "--output", str(output_path)]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    written = pd.read_csv(output_path)
    assert "is_universe_eligible" in written.columns
    assert "universe_exclusion_reason" in written.columns
    assert "universe_lagged_close" in written.columns
    assert "passes_universe_min_average_volume" in written.columns
    assert "universe_lagged_average_dollar_volume_2d" in written.columns
    assert not written["is_universe_eligible"].all()
    assert written["universe_exclusion_reason"].fillna("").str.contains(
        "below_min_price"
    ).any()
    assert "Saved dataset" in captured.out


def test_run_backtest_command_writes_output_csv(tmp_path: Path, capsys) -> None:
    """The run-backtest command should persist daily backtest results."""
    config_path = _write_pipeline_fixture(tmp_path)
    output_path = tmp_path / "backtest.csv"

    exit_code = main(
        ["run-backtest", "--config", str(config_path), "--output", str(output_path)]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_path.exists()
    written = pd.read_csv(output_path)
    assert "net_return" in written.columns
    assert "commission_cost" in written.columns
    assert "is_rebalance_date" in written.columns
    assert "Saved backtest results" in captured.out


def test_run_backtest_command_writes_stage2_execution_columns(
    tmp_path: Path, capsys
) -> None:
    """The backtest CSV should expose Stage 2 execution diagnostics."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        portfolio_overrides={
            "top_n": "2",
            "weighting": '"score"',
            "max_position_weight": "0.55",
        },
        backtest_overrides={
            "transaction_cost_bps": None,
            "rebalance_frequency": '"weekly"',
            "commission_bps": "2.0",
            "slippage_bps": "3.0",
            "max_turnover": "0.5",
        },
    )
    output_path = tmp_path / "stage2_backtest.csv"

    exit_code = main(
        ["run-backtest", "--config", str(config_path), "--output", str(output_path)]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    written = pd.read_csv(output_path)
    expected_columns = {
        "target_turnover",
        "turnover",
        "commission_cost",
        "slippage_cost",
        "gross_target_exposure",
        "target_holdings_count",
        "target_effective_weight_gap",
        "is_rebalance_date",
        "turnover_limit_applied",
    }
    assert expected_columns.issubset(set(written.columns))
    assert "Saved backtest results" in captured.out


def test_run_backtest_command_writes_benchmark_relative_columns(
    tmp_path: Path, capsys
) -> None:
    """The backtest CSV should expose benchmark-relative columns when configured."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        benchmark_overrides={
            "name": '"Synthetic Benchmark"',
            "rolling_window": "3",
        },
    )
    output_path = tmp_path / "benchmark_backtest.csv"

    exit_code = main(
        ["run-backtest", "--config", str(config_path), "--output", str(output_path)]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    written = pd.read_csv(output_path)
    expected_columns = {
        "benchmark_return",
        "excess_return",
        "benchmark_nav",
        "relative_return",
        "relative_nav",
    }
    assert expected_columns.issubset(set(written.columns))
    assert "Saved backtest results" in captured.out


def test_report_command_prints_summary_sections(tmp_path: Path, capsys) -> None:
    """The report command should print the configured pipeline summaries."""
    config_path = _write_pipeline_fixture(tmp_path)

    exit_code = main(["report", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Research Workflow" in captured.out
    assert "Data Quality Summary" in captured.out
    assert "Portfolio Constraints" in captured.out
    assert "Execution Assumptions" in captured.out
    assert "Execution Summary" in captured.out
    assert "Performance Summary" in captured.out
    assert "Risk Summary" in captured.out
    assert "Diagnostics Overview" in captured.out
    assert "IC Summary" in captured.out
    assert "Coverage Summary" in captured.out


def test_report_command_prints_stage2_execution_details(tmp_path: Path, capsys) -> None:
    """The report should surface configured Stage 2 constraints and realized execution stats."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        portfolio_overrides={
            "top_n": "2",
            "weighting": '"score"',
            "max_position_weight": "0.55",
        },
        backtest_overrides={
            "transaction_cost_bps": None,
            "rebalance_frequency": '"weekly"',
            "commission_bps": "2.0",
            "slippage_bps": "3.0",
            "max_turnover": "0.5",
        },
    )

    exit_code = main(["report", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Max Position Weight: 0.55" in captured.out
    assert "Rebalance Frequency: weekly" in captured.out
    assert "Max Turnover Per Rebalance: 0.5" in captured.out
    assert "Turnover Limit Applied Dates" in captured.out
    assert "Total Commission Cost" in captured.out


def test_report_command_writes_stage4_artifact_bundle(
    tmp_path: Path, capsys
) -> None:
    """The report command should export a Stage 4 artifact bundle with rich metadata."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        universe_overrides={
            "min_price": "96.0",
            "min_average_volume": "950.0",
            "min_average_dollar_volume": "100000.0",
            "min_listing_history_days": "2",
            "lag": "1",
            "average_volume_window": "2",
            "average_dollar_volume_window": "2",
        },
        benchmark_overrides={
            "name": '"Synthetic Benchmark"',
            "rolling_window": "3",
        },
        portfolio_overrides={
            "top_n": "2",
            "weighting": '"score"',
            "max_position_weight": "0.55",
        },
        backtest_overrides={
            "transaction_cost_bps": None,
            "rebalance_frequency": '"weekly"',
            "commission_bps": "2.0",
            "slippage_bps": "3.0",
            "max_turnover": "0.5",
        },
    )
    artifact_dir = tmp_path / "report_artifacts"

    exit_code = main(
        [
            "report",
            "--config",
            str(config_path),
            "--artifact-dir",
            str(artifact_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert (artifact_dir / "results.csv").exists()
    assert (artifact_dir / "report.txt").exists()
    assert (artifact_dir / "metadata.json").exists()
    assert (artifact_dir / "index.html").exists()
    assert (artifact_dir / "charts").exists()
    assert (artifact_dir / "charts" / "manifest.json").exists()
    assert (artifact_dir / "charts" / "nav_overview.png").exists()
    assert (artifact_dir / "charts" / "ic_cumulative.png").exists()
    assert (artifact_dir / "charts" / "coverage_timeseries.png").exists()
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    results = pd.read_csv(artifact_dir / "results.csv")
    report_text = (artifact_dir / "report.txt").read_text(encoding="utf-8")
    html_text = (artifact_dir / "index.html").read_text(encoding="utf-8")
    chart_manifest = json.loads(
        (artifact_dir / "charts" / "manifest.json").read_text(encoding="utf-8")
    )
    assert metadata["command"] == "report"
    assert "Research Workflow" in metadata["report_sections"]
    assert "Data Quality Summary" in metadata["report_sections"]
    assert "Relative Performance Summary" in metadata["report_sections"]
    assert metadata["workflow_configuration"]["benchmark"]["name"] == "Synthetic Benchmark"
    assert metadata["workflow_configuration"]["universe"]["lag"] == 1
    assert metadata["chart_bundle"]["chart_dir"] == "charts"
    assert metadata["chart_bundle"]["chart_count"] >= 8
    assert metadata["html_report_path"] == "index.html"
    assert "data_quality_summary" in metadata
    assert "diagnostics_overview" in metadata
    assert "performance_summary" in metadata
    assert metadata["relative_performance_summary"] is not None
    assert chart_manifest["command"] == "report"
    assert chart_manifest["chart_count"] >= 8
    assert any(chart["chart_id"] == "ic_cumulative" for chart in chart_manifest["charts"])
    assert any(chart["chart_id"] == "coverage_timeseries" for chart in chart_manifest["charts"])
    assert "benchmark_return" in results.columns
    assert "Data Quality Summary" in report_text
    assert "Diagnostics Overview" in report_text
    assert "AlphaForge Research Report" in html_text
    assert "charts/nav_overview.png" in html_text
    assert "Saved report artifacts" in captured.out
    assert "Saved report charts" in captured.out


def test_plot_report_command_writes_chart_bundle(tmp_path: Path, capsys) -> None:
    """The plot-report command should export a standalone PNG chart bundle."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        benchmark_overrides={
            "name": '"Synthetic Benchmark"',
            "rolling_window": "3",
        },
    )
    output_dir = tmp_path / "report_charts"

    exit_code = main(
        [
            "plot-report",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "nav_overview.png").exists()
    assert (output_dir / "drawdown.png").exists()
    assert (output_dir / "exposure_turnover.png").exists()
    assert (output_dir / "ic_series.png").exists()
    assert (output_dir / "ic_cumulative.png").exists()
    assert (output_dir / "coverage_summary.png").exists()
    assert (output_dir / "coverage_timeseries.png").exists()
    assert (output_dir / "quantile_bucket_returns.png").exists()
    assert (output_dir / "quantile_spread_timeseries.png").exists()
    assert (output_dir / "benchmark_risk.png").exists()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["command"] == "plot-report"
    assert manifest["chart_count"] >= 9
    assert any(chart["chart_id"] == "nav_overview" for chart in manifest["charts"])
    assert any(chart["chart_id"] == "benchmark_risk" for chart in manifest["charts"])
    assert any(chart["chart_id"] == "ic_cumulative" for chart in manifest["charts"])
    assert any(chart["chart_id"] == "quantile_spread_timeseries" for chart in manifest["charts"])
    assert "Saved report charts" in captured.out
    assert "Saved chart manifest" in captured.out


def test_report_command_prints_relative_performance_when_benchmark_configured(
    tmp_path: Path, capsys
) -> None:
    """The report should include relative-performance and benchmark-risk sections."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        benchmark_overrides={
            "name": '"Synthetic Benchmark"',
            "rolling_window": "3",
        },
    )

    exit_code = main(["report", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Benchmark Configuration" in captured.out
    assert "Relative Performance Summary" in captured.out
    assert "Benchmark Risk Summary" in captured.out
    assert "Tracking Error" in captured.out
    assert "Average Rolling Beta" in captured.out


def test_report_command_fails_cleanly_on_benchmark_misalignment(
    tmp_path: Path, capsys
) -> None:
    """Benchmark date misalignment should fail loudly instead of being silently coerced."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        benchmark_overrides={
            "rolling_window": "3",
        },
        benchmark_rows=[
            ("2024-01-02", 0.00),
            ("2024-01-03", 0.01),
            ("2024-01-04", 0.01),
            ("2024-01-05", 0.00),
            ("2024-01-08", 0.01),
        ],
    )

    exit_code = main(["report", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "align exactly to backtest dates" in captured.err


def test_report_command_prints_universe_summary_when_configured(
    tmp_path: Path, capsys
) -> None:
    """The report command should describe configured universe exclusions."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        universe_overrides={
            "min_price": "96.0",
            "min_average_volume": "950.0",
            "min_average_dollar_volume": "100000.0",
            "min_listing_history_days": "2",
            "lag": "1",
            "average_volume_window": "2",
            "average_dollar_volume_window": "2",
        },
    )

    exit_code = main(["report", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Universe Rules" in captured.out
    assert "Universe Summary" in captured.out
    assert "Excluded Rows" in captured.out
    assert "Exclusion Reasons" in captured.out
    assert "Eligible Symbols Per Date" in captured.out


def test_validate_data_command_prints_universe_preview_when_configured(
    tmp_path: Path, capsys
) -> None:
    """validate-data should preview configured universe rules and eligibility."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        universe_overrides={
            "min_price": "96.0",
            "min_average_volume": "950.0",
            "min_average_dollar_volume": "100000.0",
            "min_listing_history_days": "2",
            "lag": "1",
            "average_volume_window": "2",
            "average_dollar_volume_window": "2",
        },
    )

    exit_code = main(["validate-data", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Data Summary" in captured.out
    assert "Universe Rules" in captured.out
    assert "Universe Summary" in captured.out
    assert "Eligible Symbols Per Date" in captured.out


def test_sweep_signal_command_prints_summary_table(capsys) -> None:
    """The sweep-signal command should print a formatted summary table."""
    exit_code = main(
        [
            "sweep-signal",
            "--config",
            "configs/momentum_example.toml",
            "--parameter",
            "lookback",
            "--values",
            "1",
            "2",
            "3",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Signal Parameter Sweep" in captured.out
    assert "momentum_signal_1d" in captured.out
    assert "lookback" in captured.out


def test_sweep_signal_command_writes_artifact_bundle(
    tmp_path: Path, capsys
) -> None:
    """The sweep-signal command should persist a small artifact bundle."""
    artifact_dir = tmp_path / "sweep_artifacts"

    exit_code = main(
        [
            "sweep-signal",
            "--config",
            "configs/momentum_example.toml",
            "--parameter",
            "lookback",
            "--values",
            "1",
            "2",
            "3",
            "--artifact-dir",
            str(artifact_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert (artifact_dir / "results.csv").exists()
    assert (artifact_dir / "report.txt").exists()
    assert (artifact_dir / "metadata.json").exists()
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["command"] == "sweep-signal"
    assert metadata["parameter"] == "lookback"
    assert "research_context" in metadata
    assert "data_quality_summary" in metadata["research_context"]
    assert metadata["best_candidate"] is not None
    assert len(metadata["top_candidates"]) >= 1
    assert "Saved sweep artifacts" in captured.out


def test_sweep_signal_command_writes_indexed_experiment_run(
    tmp_path: Path, capsys
) -> None:
    """The sweep-signal command should create indexed experiment artifacts."""
    experiment_root = tmp_path / "sweep_runs"

    first_exit_code = main(
        [
            "sweep-signal",
            "--config",
            "configs/momentum_example.toml",
            "--parameter",
            "lookback",
            "--values",
            "1",
            "2",
            "3",
            "--experiment-root",
            str(experiment_root),
        ]
    )
    first_captured = capsys.readouterr()

    second_exit_code = main(
        [
            "sweep-signal",
            "--config",
            "configs/momentum_example.toml",
            "--parameter",
            "lookback",
            "--values",
            "1",
            "2",
            "3",
            "--experiment-root",
            str(experiment_root),
        ]
    )
    second_captured = capsys.readouterr()

    assert first_exit_code == 0
    assert second_exit_code == 0
    runs_index = experiment_root / "runs.csv"
    assert runs_index.exists()
    indexed = pd.read_csv(runs_index)
    assert len(indexed) == 2
    run_dirs = [path for path in experiment_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 2
    assert "Updated run index" in first_captured.out
    assert "Updated run index" in second_captured.out


def test_walk_forward_signal_command_prints_summary_sections(capsys) -> None:
    """The walk-forward-signal command should print summary and fold sections."""
    exit_code = main(
        [
            "walk-forward-signal",
            "--config",
            "configs/momentum_example.toml",
            "--parameter",
            "lookback",
            "--values",
            "1",
            "2",
            "3",
            "--train-periods",
            "4",
            "--test-periods",
            "2",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Walk-Forward Summary" in captured.out
    assert "Performance Summary" in captured.out
    assert "Walk-Forward Folds" in captured.out


def test_walk_forward_signal_command_supports_stage2_execution_settings(
    capsys,
) -> None:
    """Walk-forward should still run when Stage 2 execution settings are enabled."""
    exit_code = main(
        [
            "walk-forward-signal",
            "--config",
            "configs/stage2_execution_example.toml",
            "--parameter",
            "lookback",
            "--values",
            "1",
            "2",
            "3",
            "--train-periods",
            "4",
            "--test-periods",
            "2",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Walk-Forward Summary" in captured.out
    assert "Performance Summary" in captured.out
    assert "Walk-Forward Folds" in captured.out


def test_walk_forward_signal_command_writes_artifact_bundle(
    tmp_path: Path, capsys
) -> None:
    """The walk-forward-signal command should persist a small artifact bundle."""
    artifact_dir = tmp_path / "walk_forward_artifacts"

    exit_code = main(
        [
            "walk-forward-signal",
            "--config",
            "configs/momentum_example.toml",
            "--parameter",
            "lookback",
            "--values",
            "1",
            "2",
            "3",
            "--train-periods",
            "4",
            "--test-periods",
            "2",
            "--artifact-dir",
            str(artifact_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert (artifact_dir / "results.csv").exists()
    assert (artifact_dir / "report.txt").exists()
    assert (artifact_dir / "metadata.json").exists()
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["command"] == "walk-forward-signal"
    assert metadata["selection_metric"] == "cumulative_return"
    assert "overall_summary" in metadata
    assert "research_context" in metadata
    assert metadata["fold_count"] >= 1
    assert metadata["selected_parameter_values"]
    assert metadata["selection_distribution"]
    assert "Saved walk-forward artifacts" in captured.out


def test_walk_forward_signal_command_writes_indexed_experiment_run(
    tmp_path: Path, capsys
) -> None:
    """The walk-forward-signal command should create indexed experiment artifacts."""
    experiment_root = tmp_path / "walk_forward_runs"

    exit_code = main(
        [
            "walk-forward-signal",
            "--config",
            "configs/momentum_example.toml",
            "--parameter",
            "lookback",
            "--values",
            "1",
            "2",
            "3",
            "--train-periods",
            "4",
            "--test-periods",
            "2",
            "--experiment-root",
            str(experiment_root),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    runs_index = experiment_root / "runs.csv"
    assert runs_index.exists()
    indexed = pd.read_csv(runs_index)
    assert len(indexed) == 1
    assert indexed.loc[0, "command"] == "walk-forward-signal"
    assert indexed.loc[0, "selection_metric"] == "cumulative_return"
    assert pd.notna(indexed.loc[0, "overall_cumulative_return"])
    run_dirs = [path for path in experiment_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    assert "Updated run index" in captured.out


def test_indexed_run_updates_do_not_emit_future_warning(
    tmp_path: Path, capsys, recwarn
) -> None:
    """Mixed indexed run writes should not emit pandas concat FutureWarning."""
    experiment_root = tmp_path / "mixed_indexed_runs"

    sweep_exit_code = main(
        [
            "sweep-signal",
            "--config",
            "configs/momentum_example.toml",
            "--parameter",
            "lookback",
            "--values",
            "1",
            "2",
            "3",
            "--experiment-root",
            str(experiment_root),
        ]
    )
    capsys.readouterr()

    walk_forward_exit_code = main(
        [
            "walk-forward-signal",
            "--config",
            "configs/momentum_example.toml",
            "--parameter",
            "lookback",
            "--values",
            "1",
            "2",
            "3",
            "--train-periods",
            "4",
            "--test-periods",
            "2",
            "--experiment-root",
            str(experiment_root),
        ]
    )
    capsys.readouterr()

    future_warnings = [
        warning for warning in recwarn if issubclass(warning.category, FutureWarning)
    ]
    indexed = pd.read_csv(experiment_root / "runs.csv")

    assert sweep_exit_code == 0
    assert walk_forward_exit_code == 0
    assert set(indexed["command"]) == {"sweep-signal", "walk-forward-signal"}
    assert not future_warnings


def test_list_runs_command_prints_filtered_run_index(
    tmp_path: Path, capsys
) -> None:
    """The list-runs command should print filtered indexed runs."""
    experiment_root = tmp_path / "indexed_runs"

    sweep_exit_code = main(
        [
            "sweep-signal",
            "--config",
            "configs/momentum_example.toml",
            "--parameter",
            "lookback",
            "--values",
            "1",
            "2",
            "3",
            "--experiment-root",
            str(experiment_root),
        ]
    )
    capsys.readouterr()

    walk_forward_exit_code = main(
        [
            "walk-forward-signal",
            "--config",
            "configs/momentum_example.toml",
            "--parameter",
            "lookback",
            "--values",
            "1",
            "2",
            "3",
            "--train-periods",
            "4",
            "--test-periods",
            "2",
            "--experiment-root",
            str(experiment_root),
        ]
    )
    capsys.readouterr()

    exit_code = main(
        [
            "list-runs",
            "--experiment-root",
            str(experiment_root),
            "--command-name",
            "walk-forward-signal",
            "--sort-by",
            "created_at",
            "--limit",
            "1",
        ]
    )
    captured = capsys.readouterr()

    assert sweep_exit_code == 0
    assert walk_forward_exit_code == 0
    assert exit_code == 0
    assert "Indexed Runs" in captured.out
    assert "walk-forward-signal" in captured.out
    assert "sweep-signal" not in captured.out


def test_compare_runs_command_prints_selected_runs_and_fails_on_missing_run_id(
    tmp_path: Path, capsys
) -> None:
    """The compare-runs command should print enriched summaries and reject missing ids."""
    experiment_root = tmp_path / "compare_runs"

    first_exit_code = main(
        [
            "sweep-signal",
            "--config",
            "configs/momentum_example.toml",
            "--parameter",
            "lookback",
            "--values",
            "1",
            "2",
            "3",
            "--experiment-root",
            str(experiment_root),
        ]
    )
    capsys.readouterr()

    second_exit_code = main(
        [
            "walk-forward-signal",
            "--config",
            "configs/momentum_example.toml",
            "--parameter",
            "lookback",
            "--values",
            "1",
            "2",
            "3",
            "--train-periods",
            "4",
            "--test-periods",
            "2",
            "--experiment-root",
            str(experiment_root),
        ]
    )
    capsys.readouterr()

    indexed = pd.read_csv(experiment_root / "runs.csv")
    sweep_run_id = indexed.loc[indexed["command"] == "sweep-signal", "run_id"].iloc[0]
    walk_forward_run_id = indexed.loc[
        indexed["command"] == "walk-forward-signal",
        "run_id",
    ].iloc[0]

    compare_exit_code = main(
        [
            "compare-runs",
            "--experiment-root",
            str(experiment_root),
            "--run-id",
            sweep_run_id,
            walk_forward_run_id,
        ]
    )
    compare_captured = capsys.readouterr()

    missing_exit_code = main(
        [
            "compare-runs",
            "--experiment-root",
            str(experiment_root),
            "--run-id",
            "missing-run-id",
        ]
    )
    missing_captured = capsys.readouterr()

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert compare_exit_code == 0
    assert "Compared Runs" in compare_captured.out
    assert sweep_run_id in compare_captured.out
    assert walk_forward_run_id in compare_captured.out
    assert "sweep-signal" in compare_captured.out
    assert "walk-forward-signal" in compare_captured.out
    assert "summary_parameter_values" in compare_captured.out
    assert "summary_cumulative_return" in compare_captured.out
    assert "best_in_sample_candidate" in compare_captured.out
    assert "overall_out_of_sample" in compare_captured.out
    assert "Sweep Top Candidates" in compare_captured.out
    assert "Walk-Forward Folds" in compare_captured.out
    assert "Candidates Shown: 3/3" in compare_captured.out
    assert "parameter_value" in compare_captured.out
    assert "fold_index" in compare_captured.out
    assert "train_selection_score" in compare_captured.out
    assert "test_cumulative_return" in compare_captured.out
    assert missing_exit_code == 1
    assert "were not found in runs.csv" in missing_captured.err


def test_compare_runs_command_writes_artifact_bundle(
    tmp_path: Path, capsys
) -> None:
    """The compare-runs command should persist one comparison bundle."""
    experiment_root = tmp_path / "compare_runs_artifacts"

    first_exit_code = main(
        [
            "sweep-signal",
            "--config",
            "configs/momentum_example.toml",
            "--parameter",
            "lookback",
            "--values",
            "1",
            "2",
            "3",
            "--experiment-root",
            str(experiment_root),
        ]
    )
    capsys.readouterr()

    second_exit_code = main(
        [
            "walk-forward-signal",
            "--config",
            "configs/momentum_example.toml",
            "--parameter",
            "lookback",
            "--values",
            "1",
            "2",
            "3",
            "--train-periods",
            "4",
            "--test-periods",
            "2",
            "--experiment-root",
            str(experiment_root),
        ]
    )
    capsys.readouterr()

    indexed = pd.read_csv(experiment_root / "runs.csv")
    run_ids = indexed["run_id"].tolist()
    artifact_dir = tmp_path / "compare_bundle"

    compare_exit_code = main(
        [
            "compare-runs",
            "--experiment-root",
            str(experiment_root),
            "--run-id",
            *run_ids,
            "--artifact-dir",
            str(artifact_dir),
        ]
    )
    compare_captured = capsys.readouterr()

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert compare_exit_code == 0
    assert (artifact_dir / "results.csv").exists()
    assert (artifact_dir / "report.txt").exists()
    assert (artifact_dir / "metadata.json").exists()
    assert (artifact_dir / "charts").exists()
    assert (artifact_dir / "charts" / "manifest.json").exists()
    assert (artifact_dir / "charts" / "compare_summary_metrics.png").exists()
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(artifact_dir / "results.csv")
    report_text = (artifact_dir / "report.txt").read_text(encoding="utf-8")
    chart_manifest = json.loads(
        (artifact_dir / "charts" / "manifest.json").read_text(encoding="utf-8")
    )
    assert metadata["command"] == "compare-runs"
    assert metadata["run_ids"] == run_ids
    assert metadata["row_count"] == 2
    assert metadata["chart_bundle"]["chart_dir"] == "charts"
    assert metadata["chart_bundle"]["chart_count"] >= 1
    assert metadata["comparison_summary"]["commands"] == [
        "sweep-signal",
        "walk-forward-signal",
    ]
    assert metadata["comparison_summary"]["best_run_by_cumulative_return"] is not None
    assert chart_manifest["command"] == "compare-runs"
    assert chart_manifest["chart_count"] >= 1
    assert len(summary) == 2
    assert "summary_cumulative_return" in summary.columns
    assert "Sweep Top Candidates" in report_text
    assert "Walk-Forward Folds" in report_text
    assert "Saved compare artifacts" in compare_captured.out
    assert "Saved compare charts" in compare_captured.out


def test_compare_runs_command_auto_selects_filtered_sorted_runs(
    tmp_path: Path, capsys
) -> None:
    """The compare-runs command should support filtered automatic run selection."""
    experiment_root = tmp_path / "auto_compare_runs"

    first_exit_code = main(
        [
            "walk-forward-signal",
            "--config",
            "configs/momentum_example.toml",
            "--parameter",
            "lookback",
            "--values",
            "1",
            "2",
            "3",
            "--train-periods",
            "4",
            "--test-periods",
            "2",
            "--experiment-root",
            str(experiment_root),
        ]
    )
    capsys.readouterr()

    second_exit_code = main(
        [
            "walk-forward-signal",
            "--config",
            "configs/trend_example.toml",
            "--parameter",
            "short_window",
            "--values",
            "1",
            "2",
            "3",
            "--train-periods",
            "4",
            "--test-periods",
            "2",
            "--experiment-root",
            str(experiment_root),
        ]
    )
    capsys.readouterr()

    indexed = pd.read_csv(experiment_root / "runs.csv")
    walk_forward_runs = indexed.loc[
        indexed["command"] == "walk-forward-signal"
    ].copy()
    walk_forward_runs["overall_cumulative_return"] = pd.to_numeric(
        walk_forward_runs["overall_cumulative_return"],
        errors="coerce",
    )
    ordered = walk_forward_runs.sort_values(
        "overall_cumulative_return",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)
    selected_run_id = str(ordered.loc[0, "run_id"])
    omitted_run_id = str(ordered.loc[1, "run_id"])

    compare_exit_code = main(
        [
            "compare-runs",
            "--experiment-root",
            str(experiment_root),
            "--command-name",
            "walk-forward-signal",
            "--sort-by",
            "overall_cumulative_return",
            "--limit",
            "1",
        ]
    )
    compare_captured = capsys.readouterr()

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert compare_exit_code == 0
    assert "Compared Runs" in compare_captured.out
    assert selected_run_id in compare_captured.out
    assert omitted_run_id not in compare_captured.out
    assert "walk-forward-signal" in compare_captured.out
    assert "Walk-Forward Folds" in compare_captured.out
    assert "Sweep Top Candidates" not in compare_captured.out


def test_compare_runs_command_requires_limit_without_run_ids(
    tmp_path: Path, capsys
) -> None:
    """Automatic compare selection should require an explicit limit."""
    experiment_root = tmp_path / "auto_compare_requires_limit"

    setup_exit_code = main(
        [
            "sweep-signal",
            "--config",
            "configs/momentum_example.toml",
            "--parameter",
            "lookback",
            "--values",
            "1",
            "2",
            "3",
            "--experiment-root",
            str(experiment_root),
        ]
    )
    capsys.readouterr()

    compare_exit_code = main(
        [
            "compare-runs",
            "--experiment-root",
            str(experiment_root),
            "--command-name",
            "sweep-signal",
        ]
    )
    compare_captured = capsys.readouterr()

    assert setup_exit_code == 0
    assert compare_exit_code == 1
    assert "requires --limit" in compare_captured.err


def test_compare_runs_command_rank_selects_runs_by_average_rank(
    tmp_path: Path, capsys
) -> None:
    """The compare-runs command should support multi-metric average-rank selection."""
    experiment_root = tmp_path / "ranked_compare_runs"

    first_exit_code = main(
        [
            "walk-forward-signal",
            "--config",
            "configs/momentum_example.toml",
            "--parameter",
            "lookback",
            "--values",
            "1",
            "2",
            "3",
            "--train-periods",
            "4",
            "--test-periods",
            "2",
            "--experiment-root",
            str(experiment_root),
        ]
    )
    capsys.readouterr()

    second_exit_code = main(
        [
            "walk-forward-signal",
            "--config",
            "configs/trend_example.toml",
            "--parameter",
            "short_window",
            "--values",
            "1",
            "2",
            "3",
            "--train-periods",
            "4",
            "--test-periods",
            "2",
            "--experiment-root",
            str(experiment_root),
        ]
    )
    capsys.readouterr()

    indexed = pd.read_csv(experiment_root / "runs.csv")
    run_ids = indexed["run_id"].astype(str).tolist()
    compared = compare_indexed_runs(experiment_root, run_ids=run_ids)
    for column in ("summary_cumulative_return", "summary_mean_ic"):
        compared[column] = pd.to_numeric(compared[column], errors="coerce")
        compared[f"rank_{column}"] = compared[column].rank(
            method="min",
            ascending=False,
            na_option="bottom",
        )
    compared["average_rank"] = compared[
        ["rank_summary_cumulative_return", "rank_summary_mean_ic"]
    ].mean(axis=1)
    ordered = compared.sort_values(
        ["average_rank", "summary_cumulative_return", "summary_mean_ic", "created_at"],
        ascending=[True, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    selected_run_id = str(ordered.loc[0, "run_id"])
    omitted_run_id = str(ordered.loc[1, "run_id"])

    compare_exit_code = main(
        [
            "compare-runs",
            "--experiment-root",
            str(experiment_root),
            "--command-name",
            "walk-forward-signal",
            "--rank-by",
            "summary_cumulative_return",
            "summary_mean_ic",
            "--limit",
            "1",
        ]
    )
    compare_captured = capsys.readouterr()

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert compare_exit_code == 0
    assert "Ranked Compare Selection" in compare_captured.out
    assert "average_rank" in compare_captured.out
    assert selected_run_id in compare_captured.out
    assert omitted_run_id not in compare_captured.out


def test_compare_runs_command_rank_supports_metric_weights(
    tmp_path: Path, capsys
) -> None:
    """The compare-runs command should support weighted multi-metric rank selection."""
    experiment_root = tmp_path / "weighted_rank_compare_runs"

    first_exit_code = main(
        [
            "sweep-signal",
            "--config",
            "configs/momentum_example.toml",
            "--parameter",
            "lookback",
            "--values",
            "1",
            "2",
            "3",
            "--experiment-root",
            str(experiment_root),
        ]
    )
    capsys.readouterr()

    second_exit_code = main(
        [
            "sweep-signal",
            "--config",
            "configs/mean_reversion_example.toml",
            "--parameter",
            "lookback",
            "--values",
            "1",
            "2",
            "3",
            "--experiment-root",
            str(experiment_root),
        ]
    )
    capsys.readouterr()

    third_exit_code = main(
        [
            "sweep-signal",
            "--config",
            "configs/trend_example.toml",
            "--parameter",
            "short_window",
            "--values",
            "1",
            "2",
            "3",
            "--experiment-root",
            str(experiment_root),
        ]
    )
    capsys.readouterr()

    indexed = pd.read_csv(experiment_root / "runs.csv")
    run_ids = indexed["run_id"].astype(str).tolist()
    compared = compare_indexed_runs(experiment_root, run_ids=run_ids)
    for column in ("summary_cumulative_return", "summary_mean_ic"):
        compared[column] = pd.to_numeric(compared[column], errors="coerce")
        compared[f"rank_{column}"] = compared[column].rank(
            method="min",
            ascending=False,
            na_option="bottom",
        )
    top_cumulative_run_id = str(
        compared.sort_values(
            ["summary_cumulative_return", "summary_mean_ic", "created_at"],
            ascending=[False, False, False],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .loc[0, "run_id"]
    )
    top_mean_ic_run_id = str(
        compared.sort_values(
            ["summary_mean_ic", "summary_cumulative_return", "created_at"],
            ascending=[False, False, False],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .loc[0, "run_id"]
    )
    assert top_cumulative_run_id != top_mean_ic_run_id

    compared["weighted_rank_score"] = (
        compared["rank_summary_cumulative_return"] * 0.1
        + compared["rank_summary_mean_ic"] * 0.9
    )
    ordered = compared.sort_values(
        [
            "weighted_rank_score",
            "summary_cumulative_return",
            "summary_mean_ic",
            "created_at",
        ],
        ascending=[True, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    selected_run_id = str(ordered.loc[0, "run_id"])
    omitted_run_id = str(ordered.loc[1, "run_id"])

    compare_exit_code = main(
        [
            "compare-runs",
            "--experiment-root",
            str(experiment_root),
            "--command-name",
            "sweep-signal",
            "--rank-by",
            "summary_cumulative_return",
            "summary_mean_ic",
            "--rank-weight",
            "summary_cumulative_return=0.1",
            "summary_mean_ic=0.9",
            "--limit",
            "1",
        ]
    )
    compare_captured = capsys.readouterr()

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert third_exit_code == 0
    assert compare_exit_code == 0
    assert "Ranked Compare Selection" in compare_captured.out
    assert "weighted_rank_score" in compare_captured.out
    assert "weight_summary_cumulative_return" in compare_captured.out
    assert "weight_summary_mean_ic" in compare_captured.out
    assert selected_run_id in compare_captured.out
    assert omitted_run_id not in compare_captured.out


def test_compare_runs_command_requires_rank_by_when_rank_weight_is_provided(
    tmp_path: Path, capsys
) -> None:
    """Weighted compare ranking should require explicit rank metrics."""
    experiment_root = tmp_path / "compare_rank_weight_requires_rank_by"

    setup_exit_code = main(
        [
            "sweep-signal",
            "--config",
            "configs/momentum_example.toml",
            "--parameter",
            "lookback",
            "--values",
            "1",
            "2",
            "3",
            "--experiment-root",
            str(experiment_root),
        ]
    )
    capsys.readouterr()

    compare_exit_code = main(
        [
            "compare-runs",
            "--experiment-root",
            str(experiment_root),
            "--command-name",
            "sweep-signal",
            "--rank-weight",
            "summary_cumulative_return=1.0",
            "--limit",
            "1",
        ]
    )
    compare_captured = capsys.readouterr()

    assert setup_exit_code == 0
    assert compare_exit_code == 1
    assert "--rank-weight requires --rank-by" in compare_captured.err


def test_compare_runs_command_rejects_explicit_run_ids_with_rank_by(
    tmp_path: Path, capsys
) -> None:
    """Explicit compare ids should not be mixed with ranked automatic selection."""
    experiment_root = tmp_path / "compare_rank_conflict"

    setup_exit_code = main(
        [
            "sweep-signal",
            "--config",
            "configs/momentum_example.toml",
            "--parameter",
            "lookback",
            "--values",
            "1",
            "2",
            "3",
            "--experiment-root",
            str(experiment_root),
        ]
    )
    capsys.readouterr()

    indexed = pd.read_csv(experiment_root / "runs.csv")
    run_id = str(indexed.loc[0, "run_id"])

    compare_exit_code = main(
        [
            "compare-runs",
            "--experiment-root",
            str(experiment_root),
            "--run-id",
            run_id,
            "--rank-by",
            "summary_cumulative_return",
            "--limit",
            "1",
        ]
    )
    compare_captured = capsys.readouterr()

    assert setup_exit_code == 0
    assert compare_exit_code == 1
    assert "cannot combine explicit --run-id values" in compare_captured.err


def test_walk_forward_signal_command_writes_output_csv(
    tmp_path: Path, capsys
) -> None:
    """The walk-forward-signal command should persist fold-level CSV output."""
    output_path = tmp_path / "walk_forward.csv"

    exit_code = main(
        [
            "walk-forward-signal",
            "--config",
            "configs/momentum_example.toml",
            "--parameter",
            "lookback",
            "--values",
            "1",
            "2",
            "3",
            "--train-periods",
            "4",
            "--test-periods",
            "2",
            "--output",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_path.exists()
    written = pd.read_csv(output_path)
    assert "selected_parameter_value" in written.columns
    assert "Saved walk-forward results" in captured.out


def test_config_loader_rejects_invalid_cross_section_settings(tmp_path: Path) -> None:
    """Config loading should fail early on inconsistent signal or diagnostics settings."""
    data_path = tmp_path / "sample.csv"
    data_path.write_text(
        "date,symbol,open,high,low,close,volume\n"
        "2024-01-02,AAPL,100,101,99,100,1000\n",
        encoding="utf-8",
    )

    trend_config = tmp_path / "trend.toml"
    trend_config.write_text(
        """
[data]
path = "sample.csv"

[signal]
name = "trend"
short_window = 5
long_window = 5
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="signal.short_window"):
        load_pipeline_config(trend_config)

    diagnostics_config = tmp_path / "diagnostics.toml"
    diagnostics_config.write_text(
        """
[data]
path = "sample.csv"

[diagnostics]
ic_method = "kendall"
n_quantiles = 3
min_observations = 2
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="diagnostics.ic_method"):
        load_pipeline_config(diagnostics_config)


def test_config_loader_rejects_invalid_data_path_targets(tmp_path: Path) -> None:
    """Config loading should fail early on non-file or unsupported data paths."""
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()

    directory_config = tmp_path / "directory.toml"
    directory_config.write_text(
        """
[data]
path = "data_dir"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="must point to a file"):
        load_pipeline_config(directory_config)

    unsupported_path = tmp_path / "sample.json"
    unsupported_path.write_text("{}", encoding="utf-8")

    extension_config = tmp_path / "extension.toml"
    extension_config.write_text(
        """
[data]
path = "sample.json"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="supported file types"):
        load_pipeline_config(extension_config)


def test_report_command_fails_cleanly_when_diagnostics_label_is_missing(
    tmp_path: Path, capsys
) -> None:
    """The report command should return a clear error on inconsistent diagnostics config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        diagnostics_overrides={
            "forward_return_column": '"forward_return_5d"',
        },
    )

    exit_code = main(["report", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "diagnostics.forward_return_column" in captured.err


def test_build_dataset_command_creates_nested_output_directories(
    tmp_path: Path, capsys
) -> None:
    """Dataset output should create missing parent directories when needed."""
    config_path = _write_pipeline_fixture(tmp_path)
    output_path = tmp_path / "nested" / "outputs" / "dataset.csv"

    exit_code = main(
        ["build-dataset", "--config", str(config_path), "--output", str(output_path)]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_path.exists()
    assert "Saved dataset" in captured.out


def test_build_dataset_command_fails_cleanly_when_output_path_is_directory(
    tmp_path: Path, capsys
) -> None:
    """Dataset output should reject directory targets with a clear CLI error."""
    config_path = _write_pipeline_fixture(tmp_path)
    output_path = tmp_path / "existing-output-dir"
    output_path.mkdir()

    exit_code = main(
        ["build-dataset", "--config", str(config_path), "--output", str(output_path)]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Output path must be a file path" in captured.err


def _write_pipeline_fixture(
    tmp_path: Path,
    *,
    data_overrides: dict[str, str | None] | None = None,
    diagnostics_overrides: dict[str, str] | None = None,
    universe_overrides: dict[str, str] | None = None,
    portfolio_overrides: dict[str, str | None] | None = None,
    backtest_overrides: dict[str, str | None] | None = None,
    calendar_overrides: dict[str, str | None] | None = None,
    calendar_rows: list[str] | None = None,
    benchmark_overrides: dict[str, str | None] | None = None,
    benchmark_rows: list[tuple[str, float]] | None = None,
    symbol_metadata_overrides: dict[str, str | None] | None = None,
    symbol_metadata_rows: list[tuple[str, str, str]] | None = None,
    corporate_actions_overrides: dict[str, str | None] | None = None,
    corporate_actions_rows: list[tuple[str, str, str, str, str]] | None = None,
    fundamentals_overrides: dict[str, str | None] | None = None,
    fundamentals_rows: list[tuple[str, str, str, str, str]] | None = None,
) -> Path:
    """Create a small but runnable pipeline fixture for CLI workflow tests."""
    data_path = tmp_path / "sample.csv"
    data_path.write_text(
        "\n".join(
            [
                "date,symbol,open,high,low,close,volume",
                "2024-01-02,AAPL,100,101,99,100,1000",
                "2024-01-03,AAPL,110,111,109,110,1100",
                "2024-01-04,AAPL,121,122,120,121,1200",
                "2024-01-05,AAPL,133.1,134.1,132.1,133.1,1150",
                "2024-01-08,AAPL,146.41,147.41,145.41,146.41,1250",
                "2024-01-09,AAPL,161.051,162.051,160.051,161.051,1300",
                "2024-01-02,MSFT,100,101,99,100,1000",
                "2024-01-03,MSFT,95,96,94,95,900",
                "2024-01-04,MSFT,90,91,89,90,950",
                "2024-01-05,MSFT,85.5,86.5,84.5,85.5,940",
                "2024-01-08,MSFT,81.225,82.225,80.225,81.225,930",
                "2024-01-09,MSFT,77.16375,78.16375,76.16375,77.16375,920",
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "pipeline.toml"
    if calendar_rows is None:
        calendar_rows = []
    calendar_path = tmp_path / "calendar.csv"
    if calendar_rows:
        calendar_path.write_text(
            "\n".join(
                [
                    "date",
                    *calendar_rows,
                ]
            ),
            encoding="utf-8",
        )

    if benchmark_rows is None:
        benchmark_rows = [
            ("2024-01-02", 0.00),
            ("2024-01-03", 0.01),
            ("2024-01-04", 0.01),
            ("2024-01-05", 0.00),
            ("2024-01-08", 0.01),
            ("2024-01-09", 0.01),
        ]
    benchmark_path = tmp_path / "benchmark.csv"
    benchmark_path.write_text(
        "\n".join(
            [
                "date,benchmark_return",
                *[
                    f"{date},{benchmark_return}"
                    for date, benchmark_return in benchmark_rows
                ],
            ]
        ),
        encoding="utf-8",
    )
    if symbol_metadata_rows is None:
        symbol_metadata_rows = []
    symbol_metadata_path = tmp_path / "symbol_metadata.csv"
    if symbol_metadata_rows:
        symbol_metadata_path.write_text(
            "\n".join(
                [
                    "symbol,listing_date,delisting_date",
                    *[
                        f"{symbol},{listing_date},{delisting_date}"
                        for symbol, listing_date, delisting_date in symbol_metadata_rows
                    ],
                ]
            ),
            encoding="utf-8",
        )

    if corporate_actions_rows is None:
        corporate_actions_rows = []
    corporate_actions_path = tmp_path / "corporate_actions.csv"
    if corporate_actions_rows:
        corporate_actions_path.write_text(
            "\n".join(
                [
                    "symbol,ex_date,action_type,split_ratio,cash_amount",
                    *[
                        ",".join(
                            [symbol, ex_date, action_type, split_ratio, cash_amount]
                        )
                        for (
                            symbol,
                            ex_date,
                            action_type,
                            split_ratio,
                            cash_amount,
                        ) in corporate_actions_rows
                    ],
                ]
            ),
            encoding="utf-8",
        )

    if fundamentals_rows is None:
        fundamentals_rows = []
    fundamentals_path = tmp_path / "fundamentals.csv"
    if fundamentals_rows:
        fundamentals_path.write_text(
            "\n".join(
                [
                    "symbol,period_end_date,release_date,metric_name,metric_value",
                    *[
                        ",".join(
                            [
                                symbol,
                                period_end_date,
                                release_date,
                                metric_name,
                                metric_value,
                            ]
                        )
                        for (
                            symbol,
                            period_end_date,
                            release_date,
                            metric_name,
                            metric_value,
                        ) in fundamentals_rows
                    ],
                ]
            ),
            encoding="utf-8",
        )

    diagnostics_lines = [
        '[diagnostics]',
        'forward_return_column = "forward_return_1d"',
        'ic_method = "pearson"',
        'n_quantiles = 2',
        'min_observations = 2',
    ]
    if diagnostics_overrides is not None:
        diagnostics_lines = ["[diagnostics]"] + [
            f"{key} = {value}" for key, value in diagnostics_overrides.items()
        ]

    universe_lines: list[str] = []
    if universe_overrides is not None:
        universe_lines = ["[universe]"] + [
            f"{key} = {value}" for key, value in universe_overrides.items()
        ]

    calendar_lines: list[str] = []
    if calendar_rows:
        calendar_values: dict[str, str | None] = {
            "path": '"calendar.csv"',
            "name": '"Trading Calendar"',
            "date_column": '"date"',
        }
        if calendar_overrides is not None:
            calendar_values.update(calendar_overrides)
        calendar_lines = ["[calendar]"] + [
            f"{key} = {value}"
            for key, value in calendar_values.items()
            if value is not None
        ]

    benchmark_lines: list[str] = []
    if benchmark_overrides is not None:
        benchmark_values: dict[str, str | None] = {
            "path": '"benchmark.csv"',
            "name": '"Benchmark"',
            "return_column": '"benchmark_return"',
            "rolling_window": "2",
        }
        benchmark_values.update(benchmark_overrides)
        benchmark_lines = ["[benchmark]"] + [
            f"{key} = {value}"
            for key, value in benchmark_values.items()
            if value is not None
        ]

    symbol_metadata_lines: list[str] = []
    if symbol_metadata_rows:
        symbol_metadata_values: dict[str, str | None] = {
            "path": '"symbol_metadata.csv"',
            "listing_date_column": '"listing_date"',
            "delisting_date_column": '"delisting_date"',
        }
        if symbol_metadata_overrides is not None:
            symbol_metadata_values.update(symbol_metadata_overrides)
        symbol_metadata_lines = ["[symbol_metadata]"] + [
            f"{key} = {value}"
            for key, value in symbol_metadata_values.items()
            if value is not None
        ]

    corporate_actions_lines: list[str] = []
    if corporate_actions_rows:
        corporate_action_values: dict[str, str | None] = {
            "path": '"corporate_actions.csv"',
            "ex_date_column": '"ex_date"',
            "action_type_column": '"action_type"',
            "split_ratio_column": '"split_ratio"',
            "cash_amount_column": '"cash_amount"',
        }
        if corporate_actions_overrides is not None:
            corporate_action_values.update(corporate_actions_overrides)
        corporate_actions_lines = ["[corporate_actions]"] + [
            f"{key} = {value}"
            for key, value in corporate_action_values.items()
            if value is not None
        ]

    fundamentals_lines: list[str] = []
    if fundamentals_rows:
        fundamentals_values: dict[str, str | None] = {
            "path": '"fundamentals.csv"',
            "period_end_column": '"period_end_date"',
            "release_date_column": '"release_date"',
            "metric_name_column": '"metric_name"',
            "metric_value_column": '"metric_value"',
        }
        if fundamentals_overrides is not None:
            fundamentals_values.update(fundamentals_overrides)
        fundamentals_lines = ["[fundamentals]"] + [
            f"{key} = {value}"
            for key, value in fundamentals_values.items()
            if value is not None
        ]

    portfolio_values: dict[str, str | None] = {
        "construction": '"long_only"',
        "top_n": "1",
        "weighting": '"equal"',
        "exposure": "1.0",
    }
    if portfolio_overrides is not None:
        portfolio_values.update(portfolio_overrides)
    portfolio_lines = ["[portfolio]"] + [
        f"{key} = {value}"
        for key, value in portfolio_values.items()
        if value is not None
    ]

    backtest_values: dict[str, str | None] = {
        "signal_delay": "1",
        "transaction_cost_bps": "5.0",
        "initial_nav": "1.0",
    }
    if backtest_overrides is not None:
        backtest_values.update(backtest_overrides)
    backtest_lines = ["[backtest]"] + [
        f"{key} = {value}"
        for key, value in backtest_values.items()
        if value is not None
    ]

    data_values: dict[str, str | None] = {
        "path": '"sample.csv"',
        "price_adjustment": '"raw"',
    }
    if data_overrides is not None:
        data_values.update(data_overrides)
    data_lines = ["[data]"] + [
        f"{key} = {value}" for key, value in data_values.items() if value is not None
    ]

    config_path.write_text(
        "\n\n".join(
            section
            for section in [
                "\n".join(data_lines),
                """
[dataset]
forward_horizons = [1]
volatility_window = 2
average_volume_window = 2
""".strip(),
                "\n".join(calendar_lines) if calendar_lines else "",
                "\n".join(symbol_metadata_lines) if symbol_metadata_lines else "",
                "\n".join(corporate_actions_lines) if corporate_actions_lines else "",
                "\n".join(fundamentals_lines) if fundamentals_lines else "",
                "\n".join(benchmark_lines) if benchmark_lines else "",
                "\n".join(universe_lines) if universe_lines else "",
                """

[signal]
name = "momentum"
lookback = 1
""".strip(),
                "\n".join(portfolio_lines),
                "\n".join(backtest_lines),
                "\n".join(diagnostics_lines),
            ]
            if section
        ),
        encoding="utf-8",
    )
    return config_path

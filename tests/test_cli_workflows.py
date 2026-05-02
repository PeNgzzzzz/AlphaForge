"""Tests for config-driven CLI workflows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import alphaforge.cli.workflows as workflows
from alphaforge.cli import (
    artifacts,
    charts,
    comparison,
    data_loading,
    parameter_sweep,
    pipeline,
    report_context,
    report_package,
    research_metadata,
    reports,
    validation_report,
    walk_forward,
)
from alphaforge.cli.main import main
from alphaforge.cli.workflows import (
    add_signal_from_config,
    build_dataset_from_config,
    compare_indexed_runs,
    load_benchmark_returns_from_config,
    load_borrow_availability_from_config,
    load_classifications_from_config,
    load_corporate_actions_from_config,
    load_fundamentals_from_config,
    load_market_data_from_config,
    load_memberships_from_config,
    load_shares_outstanding_from_config,
    load_symbol_metadata_from_config,
    load_trading_status_from_config,
    load_trading_calendar_from_config,
)
from alphaforge.common import ConfigError, load_pipeline_config


def test_workflows_declares_public_compatibility_exports() -> None:
    """The legacy workflow module should expose an explicit public surface."""
    exported = set(workflows.__all__)
    public_names = {
        name
        for name in vars(workflows)
        if not name.startswith("_") and name != "annotations"
    }

    assert "annotations" not in exported
    assert not any(name.startswith("_") for name in exported)
    assert exported == public_names
    assert {
        "build_dataset_from_config",
        "run_backtest_from_config",
        "build_report_package",
        "run_signal_parameter_sweep",
        "run_walk_forward_parameter_selection",
        "compare_indexed_runs",
        "build_validate_data_text",
    }.issubset(exported)
    for name in workflows.__all__:
        assert hasattr(workflows, name)


def test_workflows_reexports_data_loading_helpers() -> None:
    """Legacy workflow imports should keep pointing at extracted loading helpers."""
    assert (
        workflows.load_market_data_from_config
        is data_loading.load_market_data_from_config
    )
    assert (
        workflows.load_trading_status_from_config
        is data_loading.load_trading_status_from_config
    )


def test_workflows_reexports_pipeline_helpers() -> None:
    """Legacy workflow imports should keep pointing at extracted pipeline helpers."""
    assert workflows.build_dataset_from_config is pipeline.build_dataset_from_config
    assert (
        workflows.build_dataset_from_market_data
        is pipeline.build_dataset_from_market_data
    )
    assert workflows.add_signal_from_config is pipeline.add_signal_from_config
    assert workflows.build_weights_from_config is pipeline.build_weights_from_config
    assert workflows.run_backtest_from_config is pipeline.run_backtest_from_config


def test_workflows_reexports_report_context_helpers() -> None:
    """Legacy workflow imports should keep pointing at extracted report context helpers."""
    assert workflows.build_report_context is report_context.build_report_context
    assert workflows._build_report_context is report_context.build_report_context


def test_workflows_reexports_research_metadata_helpers() -> None:
    """Legacy workflow imports should keep pointing at extracted metadata helpers."""
    assert (
        workflows.build_research_context_metadata
        is research_metadata.build_research_context_metadata
    )
    assert workflows._build_config_snapshot is research_metadata.build_config_snapshot
    assert (
        workflows._build_research_metadata_from_config
        is research_metadata.build_research_metadata_from_config
    )
    assert (
        workflows._build_dataset_feature_metadata_from_config
        is research_metadata.build_dataset_feature_metadata_from_config
    )
    assert (
        workflows._build_signal_pipeline_metadata_from_config
        is research_metadata.build_signal_pipeline_metadata_from_config
    )
    assert workflows._dataframe_records is research_metadata.dataframe_records
    assert workflows._series_to_metadata_dict is research_metadata.series_to_metadata_dict
    assert workflows._scalar_or_none is research_metadata.scalar_or_none


def test_workflows_reexports_validation_report_helpers() -> None:
    """Legacy workflow imports should keep pointing at validation report helpers."""
    assert (
        workflows.build_validate_data_text
        is validation_report.build_validate_data_text
    )


def test_workflows_reexports_parameter_sweep_helpers() -> None:
    """Legacy workflow imports should keep pointing at parameter sweep helpers."""
    assert (
        workflows.run_signal_parameter_sweep
        is parameter_sweep.run_signal_parameter_sweep
    )
    assert (
        workflows.build_sweep_artifact_metadata
        is parameter_sweep.build_sweep_artifact_metadata
    )
    assert workflows._normalize_sweep_values is parameter_sweep.normalize_sweep_values
    assert (
        workflows._replace_signal_parameter
        is parameter_sweep.replace_signal_parameter
    )


def test_workflows_reexports_artifact_helpers() -> None:
    """Legacy workflow imports should keep pointing at extracted artifact helpers."""
    assert workflows.write_dataframe is artifacts.write_dataframe
    assert workflows.write_artifact_bundle is artifacts.write_artifact_bundle


def test_workflows_reexports_chart_helpers() -> None:
    """Legacy workflow imports should keep pointing at extracted chart helpers."""
    assert workflows.write_compare_chart_bundle is charts.write_compare_chart_bundle
    assert (
        workflows.write_report_chart_bundle_from_context
        is charts.write_report_chart_bundle_from_context
    )


def test_workflows_reexports_report_helpers() -> None:
    """Legacy workflow imports should keep pointing at extracted report helpers."""
    assert workflows.build_report_metadata is reports.build_report_metadata
    assert workflows.render_report_text is reports.render_report_text
    assert workflows.describe_market_data is reports.describe_market_data
    assert workflows.describe_research_workflow is reports.describe_research_workflow
    assert workflows.write_report_html_page is reports.write_report_html_page


def test_workflows_reexports_report_package_helpers() -> None:
    """Legacy workflow imports should keep pointing at report package helpers."""
    assert workflows.build_report_text is report_package.build_report_text
    assert workflows.build_report_package is report_package.build_report_package
    assert (
        workflows.write_report_artifact_bundle
        is report_package.write_report_artifact_bundle
    )
    assert workflows.write_report_chart_bundle is report_package.write_report_chart_bundle


def test_workflows_reexports_comparison_helpers() -> None:
    """Legacy workflow imports should keep pointing at extracted comparison helpers."""
    assert workflows.compare_indexed_runs is comparison.compare_indexed_runs
    assert workflows.rank_compare_runs is comparison.rank_compare_runs
    assert workflows.build_compare_runs_report is comparison.build_compare_runs_report


def test_workflows_reexports_walk_forward_helpers() -> None:
    """Legacy workflow imports should keep pointing at extracted walk-forward helpers."""
    assert (
        workflows.run_walk_forward_parameter_selection
        is walk_forward.run_walk_forward_parameter_selection
    )
    assert (
        workflows.build_walk_forward_artifact_metadata
        is walk_forward.build_walk_forward_artifact_metadata_from_config
    )
    assert (
        workflows.normalize_walk_forward_selection_metric
        is walk_forward.normalize_walk_forward_selection_metric
    )
    assert workflows.build_walk_forward_folds is walk_forward.build_walk_forward_folds
    assert (
        workflows.evaluate_walk_forward_slice
        is walk_forward.evaluate_walk_forward_slice
    )


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


def test_load_pipeline_config_parses_universe_membership_filter(
    tmp_path: Path,
) -> None:
    """Universe filters may require lagged index membership status."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        universe_overrides={
            "required_membership_indexes": '["S&P 500"]',
            "lag": "1",
        },
        memberships_rows=[
            ("AAPL", "2024-01-02", "S&P 500", "1"),
            ("MSFT", "2024-01-02", "S&P 500", "0"),
        ],
    )

    config = load_pipeline_config(config_path)

    assert config.universe is not None
    assert config.universe.required_membership_indexes == ("S&P 500",)


def test_load_pipeline_config_parses_universe_trading_status_filter(
    tmp_path: Path,
) -> None:
    """Universe filters may require lagged trading status."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        universe_overrides={
            "require_tradable": "true",
            "lag": "1",
        },
        trading_status_rows=[
            ("AAPL", "2024-01-02", "1", ""),
            ("MSFT", "2024-01-02", "0", "halt"),
        ],
    )

    config = load_pipeline_config(config_path)

    assert config.universe is not None
    assert config.universe.require_tradable is True


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


def test_load_pipeline_config_parses_signal_cross_sectional_transform_settings(
    tmp_path: Path,
) -> None:
    """Optional signal transform settings should parse into the signal config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        signal_overrides={
            "winsorize_quantile": "0.1",
            "clip_lower_bound": "-2.0",
            "clip_upper_bound": "2.0",
            "cross_sectional_residualize_columns": '["style_beta"]',
            "cross_sectional_neutralize_group_column": '"classification_sector"',
            "cross_sectional_normalization": '"robust_zscore"',
            "cross_sectional_group_column": '"classification_sector"',
        },
    )

    config = load_pipeline_config(config_path)

    assert config.signal is not None
    assert config.signal.winsorize_quantile == pytest.approx(0.1)
    assert config.signal.clip_lower_bound == pytest.approx(-2.0)
    assert config.signal.clip_upper_bound == pytest.approx(2.0)
    assert config.signal.cross_sectional_residualize_columns == ("style_beta",)
    assert (
        config.signal.cross_sectional_neutralize_group_column
        == "classification_sector"
    )
    assert config.signal.cross_sectional_normalization == "robust_zscore"
    assert config.signal.cross_sectional_group_column == "classification_sector"


def test_load_pipeline_config_parses_diagnostics_settings(tmp_path: Path) -> None:
    """Optional diagnostics settings should parse into config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        diagnostics_overrides={
            "forward_return_column": '"forward_return_1d"',
            "ic_method": '"pearson"',
            "n_quantiles": "2",
            "min_observations": "2",
            "rolling_ic_window": "3",
            "group_columns": '["classification_sector"]',
            "exposure_columns": '["rolling_benchmark_beta_3d"]',
        },
    )

    config = load_pipeline_config(config_path)

    assert config.diagnostics.rolling_ic_window == 3
    assert config.diagnostics.group_columns == ("classification_sector",)
    assert config.diagnostics.exposure_columns == ("rolling_benchmark_beta_3d",)


def test_load_pipeline_config_parses_stage2_execution_settings(tmp_path: Path) -> None:
    """Optional Stage 2 portfolio and backtest settings should parse cleanly."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        portfolio_overrides={
            "top_n": "2",
            "max_position_weight": "0.55",
            "position_cap_column": '"rolling_average_volume_2d"',
            "group_column": '"classification_sector"',
            "max_group_weight": "0.60",
            "factor_exposure_bounds": (
                '[{ column = "rolling_average_volume_2d", '
                'min = -2000.0, max = 2000.0 }]'
            ),
        },
        backtest_overrides={
            "transaction_cost_bps": None,
            "fill_timing": '"next_close"',
            "rebalance_frequency": '"weekly"',
            "commission_bps": "2.0",
            "slippage_bps": "3.0",
            "max_trade_weight_column": '"max_trade_weight"',
            "max_participation_rate": "0.10",
            "participation_notional": "1000000.0",
            "min_trade_weight": "0.02",
            "max_turnover": "0.5",
        },
    )

    config = load_pipeline_config(config_path)

    assert config.portfolio is not None
    assert config.portfolio.max_position_weight == pytest.approx(0.55)
    assert config.portfolio.position_cap_column == "rolling_average_volume_2d"
    assert config.portfolio.group_column == "classification_sector"
    assert config.portfolio.max_group_weight == pytest.approx(0.60)
    assert len(config.portfolio.factor_exposure_bounds) == 1
    factor_bound = config.portfolio.factor_exposure_bounds[0]
    assert factor_bound.column == "rolling_average_volume_2d"
    assert factor_bound.min_exposure == pytest.approx(-2000.0)
    assert factor_bound.max_exposure == pytest.approx(2000.0)
    assert config.backtest is not None
    assert config.backtest.fill_timing == "next_close"
    assert config.backtest.rebalance_frequency == "weekly"
    assert config.backtest.transaction_cost_bps is None
    assert config.backtest.commission_bps == pytest.approx(2.0)
    assert config.backtest.slippage_bps == pytest.approx(3.0)
    assert config.backtest.max_trade_weight_column == "max_trade_weight"
    assert config.backtest.max_participation_rate == pytest.approx(0.10)
    assert config.backtest.participation_notional == pytest.approx(1000000.0)
    assert config.backtest.min_trade_weight == pytest.approx(0.02)
    assert config.backtest.max_turnover == pytest.approx(0.5)


def test_load_pipeline_config_parses_row_level_cost_columns(tmp_path: Path) -> None:
    """Optional row-level transaction cost columns should parse cleanly."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        backtest_overrides={
            "transaction_cost_bps": None,
            "commission_bps_column": '"row_commission_bps"',
            "slippage_bps_column": '"row_slippage_bps"',
        },
    )

    config = load_pipeline_config(config_path)

    assert config.backtest is not None
    assert config.backtest.transaction_cost_bps is None
    assert config.backtest.commission_bps == pytest.approx(0.0)
    assert config.backtest.slippage_bps == pytest.approx(0.0)
    assert config.backtest.commission_bps_column == "row_commission_bps"
    assert config.backtest.slippage_bps_column == "row_slippage_bps"


def test_load_pipeline_config_normalizes_scalar_choice_settings(
    tmp_path: Path,
) -> None:
    """Scalar config choices should share trimming and choice validation behavior."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        signal_overrides={
            "name": '" momentum "',
            "cross_sectional_normalization": '" rank "',
        },
        portfolio_overrides={
            "construction": '" long_short "',
            "weighting": '" score "',
            "bottom_n": "1",
        },
        backtest_overrides={
            "fill_timing": '" next_close "',
            "rebalance_frequency": '" weekly "',
        },
        diagnostics_overrides={
            "ic_method": '" spearman "',
        },
    )

    config = load_pipeline_config(config_path)

    assert config.signal is not None
    assert config.signal.name == "momentum"
    assert config.signal.cross_sectional_normalization == "rank"
    assert config.portfolio is not None
    assert config.portfolio.construction == "long_short"
    assert config.portfolio.weighting == "score"
    assert config.backtest is not None
    assert config.backtest.fill_timing == "next_close"
    assert config.backtest.rebalance_frequency == "weekly"
    assert config.diagnostics.ic_method == "spearman"


@pytest.mark.parametrize(
    ("fixture_kwargs", "match"),
    [
        (
            {"data_overrides": {"price_adjustment": '"adjusted"'}},
            "data.price_adjustment",
        ),
        ({"signal_overrides": {"name": '"carry"'}}, "signal.name"),
        (
            {"signal_overrides": {"cross_sectional_normalization": '"robust"'}},
            "signal.cross_sectional_normalization",
        ),
        (
            {"portfolio_overrides": {"construction": '"market_neutral"'}},
            "portfolio.construction",
        ),
        ({"portfolio_overrides": {"weighting": '"rank"'}}, "portfolio.weighting"),
        (
            {"backtest_overrides": {"fill_timing": '"next_open"'}},
            "backtest.fill_timing",
        ),
        (
            {"backtest_overrides": {"rebalance_frequency": '"quarterly"'}},
            "backtest.rebalance_frequency",
        ),
        (
            {"diagnostics_overrides": {"ic_method": '"kendall"'}},
            "diagnostics.ic_method",
        ),
    ],
)
def test_load_pipeline_config_rejects_invalid_scalar_choice_settings(
    tmp_path: Path,
    fixture_kwargs: dict[str, dict[str, str]],
    match: str,
) -> None:
    """Invalid scalar config choices should fail through shared choice validation."""
    config_path = _write_pipeline_fixture(tmp_path, **fixture_kwargs)

    with pytest.raises(ConfigError, match=match):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_nan_optional_float_settings(
    tmp_path: Path,
) -> None:
    """Optional float config fields should fail fast on NaN inputs."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        backtest_overrides={
            "max_turnover": "nan",
        },
    )

    with pytest.raises(
        ConfigError,
        match="backtest.max_turnover must be a non-negative float",
    ):
        load_pipeline_config(config_path)


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


def test_load_pipeline_config_parses_shares_outstanding_section(
    tmp_path: Path,
) -> None:
    """Optional shares-outstanding settings should parse into the top-level config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        shares_outstanding_overrides={
            "effective_date_column": '"effective_on"',
            "shares_outstanding_column": '"shares"',
        },
        shares_outstanding_rows=[
            ("AAPL", "2024-01-02", "15500000000"),
            ("MSFT", "2024-01-03", "7430000000"),
        ],
    )

    config = load_pipeline_config(config_path)

    assert config.shares_outstanding is not None
    assert config.shares_outstanding.effective_date_column == "effective_on"
    assert config.shares_outstanding.shares_outstanding_column == "shares"


def test_load_pipeline_config_parses_market_cap_dataset_setting(
    tmp_path: Path,
) -> None:
    """Optional market-cap dataset feature should parse as a boolean flag."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "include_market_cap": "true",
        },
        shares_outstanding_rows=[
            ("AAPL", "2024-01-02", "1000000000"),
            ("MSFT", "2024-01-02", "2000000000"),
        ],
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.include_market_cap


def test_load_pipeline_config_parses_market_cap_bucket_count(
    tmp_path: Path,
) -> None:
    """Optional market-cap bucket count should parse into dataset settings."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "include_market_cap": "true",
            "market_cap_bucket_count": "3",
        },
        shares_outstanding_rows=[
            ("AAPL", "2024-01-02", "1000000000"),
            ("MSFT", "2024-01-02", "2000000000"),
        ],
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.market_cap_bucket_count == 3


def test_load_pipeline_config_parses_classifications_section(tmp_path: Path) -> None:
    """Optional classifications settings should parse into the top-level config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        classifications_overrides={
            "effective_date_column": '"effective_on"',
            "sector_column": '"sector_name"',
            "industry_column": '"industry_name"',
        },
        classifications_rows=[
            ("AAPL", "2024-01-02", "Technology", "Hardware"),
            ("MSFT", "2024-01-03", "Technology", "Software"),
        ],
    )

    config = load_pipeline_config(config_path)

    assert config.classifications is not None
    assert config.classifications.effective_date_column == "effective_on"
    assert config.classifications.sector_column == "sector_name"
    assert config.classifications.industry_column == "industry_name"


def test_load_pipeline_config_parses_memberships_section(tmp_path: Path) -> None:
    """Optional memberships settings should parse into the top-level config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        memberships_overrides={
            "effective_date_column": '"effective_on"',
            "index_column": '"index_id"',
            "is_member_column": '"member_flag"',
        },
        memberships_rows=[
            ("AAPL", "2024-01-02", "S&P 500", "1"),
            ("MSFT", "2024-01-03", "NASDAQ 100", "0"),
        ],
    )

    config = load_pipeline_config(config_path)

    assert config.memberships is not None
    assert config.memberships.effective_date_column == "effective_on"
    assert config.memberships.index_column == "index_id"
    assert config.memberships.is_member_column == "member_flag"


def test_load_pipeline_config_parses_borrow_availability_section(
    tmp_path: Path,
) -> None:
    """Optional borrow availability settings should parse into the top-level config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        borrow_availability_overrides={
            "effective_date_column": '"effective_on"',
            "is_borrowable_column": '"borrowable"',
            "borrow_fee_bps_column": '"fee_bps"',
        },
        borrow_availability_rows=[
            ("AAPL", "2024-01-02", "1", "12.5"),
            ("MSFT", "2024-01-03", "0", ""),
        ],
    )

    config = load_pipeline_config(config_path)

    assert config.borrow_availability is not None
    assert config.borrow_availability.effective_date_column == "effective_on"
    assert config.borrow_availability.is_borrowable_column == "borrowable"
    assert config.borrow_availability.borrow_fee_bps_column == "fee_bps"


def test_load_pipeline_config_parses_trading_status_section(
    tmp_path: Path,
) -> None:
    """Optional trading status settings should parse into the top-level config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        trading_status_overrides={
            "effective_date_column": '"effective_on"',
            "is_tradable_column": '"tradable"',
            "status_reason_column": '"reason"',
        },
        trading_status_rows=[
            ("AAPL", "2024-01-02", "1", ""),
            ("MSFT", "2024-01-03", "0", "halt"),
        ],
    )

    config = load_pipeline_config(config_path)

    assert config.trading_status is not None
    assert config.trading_status.effective_date_column == "effective_on"
    assert config.trading_status.is_tradable_column == "tradable"
    assert config.trading_status.status_reason_column == "reason"


def test_load_pipeline_config_parses_dataset_fundamental_metrics(
    tmp_path: Path,
) -> None:
    """Dataset fundamentals selection should parse into the dataset config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "fundamental_metrics": '["revenue", "book_value"]',
        },
        fundamentals_rows=[
            ("AAPL", "2023-12-31", "2024-01-30", "revenue", "119000"),
        ],
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.fundamental_metrics == ("revenue", "book_value")


def test_load_pipeline_config_parses_dataset_valuation_metrics(
    tmp_path: Path,
) -> None:
    """Dataset valuation metric selection should parse into the dataset config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "valuation_metrics": '["eps", "book_value_per_share"]',
        },
        fundamentals_rows=[
            ("AAPL", "2023-12-31", "2024-01-30", "eps", "5.5"),
        ],
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.valuation_metrics == ("eps", "book_value_per_share")


def test_load_pipeline_config_parses_dataset_quality_ratio_metrics(
    tmp_path: Path,
) -> None:
    """Dataset quality ratio selections should parse into numerator/denominator pairs."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "quality_ratio_metrics": (
                '[["net_income", "total_assets"], ["gross_profit", "revenue"]]'
            ),
        },
        fundamentals_rows=[
            ("AAPL", "2023-12-31", "2024-01-30", "net_income", "100.0"),
        ],
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.quality_ratio_metrics == (
        ("net_income", "total_assets"),
        ("gross_profit", "revenue"),
    )


def test_load_pipeline_config_parses_dataset_growth_metrics(
    tmp_path: Path,
) -> None:
    """Dataset growth metric selections should parse into the dataset config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "growth_metrics": '["revenue", "gross_profit"]',
        },
        fundamentals_rows=[
            ("AAPL", "2023-12-31", "2024-01-30", "revenue", "100.0"),
        ],
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.growth_metrics == ("revenue", "gross_profit")


def test_load_pipeline_config_parses_dataset_stability_ratio_metrics(
    tmp_path: Path,
) -> None:
    """Dataset stability ratio selections should parse into metric pairs."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "stability_ratio_metrics": '[["total_debt", "total_assets"]]',
        },
        fundamentals_rows=[
            ("AAPL", "2023-12-31", "2024-01-30", "total_debt", "100.0"),
        ],
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.stability_ratio_metrics == (
        ("total_debt", "total_assets"),
    )


def test_load_pipeline_config_parses_dataset_classification_fields(
    tmp_path: Path,
) -> None:
    """Dataset classification selection should parse into the dataset config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "classification_fields": '["sector"]',
        },
        classifications_rows=[
            ("AAPL", "2024-01-02", "Technology", "Hardware"),
        ],
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.classification_fields == ("sector",)


def test_load_pipeline_config_parses_dataset_membership_indexes(
    tmp_path: Path,
) -> None:
    """Dataset membership selection should parse into the dataset config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "membership_indexes": '["S&P 500"]',
        },
        memberships_rows=[
            ("AAPL", "2024-01-02", "S&P 500", "1"),
        ],
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.membership_indexes == ("S&P 500",)


def test_load_pipeline_config_parses_dataset_borrow_fields(
    tmp_path: Path,
) -> None:
    """Dataset borrow field selection should parse into the dataset config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "borrow_fields": '["is_borrowable", "borrow_fee_bps"]',
        },
        borrow_availability_rows=[
            ("AAPL", "2024-01-02", "1", "12.5"),
        ],
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.borrow_fields == ("is_borrowable", "borrow_fee_bps")


def test_load_pipeline_config_normalizes_config_selection_lists(
    tmp_path: Path,
) -> None:
    """Config selection lists should trim values through shared validation."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "classification_fields": '[" sector "]',
            "membership_indexes": '[" S&P 500 "]',
            "borrow_fields": '[" borrow_fee_bps "]',
        },
        universe_overrides={
            "required_membership_indexes": '[" S&P 500 "]',
        },
        classifications_rows=[
            ("AAPL", "2024-01-02", "Technology", "Hardware"),
        ],
        memberships_rows=[
            ("AAPL", "2024-01-02", "S&P 500", "1"),
        ],
        borrow_availability_rows=[
            ("AAPL", "2024-01-02", "1", "12.5"),
        ],
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.classification_fields == ("sector",)
    assert config.dataset.membership_indexes == ("S&P 500",)
    assert config.dataset.borrow_fields == ("borrow_fee_bps",)
    assert config.universe is not None
    assert config.universe.required_membership_indexes == ("S&P 500",)


@pytest.mark.parametrize(
    ("dataset_overrides", "match"),
    [
        (
            {"classification_fields": '["sector", " sector "]'},
            "dataset.classification_fields must not contain duplicates",
        ),
        (
            {"borrow_fields": '["is_borrowable", " is_borrowable "]'},
            "dataset.borrow_fields must not contain duplicates",
        ),
        (
            {"membership_indexes": '["S&P 500", " S&P 500 "]'},
            "dataset.membership_indexes must not contain duplicates",
        ),
    ],
)
def test_load_pipeline_config_rejects_duplicate_config_selection_lists(
    tmp_path: Path,
    dataset_overrides: dict[str, str],
    match: str,
) -> None:
    """Config selection lists should reject duplicate values after trimming."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides=dataset_overrides,
    )

    with pytest.raises(ConfigError, match=match):
        load_pipeline_config(config_path)


@pytest.mark.parametrize(
    ("dataset_overrides", "match"),
    [
        (
            {"classification_fields": '["country"]'},
            "dataset.classification_fields must be one of",
        ),
        (
            {"borrow_fields": '["locate_quantity"]'},
            "dataset.borrow_fields must be one of",
        ),
        (
            {"classification_fields": '"sector"'},
            "dataset.classification_fields must be a list of strings",
        ),
    ],
)
def test_load_pipeline_config_rejects_invalid_config_selection_lists(
    tmp_path: Path,
    dataset_overrides: dict[str, str],
    match: str,
) -> None:
    """Config selection list helpers should keep fail-fast config errors."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides=dataset_overrides,
    )

    with pytest.raises(ConfigError, match=match):
        load_pipeline_config(config_path)


def test_load_pipeline_config_parses_dataset_benchmark_rolling_window(
    tmp_path: Path,
) -> None:
    """Optional rolling benchmark feature settings should parse into the dataset config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "benchmark_rolling_window": "3",
        },
        benchmark_overrides={},
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.benchmark_rolling_window == 3


def test_load_pipeline_config_parses_dataset_benchmark_residual_return_window(
    tmp_path: Path,
) -> None:
    """Optional benchmark residual-return settings should parse into the dataset config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "benchmark_residual_return_window": "3",
        },
        benchmark_overrides={},
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.benchmark_residual_return_window == 3


def test_load_pipeline_config_parses_dataset_higher_moments_window(
    tmp_path: Path,
) -> None:
    """Optional rolling higher-moments settings should parse into the dataset config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "higher_moments_window": "4",
        },
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.higher_moments_window == 4


def test_load_pipeline_config_parses_dataset_parkinson_volatility_window(
    tmp_path: Path,
) -> None:
    """Optional Parkinson volatility settings should parse into the dataset config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "parkinson_volatility_window": "5",
        },
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.parkinson_volatility_window == 5


def test_load_pipeline_config_parses_dataset_average_true_range_window(
    tmp_path: Path,
) -> None:
    """Optional ATR settings should parse into the dataset config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "average_true_range_window": "5",
        },
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.average_true_range_window == 5


def test_load_pipeline_config_parses_dataset_normalized_average_true_range_window(
    tmp_path: Path,
) -> None:
    """Optional normalized ATR settings should parse into the dataset config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "normalized_average_true_range_window": "5",
        },
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.normalized_average_true_range_window == 5


def test_load_pipeline_config_parses_dataset_amihud_illiquidity_window(
    tmp_path: Path,
) -> None:
    """Optional Amihud illiquidity settings should parse into the dataset config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "amihud_illiquidity_window": "5",
        },
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.amihud_illiquidity_window == 5


def test_load_pipeline_config_parses_dataset_dollar_volume_shock_window(
    tmp_path: Path,
) -> None:
    """Optional dollar-volume shock settings should parse into the dataset config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "dollar_volume_shock_window": "5",
        },
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.dollar_volume_shock_window == 5


def test_load_pipeline_config_parses_dataset_dollar_volume_zscore_window(
    tmp_path: Path,
) -> None:
    """Optional dollar-volume z-score settings should parse into the dataset config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "dollar_volume_zscore_window": "5",
        },
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.dollar_volume_zscore_window == 5


def test_load_pipeline_config_parses_dataset_volume_shock_window(
    tmp_path: Path,
) -> None:
    """Optional volume-shock settings should parse into the dataset config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "volume_shock_window": "5",
        },
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.volume_shock_window == 5


def test_load_pipeline_config_parses_dataset_relative_volume_window(
    tmp_path: Path,
) -> None:
    """Optional relative volume settings should parse into the dataset config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "relative_volume_window": "5",
        },
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.relative_volume_window == 5


def test_load_pipeline_config_parses_dataset_relative_dollar_volume_window(
    tmp_path: Path,
) -> None:
    """Optional relative dollar volume settings should parse into the dataset config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "relative_dollar_volume_window": "5",
        },
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.relative_dollar_volume_window == 5


def test_load_pipeline_config_parses_dataset_rogers_satchell_volatility_window(
    tmp_path: Path,
) -> None:
    """Optional Rogers-Satchell settings should parse into the dataset config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "rogers_satchell_volatility_window": "5",
        },
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.rogers_satchell_volatility_window == 5


def test_load_pipeline_config_parses_dataset_yang_zhang_volatility_window(
    tmp_path: Path,
) -> None:
    """Optional Yang-Zhang settings should parse into the dataset config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "yang_zhang_volatility_window": "5",
        },
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.yang_zhang_volatility_window == 5


def test_load_pipeline_config_parses_dataset_garman_klass_volatility_window(
    tmp_path: Path,
) -> None:
    """Optional Garman-Klass settings should parse into the dataset config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "garman_klass_volatility_window": "5",
        },
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.garman_klass_volatility_window == 5


def test_load_pipeline_config_parses_dataset_realized_volatility_window(
    tmp_path: Path,
) -> None:
    """Optional realized-volatility settings should parse into the dataset config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "realized_volatility_window": "5",
        },
    )

    config = load_pipeline_config(config_path)

    assert config.dataset.realized_volatility_window == 5


def test_load_pipeline_config_requires_fundamentals_for_dataset_metrics(
    tmp_path: Path,
) -> None:
    """Dataset fundamentals selection should require a fundamentals section."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "fundamental_metrics": '["revenue"]',
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.fundamental_metrics requires a \\[fundamentals\\] section",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_requires_fundamentals_for_dataset_valuation_metrics(
    tmp_path: Path,
) -> None:
    """Dataset valuation metric selection should require a fundamentals section."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "valuation_metrics": '["eps"]',
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.valuation_metrics requires a \\[fundamentals\\] section",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_requires_fundamentals_for_dataset_quality_ratio_metrics(
    tmp_path: Path,
) -> None:
    """Dataset quality ratio selection should require a fundamentals section."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "quality_ratio_metrics": '[["net_income", "total_assets"]]',
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.quality_ratio_metrics requires a \\[fundamentals\\] section",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_invalid_dataset_quality_ratio_metrics(
    tmp_path: Path,
) -> None:
    """Dataset quality ratios must be explicit numerator/denominator pairs."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "quality_ratio_metrics": '["net_income"]',
        },
        fundamentals_rows=[
            ("AAPL", "2023-12-31", "2024-01-30", "net_income", "100.0"),
        ],
    )

    with pytest.raises(
        ConfigError,
        match=(
            "dataset.quality_ratio_metrics must be a list of "
            "\\[numerator, denominator\\] string pairs"
        ),
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_duplicate_dataset_quality_ratio_metrics(
    tmp_path: Path,
) -> None:
    """Dataset quality ratio metric pairs should be unique after normalization."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "quality_ratio_metrics": (
                '[["net_income", "total_assets"], '
                '[" net_income ", " total_assets "]]'
            ),
        },
        fundamentals_rows=[
            ("AAPL", "2023-12-31", "2024-01-30", "net_income", "100.0"),
            ("AAPL", "2023-12-31", "2024-01-30", "total_assets", "200.0"),
        ],
    )

    with pytest.raises(
        ConfigError,
        match="dataset.quality_ratio_metrics must not contain duplicate metric pairs",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_same_dataset_quality_ratio_metric_pair(
    tmp_path: Path,
) -> None:
    """Dataset quality ratio pairs should require distinct numerator/denominator."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "quality_ratio_metrics": '[["total_assets", " total_assets "]]',
        },
        fundamentals_rows=[
            ("AAPL", "2023-12-31", "2024-01-30", "total_assets", "200.0"),
        ],
    )

    with pytest.raises(
        ConfigError,
        match="dataset.quality_ratio_metrics numerator and denominator must be different",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_requires_fundamentals_for_dataset_growth_metrics(
    tmp_path: Path,
) -> None:
    """Dataset growth metric selection should require a fundamentals section."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "growth_metrics": '["revenue"]',
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.growth_metrics requires a \\[fundamentals\\] section",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_requires_fundamentals_for_dataset_stability_ratios(
    tmp_path: Path,
) -> None:
    """Dataset stability ratio selection should require a fundamentals section."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "stability_ratio_metrics": '[["total_debt", "total_assets"]]',
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.stability_ratio_metrics requires a \\[fundamentals\\] section",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_requires_classifications_for_dataset_fields(
    tmp_path: Path,
) -> None:
    """Dataset classification selection should require a classifications section."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "classification_fields": '["sector"]',
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.classification_fields requires a \\[classifications\\] section",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_requires_portfolio_group_cap_pair(
    tmp_path: Path,
) -> None:
    """Portfolio group caps should require both the group column and cap."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        portfolio_overrides={
            "group_column": '"classification_sector"',
        },
    )

    with pytest.raises(
        ConfigError,
        match="portfolio.group_column and portfolio.max_group_weight",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_group_cap_above_side_exposure(
    tmp_path: Path,
) -> None:
    """Portfolio group caps should stay within the configured side exposure."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        portfolio_overrides={
            "group_column": '"classification_sector"',
            "max_group_weight": "1.2",
        },
    )

    with pytest.raises(
        ConfigError,
        match="portfolio.max_group_weight cannot exceed portfolio.exposure",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_empty_position_cap_column(
    tmp_path: Path,
) -> None:
    """Portfolio row-level cap columns should be explicit non-empty strings."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        portfolio_overrides={
            "position_cap_column": '""',
        },
    )

    with pytest.raises(ConfigError, match="portfolio.position_cap_column"):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_invalid_factor_exposure_bounds(
    tmp_path: Path,
) -> None:
    """Portfolio factor exposure bounds should be shrink-only compatible."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        portfolio_overrides={
            "factor_exposure_bounds": (
                '[{ column = "style_beta", min = 0.1, max = 0.5 }]'
            ),
        },
    )

    with pytest.raises(
        ConfigError,
        match="portfolio.factor_exposure_bounds min must be <= 0",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_requires_memberships_for_dataset_indexes(
    tmp_path: Path,
) -> None:
    """Dataset membership selection should require a memberships section."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "membership_indexes": '["S&P 500"]',
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.membership_indexes requires a \\[memberships\\] section",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_requires_borrow_availability_for_dataset_fields(
    tmp_path: Path,
) -> None:
    """Dataset borrow field selection should require a borrow section."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "borrow_fields": '["is_borrowable"]',
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.borrow_fields requires a \\[borrow_availability\\] section",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_requires_trading_status_for_tradable_filter(
    tmp_path: Path,
) -> None:
    """Trading-status universe filters should require trading status input config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        universe_overrides={
            "require_tradable": "true",
        },
    )

    with pytest.raises(ConfigError, match="requires a \\[trading_status\\] section"):
        load_pipeline_config(config_path)


def test_load_pipeline_config_requires_shares_outstanding_for_market_cap(
    tmp_path: Path,
) -> None:
    """Market-cap dataset feature should require a shares-outstanding section."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "include_market_cap": "true",
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.include_market_cap requires a \\[shares_outstanding\\] section",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_requires_market_cap_for_buckets(
    tmp_path: Path,
) -> None:
    """Market-cap buckets should require explicit market-cap construction."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "market_cap_bucket_count": "3",
        },
    )

    with pytest.raises(
        ConfigError,
        match=(
            "dataset.market_cap_bucket_count requires "
            "dataset.include_market_cap=true"
        ),
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_too_few_market_cap_buckets(
    tmp_path: Path,
) -> None:
    """Market-cap bucket count should define at least two buckets."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "include_market_cap": "true",
            "market_cap_bucket_count": "1",
        },
        shares_outstanding_rows=[
            ("AAPL", "2024-01-02", "1000000000"),
        ],
    )

    with pytest.raises(
        ConfigError,
        match="dataset.market_cap_bucket_count must be at least 2",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_requires_benchmark_for_dataset_rolling_window(
    tmp_path: Path,
) -> None:
    """Rolling benchmark features should require an explicit benchmark section."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "benchmark_rolling_window": "3",
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.benchmark_rolling_window requires a \\[benchmark\\] section",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_requires_benchmark_for_dataset_residual_returns(
    tmp_path: Path,
) -> None:
    """Benchmark residual-return features should require an explicit benchmark section."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "benchmark_residual_return_window": "3",
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.benchmark_residual_return_window requires a \\[benchmark\\] section",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_small_dataset_higher_moments_window(
    tmp_path: Path,
) -> None:
    """Dataset higher-moment windows smaller than 4 should fail during config load."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "higher_moments_window": "3",
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.higher_moments_window must be at least 4",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_small_dataset_benchmark_residual_window(
    tmp_path: Path,
) -> None:
    """Benchmark residual-return windows smaller than 2 should fail."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "benchmark_residual_return_window": "1",
        },
        benchmark_overrides={},
    )

    with pytest.raises(
        ConfigError,
        match="dataset.benchmark_residual_return_window must be at least 2",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_nonpositive_dataset_realized_volatility_window(
    tmp_path: Path,
) -> None:
    """Dataset realized-volatility windows should be positive integers."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "realized_volatility_window": "0",
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.realized_volatility_window must be a positive integer",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_nonpositive_dataset_parkinson_volatility_window(
    tmp_path: Path,
) -> None:
    """Dataset Parkinson volatility windows should be positive integers."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "parkinson_volatility_window": "0",
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.parkinson_volatility_window must be a positive integer",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_nonpositive_dataset_average_true_range_window(
    tmp_path: Path,
) -> None:
    """Dataset ATR windows should be positive integers."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "average_true_range_window": "0",
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.average_true_range_window must be a positive integer",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_nonpositive_dataset_normalized_average_true_range_window(
    tmp_path: Path,
) -> None:
    """Dataset normalized ATR windows should be positive integers."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "normalized_average_true_range_window": "0",
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.normalized_average_true_range_window must be a positive integer",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_nonpositive_dataset_amihud_illiquidity_window(
    tmp_path: Path,
) -> None:
    """Dataset Amihud illiquidity windows should be positive integers."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "amihud_illiquidity_window": "0",
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.amihud_illiquidity_window must be a positive integer",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_nonpositive_dataset_dollar_volume_shock_window(
    tmp_path: Path,
) -> None:
    """Dataset dollar-volume shock windows should be positive integers."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "dollar_volume_shock_window": "0",
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.dollar_volume_shock_window must be a positive integer",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_small_dataset_dollar_volume_zscore_window(
    tmp_path: Path,
) -> None:
    """Dataset dollar-volume z-score windows should support sample dispersion."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "dollar_volume_zscore_window": "1",
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.dollar_volume_zscore_window must be at least 2",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_nonpositive_dataset_volume_shock_window(
    tmp_path: Path,
) -> None:
    """Dataset volume-shock windows should be positive integers."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "volume_shock_window": "0",
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.volume_shock_window must be a positive integer",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_nonpositive_dataset_relative_volume_window(
    tmp_path: Path,
) -> None:
    """Dataset relative volume windows should be positive integers."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "relative_volume_window": "0",
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.relative_volume_window must be a positive integer",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_nonpositive_dataset_relative_dollar_volume_window(
    tmp_path: Path,
) -> None:
    """Dataset relative dollar volume windows should be positive integers."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "relative_dollar_volume_window": "0",
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.relative_dollar_volume_window must be a positive integer",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_nonpositive_dataset_rogers_satchell_volatility_window(
    tmp_path: Path,
) -> None:
    """Dataset Rogers-Satchell windows should be positive integers."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "rogers_satchell_volatility_window": "0",
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.rogers_satchell_volatility_window must be a positive integer",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_small_dataset_yang_zhang_volatility_window(
    tmp_path: Path,
) -> None:
    """Dataset Yang-Zhang windows should reject values smaller than 2."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "yang_zhang_volatility_window": "1",
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.yang_zhang_volatility_window must be at least 2",
    ):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_nonpositive_dataset_garman_klass_volatility_window(
    tmp_path: Path,
) -> None:
    """Dataset Garman-Klass windows should be positive integers."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "garman_klass_volatility_window": "0",
        },
    )

    with pytest.raises(
        ConfigError,
        match="dataset.garman_klass_volatility_window must be a positive integer",
    ):
        load_pipeline_config(config_path)


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


def test_load_shares_outstanding_from_config_normalizes_date_dtype(
    tmp_path: Path,
) -> None:
    """Loaded shares-outstanding dates should use stable nanosecond dtypes."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        shares_outstanding_rows=[
            ("AAPL", "2024-01-02", "15500000000"),
        ],
    )

    config = load_pipeline_config(config_path)
    shares_outstanding = load_shares_outstanding_from_config(config)

    assert pd.api.types.is_datetime64_ns_dtype(
        shares_outstanding["effective_date"]
    )


def test_load_classifications_from_config_normalizes_date_dtype(
    tmp_path: Path,
) -> None:
    """Loaded classifications dates should use stable nanosecond dtypes."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        classifications_rows=[
            ("AAPL", "2024-01-02", "Technology", "Hardware"),
            ("MSFT", "2024-01-03", "Technology", "Software"),
        ],
    )

    config = load_pipeline_config(config_path)
    classifications = load_classifications_from_config(config)

    assert pd.api.types.is_datetime64_ns_dtype(classifications["effective_date"])


def test_load_memberships_from_config_normalizes_date_dtype(tmp_path: Path) -> None:
    """Loaded memberships dates should use stable nanosecond dtypes."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        memberships_rows=[
            ("AAPL", "2024-01-03", "S&P 500", "1"),
        ],
    )

    config = load_pipeline_config(config_path)
    memberships = load_memberships_from_config(config)

    assert pd.api.types.is_datetime64_ns_dtype(memberships["effective_date"])


def test_load_borrow_availability_from_config_normalizes_date_dtype(
    tmp_path: Path,
) -> None:
    """Loaded borrow availability dates should use stable nanosecond dtypes."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        borrow_availability_rows=[
            ("AAPL", "2024-01-03", "1", "12.5"),
        ],
    )

    config = load_pipeline_config(config_path)
    borrow_availability = load_borrow_availability_from_config(config)

    assert pd.api.types.is_datetime64_ns_dtype(
        borrow_availability["effective_date"]
    )


def test_load_trading_status_from_config_normalizes_date_dtype(
    tmp_path: Path,
) -> None:
    """Loaded trading status dates should use stable nanosecond dtypes."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        trading_status_rows=[
            ("AAPL", "2024-01-03", "1", ""),
        ],
    )

    config = load_pipeline_config(config_path)
    trading_status = load_trading_status_from_config(config)

    assert pd.api.types.is_datetime64_ns_dtype(trading_status["effective_date"])


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


def test_load_shares_outstanding_from_config_canonicalizes_custom_columns(
    tmp_path: Path,
) -> None:
    """Configured shares-outstanding columns should map into the canonical schema."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        shares_outstanding_overrides={
            "effective_date_column": '"effective_on"',
            "shares_outstanding_column": '"shares"',
        },
        shares_outstanding_rows=[
            ("AAPL", "2024-01-02", "15500000000"),
            ("MSFT", "2024-01-03", "7430000000"),
        ],
    )
    shares_outstanding_path = tmp_path / "shares_outstanding.csv"
    shares_outstanding_path.write_text(
        "\n".join(
            [
                "symbol,effective_on,shares,source_name",
                "MSFT,2024-01-03,7430000000,vendor",
                "AAPL,2024-01-02,15500000000,vendor",
            ]
        ),
        encoding="utf-8",
    )

    config = load_pipeline_config(config_path)
    shares_outstanding = load_shares_outstanding_from_config(config)

    assert list(shares_outstanding.columns) == [
        "symbol",
        "effective_date",
        "shares_outstanding",
        "source_name",
    ]
    assert shares_outstanding["symbol"].tolist() == ["AAPL", "MSFT"]
    assert shares_outstanding["shares_outstanding"].tolist() == pytest.approx(
        [15_500_000_000.0, 7_430_000_000.0]
    )


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


def test_validate_data_command_prints_shares_outstanding_summary(
    tmp_path: Path, capsys
) -> None:
    """validate-data should summarize configured shares-outstanding coverage."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        shares_outstanding_rows=[
            ("AAPL", "2024-01-02", "15500000000"),
            ("MSFT", "2024-01-03", "7430000000"),
            ("AAPL", "2024-01-05", "15400000000"),
        ],
    )

    exit_code = main(["validate-data", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Shares Outstanding Configuration" in captured.out
    assert "Shares Outstanding Summary" in captured.out
    assert "Latest Shares Total: 22830000000" in captured.out


def test_validate_data_command_prints_classifications_summary(
    tmp_path: Path, capsys
) -> None:
    """validate-data should summarize configured classifications coverage."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        classifications_rows=[
            ("AAPL", "2024-01-02", "Technology", "Hardware"),
            ("MSFT", "2024-01-03", "Technology", "Software"),
            ("AAPL", "2024-01-05", "Consumer", "Retail"),
        ],
    )

    exit_code = main(["validate-data", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Classifications Configuration" in captured.out
    assert "Classifications Summary" in captured.out
    assert "Sectors: 2" in captured.out


def test_validate_data_command_prints_memberships_summary(
    tmp_path: Path, capsys
) -> None:
    """validate-data should summarize configured memberships coverage."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        memberships_rows=[
            ("AAPL", "2024-01-03", "S&P 500", "1"),
            ("MSFT", "2024-01-04", "S&P 500", "0"),
            ("AAPL", "2024-01-05", "NASDAQ 100", "1"),
        ],
    )

    exit_code = main(["validate-data", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Memberships Configuration" in captured.out
    assert "Memberships Summary" in captured.out
    assert "Indexes: 2" in captured.out


def test_validate_data_command_prints_borrow_availability_summary(
    tmp_path: Path, capsys
) -> None:
    """validate-data should summarize configured borrow coverage."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        borrow_availability_rows=[
            ("AAPL", "2024-01-03", "1", "12.5"),
            ("MSFT", "2024-01-04", "0", ""),
            ("AAPL", "2024-01-05", "1", "15.0"),
        ],
    )

    exit_code = main(["validate-data", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Borrow Availability Configuration" in captured.out
    assert "Borrow Availability Summary" in captured.out
    assert "Fee Observations: 2" in captured.out


def test_validate_data_command_prints_trading_status_summary(
    tmp_path: Path, capsys
) -> None:
    """validate-data should summarize configured trading status coverage."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        trading_status_rows=[
            ("AAPL", "2024-01-03", "1", ""),
            ("MSFT", "2024-01-04", "0", "halt"),
            ("AAPL", "2024-01-05", "1", "resume"),
        ],
    )

    exit_code = main(["validate-data", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Trading Status Configuration" in captured.out
    assert "Trading Status Summary" in captured.out
    assert "Not Tradable Rows: 1" in captured.out
    assert "Reason Observations: 2" in captured.out


def test_build_dataset_from_config_attaches_selected_fundamentals(
    tmp_path: Path,
) -> None:
    """Configured dataset fundamentals should join only the selected metrics."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "fundamental_metrics": '["revenue"]',
        },
        fundamentals_rows=[
            ("AAPL", "2023-09-30", "2024-01-03", "revenue", "100.0"),
            ("AAPL", "2023-09-30", "2024-01-03", "book_value", "50.0"),
            ("AAPL", "2023-12-31", "2024-01-05", "revenue", "120.0"),
        ],
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "fundamental_revenue" in dataset.columns
    assert "fundamental_book_value" not in dataset.columns
    revenue_by_date = dataset.loc[
        dataset["symbol"] == "AAPL",
        ["date", "fundamental_revenue"],
    ]
    assert pd.isna(
        revenue_by_date.loc[
            revenue_by_date["date"] == pd.Timestamp("2024-01-03"),
            "fundamental_revenue",
        ]
    ).all()
    assert revenue_by_date.loc[
        revenue_by_date["date"] == pd.Timestamp("2024-01-04"),
        "fundamental_revenue",
    ].iloc[0] == pytest.approx(100.0)
    assert revenue_by_date.loc[
        revenue_by_date["date"] == pd.Timestamp("2024-01-08"),
        "fundamental_revenue",
    ].iloc[0] == pytest.approx(120.0)


def test_build_dataset_from_config_attaches_valuation_metrics(
    tmp_path: Path,
) -> None:
    """Configured valuation metrics should attach PIT fundamentals and price ratios."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "valuation_metrics": '["eps"]',
        },
        fundamentals_rows=[
            ("AAPL", "2023-12-31", "2024-01-02", "eps", "5.5"),
        ],
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "fundamental_eps" in dataset.columns
    assert "valuation_eps_to_price" in dataset.columns
    aapl_by_date = dataset.loc[
        dataset["symbol"] == "AAPL",
        ["date", "valuation_eps_to_price"],
    ]
    assert pd.isna(
        aapl_by_date.loc[
            aapl_by_date["date"] == pd.Timestamp("2024-01-02"),
            "valuation_eps_to_price",
        ]
    ).all()
    assert aapl_by_date.loc[
        aapl_by_date["date"] == pd.Timestamp("2024-01-03"),
        "valuation_eps_to_price",
    ].iloc[0] == pytest.approx(5.5 / 110.0)


def test_build_dataset_from_config_attaches_quality_ratio_metrics(
    tmp_path: Path,
) -> None:
    """Configured quality ratios should attach PIT fundamentals and ratios."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "quality_ratio_metrics": '[["net_income", "total_assets"]]',
        },
        fundamentals_rows=[
            ("AAPL", "2023-12-31", "2024-01-02", "net_income", "11.0"),
            ("AAPL", "2023-12-31", "2024-01-02", "total_assets", "110.0"),
        ],
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "fundamental_net_income" in dataset.columns
    assert "fundamental_total_assets" in dataset.columns
    assert "quality_net_income_to_total_assets" in dataset.columns
    aapl_by_date = dataset.loc[
        dataset["symbol"] == "AAPL",
        ["date", "quality_net_income_to_total_assets"],
    ]
    assert pd.isna(
        aapl_by_date.loc[
            aapl_by_date["date"] == pd.Timestamp("2024-01-02"),
            "quality_net_income_to_total_assets",
        ]
    ).all()
    assert aapl_by_date.loc[
        aapl_by_date["date"] == pd.Timestamp("2024-01-03"),
        "quality_net_income_to_total_assets",
    ].iloc[0] == pytest.approx(0.1)


def test_build_dataset_from_config_attaches_growth_metrics(
    tmp_path: Path,
) -> None:
    """Configured growth metrics should attach PIT fundamental growth rates."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "growth_metrics": '["revenue"]',
        },
        fundamentals_rows=[
            ("AAPL", "2023-09-30", "2024-01-02", "revenue", "100.0"),
            ("AAPL", "2023-12-31", "2024-01-04", "revenue", "125.0"),
        ],
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "fundamental_revenue" in dataset.columns
    assert "growth_revenue" in dataset.columns
    aapl_by_date = dataset.loc[
        dataset["symbol"] == "AAPL",
        ["date", "growth_revenue"],
    ]
    assert pd.isna(
        aapl_by_date.loc[
            aapl_by_date["date"] == pd.Timestamp("2024-01-04"),
            "growth_revenue",
        ]
    ).all()
    assert aapl_by_date.loc[
        aapl_by_date["date"] == pd.Timestamp("2024-01-05"),
        "growth_revenue",
    ].iloc[0] == pytest.approx(0.25)


def test_build_dataset_from_config_attaches_stability_ratio_metrics(
    tmp_path: Path,
) -> None:
    """Configured stability ratios should attach PIT balance-sheet ratios."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "stability_ratio_metrics": '[["total_debt", "total_assets"]]',
        },
        fundamentals_rows=[
            ("AAPL", "2023-12-31", "2024-01-02", "total_debt", "44.0"),
            ("AAPL", "2023-12-31", "2024-01-02", "total_assets", "110.0"),
        ],
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "fundamental_total_debt" in dataset.columns
    assert "fundamental_total_assets" in dataset.columns
    assert "stability_total_debt_to_total_assets" in dataset.columns
    aapl_by_date = dataset.loc[
        dataset["symbol"] == "AAPL",
        ["date", "stability_total_debt_to_total_assets"],
    ]
    assert pd.isna(
        aapl_by_date.loc[
            aapl_by_date["date"] == pd.Timestamp("2024-01-02"),
            "stability_total_debt_to_total_assets",
        ]
    ).all()
    assert aapl_by_date.loc[
        aapl_by_date["date"] == pd.Timestamp("2024-01-03"),
        "stability_total_debt_to_total_assets",
    ].iloc[0] == pytest.approx(0.4)


def test_build_dataset_from_config_attaches_selected_classifications(
    tmp_path: Path,
) -> None:
    """Configured dataset classifications should join only the selected fields."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "classification_fields": '["sector"]',
        },
        classifications_rows=[
            ("AAPL", "2024-01-03", "Technology", "Hardware"),
            ("AAPL", "2024-01-06", "Consumer", "Retail"),
        ],
        calendar_rows=[
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ],
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "classification_sector" in dataset.columns
    assert "classification_industry" not in dataset.columns
    sector_by_date = dataset.loc[
        dataset["symbol"] == "AAPL",
        ["date", "classification_sector"],
    ]
    assert pd.isna(
        sector_by_date.loc[
            sector_by_date["date"] == pd.Timestamp("2024-01-02"),
            "classification_sector",
        ]
    ).all()
    assert sector_by_date.loc[
        sector_by_date["date"] == pd.Timestamp("2024-01-03"),
        "classification_sector",
    ].iloc[0] == "Technology"
    assert sector_by_date.loc[
        sector_by_date["date"] == pd.Timestamp("2024-01-08"),
        "classification_sector",
    ].iloc[0] == "Consumer"


def test_build_dataset_from_config_attaches_selected_memberships(
    tmp_path: Path,
) -> None:
    """Configured dataset memberships should join only the selected indexes."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "membership_indexes": '["S&P 500"]',
        },
        memberships_rows=[
            ("AAPL", "2024-01-03", "S&P 500", "1"),
            ("AAPL", "2024-01-06", "S&P 500", "0"),
            ("AAPL", "2024-01-04", "NASDAQ 100", "1"),
        ],
        calendar_rows=[
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ],
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "membership_s_p_500" in dataset.columns
    assert "membership_nasdaq_100" not in dataset.columns
    membership_by_date = dataset.loc[
        dataset["symbol"] == "AAPL",
        ["date", "membership_s_p_500"],
    ]
    assert pd.isna(
        membership_by_date.loc[
            membership_by_date["date"] == pd.Timestamp("2024-01-02"),
            "membership_s_p_500",
        ]
    ).all()
    assert membership_by_date.loc[
        membership_by_date["date"] == pd.Timestamp("2024-01-03"),
        "membership_s_p_500",
    ].iloc[0]
    assert not membership_by_date.loc[
        membership_by_date["date"] == pd.Timestamp("2024-01-08"),
        "membership_s_p_500",
    ].iloc[0]


def test_build_dataset_from_config_filters_required_universe_memberships(
    tmp_path: Path,
) -> None:
    """Configured universe membership requirements should drive eligibility."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        universe_overrides={
            "required_membership_indexes": '["S&P 500"]',
            "lag": "1",
        },
        memberships_rows=[
            ("AAPL", "2024-01-02", "S&P 500", "1"),
            ("MSFT", "2024-01-02", "S&P 500", "0"),
            ("MSFT", "2024-01-04", "S&P 500", "1"),
        ],
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))
    by_symbol_date = dataset.set_index(["symbol", "date"])

    assert "membership_s_p_500" in dataset.columns
    assert "universe_lagged_membership_s_p_500" in dataset.columns
    assert "passes_universe_membership_s_p_500" in dataset.columns
    assert not bool(
        by_symbol_date.loc[("MSFT", pd.Timestamp("2024-01-04")), "is_universe_eligible"]
    )
    assert (
        by_symbol_date.loc[
            ("MSFT", pd.Timestamp("2024-01-04")),
            "universe_exclusion_reason",
        ]
        == "not_member_s_p_500"
    )
    assert bool(
        by_symbol_date.loc[("MSFT", pd.Timestamp("2024-01-05")), "is_universe_eligible"]
    )


def test_build_dataset_from_config_filters_required_trading_status(
    tmp_path: Path,
) -> None:
    """Configured trading-status requirements should drive universe eligibility."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        universe_overrides={
            "require_tradable": "true",
            "lag": "1",
        },
        trading_status_rows=[
            ("AAPL", "2024-01-02", "1", ""),
            ("MSFT", "2024-01-02", "0", "halt"),
            ("MSFT", "2024-01-04", "1", "resume"),
        ],
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))
    by_symbol_date = dataset.set_index(["symbol", "date"])

    assert "trading_is_tradable" in dataset.columns
    assert "universe_lagged_trading_is_tradable" in dataset.columns
    assert "passes_universe_trading_status" in dataset.columns
    assert not bool(
        by_symbol_date.loc[("MSFT", pd.Timestamp("2024-01-04")), "is_universe_eligible"]
    )
    assert (
        by_symbol_date.loc[
            ("MSFT", pd.Timestamp("2024-01-04")),
            "universe_exclusion_reason",
        ]
        == "not_tradable"
    )
    assert bool(
        by_symbol_date.loc[("MSFT", pd.Timestamp("2024-01-05")), "is_universe_eligible"]
    )


def test_build_dataset_from_config_attaches_selected_borrow_fields(
    tmp_path: Path,
) -> None:
    """Configured dataset borrow fields should join only the selected fields."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "borrow_fields": '["is_borrowable"]',
        },
        borrow_availability_rows=[
            ("AAPL", "2024-01-03", "1", "12.5"),
            ("AAPL", "2024-01-06", "0", "25.0"),
        ],
        calendar_rows=[
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ],
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "borrow_is_borrowable" in dataset.columns
    assert "borrow_fee_bps" not in dataset.columns
    borrow_by_date = dataset.loc[
        dataset["symbol"] == "AAPL",
        ["date", "borrow_is_borrowable"],
    ]
    assert pd.isna(
        borrow_by_date.loc[
            borrow_by_date["date"] == pd.Timestamp("2024-01-02"),
            "borrow_is_borrowable",
        ]
    ).all()
    assert borrow_by_date.loc[
        borrow_by_date["date"] == pd.Timestamp("2024-01-03"),
        "borrow_is_borrowable",
    ].iloc[0]
    assert not borrow_by_date.loc[
        borrow_by_date["date"] == pd.Timestamp("2024-01-08"),
        "borrow_is_borrowable",
    ].iloc[0]


def test_build_dataset_from_config_attaches_market_cap(
    tmp_path: Path,
) -> None:
    """Configured market-cap feature should attach effective-date shares."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "include_market_cap": "true",
        },
        shares_outstanding_rows=[
            ("AAPL", "2024-01-03", "1000000000"),
            ("AAPL", "2024-01-06", "2000000000"),
        ],
        calendar_rows=[
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ],
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "shares_outstanding" in dataset.columns
    assert "market_cap" in dataset.columns
    aapl_by_date = dataset.loc[
        dataset["symbol"] == "AAPL",
        ["date", "shares_outstanding", "market_cap"],
    ]
    assert pd.isna(
        aapl_by_date.loc[
            aapl_by_date["date"] == pd.Timestamp("2024-01-02"),
            "shares_outstanding",
        ]
    ).all()
    assert aapl_by_date.loc[
        aapl_by_date["date"] == pd.Timestamp("2024-01-03"),
        "shares_outstanding",
    ].iloc[0] == pytest.approx(1_000_000_000.0)
    assert aapl_by_date.loc[
        aapl_by_date["date"] == pd.Timestamp("2024-01-08"),
        "market_cap",
    ].iloc[0] == pytest.approx(146.41 * 2_000_000_000.0)


def test_build_dataset_from_config_attaches_market_cap_buckets(
    tmp_path: Path,
) -> None:
    """Configured market-cap buckets should be available as group columns."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "include_market_cap": "true",
            "market_cap_bucket_count": "2",
        },
        shares_outstanding_rows=[
            ("AAPL", "2024-01-02", "1000000000"),
            ("MSFT", "2024-01-02", "2000000000"),
        ],
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    first_date = dataset.loc[
        dataset["date"] == pd.Timestamp("2024-01-02"),
        ["symbol", "market_cap_bucket"],
    ].set_index("symbol")
    assert first_date.loc["AAPL", "market_cap_bucket"] == 1
    assert first_date.loc["MSFT", "market_cap_bucket"] == 2
    assert str(dataset["market_cap_bucket"].dtype) == "Int64"


def test_build_dataset_from_config_attaches_rolling_benchmark_statistics(
    tmp_path: Path,
) -> None:
    """Dataset builds should attach rolling benchmark beta/correlation when configured."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "benchmark_rolling_window": "2",
        },
        benchmark_overrides={},
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "rolling_benchmark_beta_2d" in dataset.columns
    assert "rolling_benchmark_correlation_2d" in dataset.columns
    assert dataset["rolling_benchmark_beta_2d"].notna().any()


def test_build_dataset_from_config_attaches_benchmark_residual_returns(
    tmp_path: Path,
) -> None:
    """Dataset builds should attach benchmark residual returns when configured."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "benchmark_residual_return_window": "2",
        },
        benchmark_overrides={},
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "benchmark_residual_return_2d" in dataset.columns
    assert "rolling_benchmark_beta_20d" not in dataset.columns
    assert dataset["benchmark_residual_return_2d"].notna().any()


def test_build_dataset_from_config_attaches_rolling_higher_moments(
    tmp_path: Path,
) -> None:
    """Dataset builds should attach rolling skew/kurtosis when configured."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "higher_moments_window": "4",
        },
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "rolling_skew_4d" in dataset.columns
    assert "rolling_kurtosis_4d" in dataset.columns
    assert dataset["rolling_kurtosis_4d"].notna().any()


def test_build_dataset_from_config_attaches_parkinson_volatility(
    tmp_path: Path,
) -> None:
    """Dataset builds should attach Parkinson volatility when configured."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "parkinson_volatility_window": "4",
        },
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "parkinson_volatility_4d" in dataset.columns
    assert dataset["parkinson_volatility_4d"].notna().any()


def test_build_dataset_from_config_attaches_average_true_range(
    tmp_path: Path,
) -> None:
    """Dataset builds should attach ATR when configured."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "average_true_range_window": "4",
        },
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "average_true_range_4d" in dataset.columns
    assert dataset["average_true_range_4d"].notna().any()


def test_build_dataset_from_config_attaches_normalized_average_true_range(
    tmp_path: Path,
) -> None:
    """Dataset builds should attach normalized ATR when configured."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "normalized_average_true_range_window": "4",
        },
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "normalized_average_true_range_4d" in dataset.columns
    assert dataset["normalized_average_true_range_4d"].notna().any()


def test_build_dataset_from_config_attaches_amihud_illiquidity(
    tmp_path: Path,
) -> None:
    """Dataset builds should attach Amihud illiquidity when configured."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "amihud_illiquidity_window": "4",
        },
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "amihud_illiquidity_4d" in dataset.columns
    assert dataset["amihud_illiquidity_4d"].notna().any()


def test_build_dataset_from_config_attaches_dollar_volume_shock(
    tmp_path: Path,
) -> None:
    """Dataset builds should attach dollar-volume shock when configured."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "dollar_volume_shock_window": "4",
        },
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "dollar_volume_shock_4d" in dataset.columns
    assert dataset["dollar_volume_shock_4d"].notna().any()


def test_build_dataset_from_config_attaches_dollar_volume_zscore(
    tmp_path: Path,
) -> None:
    """Dataset builds should attach dollar-volume z-scores when configured."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "dollar_volume_zscore_window": "4",
        },
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "dollar_volume_zscore_4d" in dataset.columns
    assert dataset["dollar_volume_zscore_4d"].notna().any()


def test_build_dataset_from_config_attaches_volume_shock(
    tmp_path: Path,
) -> None:
    """Dataset builds should attach volume shock when configured."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "volume_shock_window": "4",
        },
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "volume_shock_4d" in dataset.columns
    assert dataset["volume_shock_4d"].notna().any()


def test_build_dataset_from_config_attaches_relative_volume(
    tmp_path: Path,
) -> None:
    """Dataset builds should attach relative volume when configured."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "relative_volume_window": "4",
        },
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "relative_volume_4d" in dataset.columns
    assert dataset["relative_volume_4d"].notna().any()


def test_build_dataset_from_config_attaches_relative_dollar_volume(
    tmp_path: Path,
) -> None:
    """Dataset builds should attach relative dollar volume when configured."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "relative_dollar_volume_window": "4",
        },
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "relative_dollar_volume_4d" in dataset.columns
    assert dataset["relative_dollar_volume_4d"].notna().any()


def test_build_dataset_from_config_attaches_rogers_satchell_volatility(
    tmp_path: Path,
) -> None:
    """Dataset builds should attach Rogers-Satchell volatility when configured."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "rogers_satchell_volatility_window": "4",
        },
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "rogers_satchell_volatility_4d" in dataset.columns
    assert dataset["rogers_satchell_volatility_4d"].notna().any()


def test_build_dataset_from_config_attaches_yang_zhang_volatility(
    tmp_path: Path,
) -> None:
    """Dataset builds should attach Yang-Zhang volatility when configured."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "yang_zhang_volatility_window": "4",
        },
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "yang_zhang_volatility_4d" in dataset.columns
    assert dataset["yang_zhang_volatility_4d"].notna().any()


def test_build_dataset_from_config_attaches_garman_klass_volatility(
    tmp_path: Path,
) -> None:
    """Dataset builds should attach Garman-Klass volatility when configured."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "garman_klass_volatility_window": "4",
        },
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "garman_klass_volatility_4d" in dataset.columns
    assert dataset["garman_klass_volatility_4d"].notna().any()


def test_build_dataset_from_config_attaches_realized_volatility_family(
    tmp_path: Path,
) -> None:
    """Dataset builds should attach realized-volatility family features when configured."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "realized_volatility_window": "4",
        },
    )

    dataset = build_dataset_from_config(load_pipeline_config(config_path))

    assert "realized_volatility_4d" in dataset.columns
    assert "downside_realized_volatility_4d" in dataset.columns
    assert "upside_realized_volatility_4d" in dataset.columns
    assert dataset["realized_volatility_4d"].notna().any()


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
    """A universe section must define at least one filtering rule."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        universe_overrides={
            "lag": "1",
        },
    )

    with pytest.raises(ConfigError, match="at least one filtering rule"):
        load_pipeline_config(config_path)


def test_load_pipeline_config_requires_memberships_for_universe_membership_filter(
    tmp_path: Path,
) -> None:
    """Membership-based universe filters should require membership input config."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        universe_overrides={
            "required_membership_indexes": '["S&P 500"]',
        },
    )

    with pytest.raises(ConfigError, match="requires a \\[memberships\\] section"):
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


def test_load_pipeline_config_rejects_mixed_legacy_and_row_costs(
    tmp_path: Path,
) -> None:
    """Legacy transaction cost fields should not mix with row-level cost columns."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        backtest_overrides={
            "commission_bps_column": '"row_commission_bps"',
        },
    )

    with pytest.raises(ConfigError, match="cannot be combined"):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_mixed_fixed_and_row_component_costs(
    tmp_path: Path,
) -> None:
    """A fixed bps component should not mix with the same row-level component."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        backtest_overrides={
            "transaction_cost_bps": None,
            "commission_bps": "1.0",
            "commission_bps_column": '"row_commission_bps"',
        },
    )

    with pytest.raises(ConfigError, match="cannot be combined"):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_unpaired_participation_cap(
    tmp_path: Path,
) -> None:
    """Participation cap settings should require both rate and notional."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        backtest_overrides={
            "max_participation_rate": "0.10",
        },
    )

    with pytest.raises(ConfigError, match="configured together"):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_invalid_participation_rate(
    tmp_path: Path,
) -> None:
    """Participation rates should stay within the supported probability range."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        backtest_overrides={
            "max_participation_rate": "1.10",
            "participation_notional": "1000000.0",
        },
    )

    with pytest.raises(ConfigError, match="max_participation_rate"):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_negative_min_trade_weight(
    tmp_path: Path,
) -> None:
    """Minimum trade-weight clipping thresholds should be non-negative."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        backtest_overrides={
            "min_trade_weight": "-0.01",
        },
    )

    with pytest.raises(ConfigError, match="min_trade_weight"):
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
            "position_cap_column": '"rolling_average_volume_2d"',
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
            "position_cap_column": '"rolling_average_volume_2d"',
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
    assert "Position Cap Column: rolling_average_volume_2d" in captured.out
    assert "Fill Timing: close" in captured.out
    assert "Rebalance Frequency: weekly" in captured.out
    assert "Max Turnover Per Rebalance: 0.5" in captured.out
    assert "Turnover Limit Applied Dates" in captured.out
    assert "Total Commission Cost" in captured.out


def test_report_command_records_portfolio_group_constraint_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report artifacts should preserve explicit portfolio group constraints."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "classification_fields": '["sector"]',
        },
        classifications_rows=[
            ("AAPL", "2024-01-02", "Technology", "Hardware"),
            ("MSFT", "2024-01-02", "Technology", "Software"),
        ],
        portfolio_overrides={
            "top_n": "2",
            "position_cap_column": '"rolling_average_volume_2d"',
            "group_column": '"classification_sector"',
            "max_group_weight": "0.40",
            "factor_exposure_bounds": (
                '[{ column = "rolling_average_volume_2d", '
                'min = -2000.0, max = 2000.0 }]'
            ),
        },
    )
    artifact_dir = tmp_path / "portfolio_group_constraint_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    report_text = (artifact_dir / "report.txt").read_text(encoding="utf-8")

    assert exit_code == 0
    assert metadata["workflow_configuration"]["portfolio"]["group_column"] == (
        "classification_sector"
    )
    assert metadata["workflow_configuration"]["portfolio"]["max_group_weight"] == (
        pytest.approx(0.40)
    )
    assert metadata["workflow_configuration"]["portfolio"]["position_cap_column"] == (
        "rolling_average_volume_2d"
    )
    assert metadata["workflow_configuration"]["portfolio"]["factor_exposure_bounds"] == [
        {
            "column": "rolling_average_volume_2d",
            "min": -2000.0,
            "max": 2000.0,
        }
    ]
    diversification_summary = metadata["portfolio_diversification_summary"]
    assert diversification_summary["periods"] > 0.0
    assert diversification_summary["average_holdings_count"] > 0.0
    assert (
        diversification_summary["average_effective_number_of_positions"]
        > 0.0
    )
    assert "Portfolio Diversification Summary" in metadata["report_sections"]
    exposure_rows = metadata["portfolio_group_exposure_summary"]["rows"]
    technology_rows = [
        row for row in exposure_rows if row["group_value"] == "Technology"
    ]
    assert technology_rows
    assert all(
        row["group_column"] == "classification_sector"
        for row in technology_rows
    )
    assert max(row["max_gross_exposure"] for row in technology_rows) <= 0.40 + 1e-12
    assert "Portfolio Group Exposure Summary" in metadata["report_sections"]
    assert "Group Column: classification_sector" in report_text
    assert "Max Group Weight: 0.4" in report_text
    assert "Position Cap Column: rolling_average_volume_2d" in report_text
    assert "Factor Exposure Bound: rolling_average_volume_2d" in report_text
    assert "Portfolio Diversification Summary" in report_text
    assert "Portfolio Group Exposure Summary" in report_text
    assert "Saved report artifacts" in captured.out


def test_report_command_records_numeric_exposure_summary_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report artifacts should preserve configured numeric exposure summaries."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        diagnostics_overrides={
            "forward_return_column": '"forward_return_1d"',
            "ic_method": '"pearson"',
            "n_quantiles": "2",
            "min_observations": "2",
            "rolling_ic_window": "3",
            "exposure_columns": '["rolling_average_volume_2d"]',
        },
        portfolio_overrides={
            "top_n": "2",
            "weighting": '"equal"',
        },
    )
    artifact_dir = tmp_path / "numeric_exposure_summary_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    report_text = (artifact_dir / "report.txt").read_text(encoding="utf-8")

    assert exit_code == 0
    assert metadata["workflow_configuration"]["diagnostics"]["exposure_columns"] == [
        "rolling_average_volume_2d"
    ]
    assert "Portfolio Numeric Exposure Summary" in metadata["report_sections"]
    exposure_rows = metadata["portfolio_numeric_exposure_summary"]["rows"]
    assert len(exposure_rows) == 1
    assert exposure_rows[0]["exposure_column"] == "rolling_average_volume_2d"
    assert exposure_rows[0]["periods"] > 0.0
    assert exposure_rows[0]["average_gross_weight_with_exposure"] > 0.0
    assert "Portfolio Numeric Exposure Summary" in report_text
    assert "rolling_average_volume_2d" in report_text
    assert "Saved report artifacts" in captured.out


def test_report_command_writes_stage4_artifact_bundle(
    tmp_path: Path, capsys
) -> None:
    """The report command should export a Stage 4 artifact bundle with rich metadata."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "forward_horizons": "[1, 2]",
            "volatility_window": "2",
            "average_volume_window": "2",
        },
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
        diagnostics_overrides={
            "forward_return_column": '"forward_return_1d"',
            "ic_method": '"pearson"',
            "n_quantiles": "2",
            "min_observations": "2",
            "rolling_ic_window": "2",
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
    assert (artifact_dir / "charts" / "ic_decay_series.png").exists()
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
    assert metadata["workflow_configuration"]["diagnostics"]["rolling_ic_window"] == 2
    assert metadata["chart_bundle"]["chart_dir"] == "charts"
    assert metadata["chart_bundle"]["chart_count"] >= 8
    assert metadata["html_report_path"] == "index.html"
    assert "data_quality_summary" in metadata
    assert "diagnostics_overview" in metadata
    assert metadata["rolling_ic_summary"]["window"] == pytest.approx(2.0)
    assert "quantile_cumulative_returns" in metadata
    assert "quantile_spread_stability" in metadata
    assert "positive_spread_ratio" in metadata["quantile_spread_stability"]
    assert [
        row["forward_return_column"]
        for row in metadata["ic_decay_summary"]["rows"]
    ] == ["forward_return_1d", "forward_return_2d"]
    assert {
        row["forward_return_column"]
        for row in metadata["ic_decay_series"]["rows"]
    } == {"forward_return_1d", "forward_return_2d"}
    assert "latest_rolling_mean_ic" in metadata["diagnostics_overview"]
    assert "performance_summary" in metadata
    assert metadata["relative_performance_summary"] is not None
    assert chart_manifest["command"] == "report"
    assert chart_manifest["chart_count"] >= 8
    assert any(chart["chart_id"] == "ic_cumulative" for chart in chart_manifest["charts"])
    assert any(chart["chart_id"] == "ic_decay_series" for chart in chart_manifest["charts"])
    assert any(chart["chart_id"] == "coverage_timeseries" for chart in chart_manifest["charts"])
    assert "benchmark_return" in results.columns
    assert "Data Quality Summary" in report_text
    assert "Diagnostics Overview" in report_text
    assert "Rolling IC Summary" in report_text
    assert "IC Decay Summary" in report_text
    assert "AlphaForge Research Report" in html_text
    assert "charts/nav_overview.png" in html_text
    assert "Saved report artifacts" in captured.out
    assert "Saved report charts" in captured.out


def test_report_command_records_fundamental_metric_selection_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record the configured fundamentals selection."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "fundamental_metrics": '["revenue"]',
        },
        fundamentals_rows=[
            ("AAPL", "2023-12-31", "2024-01-03", "revenue", "100.0"),
        ],
    )
    artifact_dir = tmp_path / "report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"]["fundamental_metrics"] == [
        "revenue"
    ]
    assert "Saved report artifacts" in captured.out


def test_report_command_records_market_cap_selection_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record configured market-cap feature provenance."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "include_market_cap": "true",
            "market_cap_bucket_count": "2",
        },
        shares_outstanding_rows=[
            ("AAPL", "2024-01-02", "1000000000"),
            ("MSFT", "2024-01-02", "2000000000"),
        ],
    )
    artifact_dir = tmp_path / "market_cap_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    feature_columns = {
        entry["column"] for entry in metadata["dataset_feature_metadata"]
    }

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"]["include_market_cap"]
    assert (
        metadata["workflow_configuration"]["dataset"]["market_cap_bucket_count"]
        == 2
    )
    assert "market_cap" in feature_columns
    assert "market_cap_bucket" in feature_columns
    assert "Saved report artifacts" in captured.out


def test_report_command_records_valuation_metric_selection_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record the configured valuation metrics."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "valuation_metrics": '["eps"]',
        },
        fundamentals_rows=[
            ("AAPL", "2023-12-31", "2024-01-02", "eps", "5.5"),
        ],
    )
    artifact_dir = tmp_path / "valuation_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"]["valuation_metrics"] == [
        "eps"
    ]
    assert "Saved report artifacts" in captured.out


def test_report_command_records_quality_ratio_selection_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record the configured quality ratios."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "quality_ratio_metrics": '[["net_income", "total_assets"]]',
        },
        fundamentals_rows=[
            ("AAPL", "2023-12-31", "2024-01-02", "net_income", "11.0"),
            ("AAPL", "2023-12-31", "2024-01-02", "total_assets", "110.0"),
        ],
    )
    artifact_dir = tmp_path / "quality_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"]["quality_ratio_metrics"] == [
        ["net_income", "total_assets"]
    ]
    assert "Saved report artifacts" in captured.out


def test_report_command_records_growth_metric_selection_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record the configured growth metrics."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "growth_metrics": '["revenue"]',
        },
        fundamentals_rows=[
            ("AAPL", "2023-09-30", "2024-01-02", "revenue", "100.0"),
            ("AAPL", "2023-12-31", "2024-01-04", "revenue", "125.0"),
        ],
    )
    artifact_dir = tmp_path / "growth_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"]["growth_metrics"] == [
        "revenue"
    ]
    assert "Saved report artifacts" in captured.out


def test_report_command_records_stability_ratio_selection_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record the configured stability ratios."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "stability_ratio_metrics": '[["total_debt", "total_assets"]]',
        },
        fundamentals_rows=[
            ("AAPL", "2023-12-31", "2024-01-02", "total_debt", "44.0"),
            ("AAPL", "2023-12-31", "2024-01-02", "total_assets", "110.0"),
        ],
    )
    artifact_dir = tmp_path / "stability_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"][
        "stability_ratio_metrics"
    ] == [["total_debt", "total_assets"]]
    assert "Saved report artifacts" in captured.out


def test_report_command_records_dataset_feature_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should include dataset feature provenance entries."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "stability_ratio_metrics": '[["total_debt", "total_assets"]]',
        },
        fundamentals_rows=[
            ("AAPL", "2023-12-31", "2024-01-02", "total_debt", "44.0"),
            ("AAPL", "2023-12-31", "2024-01-02", "total_assets", "110.0"),
        ],
    )
    artifact_dir = tmp_path / "feature_metadata_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    by_column = {
        entry["column"]: entry
        for entry in metadata["dataset_feature_metadata"]
    }

    assert exit_code == 0
    assert by_column["forward_return_1d"]["role"] == "label"
    assert by_column["fundamental_total_debt"]["source"] == "fundamentals"
    assert "next market session" in by_column["fundamental_total_debt"]["timing"]
    assert by_column["stability_total_debt_to_total_assets"]["family"] == (
        "stability_ratio"
    )
    assert metadata["feature_cache_metadata"]["materialization"] == "metadata_only"
    assert len(metadata["feature_cache_metadata"]["cache_key"]) == 64
    assert "forward_return_1d" in metadata["feature_cache_metadata"]["label_columns"]
    assert "forward_return_1d" not in metadata["feature_cache_metadata"][
        "feature_columns"
    ]
    assert "stability_total_debt_to_total_assets" in metadata[
        "feature_cache_metadata"
    ]["feature_columns"]
    assert "Saved report artifacts" in captured.out


def test_report_command_records_classification_field_selection_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record the configured classifications selection."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "classification_fields": '["sector"]',
        },
        classifications_rows=[
            ("AAPL", "2024-01-03", "Technology", "Hardware"),
        ],
    )
    artifact_dir = tmp_path / "classification_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"]["classification_fields"] == [
        "sector"
    ]
    assert "Saved report artifacts" in captured.out


def test_report_command_records_grouped_diagnostics_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should include configured grouped IC and coverage diagnostics."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "classification_fields": '["sector"]',
        },
        classifications_rows=[
            ("AAPL", "2024-01-02", "Technology", "Hardware"),
            ("MSFT", "2024-01-02", "Technology", "Software"),
            ("TSLA", "2024-01-02", "Consumer", "Automobiles"),
        ],
        diagnostics_overrides={
            "forward_return_column": '"forward_return_1d"',
            "ic_method": '"pearson"',
            "n_quantiles": "2",
            "min_observations": "2",
            "rolling_ic_window": "2",
            "group_columns": '["classification_sector"]',
        },
    )
    artifact_dir = tmp_path / "grouped_ic_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    report_text = (artifact_dir / "report.txt").read_text(encoding="utf-8")
    chart_manifest = json.loads(
        (artifact_dir / "charts" / "manifest.json").read_text(encoding="utf-8")
    )

    assert exit_code == 0
    assert (artifact_dir / "charts" / "grouped_ic_timeseries.png").exists()
    assert (artifact_dir / "charts" / "grouped_ic_summary.png").exists()
    assert (artifact_dir / "charts" / "grouped_coverage_timeseries.png").exists()
    assert (artifact_dir / "charts" / "grouped_coverage_summary.png").exists()
    assert metadata["workflow_configuration"]["diagnostics"]["group_columns"] == [
        "classification_sector"
    ]
    assert {
        row["group_column"] for row in metadata["grouped_ic_series"]["rows"]
    } == {"classification_sector"}
    assert {
        row["group_column"] for row in metadata["grouped_ic_summary"]["rows"]
    } == {"classification_sector"}
    assert {
        row["group_column"] for row in metadata["grouped_coverage_by_date"]["rows"]
    } == {"classification_sector"}
    assert {
        row["group_column"] for row in metadata["grouped_coverage_summary"]["rows"]
    } == {"classification_sector"}
    assert metadata["quantile_cumulative_returns"]["rows"]
    assert metadata["quantile_spread_stability"]["periods"] > 0
    assert "Grouped IC Summary" in metadata["report_sections"]
    assert "Grouped Coverage Summary" in metadata["report_sections"]
    assert "Cumulative Quantile Mean Forward Returns" in metadata["report_sections"]
    assert "Quantile Spread Stability" in metadata["report_sections"]
    assert any(
        chart["chart_id"] == "grouped_ic_timeseries"
        for chart in chart_manifest["charts"]
    )
    assert any(
        chart["chart_id"] == "grouped_ic_summary"
        for chart in chart_manifest["charts"]
    )
    assert any(
        chart["chart_id"] == "grouped_coverage_timeseries"
        for chart in chart_manifest["charts"]
    )
    assert any(
        chart["chart_id"] == "grouped_coverage_summary"
        for chart in chart_manifest["charts"]
    )
    assert "Grouped IC Summary" in report_text
    assert "Grouped Coverage Summary" in report_text
    assert "Cumulative Quantile Mean Forward Returns" in report_text
    assert "Quantile Spread Stability" in report_text
    assert "Saved report artifacts" in captured.out


def test_report_command_records_membership_index_selection_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record the configured membership selection."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "membership_indexes": '["S&P 500"]',
        },
        memberships_rows=[
            ("AAPL", "2024-01-03", "S&P 500", "1"),
        ],
    )
    artifact_dir = tmp_path / "membership_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"]["membership_indexes"] == [
        "S&P 500"
    ]
    assert "Saved report artifacts" in captured.out


def test_report_command_records_borrow_field_selection_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record the configured borrow selection."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "borrow_fields": '["is_borrowable"]',
        },
        borrow_availability_rows=[
            ("AAPL", "2024-01-03", "1", "12.5"),
        ],
    )
    artifact_dir = tmp_path / "borrow_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"]["borrow_fields"] == [
        "is_borrowable"
    ]
    assert "Saved report artifacts" in captured.out


def test_report_command_records_dataset_benchmark_rolling_window_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record configured rolling benchmark dataset features."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "benchmark_rolling_window": "3",
        },
        benchmark_overrides={},
    )
    artifact_dir = tmp_path / "rolling_benchmark_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"]["benchmark_rolling_window"] == 3
    assert "Saved report artifacts" in captured.out


def test_report_command_records_dataset_benchmark_residual_window_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record configured benchmark residual-return features."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "benchmark_residual_return_window": "3",
        },
        benchmark_overrides={},
    )
    artifact_dir = tmp_path / "benchmark_residual_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"][
        "benchmark_residual_return_window"
    ] == 3
    assert "Saved report artifacts" in captured.out


def test_report_command_records_dataset_higher_moments_window_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record configured rolling higher-moment features."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "higher_moments_window": "4",
        },
    )
    artifact_dir = tmp_path / "rolling_higher_moments_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"]["higher_moments_window"] == 4
    assert "Saved report artifacts" in captured.out


def test_report_command_records_dataset_parkinson_volatility_window_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record configured Parkinson volatility features."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "parkinson_volatility_window": "4",
        },
    )
    artifact_dir = tmp_path / "parkinson_volatility_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"][
        "parkinson_volatility_window"
    ] == 4
    assert "Saved report artifacts" in captured.out


def test_report_command_records_dataset_average_true_range_window_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record configured ATR features."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "average_true_range_window": "4",
        },
    )
    artifact_dir = tmp_path / "average_true_range_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"][
        "average_true_range_window"
    ] == 4
    assert "Saved report artifacts" in captured.out


def test_report_command_records_dataset_normalized_average_true_range_window_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record configured normalized ATR features."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "normalized_average_true_range_window": "4",
        },
    )
    artifact_dir = tmp_path / "normalized_average_true_range_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"][
        "normalized_average_true_range_window"
    ] == 4
    assert "Saved report artifacts" in captured.out


def test_report_command_records_dataset_amihud_illiquidity_window_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record configured Amihud illiquidity features."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "amihud_illiquidity_window": "4",
        },
    )
    artifact_dir = tmp_path / "amihud_illiquidity_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"][
        "amihud_illiquidity_window"
    ] == 4
    assert "Saved report artifacts" in captured.out


def test_report_command_records_dataset_dollar_volume_shock_window_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record configured dollar-volume shock features."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "dollar_volume_shock_window": "4",
        },
    )
    artifact_dir = tmp_path / "dollar_volume_shock_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"][
        "dollar_volume_shock_window"
    ] == 4
    assert "Saved report artifacts" in captured.out


def test_report_command_records_dataset_dollar_volume_zscore_window_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record configured dollar-volume z-score features."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "dollar_volume_zscore_window": "4",
        },
    )
    artifact_dir = tmp_path / "dollar_volume_zscore_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"][
        "dollar_volume_zscore_window"
    ] == 4
    assert "Saved report artifacts" in captured.out


def test_report_command_records_dataset_volume_shock_window_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record configured volume-shock features."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "volume_shock_window": "4",
        },
    )
    artifact_dir = tmp_path / "volume_shock_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"]["volume_shock_window"] == 4
    assert "Saved report artifacts" in captured.out


def test_report_command_records_dataset_relative_volume_window_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record configured relative volume features."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "relative_volume_window": "4",
        },
    )
    artifact_dir = tmp_path / "relative_volume_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"]["relative_volume_window"] == 4
    assert "Saved report artifacts" in captured.out


def test_report_command_records_dataset_relative_dollar_volume_window_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record configured relative dollar volume features."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "relative_dollar_volume_window": "4",
        },
    )
    artifact_dir = tmp_path / "relative_dollar_volume_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"][
        "relative_dollar_volume_window"
    ] == 4
    assert "Saved report artifacts" in captured.out


def test_report_command_records_dataset_rogers_satchell_volatility_window_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record configured Rogers-Satchell volatility features."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "rogers_satchell_volatility_window": "4",
        },
    )
    artifact_dir = tmp_path / "rogers_satchell_volatility_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"][
        "rogers_satchell_volatility_window"
    ] == 4
    assert "Saved report artifacts" in captured.out


def test_report_command_records_dataset_yang_zhang_volatility_window_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record configured Yang-Zhang volatility features."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "yang_zhang_volatility_window": "4",
        },
    )
    artifact_dir = tmp_path / "yang_zhang_volatility_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"][
        "yang_zhang_volatility_window"
    ] == 4
    assert "Saved report artifacts" in captured.out


def test_report_command_records_dataset_garman_klass_volatility_window_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record configured Garman-Klass volatility features."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "garman_klass_volatility_window": "4",
        },
    )
    artifact_dir = tmp_path / "garman_klass_volatility_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"][
        "garman_klass_volatility_window"
    ] == 4
    assert "Saved report artifacts" in captured.out


def test_report_command_records_dataset_realized_volatility_window_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record configured realized-volatility features."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "realized_volatility_window": "4",
        },
    )
    artifact_dir = tmp_path / "realized_volatility_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["workflow_configuration"]["dataset"][
        "realized_volatility_window"
    ] == 4
    assert "Saved report artifacts" in captured.out


def test_report_command_records_signal_transform_settings_in_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report metadata should record configured signal transforms and final signal column."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        dataset_overrides={
            "forward_horizons": "[1]",
            "volatility_window": "2",
            "average_volume_window": "2",
            "classification_fields": '["sector"]',
        },
        classifications_rows=[
            ("AAPL", "2024-01-02", "Technology", "Software"),
            ("MSFT", "2024-01-02", "Technology", "Software"),
        ],
        signal_overrides={
            "winsorize_quantile": "0.1",
            "clip_lower_bound": "-2.0",
            "clip_upper_bound": "2.0",
            "cross_sectional_neutralize_group_column": '"classification_sector"',
            "cross_sectional_normalization": '"robust_zscore"',
            "cross_sectional_group_column": '"classification_sector"',
        },
    )
    artifact_dir = tmp_path / "signal_transform_report_artifact"

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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    report_text = (artifact_dir / "report.txt").read_text(encoding="utf-8")

    assert exit_code == 0
    assert metadata["workflow_configuration"]["signal"][
        "winsorize_quantile"
    ] == pytest.approx(0.1)
    assert metadata["workflow_configuration"]["signal"][
        "clip_lower_bound"
    ] == pytest.approx(-2.0)
    assert metadata["workflow_configuration"]["signal"][
        "clip_upper_bound"
    ] == pytest.approx(2.0)
    assert (
        metadata["workflow_configuration"]["signal"][
            "cross_sectional_normalization"
        ]
        == "robust_zscore"
    )
    assert (
        metadata["workflow_configuration"]["signal"][
            "cross_sectional_neutralize_group_column"
        ]
        == "classification_sector"
    )
    assert (
        metadata["workflow_configuration"]["signal"][
            "cross_sectional_group_column"
        ]
        == "classification_sector"
    )
    signal_metadata = metadata["signal_pipeline_metadata"]
    assert signal_metadata["factor"]["name"] == "momentum"
    assert signal_metadata["factor"]["parameters"] == {"lookback": 1}
    assert signal_metadata["raw_signal_column"] == "momentum_signal_1d"
    assert signal_metadata["final_signal_column"] == (
        "momentum_signal_1d_winsorized_clipped_demeaned_robust_zscore"
    )
    assert [
        step["name"] for step in signal_metadata["transform_pipeline"]
    ] == ["winsorize", "clip", "demean", "robust_zscore"]
    assert signal_metadata["transform_pipeline"][0]["parameters"]["quantile"] == (
        pytest.approx(0.1)
    )
    assert signal_metadata["transform_pipeline"][1]["parameters"] == {
        "lower_bound": -2.0,
        "upper_bound": 2.0,
    }
    assert (
        signal_metadata["transform_pipeline"][2]["group_column"]
        == "classification_sector"
    )
    assert signal_metadata["transform_pipeline"][2]["group_scope"] == (
        "date_and_group"
    )
    assert signal_metadata["transform_pipeline"][2]["neutralization"] == (
        "group_demean"
    )
    assert (
        signal_metadata["transform_pipeline"][3]["group_column"]
        == "classification_sector"
    )
    assert "Signal Transform: winsorize_quantile=0.1" in report_text
    assert "clip_bounds=[-2.0, 2.0]" in report_text
    assert "cross_sectional_neutralize_group_column=classification_sector" in report_text
    assert "cross_sectional_normalization=robust_zscore" in report_text
    assert "cross_sectional_group_column=classification_sector" in report_text
    assert (
        "Signal Column: momentum_signal_1d_winsorized_clipped_demeaned_robust_zscore"
        in report_text
    )
    assert "Saved report artifacts" in captured.out


def test_add_signal_from_config_applies_transform_after_universe_masking(
    tmp_path: Path,
) -> None:
    """Cross-sectional transforms should only use rows still eligible after masking."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        signal_overrides={
            "cross_sectional_normalization": '"rank"',
        },
    )
    config = load_pipeline_config(config_path)
    dataset = pd.DataFrame(
        {
            "date": [
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
                "2024-01-03",
            ],
            "symbol": ["AAPL", "MSFT", "TSLA", "AAPL", "MSFT", "TSLA"],
            "open": [100.0, 100.0, 100.0, 110.0, 120.0, 1100.0],
            "high": [101.0, 101.0, 101.0, 111.0, 121.0, 1101.0],
            "low": [99.0, 99.0, 99.0, 109.0, 119.0, 1099.0],
            "close": [100.0, 100.0, 100.0, 110.0, 120.0, 1100.0],
            "volume": [1000, 1000, 1000, 1000, 1000, 1000],
            "is_universe_eligible": [True, True, True, True, True, False],
        }
    )

    signaled, signal_column = add_signal_from_config(dataset, config)
    day_two = signaled.loc[signaled["date"] == pd.Timestamp("2024-01-03")]

    assert signal_column == "momentum_signal_1d_rank"
    assert day_two.loc[day_two["symbol"] == "AAPL", signal_column].item() == pytest.approx(0.0)
    assert day_two.loc[day_two["symbol"] == "MSFT", signal_column].item() == pytest.approx(1.0)
    assert pd.isna(day_two.loc[day_two["symbol"] == "TSLA", signal_column].item())


def test_add_signal_from_config_applies_same_date_residualization(
    tmp_path: Path,
) -> None:
    """Configured residualization should use only same-date numeric exposures."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        signal_overrides={
            "cross_sectional_residualize_columns": '["style_exposure"]',
        },
    )
    config = load_pipeline_config(config_path)
    close_values = [100.0, 100.0, 100.0, 100.0, 109.0, 109.0, 111.0, 115.0]
    dataset = pd.DataFrame(
        {
            "date": [
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
                "2024-01-03",
                "2024-01-03",
            ],
            "symbol": [
                "AAPL",
                "MSFT",
                "NVDA",
                "TSLA",
                "AAPL",
                "MSFT",
                "NVDA",
                "TSLA",
            ],
            "open": close_values,
            "high": [value + 1.0 for value in close_values],
            "low": [value - 1.0 for value in close_values],
            "close": close_values,
            "volume": [1000] * 8,
            "style_exposure": [-1.0, 0.0, 1.0, 2.0, -1.0, 0.0, 1.0, 2.0],
        }
    )

    signaled, signal_column = add_signal_from_config(dataset, config)
    day_two = signaled.loc[signaled["date"] == pd.Timestamp("2024-01-03")]

    assert signal_column == "momentum_signal_1d_residualized"
    assert day_two[signal_column].tolist() == pytest.approx(
        [0.01, -0.01, -0.01, 0.01]
    )


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
    assert (output_dir / "ic_decay_series.png").exists()
    assert (output_dir / "coverage_summary.png").exists()
    assert (output_dir / "coverage_timeseries.png").exists()
    assert (output_dir / "quantile_bucket_returns.png").exists()
    assert (output_dir / "quantile_cumulative_returns.png").exists()
    assert (output_dir / "quantile_spread_timeseries.png").exists()
    assert (output_dir / "benchmark_risk.png").exists()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["command"] == "plot-report"
    assert manifest["chart_count"] >= 9
    assert any(chart["chart_id"] == "nav_overview" for chart in manifest["charts"])
    assert any(chart["chart_id"] == "benchmark_risk" for chart in manifest["charts"])
    assert any(chart["chart_id"] == "ic_cumulative" for chart in manifest["charts"])
    assert any(chart["chart_id"] == "ic_decay_series" for chart in manifest["charts"])
    assert any(
        chart["chart_id"] == "quantile_cumulative_returns"
        for chart in manifest["charts"]
    )
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


def test_validate_data_command_prints_membership_universe_rule(
    tmp_path: Path, capsys
) -> None:
    """Universe membership requirements should appear in validation output."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        universe_overrides={
            "required_membership_indexes": '["S&P 500"]',
            "lag": "1",
        },
        memberships_rows=[
            ("AAPL", "2024-01-02", "S&P 500", "1"),
            ("MSFT", "2024-01-02", "S&P 500", "0"),
        ],
    )

    exit_code = main(["validate-data", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Universe Rules" in captured.out
    assert "Required Membership Indexes: S&P 500" in captured.out
    assert "not_member_s_p_500" in captured.out


def test_validate_data_command_prints_trading_status_universe_rule(
    tmp_path: Path, capsys
) -> None:
    """Universe trading-status requirements should appear in validation output."""
    config_path = _write_pipeline_fixture(
        tmp_path,
        universe_overrides={
            "require_tradable": "true",
            "lag": "1",
        },
        trading_status_rows=[
            ("AAPL", "2024-01-02", "1", ""),
            ("MSFT", "2024-01-02", "0", "halt"),
        ],
    )

    exit_code = main(["validate-data", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Universe Rules" in captured.out
    assert "Require Tradable Status: true" in captured.out
    assert "not_tradable" in captured.out


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
    assert any(
        entry["column"] == "forward_return_1d"
        for entry in metadata["research_context"]["dataset_feature_metadata"]
    )
    assert metadata["research_context"]["signal_pipeline_metadata"]["factor"][
        "name"
    ] == "momentum"
    assert metadata["research_context"]["signal_pipeline_metadata"]["factor"][
        "parameters"
    ] == {"lookback": 2}
    assert metadata["research_context"]["signal_pipeline_metadata"][
        "final_signal_column"
    ] == "momentum_signal_2d"
    assert metadata["research_context"]["feature_cache_metadata"][
        "materialization"
    ] == "metadata_only"
    assert (
        len(metadata["research_context"]["feature_cache_metadata"]["cache_key"])
        == 64
    )
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
    assert any(
        entry["column"] == "forward_return_1d"
        for entry in metadata["research_context"]["dataset_feature_metadata"]
    )
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

    invalid_transform_config = tmp_path / "invalid_transform.toml"
    invalid_transform_config.write_text(
        """
[data]
path = "sample.csv"

[signal]
name = "momentum"
lookback = 1
cross_sectional_normalization = "robust"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="signal.cross_sectional_normalization"):
        load_pipeline_config(invalid_transform_config)

    group_without_normalization_config = tmp_path / "group_without_normalization.toml"
    group_without_normalization_config.write_text(
        """
[data]
path = "sample.csv"

[signal]
name = "momentum"
lookback = 1
cross_sectional_group_column = "classification_sector"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="signal.cross_sectional_group_column"):
        load_pipeline_config(group_without_normalization_config)

    invalid_neutralize_group_config = tmp_path / "invalid_neutralize_group.toml"
    invalid_neutralize_group_config.write_text(
        """
[data]
path = "sample.csv"

[signal]
name = "momentum"
lookback = 1
cross_sectional_neutralize_group_column = ""
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigError,
        match="signal.cross_sectional_neutralize_group_column",
    ):
        load_pipeline_config(invalid_neutralize_group_config)

    invalid_winsorize_config = tmp_path / "invalid_winsorize.toml"
    invalid_winsorize_config.write_text(
        """
[data]
path = "sample.csv"

[signal]
name = "momentum"
lookback = 1
winsorize_quantile = 0.5
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="signal.winsorize_quantile"):
        load_pipeline_config(invalid_winsorize_config)

    partial_clip_config = tmp_path / "partial_clip.toml"
    partial_clip_config.write_text(
        """
[data]
path = "sample.csv"

[signal]
name = "momentum"
lookback = 1
clip_lower_bound = -2.0
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="signal.clip_lower_bound"):
        load_pipeline_config(partial_clip_config)

    invalid_clip_config = tmp_path / "invalid_clip.toml"
    invalid_clip_config.write_text(
        """
[data]
path = "sample.csv"

[signal]
name = "momentum"
lookback = 1
clip_lower_bound = 2.0
clip_upper_bound = 2.0
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="signal.clip_lower_bound"):
        load_pipeline_config(invalid_clip_config)

    invalid_residualize_config = tmp_path / "invalid_residualize.toml"
    invalid_residualize_config.write_text(
        """
[data]
path = "sample.csv"

[signal]
name = "momentum"
lookback = 1
cross_sectional_residualize_columns = ["style_beta", "style_beta"]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigError,
        match="signal.cross_sectional_residualize_columns",
    ):
        load_pipeline_config(invalid_residualize_config)

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

    rolling_ic_config = tmp_path / "rolling_ic.toml"
    rolling_ic_config.write_text(
        """
[data]
path = "sample.csv"

[diagnostics]
rolling_ic_window = 1
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="diagnostics.rolling_ic_window"):
        load_pipeline_config(rolling_ic_config)

    grouped_ic_config = tmp_path / "grouped_ic.toml"
    grouped_ic_config.write_text(
        """
[data]
path = "sample.csv"

[diagnostics]
group_columns = ["classification_sector", "classification_sector"]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="diagnostics.group_columns"):
        load_pipeline_config(grouped_ic_config)


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
    signal_overrides: dict[str, str] | None = None,
    dataset_overrides: dict[str, str] | None = None,
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
    shares_outstanding_overrides: dict[str, str | None] | None = None,
    shares_outstanding_rows: list[tuple[str, str, str]] | None = None,
    classifications_overrides: dict[str, str | None] | None = None,
    classifications_rows: list[tuple[str, str, str, str]] | None = None,
    memberships_overrides: dict[str, str | None] | None = None,
    memberships_rows: list[tuple[str, str, str, str]] | None = None,
    borrow_availability_overrides: dict[str, str | None] | None = None,
    borrow_availability_rows: list[tuple[str, str, str, str]] | None = None,
    trading_status_overrides: dict[str, str | None] | None = None,
    trading_status_rows: list[tuple[str, str, str, str]] | None = None,
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

    if shares_outstanding_rows is None:
        shares_outstanding_rows = []
    shares_outstanding_path = tmp_path / "shares_outstanding.csv"
    if shares_outstanding_rows:
        shares_outstanding_path.write_text(
            "\n".join(
                [
                    "symbol,effective_date,shares_outstanding",
                    *[
                        ",".join(
                            [
                                symbol,
                                effective_date,
                                shares_outstanding,
                            ]
                        )
                        for (
                            symbol,
                            effective_date,
                            shares_outstanding,
                        ) in shares_outstanding_rows
                    ],
                ]
            ),
            encoding="utf-8",
        )

    if classifications_rows is None:
        classifications_rows = []
    classifications_path = tmp_path / "classifications.csv"
    if classifications_rows:
        classifications_path.write_text(
            "\n".join(
                [
                    "symbol,effective_date,sector,industry",
                    *[
                        ",".join(
                            [
                                symbol,
                                effective_date,
                                sector,
                                industry,
                            ]
                        )
                        for (
                            symbol,
                            effective_date,
                            sector,
                            industry,
                        ) in classifications_rows
                    ],
                ]
            ),
            encoding="utf-8",
        )

    if memberships_rows is None:
        memberships_rows = []
    memberships_path = tmp_path / "memberships.csv"
    if memberships_rows:
        memberships_path.write_text(
            "\n".join(
                [
                    "symbol,effective_date,index_name,is_member",
                    *[
                        ",".join(
                            [
                                symbol,
                                effective_date,
                                index_name,
                                is_member,
                            ]
                        )
                        for (
                            symbol,
                            effective_date,
                            index_name,
                            is_member,
                        ) in memberships_rows
                    ],
                ]
            ),
            encoding="utf-8",
        )

    if borrow_availability_rows is None:
        borrow_availability_rows = []
    borrow_availability_path = tmp_path / "borrow_availability.csv"
    if borrow_availability_rows:
        borrow_availability_path.write_text(
            "\n".join(
                [
                    "symbol,effective_date,is_borrowable,borrow_fee_bps",
                    *[
                        ",".join(
                            [
                                symbol,
                                effective_date,
                                is_borrowable,
                                borrow_fee_bps,
                            ]
                        )
                        for (
                            symbol,
                            effective_date,
                            is_borrowable,
                            borrow_fee_bps,
                        ) in borrow_availability_rows
                    ],
                ]
            ),
            encoding="utf-8",
        )

    if trading_status_rows is None:
        trading_status_rows = []
    trading_status_path = tmp_path / "trading_status.csv"
    if trading_status_rows:
        trading_status_path.write_text(
            "\n".join(
                [
                    "symbol,effective_date,is_tradable,status_reason",
                    *[
                        ",".join(
                            [
                                symbol,
                                effective_date,
                                is_tradable,
                                status_reason,
                            ]
                        )
                        for (
                            symbol,
                            effective_date,
                            is_tradable,
                            status_reason,
                        ) in trading_status_rows
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

    shares_outstanding_lines: list[str] = []
    if shares_outstanding_rows:
        shares_outstanding_values: dict[str, str | None] = {
            "path": '"shares_outstanding.csv"',
            "effective_date_column": '"effective_date"',
            "shares_outstanding_column": '"shares_outstanding"',
        }
        if shares_outstanding_overrides is not None:
            shares_outstanding_values.update(shares_outstanding_overrides)
        shares_outstanding_lines = ["[shares_outstanding]"] + [
            f"{key} = {value}"
            for key, value in shares_outstanding_values.items()
            if value is not None
        ]

    classifications_lines: list[str] = []
    if classifications_rows:
        classifications_values: dict[str, str | None] = {
            "path": '"classifications.csv"',
            "effective_date_column": '"effective_date"',
            "sector_column": '"sector"',
            "industry_column": '"industry"',
        }
        if classifications_overrides is not None:
            classifications_values.update(classifications_overrides)
        classifications_lines = ["[classifications]"] + [
            f"{key} = {value}"
            for key, value in classifications_values.items()
            if value is not None
        ]

    memberships_lines: list[str] = []
    if memberships_rows:
        memberships_values: dict[str, str | None] = {
            "path": '"memberships.csv"',
            "effective_date_column": '"effective_date"',
            "index_column": '"index_name"',
            "is_member_column": '"is_member"',
        }
        if memberships_overrides is not None:
            memberships_values.update(memberships_overrides)
        memberships_lines = ["[memberships]"] + [
            f"{key} = {value}"
            for key, value in memberships_values.items()
            if value is not None
        ]

    borrow_availability_lines: list[str] = []
    if borrow_availability_rows:
        borrow_availability_values: dict[str, str | None] = {
            "path": '"borrow_availability.csv"',
            "effective_date_column": '"effective_date"',
            "is_borrowable_column": '"is_borrowable"',
            "borrow_fee_bps_column": '"borrow_fee_bps"',
        }
        if borrow_availability_overrides is not None:
            borrow_availability_values.update(borrow_availability_overrides)
        borrow_availability_lines = ["[borrow_availability]"] + [
            f"{key} = {value}"
            for key, value in borrow_availability_values.items()
            if value is not None
        ]

    trading_status_lines: list[str] = []
    if trading_status_rows:
        trading_status_values: dict[str, str | None] = {
            "path": '"trading_status.csv"',
            "effective_date_column": '"effective_date"',
            "is_tradable_column": '"is_tradable"',
            "status_reason_column": '"status_reason"',
        }
        if trading_status_overrides is not None:
            trading_status_values.update(trading_status_overrides)
        trading_status_lines = ["[trading_status]"] + [
            f"{key} = {value}"
            for key, value in trading_status_values.items()
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

    dataset_values: dict[str, str] = {
        "forward_horizons": "[1]",
        "volatility_window": "2",
        "average_volume_window": "2",
    }
    if dataset_overrides is not None:
        dataset_values.update(dataset_overrides)
    dataset_lines = ["[dataset]"] + [
        f"{key} = {value}" for key, value in dataset_values.items()
    ]

    config_path.write_text(
        "\n\n".join(
            section
            for section in [
                "\n".join(data_lines),
                "\n".join(dataset_lines),
                "\n".join(calendar_lines) if calendar_lines else "",
                "\n".join(symbol_metadata_lines) if symbol_metadata_lines else "",
                "\n".join(corporate_actions_lines) if corporate_actions_lines else "",
                "\n".join(fundamentals_lines) if fundamentals_lines else "",
                "\n".join(shares_outstanding_lines)
                if shares_outstanding_lines
                else "",
                "\n".join(classifications_lines) if classifications_lines else "",
                "\n".join(memberships_lines) if memberships_lines else "",
                "\n".join(borrow_availability_lines)
                if borrow_availability_lines
                else "",
                "\n".join(trading_status_lines) if trading_status_lines else "",
                "\n".join(benchmark_lines) if benchmark_lines else "",
                "\n".join(universe_lines) if universe_lines else "",
                "\n".join(
                    [
                        "[signal]",
                        *[
                            f"{key} = {value}"
                            for key, value in (
                                {
                                    "name": '"momentum"',
                                    "lookback": "1",
                                    **(
                                        signal_overrides
                                        if signal_overrides is not None
                                        else {}
                                    ),
                                }
                            ).items()
                        ],
                    ]
                ),
                "\n".join(portfolio_lines),
                "\n".join(backtest_lines),
                "\n".join(diagnostics_lines),
            ]
            if section
        ),
        encoding="utf-8",
    )
    return config_path

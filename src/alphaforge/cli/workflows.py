"""Config-driven CLI workflows for AlphaForge."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from alphaforge.analytics import (
    compute_ic_series,
    compute_quantile_bucket_returns,
    compute_quantile_spread_series,
    compute_signal_coverage_by_date,
    format_performance_summary,
    format_relative_performance_summary,
    save_compare_summary_chart,
    save_coverage_summary_chart,
    save_coverage_timeseries_chart,
    save_drawdown_chart,
    save_exposure_turnover_chart,
    save_ic_cumulative_chart,
    save_ic_series_chart,
    save_nav_overview_chart,
    save_quantile_bucket_chart,
    save_quantile_spread_chart,
    save_rolling_benchmark_risk_chart,
    summarize_backtest,
    summarize_ic,
    summarize_relative_performance,
    summarize_signal_coverage,
    validate_parameter_sweep_results,
    validate_walk_forward_results,
)
from alphaforge.backtest import run_daily_backtest
from alphaforge.common import AlphaForgeConfig
from alphaforge.data import (
    apply_split_adjustments,
    ensure_dates_on_trading_calendar,
    load_benchmark_returns,
    load_borrow_availability,
    load_classifications,
    load_corporate_actions,
    load_fundamentals,
    load_memberships,
    load_ohlcv,
    load_symbol_metadata,
    load_trading_calendar,
)
from alphaforge.features import build_research_dataset
from alphaforge.portfolio import build_long_only_weights, build_long_short_weights
from alphaforge.risk import (
    compute_rolling_benchmark_risk,
    format_benchmark_risk_summary,
    format_risk_summary,
    summarize_risk,
    summarize_rolling_benchmark_risk,
)
from alphaforge.signals import (
    apply_cross_sectional_signal_transform,
    add_mean_reversion_signal,
    add_momentum_signal,
    add_trend_signal,
)


class WorkflowError(ValueError):
    """Raised when a CLI workflow cannot run from the provided config."""


def load_market_data_from_config(config: AlphaForgeConfig) -> pd.DataFrame:
    """Load and validate raw market data from the configured path."""
    market_data = load_ohlcv(config.data.path)
    if config.data.price_adjustment == "raw":
        return market_data

    corporate_actions = load_corporate_actions_from_config(config)
    return apply_split_adjustments(
        market_data,
        corporate_actions,
        ohlcv_source=str(config.data.path),
        corporate_actions_source=str(config.corporate_actions.path),
    )


def load_trading_calendar_from_config(config: AlphaForgeConfig) -> pd.DataFrame:
    """Load and validate the optional trading calendar from config."""
    calendar_config = config.calendar
    if calendar_config is None:
        raise WorkflowError("The config does not include a [calendar] section.")

    return load_trading_calendar(
        calendar_config.path,
        date_column=calendar_config.date_column,
    )


def load_symbol_metadata_from_config(config: AlphaForgeConfig) -> pd.DataFrame:
    """Load and validate the optional symbol metadata from config."""
    symbol_metadata_config = config.symbol_metadata
    if symbol_metadata_config is None:
        raise WorkflowError("The config does not include a [symbol_metadata] section.")

    return load_symbol_metadata(
        symbol_metadata_config.path,
        listing_date_column=symbol_metadata_config.listing_date_column,
        delisting_date_column=symbol_metadata_config.delisting_date_column,
    )


def load_corporate_actions_from_config(config: AlphaForgeConfig) -> pd.DataFrame:
    """Load and validate the optional corporate-actions input from config."""
    corporate_actions_config = config.corporate_actions
    if corporate_actions_config is None:
        raise WorkflowError("The config does not include a [corporate_actions] section.")

    return load_corporate_actions(
        corporate_actions_config.path,
        ex_date_column=corporate_actions_config.ex_date_column,
        action_type_column=corporate_actions_config.action_type_column,
        split_ratio_column=corporate_actions_config.split_ratio_column,
        cash_amount_column=corporate_actions_config.cash_amount_column,
    )


def load_fundamentals_from_config(config: AlphaForgeConfig) -> pd.DataFrame:
    """Load and validate the optional fundamentals input from config."""
    fundamentals_config = config.fundamentals
    if fundamentals_config is None:
        raise WorkflowError("The config does not include a [fundamentals] section.")

    return load_fundamentals(
        fundamentals_config.path,
        period_end_column=fundamentals_config.period_end_column,
        release_date_column=fundamentals_config.release_date_column,
        metric_name_column=fundamentals_config.metric_name_column,
        metric_value_column=fundamentals_config.metric_value_column,
    )


def load_classifications_from_config(config: AlphaForgeConfig) -> pd.DataFrame:
    """Load and validate the optional classifications input from config."""
    classifications_config = config.classifications
    if classifications_config is None:
        raise WorkflowError("The config does not include a [classifications] section.")

    return load_classifications(
        classifications_config.path,
        effective_date_column=classifications_config.effective_date_column,
        sector_column=classifications_config.sector_column,
        industry_column=classifications_config.industry_column,
    )


def load_memberships_from_config(config: AlphaForgeConfig) -> pd.DataFrame:
    """Load and validate the optional memberships input from config."""
    memberships_config = config.memberships
    if memberships_config is None:
        raise WorkflowError("The config does not include a [memberships] section.")

    return load_memberships(
        memberships_config.path,
        effective_date_column=memberships_config.effective_date_column,
        index_column=memberships_config.index_column,
        is_member_column=memberships_config.is_member_column,
    )


def load_borrow_availability_from_config(config: AlphaForgeConfig) -> pd.DataFrame:
    """Load and validate the optional borrow availability input from config."""
    borrow_availability_config = config.borrow_availability
    if borrow_availability_config is None:
        raise WorkflowError(
            "The config does not include a [borrow_availability] section."
        )

    return load_borrow_availability(
        borrow_availability_config.path,
        effective_date_column=borrow_availability_config.effective_date_column,
        is_borrowable_column=borrow_availability_config.is_borrowable_column,
        borrow_fee_bps_column=borrow_availability_config.borrow_fee_bps_column,
    )


def load_benchmark_returns_from_config(config: AlphaForgeConfig) -> pd.DataFrame:
    """Load and validate the optional benchmark return series from config."""
    benchmark_config = config.benchmark
    if benchmark_config is None:
        raise WorkflowError("The config does not include a [benchmark] section.")

    return load_benchmark_returns(
        benchmark_config.path,
        return_column=benchmark_config.return_column,
    )


def build_dataset_from_config(config: AlphaForgeConfig) -> pd.DataFrame:
    """Build the research dataset from config-driven dataset settings."""
    market_data = load_market_data_from_config(config)
    trading_calendar = (
        load_trading_calendar_from_config(config)
        if config.calendar is not None
        else None
    )
    symbol_metadata = (
        load_symbol_metadata_from_config(config)
        if config.symbol_metadata is not None
        else None
    )
    fundamentals = (
        load_fundamentals_from_config(config)
        if config.dataset.fundamental_metrics
        else None
    )
    classifications = (
        load_classifications_from_config(config)
        if config.dataset.classification_fields
        else None
    )
    memberships = (
        load_memberships_from_config(config)
        if config.dataset.membership_indexes
        else None
    )
    borrow_availability = (
        load_borrow_availability_from_config(config)
        if config.dataset.borrow_fields
        else None
    )
    benchmark_returns = (
        load_benchmark_returns_from_config(config)
        if config.dataset.benchmark_rolling_window is not None
        else None
    )
    return build_dataset_from_market_data(
        market_data,
        config=config,
        trading_calendar=trading_calendar,
        symbol_metadata=symbol_metadata,
        fundamentals=fundamentals,
        classifications=classifications,
        memberships=memberships,
        borrow_availability=borrow_availability,
        benchmark_returns=benchmark_returns,
    )


def build_dataset_from_market_data(
    market_data: pd.DataFrame,
    *,
    config: AlphaForgeConfig,
    trading_calendar: pd.DataFrame | None = None,
    symbol_metadata: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    classifications: pd.DataFrame | None = None,
    memberships: pd.DataFrame | None = None,
    borrow_availability: pd.DataFrame | None = None,
    benchmark_returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the research dataset from already-loaded market data."""
    universe_config = config.universe
    return build_research_dataset(
        market_data,
        trading_calendar=trading_calendar,
        symbol_metadata=symbol_metadata,
        fundamentals=fundamentals,
        classifications=classifications,
        memberships=memberships,
        borrow_availability=borrow_availability,
        benchmark_returns=benchmark_returns,
        fundamental_metrics=(
            config.dataset.fundamental_metrics if fundamentals is not None else None
        ),
        classification_fields=(
            config.dataset.classification_fields if classifications is not None else None
        ),
        membership_indexes=(
            config.dataset.membership_indexes if memberships is not None else None
        ),
        borrow_fields=(
            config.dataset.borrow_fields
            if borrow_availability is not None
            else None
        ),
        garman_klass_volatility_window=config.dataset.garman_klass_volatility_window,
        parkinson_volatility_window=config.dataset.parkinson_volatility_window,
        rogers_satchell_volatility_window=(
            config.dataset.rogers_satchell_volatility_window
        ),
        realized_volatility_window=config.dataset.realized_volatility_window,
        higher_moments_window=config.dataset.higher_moments_window,
        benchmark_rolling_window=config.dataset.benchmark_rolling_window,
        forward_horizons=config.dataset.forward_horizons,
        volatility_window=config.dataset.volatility_window,
        average_volume_window=config.dataset.average_volume_window,
        minimum_price=(
            universe_config.min_price if universe_config is not None else None
        ),
        minimum_average_volume=(
            universe_config.min_average_volume if universe_config is not None else None
        ),
        minimum_average_dollar_volume=(
            universe_config.min_average_dollar_volume
            if universe_config is not None
            else None
        ),
        minimum_listing_history_days=(
            universe_config.min_listing_history_days
            if universe_config is not None
            else None
        ),
        universe_lag=universe_config.lag if universe_config is not None else 1,
        universe_average_volume_window=(
            universe_config.average_volume_window if universe_config is not None else None
        ),
        universe_average_dollar_volume_window=(
            universe_config.average_dollar_volume_window
            if universe_config is not None
            else None
        ),
    )


def add_signal_from_config(
    dataset: pd.DataFrame,
    config: AlphaForgeConfig,
) -> tuple[pd.DataFrame, str]:
    """Append the configured signal to the research dataset."""
    signal_config = require_signal_config(config)

    if signal_config.name == "momentum":
        lookback = signal_config.lookback or 1
        signal_column = f"momentum_signal_{lookback}d"
        signaled = add_momentum_signal(dataset, lookback=lookback)
        masked = _mask_ineligible_signal_rows(
            signaled,
            signal_column=signal_column,
        )
        return _apply_signal_transforms_from_config(
            masked,
            signal_column=signal_column,
            config=config,
        )

    if signal_config.name == "mean_reversion":
        lookback = signal_config.lookback or 1
        signal_column = f"mean_reversion_signal_{lookback}d"
        signaled = add_mean_reversion_signal(dataset, lookback=lookback)
        masked = _mask_ineligible_signal_rows(
            signaled,
            signal_column=signal_column,
        )
        return _apply_signal_transforms_from_config(
            masked,
            signal_column=signal_column,
            config=config,
        )

    short_window = signal_config.short_window or 20
    long_window = signal_config.long_window or 60
    signal_column = f"trend_signal_{short_window}_{long_window}d"
    signaled = add_trend_signal(
        dataset,
        short_window=short_window,
        long_window=long_window,
    )
    masked = _mask_ineligible_signal_rows(
        signaled,
        signal_column=signal_column,
    )
    return _apply_signal_transforms_from_config(
        masked,
        signal_column=signal_column,
        config=config,
    )


def build_weights_from_config(
    frame: pd.DataFrame,
    *,
    score_column: str,
    config: AlphaForgeConfig,
) -> pd.DataFrame:
    """Build portfolio weights from the configured portfolio settings."""
    portfolio_config = require_portfolio_config(config)

    if portfolio_config.construction == "long_only":
        return build_long_only_weights(
            frame,
            score_column=score_column,
            top_n=portfolio_config.top_n,
            weighting=portfolio_config.weighting,
            exposure=portfolio_config.exposure,
            max_position_weight=portfolio_config.max_position_weight,
        )

    return build_long_short_weights(
        frame,
        score_column=score_column,
        top_n=portfolio_config.top_n,
        bottom_n=portfolio_config.bottom_n,
        weighting=portfolio_config.weighting,
        long_exposure=portfolio_config.long_exposure,
        short_exposure=portfolio_config.short_exposure,
        max_position_weight=portfolio_config.max_position_weight,
    )


def run_backtest_from_config(config: AlphaForgeConfig) -> pd.DataFrame:
    """Run the configured research, signal, weighting, and backtest workflow."""
    dataset = build_dataset_from_config(config)
    signaled, signal_column = add_signal_from_config(dataset, config)
    weighted = build_weights_from_config(signaled, score_column=signal_column, config=config)
    backtest = _run_backtest_with_config(weighted, config=config)
    return _maybe_attach_benchmark_to_backtest(backtest, config=config)


def build_report_text(config: AlphaForgeConfig) -> str:
    """Run the full configured pipeline and render a text report."""
    _, report_text, _ = build_report_package(config)
    return report_text


def build_report_package(
    config: AlphaForgeConfig,
    *,
    config_path: str | None = None,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """Build the backtest, text report, and metadata for one configured research run."""
    context = _build_report_context(config)
    report_text = _render_report_text(context, config=config)
    metadata = _build_report_metadata(
        context,
        config=config,
        config_path=config_path,
    )
    return context["backtest"], report_text, metadata


def write_report_artifact_bundle(
    config: AlphaForgeConfig,
    artifact_dir: str | Path,
    *,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Write one full report artifact bundle, including static chart outputs."""
    context = _build_report_context(config)
    report_text = _render_report_text(context, config=config)
    metadata = _build_report_metadata(
        context,
        config=config,
        config_path=config_path,
    )
    artifact_paths = write_artifact_bundle(
        artifact_dir,
        results=context["backtest"],
        report_text=report_text,
        metadata=metadata,
    )
    chart_bundle = _write_report_chart_bundle_from_context(
        context,
        output_dir=Path(artifact_paths["artifact_dir"]) / "charts",
        config=config,
        config_path=config_path,
        command_name="report",
    )
    enriched_metadata = dict(metadata)
    enriched_metadata["chart_bundle"] = {
        "chart_dir": "charts",
        "manifest_path": str(Path("charts") / "manifest.json"),
        "chart_count": chart_bundle["chart_count"],
        "charts": chart_bundle["charts"],
    }
    html_path = _write_report_html_page(
        report_text=report_text,
        metadata=enriched_metadata,
        artifact_dir=artifact_paths["artifact_dir"],
    )
    enriched_metadata["html_report_path"] = html_path.name
    write_json(enriched_metadata, artifact_paths["metadata_path"])
    artifact_paths["chart_dir"] = chart_bundle["chart_dir"]
    artifact_paths["chart_manifest_path"] = chart_bundle["manifest_path"]
    artifact_paths["html_path"] = html_path
    return artifact_paths


def write_report_chart_bundle(
    config: AlphaForgeConfig,
    output_dir: str | Path,
    *,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Write one static chart bundle from the configured report context."""
    context = _build_report_context(config)
    return _write_report_chart_bundle_from_context(
        context,
        output_dir=output_dir,
        config=config,
        config_path=config_path,
        command_name="plot-report",
    )


def write_compare_artifact_bundle(
    artifact_dir: str | Path,
    *,
    results: pd.DataFrame,
    report_text: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Write one compare-runs artifact bundle plus static multi-run charts."""
    artifact_paths = write_artifact_bundle(
        artifact_dir,
        results=results,
        report_text=report_text,
        metadata=metadata,
    )
    chart_bundle = _write_compare_chart_bundle(
        results,
        output_dir=Path(artifact_paths["artifact_dir"]) / "charts",
    )
    enriched_metadata = dict(metadata)
    enriched_metadata["chart_bundle"] = {
        "chart_dir": "charts",
        "manifest_path": str(Path("charts") / "manifest.json"),
        "chart_count": chart_bundle["chart_count"],
        "charts": chart_bundle["charts"],
    }
    write_json(enriched_metadata, artifact_paths["metadata_path"])
    artifact_paths["chart_dir"] = chart_bundle["chart_dir"]
    artifact_paths["chart_manifest_path"] = chart_bundle["manifest_path"]
    return artifact_paths


def build_validate_data_text(config: AlphaForgeConfig) -> str:
    """Render validation output for market data and optional universe filters."""
    market_data = load_market_data_from_config(config)
    trading_calendar = (
        load_trading_calendar_from_config(config)
        if config.calendar is not None
        else None
    )
    symbol_metadata = (
        load_symbol_metadata_from_config(config)
        if config.symbol_metadata is not None
        else None
    )
    corporate_actions = (
        load_corporate_actions_from_config(config)
        if config.corporate_actions is not None
        else None
    )
    fundamentals = (
        load_fundamentals_from_config(config)
        if config.fundamentals is not None
        else None
    )
    classifications = (
        load_classifications_from_config(config)
        if config.classifications is not None
        else None
    )
    memberships = (
        load_memberships_from_config(config)
        if config.memberships is not None
        else None
    )
    borrow_availability = (
        load_borrow_availability_from_config(config)
        if config.borrow_availability is not None
        else None
    )
    if trading_calendar is not None:
        ensure_dates_on_trading_calendar(
            market_data["date"],
            trading_calendar,
            source="market data",
        )
    if trading_calendar is not None and corporate_actions is not None:
        ensure_dates_on_trading_calendar(
            corporate_actions["ex_date"],
            trading_calendar,
            source="corporate actions",
        )
    sections = [
        describe_market_data(market_data),
        describe_data_quality(market_data),
    ]
    if config.calendar is not None:
        sections.append(describe_trading_calendar_configuration(config))
        sections.append(describe_trading_calendar_data(trading_calendar, config=config))
    if config.symbol_metadata is not None:
        sections.append(describe_symbol_metadata_configuration(config))
        sections.append(describe_symbol_metadata_data(symbol_metadata, config=config))
    if config.corporate_actions is not None:
        sections.append(describe_corporate_actions_configuration(config))
        sections.append(
            describe_corporate_actions_data(corporate_actions, config=config)
        )
    if config.fundamentals is not None:
        sections.append(describe_fundamentals_configuration(config))
        sections.append(describe_fundamentals_data(fundamentals, config=config))
    if config.classifications is not None:
        sections.append(describe_classifications_configuration(config))
        sections.append(describe_classifications_data(classifications, config=config))
    if config.memberships is not None:
        sections.append(describe_memberships_configuration(config))
        sections.append(describe_memberships_data(memberships, config=config))
    if config.borrow_availability is not None:
        sections.append(describe_borrow_availability_configuration(config))
        sections.append(
            describe_borrow_availability_data(
                borrow_availability,
                config=config,
            )
        )
    if config.benchmark is not None:
        benchmark_data = load_benchmark_returns_from_config(config)
        if trading_calendar is not None:
            ensure_dates_on_trading_calendar(
                benchmark_data["date"],
                trading_calendar,
                source="benchmark data",
            )
        sections.append(describe_benchmark_configuration(config))
        sections.append(describe_benchmark_data(benchmark_data, config=config))
    if config.universe is not None:
        dataset = build_dataset_from_market_data(
            market_data,
            config=config,
            trading_calendar=trading_calendar,
            symbol_metadata=symbol_metadata,
            fundamentals=(
                fundamentals if config.dataset.fundamental_metrics else None
            ),
            classifications=(
                classifications if config.dataset.classification_fields else None
            ),
            memberships=(
                memberships if config.dataset.membership_indexes else None
            ),
            borrow_availability=(
                borrow_availability if config.dataset.borrow_fields else None
            ),
        )
        sections.append(describe_universe_configuration(config))
        sections.append(describe_universe_eligibility(dataset))
    return "\n\n".join(section for section in sections if section)


def run_signal_parameter_sweep(
    config: AlphaForgeConfig,
    *,
    parameter_name: str,
    values: list[int],
) -> pd.DataFrame:
    """Run a simple signal parameter sweep against a fixed pipeline config."""
    portfolio_config = require_portfolio_config(config)
    candidate_values = _normalize_sweep_values(values)
    dataset = build_dataset_from_config(config)

    rows = []
    for parameter_value in candidate_values:
        candidate_config = _replace_signal_parameter(
            config,
            parameter_name=parameter_name,
            parameter_value=parameter_value,
        )
        signaled, signal_column = add_signal_from_config(dataset, candidate_config)
        _validate_diagnostics_column(signaled, candidate_config)
        weighted = build_weights_from_config(
            signaled,
            score_column=signal_column,
            config=candidate_config,
        )
        backtest = _run_backtest_with_config(weighted, config=config)
        performance_summary = summarize_backtest(backtest)
        ic_series = compute_ic_series(
            signaled,
            signal_column=signal_column,
            forward_return_column=config.diagnostics.forward_return_column,
            method=config.diagnostics.ic_method,
            min_observations=config.diagnostics.min_observations,
        )
        ic_summary = summarize_ic(ic_series)
        coverage_summary = summarize_signal_coverage(
            signaled,
            signal_column=signal_column,
            forward_return_column=config.diagnostics.forward_return_column,
        )
        rows.append(
            {
                "parameter_name": parameter_name,
                "parameter_value": float(parameter_value),
                "signal_column": signal_column,
                "cumulative_return": performance_summary["cumulative_return"],
                "max_drawdown": performance_summary["max_drawdown"],
                "sharpe_ratio": performance_summary["sharpe_ratio"],
                "average_turnover": performance_summary["average_turnover"],
                "hit_rate": performance_summary["hit_rate"],
                "mean_ic": ic_summary["mean_ic"],
                "ic_ir": ic_summary["ic_ir"],
                "joint_coverage_ratio": coverage_summary["joint_coverage_ratio"],
                "construction": portfolio_config.construction,
            }
        )

    return pd.DataFrame(rows)


def run_walk_forward_parameter_selection(
    config: AlphaForgeConfig,
    *,
    parameter_name: str,
    values: list[int],
    train_periods: int,
    test_periods: int,
    selection_metric: str = "cumulative_return",
) -> tuple[pd.DataFrame, pd.Series]:
    """Run a rolling walk-forward evaluation with in-sample parameter selection."""
    backtest_config = require_backtest_config(config)
    candidate_values = _normalize_sweep_values(values)
    train_periods = _normalize_positive_int(
        train_periods,
        parameter_name="train_periods",
    )
    test_periods = _normalize_positive_int(
        test_periods,
        parameter_name="test_periods",
    )
    selection_metric = _normalize_walk_forward_selection_metric(selection_metric)

    dataset = build_dataset_from_config(config)
    unique_dates = _extract_unique_dates(dataset)
    folds = _build_walk_forward_folds(
        unique_dates,
        train_periods=train_periods,
        test_periods=test_periods,
    )

    candidates = []
    for parameter_value in candidate_values:
        candidate_config = _replace_signal_parameter(
            config,
            parameter_name=parameter_name,
            parameter_value=parameter_value,
        )
        signaled, signal_column = add_signal_from_config(dataset, candidate_config)
        _validate_diagnostics_column(signaled, candidate_config)
        weighted = build_weights_from_config(
            signaled,
            score_column=signal_column,
            config=candidate_config,
        )
        candidates.append(
            {
                "parameter_value": parameter_value,
                "signal_column": signal_column,
                "signaled": signaled,
                "weighted": weighted,
            }
        )

    fold_rows = []
    oos_backtests: list[pd.DataFrame] = []
    for fold_index, fold in enumerate(folds, start=1):
        best_candidate = None
        best_score = float("-inf")
        best_train_evaluation = None

        for candidate in candidates:
            train_evaluation = _evaluate_walk_forward_slice(
                signaled=candidate["signaled"],
                weighted=candidate["weighted"],
                signal_column=candidate["signal_column"],
                config=config,
                evaluation_dates=fold["train_dates"],
            )
            candidate_score = _extract_walk_forward_selection_score(
                train_evaluation,
                selection_metric=selection_metric,
            )
            if candidate_score > best_score:
                best_candidate = candidate
                best_score = candidate_score
                best_train_evaluation = train_evaluation

        if best_candidate is None or best_train_evaluation is None:
            raise WorkflowError("walk-forward selection could not choose a valid candidate.")

        test_evaluation = _evaluate_walk_forward_slice(
            signaled=best_candidate["signaled"],
            weighted=best_candidate["weighted"],
            signal_column=best_candidate["signal_column"],
            config=config,
            evaluation_dates=fold["test_dates"],
        )
        oos_backtests.append(test_evaluation["backtest"])

        fold_rows.append(
            {
                "fold_index": float(fold_index),
                "parameter_name": parameter_name,
                "selected_parameter_value": float(best_candidate["parameter_value"]),
                "signal_column": best_candidate["signal_column"],
                "selection_metric": selection_metric,
                "train_start": fold["train_dates"][0].date().isoformat(),
                "train_end": fold["train_dates"][-1].date().isoformat(),
                "test_start": fold["test_dates"][0].date().isoformat(),
                "test_end": fold["test_dates"][-1].date().isoformat(),
                "train_selection_score": best_score,
                "train_cumulative_return": best_train_evaluation["performance_summary"][
                    "cumulative_return"
                ],
                "train_mean_ic": best_train_evaluation["ic_summary"]["mean_ic"],
                "test_cumulative_return": test_evaluation["performance_summary"][
                    "cumulative_return"
                ],
                "test_max_drawdown": test_evaluation["performance_summary"][
                    "max_drawdown"
                ],
                "test_sharpe_ratio": test_evaluation["performance_summary"][
                    "sharpe_ratio"
                ],
                "test_mean_ic": test_evaluation["ic_summary"]["mean_ic"],
                "test_joint_coverage_ratio": test_evaluation["coverage_summary"][
                    "joint_coverage_ratio"
                ],
            }
        )

    combined_oos_backtest = pd.concat(oos_backtests, ignore_index=True).sort_values(
        "date",
        kind="mergesort",
    )
    overall_summary = summarize_backtest(combined_oos_backtest.reset_index(drop=True))
    return pd.DataFrame(fold_rows), overall_summary


def _build_report_context(config: AlphaForgeConfig) -> dict[str, Any]:
    """Build the shared report context once so text and metadata stay aligned."""
    market_data = load_market_data_from_config(config)
    benchmark_data = (
        load_benchmark_returns_from_config(config)
        if config.benchmark is not None
        else None
    )
    dataset = build_dataset_from_market_data(
        market_data,
        config=config,
        benchmark_returns=(
            benchmark_data
            if config.dataset.benchmark_rolling_window is not None
            else None
        ),
    )
    signaled, signal_column = add_signal_from_config(dataset, config)
    _validate_diagnostics_column(signaled, config)
    weighted = build_weights_from_config(
        signaled,
        score_column=signal_column,
        config=config,
    )
    backtest = _run_backtest_with_config(weighted, config=config)
    backtest = _maybe_attach_benchmark_to_backtest(
        backtest,
        config=config,
        benchmark_data=benchmark_data,
    )

    performance_summary = summarize_backtest(backtest)
    relative_performance_summary = (
        summarize_relative_performance(backtest)
        if benchmark_data is not None
        else None
    )
    risk_summary = summarize_risk(backtest)
    benchmark_risk_summary = None
    rolling_benchmark_risk = None
    if benchmark_data is not None:
        rolling_benchmark_risk = compute_rolling_benchmark_risk(
            backtest.loc[:, ["date", "net_return"]],
            benchmark_data,
            return_column="net_return",
            benchmark_return_column="benchmark_return",
            window=config.benchmark.rolling_window,
        )
        benchmark_risk_summary = summarize_rolling_benchmark_risk(
            rolling_benchmark_risk
        )

    ic_series = compute_ic_series(
        signaled,
        signal_column=signal_column,
        forward_return_column=config.diagnostics.forward_return_column,
        method=config.diagnostics.ic_method,
        min_observations=config.diagnostics.min_observations,
    )
    ic_summary = summarize_ic(ic_series)
    quantile_summary = compute_quantile_bucket_returns(
        signaled,
        signal_column=signal_column,
        forward_return_column=config.diagnostics.forward_return_column,
        n_quantiles=config.diagnostics.n_quantiles,
        min_observations=config.diagnostics.min_observations,
    )
    quantile_spread_series = compute_quantile_spread_series(
        signaled,
        signal_column=signal_column,
        forward_return_column=config.diagnostics.forward_return_column,
        n_quantiles=config.diagnostics.n_quantiles,
        min_observations=config.diagnostics.min_observations,
    )
    coverage_summary = summarize_signal_coverage(
        signaled,
        signal_column=signal_column,
        forward_return_column=config.diagnostics.forward_return_column,
    )
    coverage_by_date = compute_signal_coverage_by_date(
        signaled,
        signal_column=signal_column,
        forward_return_column=config.diagnostics.forward_return_column,
    )

    return {
        "market_data": market_data,
        "benchmark_data": benchmark_data,
        "dataset": dataset,
        "signaled": signaled,
        "signal_column": signal_column,
        "weighted": weighted,
        "backtest": backtest,
        "rolling_benchmark_risk": rolling_benchmark_risk,
        "performance_summary": performance_summary,
        "relative_performance_summary": relative_performance_summary,
        "risk_summary": risk_summary,
        "benchmark_risk_summary": benchmark_risk_summary,
        "ic_series": ic_series,
        "ic_summary": ic_summary,
        "quantile_summary": quantile_summary,
        "quantile_spread_series": quantile_spread_series,
        "coverage_summary": coverage_summary,
        "coverage_by_date": coverage_by_date,
    }


def _render_report_text(context: dict[str, Any], *, config: AlphaForgeConfig) -> str:
    """Render the final Stage 4 report text from a precomputed context."""
    quantile_summary = context["quantile_summary"]
    quantile_text = (
        quantile_summary.to_string(index=False)
        if not quantile_summary.empty
        else "No quantile buckets produced for the configured signal/label coverage."
    )

    sections = [
        describe_research_workflow(context, config=config),
        describe_market_data(context["market_data"]),
        describe_data_quality(context["market_data"]),
        describe_benchmark_configuration(config),
        describe_benchmark_data(context["benchmark_data"], config=config),
        describe_universe_configuration(config),
        describe_universe_eligibility(context["dataset"]),
        describe_portfolio_constraints(config),
        describe_execution_configuration(config),
        describe_execution_results(context["backtest"]),
        format_performance_summary(context["performance_summary"]),
        (
            format_relative_performance_summary(context["relative_performance_summary"])
            if context["relative_performance_summary"] is not None
            else ""
        ),
        format_risk_summary(context["risk_summary"]),
        (
            format_benchmark_risk_summary(
                context["benchmark_risk_summary"],
                window=config.benchmark.rolling_window,
            )
            if context["benchmark_risk_summary"] is not None
            else ""
        ),
        describe_diagnostics_overview(
            context["ic_summary"],
            context["coverage_summary"],
            quantile_summary,
        ),
        "IC Summary\n" + context["ic_summary"].to_string(),
        "Quantile Bucket Returns\n" + quantile_text,
        "Coverage Summary\n" + context["coverage_summary"].to_string(),
    ]
    return "\n\n".join(section for section in sections if section)


def _build_report_metadata(
    context: dict[str, Any],
    *,
    config: AlphaForgeConfig,
    config_path: str | None,
) -> dict[str, Any]:
    """Build Stage 4 report metadata with research-relevant diagnostics."""
    quantile_summary = context["quantile_summary"]
    diagnostics_overview = _summarize_diagnostics_overview(
        context["ic_summary"],
        context["coverage_summary"],
        quantile_summary,
    )
    report_sections = [
        "Research Workflow",
        "Data Summary",
        "Data Quality Summary",
        "Benchmark Configuration" if config.benchmark is not None else None,
        "Benchmark Summary" if config.benchmark is not None else None,
        "Universe Rules" if config.universe is not None else None,
        "Universe Summary" if config.universe is not None else None,
        "Portfolio Constraints",
        "Execution Assumptions",
        "Execution Summary",
        "Performance Summary",
        (
            "Relative Performance Summary"
            if context["relative_performance_summary"] is not None
            else None
        ),
        "Risk Summary",
        (
            "Benchmark Risk Summary"
            if context["benchmark_risk_summary"] is not None
            else None
        ),
        "Diagnostics Overview",
        "IC Summary",
        "Quantile Bucket Returns",
        "Coverage Summary",
    ]

    return {
        "command": "report",
        "config": config_path or "",
        "row_count": int(len(context["backtest"])),
        "report_sections": [section for section in report_sections if section],
        "workflow_configuration": _build_config_snapshot(config),
        "data_summary": _summarize_market_data(context["market_data"]),
        "data_quality_summary": _summarize_data_quality(context["market_data"]),
        "benchmark_summary": (
            _summarize_benchmark_data(context["benchmark_data"])
            if context["benchmark_data"] is not None
            else None
        ),
        "universe_summary": _summarize_universe_eligibility(context["dataset"]),
        "performance_summary": _series_to_metadata_dict(context["performance_summary"]),
        "relative_performance_summary": (
            _series_to_metadata_dict(context["relative_performance_summary"])
            if context["relative_performance_summary"] is not None
            else None
        ),
        "risk_summary": _series_to_metadata_dict(context["risk_summary"]),
        "benchmark_risk_summary": (
            _series_to_metadata_dict(context["benchmark_risk_summary"])
            if context["benchmark_risk_summary"] is not None
            else None
        ),
        "diagnostics_overview": diagnostics_overview,
        "ic_summary": _series_to_metadata_dict(context["ic_summary"]),
        "coverage_summary": _series_to_metadata_dict(context["coverage_summary"]),
        "quantile_bucket_summary": {
            "rows": _dataframe_records(quantile_summary),
            "top_bottom_spread": diagnostics_overview.get("top_bottom_quantile_spread"),
        },
    }


def describe_market_data(frame: pd.DataFrame) -> str:
    """Render a concise market data summary."""
    summary = _summarize_market_data(frame)
    return "\n".join(
        [
            "Data Summary",
            f"Rows: {summary['rows']}",
            f"Symbols: {summary['symbols']}",
            f"Trading Dates: {summary['trading_dates']}",
            f"Date Range: {summary['start_date']} -> {summary['end_date']}",
        ]
    )


def describe_research_workflow(
    context: dict[str, Any],
    *,
    config: AlphaForgeConfig,
) -> str:
    """Render a concise research-workflow overview for interview-ready reports."""
    signal = require_signal_config(config)
    portfolio = require_portfolio_config(config)
    backtest = require_backtest_config(config)

    if signal.name == "momentum":
        signal_text = f"momentum (lookback={signal.lookback})"
    elif signal.name == "mean_reversion":
        signal_text = f"mean_reversion (lookback={signal.lookback})"
    else:
        signal_text = (
            f"trend (short_window={signal.short_window}, long_window={signal.long_window})"
        )
    signal_transform_text = _describe_signal_transform(signal)

    portfolio_text = (
        f"{portfolio.construction}, top_n={portfolio.top_n}, weighting={portfolio.weighting}"
    )
    if portfolio.construction == "long_short":
        bottom_n = portfolio.bottom_n if portfolio.bottom_n is not None else portfolio.top_n
        portfolio_text += f", bottom_n={bottom_n}"

    benchmark_text = config.benchmark.name if config.benchmark is not None else "None"
    universe_text = "enabled" if config.universe is not None else "disabled"

    return "\n".join(
        [
            "Research Workflow",
            f"Market Data File: {config.data.path.name}",
            f"Price Adjustment: {config.data.price_adjustment}",
            f"Benchmark: {benchmark_text}",
            f"Signal: {signal_text}",
            f"Signal Transform: {signal_transform_text}",
            f"Signal Column: {context['signal_column']}",
            f"Label Column: {config.diagnostics.forward_return_column}",
            f"Universe Filters: {universe_text}",
            f"Portfolio: {portfolio_text}",
            f"Rebalance Frequency: {backtest.rebalance_frequency}",
        ]
    )


def describe_data_quality(frame: pd.DataFrame) -> str:
    """Render a concise market-data quality summary for reports and validation."""
    summary = _summarize_data_quality(frame)
    return "\n".join(
        [
            "Data Quality Summary",
            "Rows Per Symbol (min/avg/max): "
            f"{summary['rows_per_symbol_min']}/"
            f"{summary['rows_per_symbol_avg']:.2f}/"
            f"{summary['rows_per_symbol_max']}",
            "Symbols Per Date (min/avg/max): "
            f"{summary['symbols_per_date_min']}/"
            f"{summary['symbols_per_date_avg']:.2f}/"
            f"{summary['symbols_per_date_max']}",
            f"Close Range: {summary['close_min']:.2f} -> {summary['close_max']:.2f}",
            f"Volume Range: {summary['volume_min']:.2f} -> {summary['volume_max']:.2f}",
        ]
    )


def describe_diagnostics_overview(
    ic_summary: pd.Series,
    coverage_summary: pd.Series,
    quantile_summary: pd.DataFrame,
) -> str:
    """Render a compact diagnostics overview ahead of the raw tables."""
    summary = _summarize_diagnostics_overview(
        ic_summary,
        coverage_summary,
        quantile_summary,
    )
    lines = [
        "Diagnostics Overview",
        f"Mean IC: {summary['mean_ic']:.2f}" if summary["mean_ic"] is not None else "Mean IC: NaN",
        f"IC IR: {summary['ic_ir']:.2f}" if summary["ic_ir"] is not None else "IC IR: NaN",
        "Joint Coverage Ratio: "
        f"{summary['joint_coverage_ratio']:.2%}"
        if summary["joint_coverage_ratio"] is not None
        else "Joint Coverage Ratio: NaN",
        "Average Daily Usable Rows: "
        f"{summary['average_daily_usable_rows']:.2f}"
        if summary["average_daily_usable_rows"] is not None
        else "Average Daily Usable Rows: NaN",
    ]
    top_bottom_spread = summary["top_bottom_quantile_spread"]
    if top_bottom_spread is not None:
        lines.append(f"Top-Bottom Quantile Spread: {top_bottom_spread:.2%}")
    return "\n".join(lines)


def describe_universe_configuration(config: AlphaForgeConfig) -> str:
    """Render the configured tradability-aware universe rules."""
    if config.universe is None:
        return ""

    universe = config.universe
    lines = [
        "Universe Rules",
        f"Lag: {universe.lag} day(s)",
    ]
    if universe.min_price is not None:
        lines.append(f"Minimum Price: {universe.min_price}")
    if universe.min_average_volume is not None:
        average_volume_window = (
            universe.average_volume_window or config.dataset.average_volume_window
        )
        lines.append(
            f"Minimum Average Volume ({average_volume_window}d): {universe.min_average_volume}"
        )
    if universe.min_average_dollar_volume is not None:
        average_dollar_volume_window = (
            universe.average_dollar_volume_window
            or config.dataset.average_volume_window
        )
        lines.append(
            "Minimum Average Dollar Volume "
            f"({average_dollar_volume_window}d): {universe.min_average_dollar_volume}"
        )
    if universe.min_listing_history_days is not None:
        lines.append(
            f"Minimum Listing History: {universe.min_listing_history_days} day(s)"
        )
    return "\n".join(lines)


def describe_benchmark_configuration(config: AlphaForgeConfig) -> str:
    """Render the configured benchmark settings."""
    if config.benchmark is None:
        return ""

    benchmark = config.benchmark
    return "\n".join(
        [
            "Benchmark Configuration",
            f"Name: {benchmark.name}",
            f"Return Column: {benchmark.return_column}",
            f"Rolling Window: {benchmark.rolling_window}",
        ]
    )


def describe_benchmark_data(
    frame: pd.DataFrame | None,
    *,
    config: AlphaForgeConfig,
) -> str:
    """Render a concise benchmark return-series summary."""
    if frame is None or config.benchmark is None:
        return ""

    summary = _summarize_benchmark_data(frame)
    return "\n".join(
        [
            "Benchmark Summary",
            f"Rows: {summary['rows']}",
            f"Date Range: {summary['start_date']} -> {summary['end_date']}",
            f"Average Daily Return: {_format_percent_or_nan(summary['average_daily_return'])}",
            "Realized Volatility: "
            f"{_format_percent_or_nan(summary['realized_volatility'])}",
        ]
    )


def describe_symbol_metadata_configuration(config: AlphaForgeConfig) -> str:
    """Render the configured symbol metadata settings."""
    if config.symbol_metadata is None:
        return ""

    symbol_metadata = config.symbol_metadata
    return "\n".join(
        [
            "Symbol Metadata Configuration",
            f"Listing Date Column: {symbol_metadata.listing_date_column}",
            f"Delisting Date Column: {symbol_metadata.delisting_date_column}",
        ]
    )


def describe_symbol_metadata_data(
    frame: pd.DataFrame | None,
    *,
    config: AlphaForgeConfig,
) -> str:
    """Render a concise symbol metadata summary."""
    if frame is None or config.symbol_metadata is None:
        return ""

    summary = _summarize_symbol_metadata_data(frame)
    return "\n".join(
        [
            "Symbol Metadata Summary",
            f"Symbols: {summary['symbols']}",
            f"Listed Date Range: {summary['listing_start_date']} -> {summary['listing_end_date']}",
            f"Active Symbols: {summary['active_symbols']}",
            f"Delisted Symbols: {summary['delisted_symbols']}",
        ]
    )


def describe_corporate_actions_configuration(config: AlphaForgeConfig) -> str:
    """Render the configured corporate-actions settings."""
    if config.corporate_actions is None:
        return ""

    corporate_actions = config.corporate_actions
    return "\n".join(
        [
            "Corporate Actions Configuration",
            f"Ex-Date Column: {corporate_actions.ex_date_column}",
            f"Action Type Column: {corporate_actions.action_type_column}",
            f"Split Ratio Column: {corporate_actions.split_ratio_column}",
            f"Cash Amount Column: {corporate_actions.cash_amount_column}",
        ]
    )


def describe_corporate_actions_data(
    frame: pd.DataFrame | None,
    *,
    config: AlphaForgeConfig,
) -> str:
    """Render a concise corporate-actions summary."""
    if frame is None or config.corporate_actions is None:
        return ""

    summary = _summarize_corporate_actions_data(frame)
    return "\n".join(
        [
            "Corporate Actions Summary",
            f"Rows: {summary['rows']}",
            f"Symbols: {summary['symbols']}",
            f"Ex-Date Range: {summary['start_date']} -> {summary['end_date']}",
            f"Splits: {summary['split_actions']}",
            f"Cash Dividends: {summary['cash_dividend_actions']}",
        ]
    )


def describe_fundamentals_configuration(config: AlphaForgeConfig) -> str:
    """Render the configured fundamentals settings."""
    if config.fundamentals is None:
        return ""

    fundamentals = config.fundamentals
    return "\n".join(
        [
            "Fundamentals Configuration",
            f"Period End Column: {fundamentals.period_end_column}",
            f"Release Date Column: {fundamentals.release_date_column}",
            f"Metric Name Column: {fundamentals.metric_name_column}",
            f"Metric Value Column: {fundamentals.metric_value_column}",
        ]
    )


def describe_fundamentals_data(
    frame: pd.DataFrame | None,
    *,
    config: AlphaForgeConfig,
) -> str:
    """Render a concise fundamentals summary."""
    if frame is None or config.fundamentals is None:
        return ""

    summary = _summarize_fundamentals_data(frame)
    return "\n".join(
        [
            "Fundamentals Summary",
            f"Rows: {summary['rows']}",
            f"Symbols: {summary['symbols']}",
            f"Metrics: {summary['metrics']}",
            f"Period-End Range: {summary['period_start_date']} -> {summary['period_end_date']}",
            f"Release-Date Range: {summary['release_start_date']} -> {summary['release_end_date']}",
        ]
    )


def describe_classifications_configuration(config: AlphaForgeConfig) -> str:
    """Render the configured classifications settings."""
    if config.classifications is None:
        return ""

    classifications = config.classifications
    return "\n".join(
        [
            "Classifications Configuration",
            f"Effective Date Column: {classifications.effective_date_column}",
            f"Sector Column: {classifications.sector_column}",
            f"Industry Column: {classifications.industry_column}",
        ]
    )


def describe_classifications_data(
    frame: pd.DataFrame | None,
    *,
    config: AlphaForgeConfig,
) -> str:
    """Render a concise classifications summary."""
    if frame is None or config.classifications is None:
        return ""

    summary = _summarize_classifications_data(frame)
    return "\n".join(
        [
            "Classifications Summary",
            f"Rows: {summary['rows']}",
            f"Symbols: {summary['symbols']}",
            f"Effective Date Range: {summary['start_date']} -> {summary['end_date']}",
            f"Sectors: {summary['sectors']}",
            f"Industries: {summary['industries']}",
        ]
    )


def describe_memberships_configuration(config: AlphaForgeConfig) -> str:
    """Render the configured memberships settings."""
    if config.memberships is None:
        return ""

    memberships = config.memberships
    return "\n".join(
        [
            "Memberships Configuration",
            f"Effective Date Column: {memberships.effective_date_column}",
            f"Index Column: {memberships.index_column}",
            f"Is Member Column: {memberships.is_member_column}",
        ]
    )


def describe_memberships_data(
    frame: pd.DataFrame | None,
    *,
    config: AlphaForgeConfig,
) -> str:
    """Render a concise memberships summary."""
    if frame is None or config.memberships is None:
        return ""

    summary = _summarize_memberships_data(frame)
    return "\n".join(
        [
            "Memberships Summary",
            f"Rows: {summary['rows']}",
            f"Symbols: {summary['symbols']}",
            f"Indexes: {summary['indexes']}",
            f"Effective Date Range: {summary['start_date']} -> {summary['end_date']}",
            f"Member Rows: {summary['member_rows']}",
            f"Non-Member Rows: {summary['non_member_rows']}",
        ]
    )


def describe_borrow_availability_configuration(config: AlphaForgeConfig) -> str:
    """Render the configured borrow availability settings."""
    if config.borrow_availability is None:
        return ""

    borrow_availability = config.borrow_availability
    return "\n".join(
        [
            "Borrow Availability Configuration",
            f"Effective Date Column: {borrow_availability.effective_date_column}",
            f"Is Borrowable Column: {borrow_availability.is_borrowable_column}",
            f"Borrow Fee Bps Column: {borrow_availability.borrow_fee_bps_column}",
        ]
    )


def describe_borrow_availability_data(
    frame: pd.DataFrame | None,
    *,
    config: AlphaForgeConfig,
) -> str:
    """Render a concise borrow availability summary."""
    if frame is None or config.borrow_availability is None:
        return ""

    summary = _summarize_borrow_availability_data(frame)
    return "\n".join(
        [
            "Borrow Availability Summary",
            f"Rows: {summary['rows']}",
            f"Symbols: {summary['symbols']}",
            f"Effective Date Range: {summary['start_date']} -> {summary['end_date']}",
            f"Borrowable Rows: {summary['borrowable_rows']}",
            f"Not Borrowable Rows: {summary['not_borrowable_rows']}",
            f"Fee Observations: {summary['fee_observations']}",
        ]
    )


def describe_trading_calendar_configuration(config: AlphaForgeConfig) -> str:
    """Render the configured trading calendar settings."""
    if config.calendar is None:
        return ""

    calendar = config.calendar
    return "\n".join(
        [
            "Trading Calendar Configuration",
            f"Name: {calendar.name}",
            f"Date Column: {calendar.date_column}",
        ]
    )


def describe_trading_calendar_data(
    frame: pd.DataFrame | None,
    *,
    config: AlphaForgeConfig,
) -> str:
    """Render a concise trading calendar summary."""
    if frame is None or config.calendar is None:
        return ""

    summary = _summarize_trading_calendar_data(frame)
    return "\n".join(
        [
            "Trading Calendar Summary",
            f"Sessions: {summary['sessions']}",
            f"Date Range: {summary['start_date']} -> {summary['end_date']}",
        ]
    )


def describe_universe_eligibility(frame: pd.DataFrame) -> str:
    """Render a concise summary of lagged universe eligibility."""
    summary = _summarize_universe_eligibility(frame)
    if summary is None:
        return ""

    lines = [
        "Universe Summary",
        f"Eligible Rows: {summary['eligible_rows']}/{summary['total_rows']}",
        f"Excluded Rows: {summary['excluded_rows']}/{summary['total_rows']}",
        "Eligible Symbols Ever: "
        f"{summary['eligible_symbols']}/{summary['total_symbols']}",
        "Eligible Symbols Per Date (min/avg/max): "
        f"{summary['eligible_symbols_per_date_min']}/"
        f"{summary['eligible_symbols_per_date_avg']:.2f}/"
        f"{summary['eligible_symbols_per_date_max']}",
        f"First Eligible Date: {summary['first_eligible_date']}",
    ]

    reason_counts = summary["exclusion_reasons"]
    if reason_counts:
        lines.append(
            "Exclusion Reasons: "
            + ", ".join(f"{reason}={count}" for reason, count in reason_counts.items())
        )
    return "\n".join(lines)


def describe_portfolio_constraints(config: AlphaForgeConfig) -> str:
    """Render the configured portfolio construction rules and simple constraints."""
    if config.portfolio is None:
        return ""

    portfolio = config.portfolio
    lines = [
        "Portfolio Constraints",
        f"Construction: {portfolio.construction}",
        f"Top N: {portfolio.top_n}",
        f"Weighting: {portfolio.weighting}",
    ]
    if portfolio.construction == "long_only":
        lines.append(f"Target Exposure: {portfolio.exposure}")
    else:
        bottom_n = portfolio.bottom_n if portfolio.bottom_n is not None else portfolio.top_n
        lines.extend(
            [
                f"Bottom N: {bottom_n}",
                f"Long Exposure Target: {portfolio.long_exposure}",
                f"Short Exposure Target: {portfolio.short_exposure}",
            ]
        )
    if portfolio.max_position_weight is not None:
        lines.append(f"Max Position Weight: {portfolio.max_position_weight}")
    return "\n".join(lines)


def describe_execution_configuration(config: AlphaForgeConfig) -> str:
    """Render the configured backtest execution assumptions."""
    if config.backtest is None:
        return ""

    backtest = config.backtest
    lines = [
        "Execution Assumptions",
        f"Signal Delay: {backtest.signal_delay} day(s)",
        f"Rebalance Frequency: {backtest.rebalance_frequency}",
        f"Initial NAV: {backtest.initial_nav}",
    ]
    if backtest.transaction_cost_bps is not None:
        lines.append(
            "Transaction Cost Model: "
            f"legacy total {backtest.transaction_cost_bps} bps"
        )
    else:
        lines.extend(
            [
                f"Commission: {backtest.commission_bps} bps",
                f"Slippage: {backtest.slippage_bps} bps",
            ]
        )
    if backtest.max_turnover is not None:
        lines.append(f"Max Turnover Per Rebalance: {backtest.max_turnover}")
    return "\n".join(lines)


def describe_execution_results(backtest: pd.DataFrame) -> str:
    """Render a concise summary of realized execution constraints and costs."""
    if backtest.empty:
        return ""

    summary = _summarize_execution_results(backtest)
    lines = [
        "Execution Summary",
        f"Rebalance Dates: {summary['rebalance_dates']}/{summary['periods']}",
        "Turnover Limit Applied Dates: "
        f"{summary['turnover_limit_dates']}/{summary['periods']}",
        f"Average Target Turnover: {summary['average_target_turnover']:.2f}",
        f"Average Realized Turnover: {summary['average_realized_turnover']:.2f}",
        "Average Gross Target Exposure: "
        f"{summary['average_gross_target_exposure']:.2f}",
        f"Average Gross Exposure: {summary['average_gross_exposure']:.2f}",
        f"Average Target Holdings: {summary['average_target_holdings']:.2f}",
        f"Average Realized Holdings: {summary['average_realized_holdings']:.2f}",
        "Average Target-Effective Weight Gap: "
        f"{summary['average_target_effective_weight_gap']:.2f}",
        f"Total Commission Cost: {_format_percent_or_nan(summary['total_commission_cost'])}",
        f"Total Slippage Cost: {_format_percent_or_nan(summary['total_slippage_cost'])}",
        "Total Transaction Cost: "
        f"{_format_percent_or_nan(summary['total_transaction_cost'])}",
    ]
    return "\n".join(lines)


def build_research_context_metadata(config: AlphaForgeConfig) -> dict[str, Any]:
    """Build lightweight research-context metadata for artifact bundles."""
    market_data = load_market_data_from_config(config)
    benchmark_data = (
        load_benchmark_returns_from_config(config)
        if config.benchmark is not None
        else None
    )
    dataset = (
        build_dataset_from_market_data(market_data, config=config)
        if config.universe is not None
        else None
    )
    return {
        "workflow_configuration": _build_config_snapshot(config),
        "data_summary": _summarize_market_data(market_data),
        "data_quality_summary": _summarize_data_quality(market_data),
        "benchmark_summary": (
            _summarize_benchmark_data(benchmark_data)
            if benchmark_data is not None
            else None
        ),
        "universe_summary": (
            _summarize_universe_eligibility(dataset)
            if dataset is not None
            else None
        ),
    }


def build_sweep_artifact_metadata(
    config: AlphaForgeConfig,
    *,
    config_path: str,
    parameter_name: str,
    values: list[int],
    results: pd.DataFrame,
) -> dict[str, Any]:
    """Build enriched Stage 4 metadata for a sweep artifact bundle."""
    ranked_results = results.sort_values(
        ["cumulative_return", "sharpe_ratio", "mean_ic", "parameter_value"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return {
        "command": "sweep-signal",
        "config": config_path,
        "parameter": parameter_name,
        "values": values,
        "row_count": int(len(results)),
        "research_context": build_research_context_metadata(config),
        "best_candidate": (
            _dataframe_records(ranked_results.head(1))[0]
            if not ranked_results.empty
            else None
        ),
        "top_candidates": _dataframe_records(ranked_results.head(3)),
    }


def build_walk_forward_artifact_metadata(
    config: AlphaForgeConfig,
    *,
    config_path: str,
    parameter_name: str,
    values: list[int],
    train_periods: int,
    test_periods: int,
    selection_metric: str,
    fold_results: pd.DataFrame,
    overall_summary: pd.Series,
) -> dict[str, Any]:
    """Build enriched Stage 4 metadata for a walk-forward artifact bundle."""
    selected_values = pd.Series(dtype=float)
    if "selected_parameter_value" in fold_results.columns:
        selected_values = pd.to_numeric(
            fold_results["selected_parameter_value"],
            errors="coerce",
        ).dropna()

    selection_distribution: dict[str, int] = {}
    if not selected_values.empty:
        value_counts = selected_values.value_counts(sort=False).sort_index()
        selection_distribution = {
            _format_compact_numeric(float(value)): int(count)
            for value, count in value_counts.items()
        }

    return {
        "command": "walk-forward-signal",
        "config": config_path,
        "parameter": parameter_name,
        "values": values,
        "train_periods": train_periods,
        "test_periods": test_periods,
        "selection_metric": selection_metric,
        "row_count": int(len(fold_results)),
        "overall_summary": _series_to_metadata_dict(overall_summary),
        "research_context": build_research_context_metadata(config),
        "fold_count": int(len(fold_results)),
        "selected_parameter_values": [
            _scalar_or_none(value)
            for value in sorted(selected_values.unique().tolist())
        ],
        "selection_distribution": selection_distribution,
        "test_period_start": (
            str(fold_results["test_start"].min())
            if "test_start" in fold_results.columns and not fold_results.empty
            else None
        ),
        "test_period_end": (
            str(fold_results["test_end"].max())
            if "test_end" in fold_results.columns and not fold_results.empty
            else None
        ),
    }


def build_compare_artifact_metadata(
    *,
    experiment_root: str,
    run_ids: list[str],
    selection_mode: str,
    command_name_filter: str | None,
    parameter_filter: str | None,
    rank_by: list[str] | None,
    rank_weight: list[str] | None,
    sort_by: str | None,
    ascending: bool | None,
    limit: int | None,
    results: pd.DataFrame,
) -> dict[str, Any]:
    """Build enriched Stage 4 metadata for a compare-runs artifact bundle."""
    return {
        "command": "compare-runs",
        "experiment_root": experiment_root,
        "run_ids": run_ids,
        "selection_mode": selection_mode,
        "command_name_filter": command_name_filter,
        "parameter_filter": parameter_filter,
        "rank_by": rank_by,
        "rank_weight": rank_weight,
        "sort_by": sort_by,
        "ascending": ascending,
        "limit": limit,
        "row_count": int(len(results)),
        "comparison_summary": _summarize_compare_artifact_results(results),
    }


def _summarize_market_data(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize core market-data coverage."""
    unique_dates = frame["date"].drop_duplicates().sort_values(kind="mergesort")
    return {
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "trading_dates": int(unique_dates.nunique()),
        "start_date": frame["date"].min().date().isoformat(),
        "end_date": frame["date"].max().date().isoformat(),
    }


def _summarize_data_quality(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize simple coverage and range diagnostics for daily OHLCV data."""
    rows_per_symbol = frame.groupby("symbol", sort=True).size()
    symbols_per_date = frame.groupby("date", sort=True)["symbol"].nunique()
    return {
        "rows_per_symbol_min": int(rows_per_symbol.min()),
        "rows_per_symbol_avg": float(rows_per_symbol.mean()),
        "rows_per_symbol_max": int(rows_per_symbol.max()),
        "symbols_per_date_min": int(symbols_per_date.min()),
        "symbols_per_date_avg": float(symbols_per_date.mean()),
        "symbols_per_date_max": int(symbols_per_date.max()),
        "close_min": float(frame["close"].min()),
        "close_max": float(frame["close"].max()),
        "volume_min": float(frame["volume"].min()),
        "volume_max": float(frame["volume"].max()),
    }


def _summarize_benchmark_data(frame: pd.DataFrame | None) -> dict[str, Any] | None:
    """Summarize a benchmark return series for reporting and metadata."""
    if frame is None:
        return None

    realized_volatility = frame["benchmark_return"].std(ddof=1)
    return {
        "rows": int(len(frame)),
        "start_date": frame["date"].min().date().isoformat(),
        "end_date": frame["date"].max().date().isoformat(),
        "average_daily_return": float(frame["benchmark_return"].mean()),
        "realized_volatility": (
            float(realized_volatility * (252.0**0.5))
            if pd.notna(realized_volatility)
            else None
        ),
    }


def _summarize_trading_calendar_data(
    frame: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """Summarize trading calendar coverage for reporting and validation."""
    if frame is None:
        return None

    return {
        "sessions": int(len(frame)),
        "start_date": frame["date"].min().date().isoformat(),
        "end_date": frame["date"].max().date().isoformat(),
    }


def _summarize_symbol_metadata_data(
    frame: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """Summarize symbol metadata coverage for reporting and validation."""
    if frame is None:
        return None

    delisted_symbols = int(frame["delisting_date"].notna().sum())
    return {
        "symbols": int(frame["symbol"].nunique()),
        "listing_start_date": frame["listing_date"].min().date().isoformat(),
        "listing_end_date": frame["listing_date"].max().date().isoformat(),
        "active_symbols": int(len(frame) - delisted_symbols),
        "delisted_symbols": delisted_symbols,
    }


def _summarize_corporate_actions_data(
    frame: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """Summarize corporate-action coverage for validation output."""
    if frame is None:
        return None

    return {
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "start_date": frame["ex_date"].min().date().isoformat(),
        "end_date": frame["ex_date"].max().date().isoformat(),
        "split_actions": int(frame["action_type"].eq("split").sum()),
        "cash_dividend_actions": int(
            frame["action_type"].eq("cash_dividend").sum()
        ),
    }


def _summarize_fundamentals_data(
    frame: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """Summarize long-form fundamentals coverage for validation output."""
    if frame is None:
        return None

    return {
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "metrics": int(frame["metric_name"].nunique()),
        "period_start_date": frame["period_end_date"].min().date().isoformat(),
        "period_end_date": frame["period_end_date"].max().date().isoformat(),
        "release_start_date": frame["release_date"].min().date().isoformat(),
        "release_end_date": frame["release_date"].max().date().isoformat(),
    }


def _summarize_classifications_data(
    frame: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """Summarize sector/industry classifications coverage."""
    if frame is None:
        return None

    return {
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "start_date": frame["effective_date"].min().date().isoformat(),
        "end_date": frame["effective_date"].max().date().isoformat(),
        "sectors": int(frame["sector"].nunique()),
        "industries": int(frame["industry"].nunique()),
    }


def _summarize_memberships_data(
    frame: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """Summarize index membership coverage."""
    if frame is None:
        return None

    return {
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "indexes": int(frame["index_name"].nunique()),
        "start_date": frame["effective_date"].min().date().isoformat(),
        "end_date": frame["effective_date"].max().date().isoformat(),
        "member_rows": int(frame["is_member"].sum()),
        "non_member_rows": int((~frame["is_member"]).sum()),
    }


def _summarize_borrow_availability_data(
    frame: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """Summarize borrow availability coverage."""
    if frame is None:
        return None

    return {
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "start_date": frame["effective_date"].min().date().isoformat(),
        "end_date": frame["effective_date"].max().date().isoformat(),
        "borrowable_rows": int(frame["is_borrowable"].sum()),
        "not_borrowable_rows": int((~frame["is_borrowable"]).sum()),
        "fee_observations": int(frame["borrow_fee_bps"].notna().sum()),
    }


def _summarize_universe_eligibility(frame: pd.DataFrame | None) -> dict[str, Any] | None:
    """Summarize lagged universe eligibility in a metadata-friendly form."""
    if frame is None or "is_universe_eligible" not in frame.columns:
        return None

    eligible = frame["is_universe_eligible"].fillna(False).astype(bool)
    excluded = ~eligible
    total_rows = int(len(frame))
    eligible_rows = int(eligible.sum())
    per_date_eligible_symbols = (
        frame.loc[eligible]
        .groupby("date", sort=True)["symbol"]
        .nunique()
        .reindex(frame["date"].drop_duplicates().sort_values(), fill_value=0)
    )
    exclusion_reasons: dict[str, int] = {}
    if excluded.any() and "universe_exclusion_reason" in frame.columns:
        reason_counts = (
            frame.loc[excluded, "universe_exclusion_reason"]
            .dropna()
            .astype(str)
            .str.split(";")
            .explode()
        )
        reason_counts = reason_counts.loc[reason_counts != ""].value_counts()
        exclusion_reasons = {str(reason): int(count) for reason, count in reason_counts.items()}

    return {
        "total_rows": total_rows,
        "eligible_rows": eligible_rows,
        "excluded_rows": total_rows - eligible_rows,
        "eligible_symbols": int(frame.loc[eligible, "symbol"].nunique()),
        "total_symbols": int(frame["symbol"].nunique()),
        "eligible_symbols_per_date_min": int(per_date_eligible_symbols.min()),
        "eligible_symbols_per_date_avg": float(per_date_eligible_symbols.mean()),
        "eligible_symbols_per_date_max": int(per_date_eligible_symbols.max()),
        "first_eligible_date": (
            frame.loc[eligible, "date"].min().date().isoformat()
            if eligible_rows > 0
            else "None"
        ),
        "exclusion_reasons": exclusion_reasons,
    }


def _summarize_execution_results(backtest: pd.DataFrame) -> dict[str, Any]:
    """Summarize realized execution behavior for reports and metadata."""
    return {
        "periods": int(len(backtest)),
        "rebalance_dates": int(
            backtest["is_rebalance_date"].fillna(False).astype(bool).sum()
        ),
        "turnover_limit_dates": int(
            backtest["turnover_limit_applied"].fillna(False).astype(bool).sum()
        ),
        "average_target_turnover": float(backtest["target_turnover"].mean()),
        "average_realized_turnover": float(backtest["turnover"].mean()),
        "average_gross_target_exposure": float(backtest["gross_target_exposure"].mean()),
        "average_gross_exposure": float(backtest["gross_exposure"].mean()),
        "average_target_holdings": float(backtest["target_holdings_count"].mean()),
        "average_realized_holdings": float(backtest["holdings_count"].mean()),
        "average_target_effective_weight_gap": float(
            backtest["target_effective_weight_gap"].mean()
        ),
        "total_commission_cost": float(backtest["commission_cost"].sum()),
        "total_slippage_cost": float(backtest["slippage_cost"].sum()),
        "total_transaction_cost": float(backtest["transaction_cost"].sum()),
    }


def _summarize_diagnostics_overview(
    ic_summary: pd.Series,
    coverage_summary: pd.Series,
    quantile_summary: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize the key diagnostics headline numbers."""
    top_bottom_quantile_spread = None
    if not quantile_summary.empty and "quantile" in quantile_summary.columns:
        sorted_quantiles = quantile_summary.sort_values("quantile", kind="mergesort")
        if len(sorted_quantiles) >= 2:
            top_bottom_quantile_spread = float(
                sorted_quantiles.iloc[-1]["mean_forward_return"]
                - sorted_quantiles.iloc[0]["mean_forward_return"]
            )

    return {
        "mean_ic": _scalar_or_none(ic_summary.get("mean_ic")),
        "ic_ir": _scalar_or_none(ic_summary.get("ic_ir")),
        "joint_coverage_ratio": _scalar_or_none(
            coverage_summary.get("joint_coverage_ratio")
        ),
        "average_daily_usable_rows": _scalar_or_none(
            coverage_summary.get("average_daily_usable_rows")
        ),
        "top_bottom_quantile_spread": top_bottom_quantile_spread,
    }


def _summarize_compare_artifact_results(results: pd.DataFrame) -> dict[str, Any]:
    """Summarize the headline outputs from a compare-runs result table."""
    if results.empty:
        return {
            "commands": [],
            "summary_scopes": [],
            "best_run_by_cumulative_return": None,
            "best_run_by_mean_ic": None,
        }

    summary: dict[str, Any] = {
        "commands": sorted(results["command"].dropna().astype(str).unique().tolist())
        if "command" in results.columns
        else [],
        "summary_scopes": (
            sorted(results["summary_scope"].dropna().astype(str).unique().tolist())
            if "summary_scope" in results.columns
            else []
        ),
        "best_run_by_cumulative_return": None,
        "best_run_by_mean_ic": None,
    }

    if "summary_cumulative_return" in results.columns:
        sorted_by_return = results.assign(
            summary_cumulative_return=pd.to_numeric(
                results["summary_cumulative_return"],
                errors="coerce",
            )
        ).sort_values(
            ["summary_cumulative_return", "created_at"],
            ascending=[False, False],
            kind="mergesort",
        )
        sorted_by_return = sorted_by_return.loc[
            sorted_by_return["summary_cumulative_return"].notna()
        ]
        if not sorted_by_return.empty:
            summary["best_run_by_cumulative_return"] = _dataframe_records(
                sorted_by_return.loc[
                    :,
                    [
                        column
                        for column in [
                            "run_id",
                            "command",
                            "summary_scope",
                            "summary_cumulative_return",
                            "summary_sharpe_ratio",
                            "summary_mean_ic",
                        ]
                        if column in sorted_by_return.columns
                    ],
                ].head(1)
            )[0]

    if "summary_mean_ic" in results.columns:
        sorted_by_ic = results.assign(
            summary_mean_ic=pd.to_numeric(results["summary_mean_ic"], errors="coerce")
        ).sort_values(
            ["summary_mean_ic", "created_at"],
            ascending=[False, False],
            kind="mergesort",
        )
        sorted_by_ic = sorted_by_ic.loc[sorted_by_ic["summary_mean_ic"].notna()]
        if not sorted_by_ic.empty:
            summary["best_run_by_mean_ic"] = _dataframe_records(
                sorted_by_ic.loc[
                    :,
                    [
                        column
                        for column in [
                            "run_id",
                            "command",
                            "summary_scope",
                            "summary_cumulative_return",
                            "summary_sharpe_ratio",
                            "summary_mean_ic",
                        ]
                        if column in sorted_by_ic.columns
                    ],
                ].head(1)
            )[0]

    return summary


def _build_config_snapshot(config: AlphaForgeConfig) -> dict[str, Any]:
    """Build a metadata-friendly snapshot of the active research configuration."""
    snapshot: dict[str, Any] = {
        "data": {
            "path": str(config.data.path),
            "price_adjustment": config.data.price_adjustment,
        },
        "dataset": {
            "forward_horizons": list(config.dataset.forward_horizons),
            "volatility_window": config.dataset.volatility_window,
            "average_volume_window": config.dataset.average_volume_window,
            "garman_klass_volatility_window": config.dataset.garman_klass_volatility_window,
            "parkinson_volatility_window": config.dataset.parkinson_volatility_window,
            "rogers_satchell_volatility_window": (
                config.dataset.rogers_satchell_volatility_window
            ),
            "realized_volatility_window": config.dataset.realized_volatility_window,
            "higher_moments_window": config.dataset.higher_moments_window,
            "benchmark_rolling_window": config.dataset.benchmark_rolling_window,
            "fundamental_metrics": list(config.dataset.fundamental_metrics),
            "classification_fields": list(config.dataset.classification_fields),
            "membership_indexes": list(config.dataset.membership_indexes),
            "borrow_fields": list(config.dataset.borrow_fields),
        },
        "signal": None,
        "portfolio": None,
        "backtest": None,
        "benchmark": None,
        "universe": None,
        "diagnostics": {
            "forward_return_column": config.diagnostics.forward_return_column,
            "ic_method": config.diagnostics.ic_method,
            "n_quantiles": config.diagnostics.n_quantiles,
            "min_observations": config.diagnostics.min_observations,
        },
    }

    if config.signal is not None:
        snapshot["signal"] = {
            "name": config.signal.name,
            "lookback": config.signal.lookback,
            "short_window": config.signal.short_window,
            "long_window": config.signal.long_window,
            "winsorize_quantile": config.signal.winsorize_quantile,
            "cross_sectional_normalization": (
                config.signal.cross_sectional_normalization
            ),
        }
    if config.portfolio is not None:
        snapshot["portfolio"] = {
            "construction": config.portfolio.construction,
            "top_n": config.portfolio.top_n,
            "bottom_n": config.portfolio.bottom_n,
            "weighting": config.portfolio.weighting,
            "exposure": config.portfolio.exposure,
            "long_exposure": config.portfolio.long_exposure,
            "short_exposure": config.portfolio.short_exposure,
            "max_position_weight": config.portfolio.max_position_weight,
        }
    if config.backtest is not None:
        snapshot["backtest"] = {
            "signal_delay": config.backtest.signal_delay,
            "rebalance_frequency": config.backtest.rebalance_frequency,
            "transaction_cost_bps": config.backtest.transaction_cost_bps,
            "commission_bps": config.backtest.commission_bps,
            "slippage_bps": config.backtest.slippage_bps,
            "max_turnover": config.backtest.max_turnover,
            "initial_nav": config.backtest.initial_nav,
        }
    if config.symbol_metadata is not None:
        snapshot["symbol_metadata"] = {
            "path": str(config.symbol_metadata.path),
            "listing_date_column": config.symbol_metadata.listing_date_column,
            "delisting_date_column": config.symbol_metadata.delisting_date_column,
        }
    if config.fundamentals is not None:
        snapshot["fundamentals"] = {
            "path": str(config.fundamentals.path),
            "period_end_column": config.fundamentals.period_end_column,
            "release_date_column": config.fundamentals.release_date_column,
            "metric_name_column": config.fundamentals.metric_name_column,
            "metric_value_column": config.fundamentals.metric_value_column,
        }
    if config.classifications is not None:
        snapshot["classifications"] = {
            "path": str(config.classifications.path),
            "effective_date_column": config.classifications.effective_date_column,
            "sector_column": config.classifications.sector_column,
            "industry_column": config.classifications.industry_column,
        }
    if config.memberships is not None:
        snapshot["memberships"] = {
            "path": str(config.memberships.path),
            "effective_date_column": config.memberships.effective_date_column,
            "index_column": config.memberships.index_column,
            "is_member_column": config.memberships.is_member_column,
        }
    if config.borrow_availability is not None:
        snapshot["borrow_availability"] = {
            "path": str(config.borrow_availability.path),
            "effective_date_column": config.borrow_availability.effective_date_column,
            "is_borrowable_column": (
                config.borrow_availability.is_borrowable_column
            ),
            "borrow_fee_bps_column": (
                config.borrow_availability.borrow_fee_bps_column
            ),
        }
    if config.calendar is not None:
        snapshot["calendar"] = {
            "path": str(config.calendar.path),
            "name": config.calendar.name,
            "date_column": config.calendar.date_column,
        }
    if config.benchmark is not None:
        snapshot["benchmark"] = {
            "path": str(config.benchmark.path),
            "name": config.benchmark.name,
            "return_column": config.benchmark.return_column,
            "rolling_window": config.benchmark.rolling_window,
        }
    if config.universe is not None:
        snapshot["universe"] = {
            "min_price": config.universe.min_price,
            "min_average_volume": config.universe.min_average_volume,
            "min_average_dollar_volume": config.universe.min_average_dollar_volume,
            "min_listing_history_days": config.universe.min_listing_history_days,
            "lag": config.universe.lag,
            "average_volume_window": config.universe.average_volume_window,
            "average_dollar_volume_window": config.universe.average_dollar_volume_window,
        }
    return snapshot


def _series_to_metadata_dict(summary: pd.Series) -> dict[str, Any]:
    """Convert a summary series into JSON-safe scalar metadata."""
    return {
        str(key): _scalar_or_none(value)
        for key, value in summary.items()
    }


def _dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a small DataFrame into JSON-safe record dictionaries."""
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        records.append({str(key): _scalar_or_none(value) for key, value in row.items()})
    return records


def _scalar_or_none(value: Any) -> Any:
    """Convert pandas/numpy scalars into JSON-safe Python values."""
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def _mask_ineligible_signal_rows(
    frame: pd.DataFrame,
    *,
    signal_column: str,
) -> pd.DataFrame:
    """Set signal values to missing when the lagged universe filter rejects a row."""
    if (
        signal_column not in frame.columns
        or "is_universe_eligible" not in frame.columns
    ):
        return frame

    masked = frame.copy()
    eligible = masked["is_universe_eligible"].fillna(False).astype(bool)
    masked.loc[~eligible, signal_column] = float("nan")
    return masked


def _apply_signal_transforms_from_config(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    config: AlphaForgeConfig,
) -> tuple[pd.DataFrame, str]:
    """Apply any configured within-date signal transforms after universe masking."""
    signal_config = require_signal_config(config)
    if (
        signal_config.winsorize_quantile is None
        and signal_config.cross_sectional_normalization == "none"
    ):
        return frame, signal_column

    return apply_cross_sectional_signal_transform(
        frame,
        score_column=signal_column,
        winsorize_quantile=signal_config.winsorize_quantile,
        normalization=signal_config.cross_sectional_normalization,
    )


def _describe_signal_transform(signal: Any) -> str:
    """Render the configured signal-transform pipeline succinctly."""
    parts: list[str] = []
    if signal.winsorize_quantile is not None:
        parts.append(f"winsorize_quantile={signal.winsorize_quantile}")
    if signal.cross_sectional_normalization != "none":
        parts.append(
            "cross_sectional_normalization="
            f"{signal.cross_sectional_normalization}"
        )
    if not parts:
        return "none"
    return ", ".join(parts)


def _maybe_attach_benchmark_to_backtest(
    backtest: pd.DataFrame,
    *,
    config: AlphaForgeConfig,
    benchmark_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach aligned benchmark returns and relative NAV diagnostics when configured."""
    if config.benchmark is None:
        return backtest

    aligned_benchmark = (
        benchmark_data
        if benchmark_data is not None
        else load_benchmark_returns_from_config(config)
    )
    attached = _align_benchmark_to_backtest(backtest, aligned_benchmark)
    attached["excess_return"] = attached["net_return"] - attached["benchmark_return"]

    benchmark_invalid = attached["benchmark_return"] <= -1.0
    if benchmark_invalid.any():
        raise WorkflowError("benchmark_return values must be greater than -1.0.")

    initial_nav = require_backtest_config(config).initial_nav
    attached["benchmark_nav"] = initial_nav * (1.0 + attached["benchmark_return"]).cumprod()
    attached["relative_return"] = (
        (1.0 + attached["net_return"]).div(1.0 + attached["benchmark_return"]).sub(1.0)
    )
    attached["relative_nav"] = initial_nav * (1.0 + attached["relative_return"]).cumprod()
    return attached


def _align_benchmark_to_backtest(
    backtest: pd.DataFrame,
    benchmark_data: pd.DataFrame,
) -> pd.DataFrame:
    """Require exact benchmark/date alignment before augmenting backtest output."""
    if "date" not in backtest.columns:
        raise WorkflowError("Backtest results are missing 'date'.")

    strategy_dates = (
        pd.to_datetime(backtest["date"], errors="coerce")
        .sort_values(kind="mergesort")
        .reset_index(drop=True)
    )
    if strategy_dates.isna().any():
        raise WorkflowError("Backtest results contain invalid date values.")

    benchmark_dates = benchmark_data["date"].reset_index(drop=True)
    if not strategy_dates.equals(benchmark_dates):
        strategy_set = set(strategy_dates.tolist())
        benchmark_set = set(benchmark_dates.tolist())
        missing_dates = strategy_set - benchmark_set
        extra_dates = benchmark_set - strategy_set
        problems: list[str] = []
        if missing_dates:
            problems.append(f"missing {len(missing_dates)} date(s)")
        if extra_dates:
            problems.append(f"extra {len(extra_dates)} date(s)")
        detail = ""
        if problems:
            detail = " (" + "; ".join(problems) + ")"
        raise WorkflowError(
            "benchmark returns must align exactly to backtest dates" + detail + "."
        )

    attached = backtest.sort_values("date", kind="mergesort").reset_index(drop=True).copy()
    attached["benchmark_return"] = benchmark_data["benchmark_return"].to_numpy()
    return attached


def _run_backtest_with_config(
    frame: pd.DataFrame,
    *,
    config: AlphaForgeConfig,
) -> pd.DataFrame:
    """Run the shared backtest workflow using the config-driven execution settings."""
    backtest_config = require_backtest_config(config)
    return run_daily_backtest(
        frame,
        signal_delay=backtest_config.signal_delay,
        rebalance_frequency=backtest_config.rebalance_frequency,
        transaction_cost_bps=backtest_config.transaction_cost_bps,
        commission_bps=backtest_config.commission_bps,
        slippage_bps=backtest_config.slippage_bps,
        max_turnover=backtest_config.max_turnover,
        initial_nav=backtest_config.initial_nav,
    )


def _write_report_chart_bundle_from_context(
    context: dict[str, Any],
    *,
    output_dir: str | Path,
    config: AlphaForgeConfig,
    config_path: str | None,
    command_name: str,
) -> dict[str, Any]:
    """Write a stable set of static report charts plus a small manifest."""
    chart_dir = Path(output_dir)
    if chart_dir.exists() and not chart_dir.is_dir():
        raise WorkflowError(
            f"Chart output path must be a directory path, not an existing file: {chart_dir}"
        )

    try:
        chart_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise WorkflowError(f"Failed to create chart directory {chart_dir}: {exc}") from exc

    chart_entries: list[dict[str, Any]] = []
    backtest = context["backtest"]

    nav_path = save_nav_overview_chart(backtest, chart_dir / "nav_overview.png")
    chart_entries.append(
        _build_chart_manifest_entry(
            chart_id="nav_overview",
            title="NAV Overview",
            filename=nav_path.name,
            description="Strategy net/gross NAV plus benchmark-relative NAV when available.",
        )
    )

    drawdown_path = save_drawdown_chart(backtest, chart_dir / "drawdown.png")
    chart_entries.append(
        _build_chart_manifest_entry(
            chart_id="drawdown",
            title="Drawdown",
            filename=drawdown_path.name,
            description="Net-return drawdown path from the configured backtest.",
        )
    )

    exposure_turnover_path = save_exposure_turnover_chart(
        backtest,
        chart_dir / "exposure_turnover.png",
    )
    chart_entries.append(
        _build_chart_manifest_entry(
            chart_id="exposure_turnover",
            title="Exposure And Turnover",
            filename=exposure_turnover_path.name,
            description="Gross/net exposure plus target vs realized turnover.",
        )
    )

    ic_series_path = save_ic_series_chart(context["ic_series"], chart_dir / "ic_series.png")
    chart_entries.append(
        _build_chart_manifest_entry(
            chart_id="ic_series",
            title="IC Series",
            filename=ic_series_path.name,
            description="Per-date IC values and observation counts.",
        )
    )

    ic_cumulative_path = save_ic_cumulative_chart(
        context["ic_series"],
        chart_dir / "ic_cumulative.png",
    )
    chart_entries.append(
        _build_chart_manifest_entry(
            chart_id="ic_cumulative",
            title="Cumulative IC",
            filename=ic_cumulative_path.name,
            description="Cumulative sum of per-date IC values with missing periods treated as zero.",
        )
    )

    coverage_summary_path = save_coverage_summary_chart(
        context["coverage_summary"],
        chart_dir / "coverage_summary.png",
    )
    chart_entries.append(
        _build_chart_manifest_entry(
            chart_id="coverage_summary",
            title="Coverage Summary",
            filename=coverage_summary_path.name,
            description="Signal, forward-return, and joint usable coverage ratios.",
        )
    )

    coverage_timeseries_path = save_coverage_timeseries_chart(
        context["coverage_by_date"],
        chart_dir / "coverage_timeseries.png",
    )
    chart_entries.append(
        _build_chart_manifest_entry(
            chart_id="coverage_timeseries",
            title="Coverage Through Time",
            filename=coverage_timeseries_path.name,
            description="Per-date signal, label, and joint usable coverage ratios.",
        )
    )

    quantile_summary = context["quantile_summary"]
    if not quantile_summary.empty:
        quantile_path = save_quantile_bucket_chart(
            quantile_summary,
            chart_dir / "quantile_bucket_returns.png",
        )
        chart_entries.append(
            _build_chart_manifest_entry(
                chart_id="quantile_bucket_returns",
                title="Quantile Bucket Returns",
                filename=quantile_path.name,
                description="Mean forward returns by within-date signal quantile.",
            )
        )

    quantile_spread_series = context["quantile_spread_series"]
    if not quantile_spread_series.empty:
        quantile_spread_path = save_quantile_spread_chart(
            quantile_spread_series,
            chart_dir / "quantile_spread_timeseries.png",
        )
        chart_entries.append(
            _build_chart_manifest_entry(
                chart_id="quantile_spread_timeseries",
                title="Top-Bottom Quantile Spread",
                filename=quantile_spread_path.name,
                description="Per-date top-minus-bottom quantile forward-return spread.",
            )
        )

    rolling_benchmark_risk = context["rolling_benchmark_risk"]
    if rolling_benchmark_risk is not None:
        benchmark_risk_path = save_rolling_benchmark_risk_chart(
            rolling_benchmark_risk,
            chart_dir / "benchmark_risk.png",
        )
        chart_entries.append(
            _build_chart_manifest_entry(
                chart_id="benchmark_risk",
                title="Benchmark Risk",
                filename=benchmark_risk_path.name,
                description="Rolling beta and rolling correlation versus the configured benchmark.",
            )
        )

    manifest = {
        "command": command_name,
        "config": config_path or "",
        "chart_count": len(chart_entries),
        "workflow_configuration": _build_config_snapshot(config),
        "charts": chart_entries,
    }
    manifest_path = write_json(manifest, chart_dir / "manifest.json")
    return {
        "chart_dir": chart_dir,
        "manifest_path": manifest_path,
        "chart_count": len(chart_entries),
        "charts": chart_entries,
    }


def _write_compare_chart_bundle(
    results: pd.DataFrame,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write a static chart bundle for a compare-runs artifact."""
    chart_dir = Path(output_dir)
    if chart_dir.exists() and not chart_dir.is_dir():
        raise WorkflowError(
            f"Chart output path must be a directory path, not an existing file: {chart_dir}"
        )

    try:
        chart_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise WorkflowError(f"Failed to create chart directory {chart_dir}: {exc}") from exc

    compare_chart_path = save_compare_summary_chart(
        results,
        chart_dir / "compare_summary_metrics.png",
    )
    chart_entries = [
        _build_chart_manifest_entry(
            chart_id="compare_summary_metrics",
            title="Compare Summary Metrics",
            filename=compare_chart_path.name,
            description="Multi-run comparison across cumulative return, Sharpe ratio, and mean IC.",
        )
    ]
    manifest = {
        "command": "compare-runs",
        "chart_count": len(chart_entries),
        "charts": chart_entries,
    }
    manifest_path = write_json(manifest, chart_dir / "manifest.json")
    return {
        "chart_dir": chart_dir,
        "manifest_path": manifest_path,
        "chart_count": len(chart_entries),
        "charts": chart_entries,
    }


def _write_report_html_page(
    *,
    report_text: str,
    metadata: dict[str, Any],
    artifact_dir: Path,
) -> Path:
    """Write one self-contained HTML report page for a report artifact bundle."""
    chart_bundle = metadata.get("chart_bundle", {})
    chart_entries = chart_bundle.get("charts", []) if isinstance(chart_bundle, dict) else []
    performance_summary = metadata.get("performance_summary")
    relative_performance_summary = metadata.get("relative_performance_summary")
    diagnostics_overview = metadata.get("diagnostics_overview")
    sections = [
        _render_html_summary_block(
            "Performance Summary",
            performance_summary,
            formatter=_format_html_metric_value,
        ),
        _render_html_summary_block(
            "Relative Performance Summary",
            relative_performance_summary,
            formatter=_format_html_metric_value,
        ),
        _render_html_summary_block(
            "Diagnostics Overview",
            diagnostics_overview,
            formatter=_format_html_metric_value,
        ),
    ]
    cards_html = "".join(section for section in sections if section)
    charts_html = "".join(
        _render_report_chart_card(chart)
        for chart in chart_entries
        if isinstance(chart, dict)
    )
    title = "AlphaForge Research Report"
    config_path = escape(str(metadata.get("config", "")))
    command_name = escape(str(metadata.get("command", "report")))
    report_body = escape(report_text)

    html_text = "\n".join(
        [
            "<!DOCTYPE html>",
            "<html lang=\"en\">",
            "<head>",
            "  <meta charset=\"utf-8\">",
            f"  <title>{title}</title>",
            "  <style>",
            "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #F6F8FA; color: #102A43; }",
            "    main { max-width: 1200px; margin: 0 auto; padding: 32px 24px 56px; }",
            "    h1, h2, h3 { margin: 0 0 12px; }",
            "    p { margin: 0; line-height: 1.5; }",
            "    .hero { background: linear-gradient(135deg, #0B4F6C, #1F9D8B); color: white; border-radius: 18px; padding: 24px; margin-bottom: 24px; }",
            "    .meta { margin-top: 8px; opacity: 0.88; }",
            "    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 24px; }",
            "    .card { background: white; border-radius: 16px; padding: 18px; box-shadow: 0 8px 24px rgba(16, 42, 67, 0.08); }",
            "    .metrics { display: grid; grid-template-columns: 1fr auto; row-gap: 8px; column-gap: 16px; }",
            "    .charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; margin-bottom: 24px; }",
            "    .chart-card img { width: 100%; height: auto; border-radius: 12px; background: #FFF; }",
            "    pre { white-space: pre-wrap; word-break: break-word; background: #0F172A; color: #E2E8F0; border-radius: 16px; padding: 20px; overflow-x: auto; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <main>",
            "    <section class=\"hero\">",
            f"      <h1>{title}</h1>",
            f"      <p class=\"meta\">Command: {command_name}</p>",
            f"      <p class=\"meta\">Config: {config_path}</p>",
            "    </section>",
            (
                f"    <section class=\"grid\">{cards_html}</section>"
                if cards_html
                else ""
            ),
            "    <section>",
            "      <h2>Chart Gallery</h2>",
            (
                f"      <div class=\"charts\">{charts_html}</div>"
                if charts_html
                else "      <p>No charts were generated for this artifact.</p>"
            ),
            "    </section>",
            "    <section>",
            "      <h2>Text Report</h2>",
            f"      <pre>{report_body}</pre>",
            "    </section>",
            "  </main>",
            "</body>",
            "</html>",
        ]
    )
    return write_text(html_text, artifact_dir / "index.html")


def _render_html_summary_block(
    title: str,
    summary: Any,
    *,
    formatter: Any,
) -> str:
    """Render one metric summary dict/series into an HTML card."""
    if summary is None:
        return ""
    items: list[tuple[str, Any]]
    if isinstance(summary, pd.Series):
        items = list(summary.items())
    elif isinstance(summary, dict):
        items = list(summary.items())
    else:
        return ""

    metrics_html = "".join(
        [
            f"<div>{escape(str(key))}</div><div>{escape(formatter(key, value))}</div>"
            for key, value in items
        ]
    )
    return (
        f"<article class=\"card\">"
        f"<h3>{escape(title)}</h3>"
        f"<div class=\"metrics\">{metrics_html}</div>"
        f"</article>"
    )


def _render_report_chart_card(chart: dict[str, Any]) -> str:
    """Render one chart card for the HTML report page."""
    title = escape(str(chart.get("title", "")))
    description = escape(str(chart.get("description", "")))
    path = escape(str(Path("charts") / str(chart.get("filename", ""))))
    return (
        "<article class=\"card chart-card\">"
        f"<h3>{title}</h3>"
        f"<p>{description}</p>"
        f"<img src=\"{path}\" alt=\"{title}\">"
        "</article>"
    )


def _format_html_metric_value(key: Any, value: Any) -> str:
    """Format one metric value for compact HTML summary cards."""
    scalar = _scalar_or_none(value)
    if scalar is None:
        return "NaN"
    if isinstance(scalar, bool):
        return "true" if scalar else "false"
    if isinstance(scalar, int):
        return str(scalar)
    if isinstance(scalar, float):
        key_text = str(key)
        if any(
            token in key_text
            for token in [
                "return",
                "volatility",
                "drawdown",
                "coverage_ratio",
                "hit_rate",
                "value_at_risk",
                "conditional_value_at_risk",
            ]
        ):
            return f"{scalar:.2%}"
        return f"{scalar:.2f}"
    return str(scalar)


def _build_chart_manifest_entry(
    *,
    chart_id: str,
    title: str,
    filename: str,
    description: str,
) -> dict[str, Any]:
    """Build one JSON-safe manifest entry for a generated chart file."""
    return {
        "chart_id": chart_id,
        "title": title,
        "filename": filename,
        "path": filename,
        "description": description,
    }


def write_dataframe(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to CSV, creating parent directories as needed."""
    output_path = Path(path)
    if output_path.exists() and output_path.is_dir():
        raise WorkflowError(
            f"Output path must be a file path, not an existing directory: {output_path}"
        )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
    except OSError as exc:
        raise WorkflowError(f"Failed to write CSV output to {output_path}: {exc}") from exc
    return output_path


def write_text(text: str, path: str | Path) -> Path:
    """Write plain text to disk, creating parent directories as needed."""
    output_path = Path(path)
    if output_path.exists() and output_path.is_dir():
        raise WorkflowError(
            f"Output path must be a file path, not an existing directory: {output_path}"
        )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise WorkflowError(f"Failed to write text output to {output_path}: {exc}") from exc
    return output_path


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    """Write JSON to disk, creating parent directories as needed."""
    output_path = Path(path)
    if output_path.exists() and output_path.is_dir():
        raise WorkflowError(
            f"Output path must be a file path, not an existing directory: {output_path}"
        )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                _make_json_safe(data),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise WorkflowError(f"Failed to write JSON output to {output_path}: {exc}") from exc
    return output_path


def write_artifact_bundle(
    artifact_dir: str | Path,
    *,
    results: pd.DataFrame,
    report_text: str,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    """Write a small reproducible artifact bundle for a CLI experiment run."""
    bundle_dir = Path(artifact_dir)
    if bundle_dir.exists() and not bundle_dir.is_dir():
        raise WorkflowError(
            f"Artifact path must be a directory path, not an existing file: {bundle_dir}"
        )

    try:
        bundle_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise WorkflowError(f"Failed to create artifact directory {bundle_dir}: {exc}") from exc

    results_path = write_dataframe(results, bundle_dir / "results.csv")
    report_path = write_text(report_text, bundle_dir / "report.txt")
    metadata_path = write_json(metadata, bundle_dir / "metadata.json")
    return {
        "artifact_dir": bundle_dir,
        "results_path": results_path,
        "report_path": report_path,
        "metadata_path": metadata_path,
    }


def write_indexed_artifact_bundle(
    experiment_root: str | Path,
    *,
    command_name: str,
    results: pd.DataFrame,
    report_text: str,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    """Write an artifact bundle under an experiment root and append a run index."""
    root_dir = Path(experiment_root)
    if root_dir.exists() and not root_dir.is_dir():
        raise WorkflowError(
            f"Experiment root must be a directory path, not an existing file: {root_dir}"
        )

    try:
        root_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise WorkflowError(f"Failed to create experiment root {root_dir}: {exc}") from exc

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = _build_run_id(command_name=command_name, metadata=metadata)
    bundle_dir = root_dir / run_id

    enriched_metadata = dict(metadata)
    enriched_metadata["run_id"] = run_id
    enriched_metadata["created_at"] = created_at
    enriched_metadata["artifact_dir"] = str(bundle_dir)

    artifact_paths = write_artifact_bundle(
        bundle_dir,
        results=results,
        report_text=report_text,
        metadata=enriched_metadata,
    )
    index_path = _append_run_index(
        root_dir / "runs.csv",
        _build_run_index_row(
            command_name=command_name,
            metadata=enriched_metadata,
            artifact_paths=artifact_paths,
        ),
    )
    artifact_paths["index_path"] = index_path
    return artifact_paths


def load_run_index(experiment_root: str | Path) -> pd.DataFrame:
    """Load and validate a lightweight experiment run index."""
    root_dir = Path(experiment_root)
    if not root_dir.exists():
        raise WorkflowError(f"Experiment root does not exist: {root_dir}")
    if not root_dir.is_dir():
        raise WorkflowError(f"Experiment root must be a directory: {root_dir}")

    index_path = root_dir / "runs.csv"
    if not index_path.exists():
        raise WorkflowError(f"Experiment root does not contain runs.csv: {index_path}")
    if not index_path.is_file():
        raise WorkflowError(f"Run index path must be a file: {index_path}")

    dataset = pd.read_csv(index_path)
    required_columns = [
        "run_id",
        "created_at",
        "command",
        "config",
        "parameter",
        "values",
        "row_count",
        "overall_cumulative_return",
        "artifact_dir",
        "results_path",
        "report_path",
        "metadata_path",
    ]
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise WorkflowError(f"runs.csv is missing required columns: {missing_text}.")
    if dataset.empty:
        raise WorkflowError(f"runs.csv does not contain any experiment runs: {index_path}")

    return dataset


def list_indexed_runs(
    experiment_root: str | Path,
    *,
    command_name: str | None = None,
    parameter_name: str | None = None,
    sort_by: str = "created_at",
    ascending: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    """List indexed runs with optional filtering and sorting."""
    dataset = load_run_index(experiment_root).copy()

    if command_name is not None:
        dataset = dataset.loc[dataset["command"] == command_name].copy()
    if parameter_name is not None:
        dataset = dataset.loc[dataset["parameter"] == parameter_name].copy()
    if dataset.empty:
        raise WorkflowError("No indexed runs match the requested filters.")

    sort_by = _normalize_run_index_sort_key(sort_by)
    if sort_by in {"row_count", "overall_cumulative_return"}:
        dataset[sort_by] = pd.to_numeric(dataset[sort_by], errors="coerce")
    dataset = dataset.sort_values(sort_by, ascending=ascending, kind="mergesort")

    if limit is not None:
        limit = _normalize_positive_int(limit, parameter_name="limit")
        dataset = dataset.head(limit)

    columns = [
        "run_id",
        "created_at",
        "command",
        "parameter",
        "values",
        "selection_metric",
        "row_count",
        "overall_cumulative_return",
        "artifact_dir",
    ]
    return dataset.loc[:, columns].reset_index(drop=True)


def rank_compare_runs(
    experiment_root: str | Path,
    *,
    command_name: str | None = None,
    parameter_name: str | None = None,
    rank_by: list[str],
    rank_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Rank filtered runs by one or more enriched comparison metrics."""
    normalized_rank_by = _normalize_compare_rank_by(rank_by)
    normalized_rank_weights = _normalize_compare_rank_weights(
        normalized_rank_by,
        rank_weights=rank_weights,
    )
    dataset = load_run_index(experiment_root).copy()

    if command_name is not None:
        dataset = dataset.loc[dataset["command"] == command_name].copy()
    if parameter_name is not None:
        dataset = dataset.loc[dataset["parameter"] == parameter_name].copy()
    if dataset.empty:
        raise WorkflowError("No indexed runs match the requested filters.")

    compared = compare_indexed_runs(
        experiment_root,
        run_ids=dataset["run_id"].astype(str).tolist(),
    ).copy()
    rank_columns = []
    for metric in normalized_rank_by:
        compared[metric] = pd.to_numeric(compared[metric], errors="coerce")
        rank_column = f"rank_{metric}"
        compared[rank_column] = compared[metric].rank(
            method="min",
            ascending=False,
            na_option="bottom",
        )
        rank_columns.append(rank_column)

    compared["average_rank"] = compared.loc[:, rank_columns].mean(axis=1)
    sort_columns = ["average_rank", *normalized_rank_by, "created_at"]
    ascending = [True, *([False] * len(normalized_rank_by)), False]
    weight_columns: list[str] = []
    if normalized_rank_weights is not None:
        weighted_score = 0.0
        for metric in normalized_rank_by:
            weight_column = f"weight_{metric}"
            compared[weight_column] = normalized_rank_weights[metric]
            weight_columns.append(weight_column)
            weighted_score += compared[f"rank_{metric}"] * normalized_rank_weights[metric]
        compared["weighted_rank_score"] = weighted_score
        sort_columns = ["weighted_rank_score", *normalized_rank_by, "created_at"]
        ascending = [True, *([False] * len(normalized_rank_by)), False]

    compared = compared.sort_values(
        sort_columns,
        ascending=ascending,
        kind="mergesort",
    ).reset_index(drop=True)

    columns = [
        "run_id",
        "created_at",
        "command",
        "parameter",
        "summary_scope",
        *normalized_rank_by,
        *rank_columns,
        "average_rank",
        *weight_columns,
        "weighted_rank_score" if normalized_rank_weights is not None else None,
        "artifact_dir",
    ]
    return compared.loc[:, [column for column in columns if column is not None]]


def compare_indexed_runs(
    experiment_root: str | Path,
    *,
    run_ids: list[str],
) -> pd.DataFrame:
    """Select specific indexed runs and enrich them with bundle-level summary metrics."""
    subset = _select_indexed_runs_for_comparison(
        experiment_root,
        run_ids=run_ids,
    )

    rows = [
        _build_compared_run_row(row)
        for _, row in subset.iterrows()
    ]
    return pd.DataFrame(rows)


def build_compare_runs_report(
    experiment_root: str | Path,
    *,
    run_ids: list[str],
    sweep_top_k: int = 3,
) -> str:
    """Render a compare-runs report with one summary table plus command-specific details."""
    subset = _select_indexed_runs_for_comparison(
        experiment_root,
        run_ids=run_ids,
    )
    sweep_top_k = _normalize_positive_int(
        sweep_top_k,
        parameter_name="sweep_top_k",
    )

    summary_rows: list[dict[str, Any]] = []
    detail_sections: list[str] = []
    for _, row in subset.iterrows():
        metadata, results = _load_compared_run_bundle(row)
        summary_rows.append(
            _build_compared_run_row_from_bundle(
                row,
                metadata=metadata,
                results=results,
            )
        )
        detail_sections.append(
            _build_compare_run_detail_section(
                row,
                results=results,
                sweep_top_k=sweep_top_k,
            )
        )

    sections = [format_run_index_table(pd.DataFrame(summary_rows), title="Compared Runs")]
    sections.extend(detail_sections)
    return "\n\n".join(sections)


def format_run_index_table(frame: pd.DataFrame, *, title: str) -> str:
    """Format a run index query result as plain text."""
    dataset = frame.copy()
    if "row_count" in dataset.columns:
        dataset["row_count"] = pd.to_numeric(dataset["row_count"], errors="coerce").map(
            lambda value: "NaN" if pd.isna(value) else str(int(value))
        )
    if "overall_cumulative_return" in dataset.columns:
        dataset["overall_cumulative_return"] = pd.to_numeric(
            dataset["overall_cumulative_return"],
            errors="coerce",
        ).map(_format_percent_or_nan)
    if "summary_cumulative_return" in dataset.columns:
        dataset["summary_cumulative_return"] = pd.to_numeric(
            dataset["summary_cumulative_return"],
            errors="coerce",
        ).map(_format_percent_or_nan)
    for column in ("summary_sharpe_ratio", "summary_mean_ic"):
        if column in dataset.columns:
            dataset[column] = pd.to_numeric(dataset[column], errors="coerce").map(
                _format_number_or_nan
            )
    numeric_rank_columns = [
        column
        for column in dataset.columns
        if column == "average_rank"
        or column == "weighted_rank_score"
        or column.startswith("rank_")
        or column.startswith("weight_")
    ]
    for column in numeric_rank_columns:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce").map(
            _format_number_or_nan
        )
    return title + "\n" + dataset.to_string(index=False)


def _select_indexed_runs_for_comparison(
    experiment_root: str | Path,
    *,
    run_ids: list[str],
) -> pd.DataFrame:
    """Load and order the requested run ids for comparison."""
    if not run_ids:
        raise WorkflowError("compare-runs requires at least one run_id.")

    normalized_run_ids = []
    seen: set[str] = set()
    for run_id in run_ids:
        if not isinstance(run_id, str) or not run_id.strip():
            raise WorkflowError("run_id values must be non-empty strings.")
        normalized_run_id = run_id.strip()
        if normalized_run_id in seen:
            raise WorkflowError(
                f"run_id values must be unique; received duplicate {normalized_run_id}."
            )
        normalized_run_ids.append(normalized_run_id)
        seen.add(normalized_run_id)

    dataset = load_run_index(experiment_root).copy()
    subset = dataset.loc[dataset["run_id"].isin(normalized_run_ids)].copy()

    missing_run_ids = [
        run_id for run_id in normalized_run_ids if run_id not in set(subset["run_id"])
    ]
    if missing_run_ids:
        missing_text = ", ".join(missing_run_ids)
        raise WorkflowError(
            f"Requested run_id values were not found in runs.csv: {missing_text}."
        )

    order = {run_id: index for index, run_id in enumerate(normalized_run_ids)}
    subset["run_order"] = subset["run_id"].map(order)
    return subset.sort_values("run_order", kind="mergesort").reset_index(drop=True)


def _normalize_compare_rank_by(rank_by: list[str]) -> list[str]:
    """Validate supported multi-metric compare ranking inputs."""
    if not rank_by:
        raise WorkflowError("rank_by must contain at least one metric.")

    allowed = {
        "summary_cumulative_return",
        "summary_sharpe_ratio",
        "summary_mean_ic",
    }
    normalized: list[str] = []
    seen: set[str] = set()
    for metric in rank_by:
        if metric not in allowed:
            allowed_text = ", ".join(sorted(allowed))
            raise WorkflowError(f"rank_by must be one of {{{allowed_text}}}.")
        if metric in seen:
            raise WorkflowError(f"rank_by metrics must be unique; received duplicate {metric}.")
        normalized.append(metric)
        seen.add(metric)
    return normalized


def _normalize_compare_rank_weights(
    rank_by: list[str],
    *,
    rank_weights: dict[str, float] | None,
) -> dict[str, float] | None:
    """Validate and normalize compare-runs rank weights."""
    if rank_weights is None:
        return None

    rank_by_set = set(rank_by)
    weight_keys = set(rank_weights)
    if weight_keys != rank_by_set:
        missing = sorted(rank_by_set - weight_keys)
        extra = sorted(weight_keys - rank_by_set)
        problems: list[str] = []
        if missing:
            problems.append("missing " + ", ".join(missing))
        if extra:
            problems.append("unexpected " + ", ".join(extra))
        raise WorkflowError(
            "rank_weights must match rank_by exactly: " + "; ".join(problems) + "."
        )

    normalized: dict[str, float] = {}
    total_weight = 0.0
    for metric in rank_by:
        weight = rank_weights[metric]
        if pd.isna(weight) or weight <= 0:
            raise WorkflowError("rank_weights values must be positive finite numbers.")
        normalized[metric] = float(weight)
        total_weight += float(weight)

    if total_weight <= 0:
        raise WorkflowError("rank_weights values must sum to a positive number.")

    return {
        metric: weight / total_weight
        for metric, weight in normalized.items()
    }


def _make_json_safe(value: Any) -> Any:
    """Recursively convert pandas/NaN-containing values into JSON-safe objects."""
    if isinstance(value, dict):
        return {str(key): _make_json_safe(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_make_json_safe(item) for item in value]
    return _scalar_or_none(value)


def _build_run_id(*, command_name: str, metadata: dict[str, Any]) -> str:
    """Build a timestamped run identifier suitable for directory names."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    parameter = str(metadata.get("parameter", "run"))
    slug = _slugify(f"{command_name}-{parameter}")
    return f"{timestamp}-{slug}"


def _build_run_index_row(
    *,
    command_name: str,
    metadata: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    """Build one flattened row for the experiment run index."""
    overall_summary = metadata.get("overall_summary")
    overall_cumulative_return = None
    if isinstance(overall_summary, dict):
        overall_cumulative_return = overall_summary.get("cumulative_return")

    values = metadata.get("values")
    values_text = ""
    if isinstance(values, list):
        values_text = ",".join(str(value) for value in values)

    return {
        "run_id": metadata.get("run_id", ""),
        "created_at": metadata.get("created_at", ""),
        "command": command_name,
        "config": metadata.get("config", ""),
        "parameter": metadata.get("parameter", ""),
        "values": values_text,
        "selection_metric": metadata.get("selection_metric", ""),
        "train_periods": metadata.get("train_periods", ""),
        "test_periods": metadata.get("test_periods", ""),
        "row_count": metadata.get("row_count", ""),
        "overall_cumulative_return": overall_cumulative_return,
        "artifact_dir": str(artifact_paths["artifact_dir"]),
        "results_path": str(artifact_paths["results_path"]),
        "report_path": str(artifact_paths["report_path"]),
        "metadata_path": str(artifact_paths["metadata_path"]),
    }


def _build_compared_run_row(index_row: pd.Series) -> dict[str, Any]:
    """Build a richer comparison row by reading a run's bundle files."""
    metadata, results = _load_compared_run_bundle(index_row)
    return _build_compared_run_row_from_bundle(
        index_row,
        metadata=metadata,
        results=results,
    )


def _build_compared_run_row_from_bundle(
    index_row: pd.Series,
    *,
    metadata: dict[str, Any],
    results: pd.DataFrame,
) -> dict[str, Any]:
    """Build one summary row from already-loaded run bundle artifacts."""
    base_row = {
        "run_id": str(index_row["run_id"]),
        "created_at": str(index_row["created_at"]),
        "command": str(index_row["command"]),
        "config": str(index_row["config"]),
        "parameter": str(index_row["parameter"]),
        "candidate_values": _stringify_candidate_values(metadata.get("values")),
        "selection_metric": _stringify_optional_text(index_row.get("selection_metric")),
        "row_count": index_row.get("row_count"),
        "artifact_dir": str(index_row["artifact_dir"]),
    }

    command_name = str(index_row["command"])
    if command_name == "sweep-signal":
        return {
            **base_row,
            **_summarize_sweep_run(results),
        }
    if command_name == "walk-forward-signal":
        return {
            **base_row,
            **_summarize_walk_forward_run(results, metadata=metadata),
        }

    raise WorkflowError(f"Unsupported command in runs.csv: {command_name}")


def _load_compared_run_bundle(index_row: pd.Series) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load one run's metadata and results artifacts from its index row."""
    metadata = _load_json_file(
        Path(str(index_row["metadata_path"])),
        description="run metadata",
    )
    results = _load_csv_file(
        Path(str(index_row["results_path"])),
        description="run results",
    )
    return metadata, results


def _build_compare_run_detail_section(
    index_row: pd.Series,
    *,
    results: pd.DataFrame,
    sweep_top_k: int,
) -> str:
    """Build one command-specific detail section for compare-runs."""
    run_id = str(index_row["run_id"])
    parameter = str(index_row["parameter"])
    command_name = str(index_row["command"])

    if command_name == "sweep-signal":
        return _format_sweep_top_candidates_section(
            run_id=run_id,
            parameter=parameter,
            results=results,
            top_k=sweep_top_k,
        )
    if command_name == "walk-forward-signal":
        return _format_walk_forward_folds_section(
            run_id=run_id,
            parameter=parameter,
            results=results,
        )

    raise WorkflowError(f"Unsupported command in runs.csv: {command_name}")


def _summarize_sweep_run(results: pd.DataFrame) -> dict[str, Any]:
    """Summarize one sweep run from its results table."""
    required_columns = [
        "parameter_value",
        "cumulative_return",
        "sharpe_ratio",
        "mean_ic",
    ]
    _ensure_required_columns(results, required_columns, description="sweep results")

    dataset = results.copy()
    for column in required_columns:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
    best_index = dataset["cumulative_return"].idxmax()
    best_row = dataset.loc[best_index]
    return {
        "summary_parameter_values": _format_compact_numeric(best_row["parameter_value"]),
        "summary_scope": "best_in_sample_candidate",
        "summary_cumulative_return": best_row["cumulative_return"],
        "summary_sharpe_ratio": best_row["sharpe_ratio"],
        "summary_mean_ic": best_row["mean_ic"],
    }


def _summarize_walk_forward_run(
    results: pd.DataFrame,
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Summarize one walk-forward run from its results table and metadata."""
    required_columns = [
        "selected_parameter_value",
        "test_mean_ic",
    ]
    _ensure_required_columns(results, required_columns, description="walk-forward results")

    dataset = results.copy()
    dataset["selected_parameter_value"] = pd.to_numeric(
        dataset["selected_parameter_value"],
        errors="coerce",
    )
    dataset["test_mean_ic"] = pd.to_numeric(dataset["test_mean_ic"], errors="coerce")

    overall_summary = metadata.get("overall_summary")
    if not isinstance(overall_summary, dict):
        raise WorkflowError("walk-forward metadata.json is missing overall_summary.")

    summary_parameter_values = ",".join(
        _format_compact_numeric(value)
        for value in sorted(dataset["selected_parameter_value"].dropna().unique())
    )
    return {
        "summary_parameter_values": summary_parameter_values,
        "summary_scope": "overall_out_of_sample",
        "summary_cumulative_return": overall_summary.get("cumulative_return"),
        "summary_sharpe_ratio": overall_summary.get("sharpe_ratio"),
        "summary_mean_ic": dataset["test_mean_ic"].mean(),
    }


def _format_sweep_top_candidates_section(
    *,
    run_id: str,
    parameter: str,
    results: pd.DataFrame,
    top_k: int,
) -> str:
    """Format the top-k in-sample sweep candidates for one compared run."""
    dataset = validate_parameter_sweep_results(results)
    top_candidates = (
        dataset.sort_values(
            ["cumulative_return", "sharpe_ratio", "mean_ic", "parameter_value"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
        .head(top_k)
        .reset_index(drop=True)
    )
    formatted = top_candidates.loc[
        :,
        [
            "parameter_value",
            "cumulative_return",
            "max_drawdown",
            "sharpe_ratio",
            "mean_ic",
            "joint_coverage_ratio",
        ],
    ].copy()
    formatted["parameter_value"] = formatted["parameter_value"].map(
        _format_compact_numeric
    )
    for column in ("cumulative_return", "max_drawdown", "joint_coverage_ratio"):
        formatted[column] = formatted[column].map(_format_percent_or_nan)
    for column in ("sharpe_ratio", "mean_ic"):
        formatted[column] = formatted[column].map(_format_number_or_nan)

    return "\n".join(
        [
            f"Sweep Top Candidates: {run_id}",
            f"Parameter: {parameter}",
            f"Candidates Shown: {len(formatted)}/{len(dataset)}",
            formatted.to_string(index=False),
        ]
    )


def _format_walk_forward_folds_section(
    *,
    run_id: str,
    parameter: str,
    results: pd.DataFrame,
) -> str:
    """Format fold-level walk-forward diagnostics for one compared run."""
    dataset = validate_walk_forward_results(results)
    formatted = dataset.loc[
        :,
        [
            "fold_index",
            "selected_parameter_value",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "train_selection_score",
            "test_cumulative_return",
            "test_sharpe_ratio",
            "test_mean_ic",
            "test_joint_coverage_ratio",
        ],
    ].copy()
    formatted["fold_index"] = formatted["fold_index"].map(_format_compact_numeric)
    formatted["selected_parameter_value"] = formatted["selected_parameter_value"].map(
        _format_compact_numeric
    )

    selection_metric = str(dataset.loc[0, "selection_metric"])
    if selection_metric == "cumulative_return":
        formatted["train_selection_score"] = formatted["train_selection_score"].map(
            _format_percent_or_nan
        )
    else:
        formatted["train_selection_score"] = formatted["train_selection_score"].map(
            _format_number_or_nan
        )

    for column in ("test_cumulative_return", "test_joint_coverage_ratio"):
        formatted[column] = formatted[column].map(_format_percent_or_nan)
    for column in ("test_sharpe_ratio", "test_mean_ic"):
        formatted[column] = formatted[column].map(_format_number_or_nan)

    return "\n".join(
        [
            f"Walk-Forward Folds: {run_id}",
            f"Parameter: {parameter}",
            f"Selection Metric: {selection_metric}",
            formatted.to_string(index=False),
        ]
    )


def _load_json_file(path: Path, *, description: str) -> dict[str, Any]:
    """Load one JSON document from disk."""
    if not path.exists() or not path.is_file():
        raise WorkflowError(f"{description} file does not exist: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"Failed to read {description} file: {path}") from exc


def _load_csv_file(path: Path, *, description: str) -> pd.DataFrame:
    """Load one CSV document from disk."""
    if not path.exists() or not path.is_file():
        raise WorkflowError(f"{description} file does not exist: {path}")
    try:
        dataset = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise WorkflowError(f"Failed to read {description} file: {path}") from exc
    if dataset.empty:
        raise WorkflowError(f"{description} file does not contain any rows: {path}")
    return dataset


def _ensure_required_columns(
    frame: pd.DataFrame,
    required_columns: list[str],
    *,
    description: str,
) -> None:
    """Ensure a CSV-backed comparison frame has the required columns."""
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise WorkflowError(f"{description} are missing required columns: {missing_text}.")


def _stringify_candidate_values(values: Any) -> str:
    """Render candidate parameter values into a compact string."""
    if isinstance(values, list):
        return ",".join(str(value) for value in values)
    if pd.isna(values):
        return ""
    return str(values)


def _stringify_optional_text(value: Any) -> str:
    """Render optional textual values while preserving missing entries as empty strings."""
    if pd.isna(value):
        return ""
    return str(value)


def _append_run_index(path: Path, row: dict[str, Any]) -> Path:
    """Append one row to an experiment run index CSV."""
    if path.exists() and path.is_dir():
        raise WorkflowError(f"Run index path must be a file path, not a directory: {path}")

    new_row = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        all_columns = existing.columns.union(new_row.columns, sort=False)
        updated = existing.reindex(columns=all_columns).copy()
        updated.loc[len(updated)] = new_row.reindex(columns=all_columns).iloc[0]
    else:
        updated = new_row

    write_dataframe(updated, path)
    return path


def _slugify(value: str) -> str:
    """Convert free-form text into a conservative filesystem slug."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return slug or "run"


def _normalize_run_index_sort_key(sort_by: str) -> str:
    """Validate supported run index sort keys."""
    if sort_by not in {"created_at", "command", "parameter", "row_count", "overall_cumulative_return"}:
        raise WorkflowError(
            "sort_by must be one of {'created_at', 'command', 'parameter', 'row_count', 'overall_cumulative_return'}."
        )
    return sort_by


def _format_percent_or_nan(value: float) -> str:
    """Format a decimal value as a percentage, preserving missing values."""
    if pd.isna(value):
        return "NaN"
    return f"{value:.2%}"


def _format_number_or_nan(value: float) -> str:
    """Format a scalar numeric value, preserving missing values."""
    if pd.isna(value):
        return "NaN"
    return f"{value:.2f}"


def _format_compact_numeric(value: float) -> str:
    """Render numeric parameter values without unnecessary decimals."""
    if pd.isna(value):
        return "NaN"
    return str(int(value))


def require_signal_config(config: AlphaForgeConfig):
    """Require a signal section for signal-dependent workflows."""
    if config.signal is None:
        raise WorkflowError("The config must include a [signal] section for this command.")
    return config.signal


def require_portfolio_config(config: AlphaForgeConfig):
    """Require a portfolio section for portfolio-dependent workflows."""
    if config.portfolio is None:
        raise WorkflowError(
            "The config must include a [portfolio] section for this command."
        )
    return config.portfolio


def require_backtest_config(config: AlphaForgeConfig):
    """Require a backtest section for backtest-dependent workflows."""
    if config.backtest is None:
        raise WorkflowError(
            "The config must include a [backtest] section for this command."
        )
    return config.backtest


def _replace_signal_parameter(
    config: AlphaForgeConfig,
    *,
    parameter_name: str,
    parameter_value: int,
) -> AlphaForgeConfig:
    """Return a copy of config with one supported signal parameter replaced."""
    signal_config = require_signal_config(config)
    parameter_value = _normalize_positive_int(
        parameter_value,
        parameter_name=f"sweep value for {parameter_name}",
    )

    if signal_config.name in {"momentum", "mean_reversion"}:
        if parameter_name != "lookback":
            raise WorkflowError(
                f"signal '{signal_config.name}' only supports sweeping the 'lookback' parameter."
            )
        return replace(
            config,
            signal=replace(signal_config, lookback=parameter_value),
        )

    if parameter_name == "short_window":
        long_window = signal_config.long_window or 60
        if parameter_value >= long_window:
            raise WorkflowError(
                "sweep short_window values must stay smaller than the configured long_window."
            )
        return replace(
            config,
            signal=replace(signal_config, short_window=parameter_value),
        )

    if parameter_name == "long_window":
        short_window = signal_config.short_window or 20
        if parameter_value <= short_window:
            raise WorkflowError(
                "sweep long_window values must stay larger than the configured short_window."
            )
        return replace(
            config,
            signal=replace(signal_config, long_window=parameter_value),
        )

    raise WorkflowError(
        "trend signals only support sweeping 'short_window' or 'long_window'."
    )


def _evaluate_walk_forward_slice(
    *,
    signaled: pd.DataFrame,
    weighted: pd.DataFrame,
    signal_column: str,
    config: AlphaForgeConfig,
    evaluation_dates: list[pd.Timestamp],
) -> dict[str, object]:
    """Evaluate one date slice from a precomputed candidate panel."""
    if not evaluation_dates:
        raise WorkflowError("walk-forward evaluation dates must not be empty.")

    backtest_config = require_backtest_config(config)

    history_periods: int | None = backtest_config.signal_delay + 1
    if (
        backtest_config.rebalance_frequency != "daily"
        or backtest_config.max_turnover is not None
    ):
        history_periods = None

    augmented_dates = _select_augmented_dates(
        weighted,
        evaluation_dates=evaluation_dates,
        history_periods=history_periods,
    )
    weighted_slice = weighted.loc[weighted["date"].isin(augmented_dates)].copy()
    backtest = _run_backtest_with_config(weighted_slice, config=config)
    filtered_backtest = (
        backtest.loc[backtest["date"].isin(evaluation_dates)]
        .sort_values("date", kind="mergesort")
        .reset_index(drop=True)
    )
    if filtered_backtest.empty:
        raise WorkflowError("walk-forward backtest evaluation produced no rows.")

    diagnostics_slice = (
        signaled.loc[signaled["date"].isin(evaluation_dates)]
        .sort_values(["date", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
    ic_series = compute_ic_series(
        diagnostics_slice,
        signal_column=signal_column,
        forward_return_column=config.diagnostics.forward_return_column,
        method=config.diagnostics.ic_method,
        min_observations=config.diagnostics.min_observations,
    )

    return {
        "backtest": filtered_backtest,
        "performance_summary": summarize_backtest(filtered_backtest),
        "ic_summary": summarize_ic(ic_series),
        "coverage_summary": summarize_signal_coverage(
            diagnostics_slice,
            signal_column=signal_column,
            forward_return_column=config.diagnostics.forward_return_column,
        ),
    }


def _extract_walk_forward_selection_score(
    evaluation: dict[str, object],
    *,
    selection_metric: str,
) -> float:
    """Extract one train-slice selection metric from a fold evaluation."""
    performance_summary = evaluation["performance_summary"]
    ic_summary = evaluation["ic_summary"]

    if not isinstance(performance_summary, pd.Series) or not isinstance(
        ic_summary, pd.Series
    ):
        raise WorkflowError("walk-forward evaluation summaries must be pandas Series.")

    if selection_metric == "cumulative_return":
        score = performance_summary["cumulative_return"]
    elif selection_metric == "sharpe_ratio":
        score = performance_summary["sharpe_ratio"]
    else:
        score = ic_summary["mean_ic"]

    if pd.isna(score):
        return float("-inf")
    return float(score)


def _normalize_walk_forward_selection_metric(selection_metric: str) -> str:
    """Validate the train-slice metric used for walk-forward selection."""
    if selection_metric not in {"cumulative_return", "sharpe_ratio", "mean_ic"}:
        raise WorkflowError(
            "selection_metric must be one of {'cumulative_return', 'sharpe_ratio', 'mean_ic'}."
        )
    return selection_metric


def _build_walk_forward_folds(
    unique_dates: list[pd.Timestamp],
    *,
    train_periods: int,
    test_periods: int,
) -> list[dict[str, list[pd.Timestamp]]]:
    """Create rolling fixed-length train/test folds over unique dates."""
    if len(unique_dates) < train_periods + test_periods:
        raise WorkflowError(
            "walk-forward requires at least train_periods + test_periods unique dates."
        )

    folds: list[dict[str, list[pd.Timestamp]]] = []
    split_index = train_periods
    while split_index + test_periods <= len(unique_dates):
        folds.append(
            {
                "train_dates": unique_dates[split_index - train_periods : split_index],
                "test_dates": unique_dates[split_index : split_index + test_periods],
            }
        )
        split_index += test_periods

    if not folds:
        raise WorkflowError("walk-forward configuration produced no valid folds.")

    return folds


def _extract_unique_dates(frame: pd.DataFrame) -> list[pd.Timestamp]:
    """Extract sorted unique dates from a dated panel."""
    if "date" not in frame.columns:
        raise WorkflowError("dated workflow inputs must contain a 'date' column.")

    unique_dates = (
        pd.to_datetime(frame["date"], errors="coerce")
        .drop_duplicates()
        .sort_values(kind="mergesort")
        .tolist()
    )
    if not unique_dates:
        raise WorkflowError("dated workflow inputs must contain at least one date.")
    return unique_dates


def _select_augmented_dates(
    frame: pd.DataFrame,
    *,
    evaluation_dates: list[pd.Timestamp],
    history_periods: int | None,
) -> list[pd.Timestamp]:
    """Select evaluation dates plus enough prior history for conservative backtests."""
    all_dates = _extract_unique_dates(frame)
    start_date = evaluation_dates[0]
    end_date = evaluation_dates[-1]
    try:
        start_index = all_dates.index(start_date)
        end_index = all_dates.index(end_date)
    except ValueError as exc:
        raise WorkflowError("walk-forward evaluation dates must exist in the input panel.") from exc

    if history_periods is None:
        augmented_start = 0
    else:
        augmented_start = max(0, start_index - history_periods)
    return all_dates[augmented_start : end_index + 1]


def _normalize_sweep_values(values: list[int]) -> list[int]:
    """Validate and normalize an ordered list of sweep candidates."""
    if not values:
        raise WorkflowError("sweep values must contain at least one positive integer.")

    normalized: list[int] = []
    seen: set[int] = set()
    for value in values:
        numeric_value = _normalize_positive_int(value, parameter_name="sweep value")
        if numeric_value in seen:
            raise WorkflowError(
                f"sweep values must be unique; received duplicate value {numeric_value}."
            )
        normalized.append(numeric_value)
        seen.add(numeric_value)
    return normalized


def _normalize_positive_int(value: int, *, parameter_name: str) -> int:
    """Validate positive integer workflow parameters."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise WorkflowError(f"{parameter_name} must be a positive integer.")
    return value


def _validate_diagnostics_column(
    dataset: pd.DataFrame,
    config: AlphaForgeConfig,
) -> None:
    """Ensure the configured diagnostics label exists in the built dataset."""
    forward_return_column = config.diagnostics.forward_return_column
    if forward_return_column not in dataset.columns:
        raise WorkflowError(
            "The configured diagnostics.forward_return_column is not present in the built dataset: "
            f"{forward_return_column}."
        )

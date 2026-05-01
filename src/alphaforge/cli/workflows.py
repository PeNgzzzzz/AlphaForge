"""Config-driven CLI workflows for AlphaForge."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from alphaforge.analytics import (
    compute_grouped_ic_series,
    compute_ic_decay_series,
    compute_ic_decay_summary,
    compute_ic_series,
    compute_quantile_bucket_returns,
    compute_quantile_cumulative_returns,
    compute_quantile_spread_series,
    compute_rolling_ic_series,
    compute_signal_coverage_by_date,
    compute_signal_coverage_by_date_and_group,
    summarize_backtest,
    summarize_grouped_ic,
    summarize_ic,
    summarize_quantile_spread_stability,
    summarize_relative_performance,
    summarize_rolling_ic,
    summarize_signal_coverage,
    summarize_signal_coverage_by_group,
)
from alphaforge.backtest import run_daily_backtest
from alphaforge.cli.artifacts import (
    write_artifact_bundle,
    write_dataframe,
    write_indexed_artifact_bundle,
    write_json,
)
from alphaforge.cli.charts import (
    write_compare_chart_bundle,
    write_report_chart_bundle_from_context,
)
from alphaforge.cli.comparison import (
    build_compare_artifact_metadata,
    build_compare_runs_report,
    compare_indexed_runs,
    format_run_index_table,
    list_indexed_runs,
    rank_compare_runs,
    write_compare_artifact_bundle,
)
from alphaforge.cli.data_loading import (
    dataset_membership_indexes as _dataset_membership_indexes,
    dataset_requires_benchmark_returns as _dataset_requires_benchmark_returns,
    dataset_requires_fundamentals as _dataset_requires_fundamentals,
    dataset_requires_memberships as _dataset_requires_memberships,
    dataset_requires_shares_outstanding as _dataset_requires_shares_outstanding,
    dataset_requires_trading_status as _dataset_requires_trading_status,
    load_benchmark_returns_from_config,
    load_borrow_availability_from_config,
    load_classifications_from_config,
    load_corporate_actions_from_config,
    load_fundamentals_from_config,
    load_market_data_from_config,
    load_memberships_from_config,
    load_shares_outstanding_from_config,
    load_symbol_metadata_from_config,
    load_trading_calendar_from_config,
    load_trading_status_from_config,
)
from alphaforge.cli.errors import WorkflowError
from alphaforge.cli.reports import (
    _describe_signal_transform,
    _format_number_or_nan,
    _format_percent_or_nan,
    _summarize_benchmark_data,
    _summarize_data_quality,
    _summarize_market_data,
    _summarize_universe_eligibility,
    describe_benchmark_configuration,
    describe_benchmark_data,
    describe_borrow_availability_configuration,
    describe_borrow_availability_data,
    describe_classifications_configuration,
    describe_classifications_data,
    describe_corporate_actions_configuration,
    describe_corporate_actions_data,
    describe_data_quality,
    describe_diagnostics_overview,
    describe_execution_configuration,
    describe_execution_results,
    describe_fundamentals_configuration,
    describe_fundamentals_data,
    describe_market_data,
    describe_memberships_configuration,
    describe_memberships_data,
    describe_portfolio_constraints,
    describe_research_workflow,
    describe_shares_outstanding_configuration,
    describe_shares_outstanding_data,
    describe_symbol_metadata_configuration,
    describe_symbol_metadata_data,
    describe_trading_calendar_configuration,
    describe_trading_calendar_data,
    describe_trading_status_configuration,
    describe_trading_status_data,
    describe_universe_configuration,
    describe_universe_eligibility,
    build_report_metadata,
    render_report_text,
    write_report_html_page,
)
from alphaforge.cli.validation import normalize_positive_int
from alphaforge.cli.walk_forward import (
    build_walk_forward_artifact_metadata as _build_walk_forward_artifact_metadata,
    build_walk_forward_folds,
    evaluate_walk_forward_slice,
    extract_unique_dates,
    extract_walk_forward_selection_score,
    normalize_walk_forward_selection_metric,
)
from alphaforge.common import AlphaForgeConfig
from alphaforge.data import ensure_dates_on_trading_calendar
from alphaforge.features import (
    build_research_dataset,
    build_research_dataset_feature_metadata,
    build_research_feature_cache_metadata,
)
from alphaforge.portfolio import build_long_only_weights, build_long_short_weights
from alphaforge.risk import (
    compute_rolling_benchmark_risk,
    summarize_group_exposure,
    summarize_numeric_exposures,
    summarize_portfolio_diversification,
    summarize_risk,
    summarize_rolling_benchmark_risk,
)
from alphaforge.signals import (
    apply_cross_sectional_signal_transform,
    build_factor_signal,
    build_signal_pipeline_metadata,
)


def build_dataset_from_config(
    config: AlphaForgeConfig,
    *,
    market_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the research dataset from config-driven dataset settings."""
    if market_data is None:
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
        if _dataset_requires_fundamentals(config)
        else None
    )
    classifications = (
        load_classifications_from_config(config)
        if config.dataset.classification_fields
        else None
    )
    memberships = (
        load_memberships_from_config(config)
        if _dataset_requires_memberships(config)
        else None
    )
    borrow_availability = (
        load_borrow_availability_from_config(config)
        if config.dataset.borrow_fields
        else None
    )
    trading_status = (
        load_trading_status_from_config(config)
        if _dataset_requires_trading_status(config)
        else None
    )
    shares_outstanding = (
        load_shares_outstanding_from_config(config)
        if _dataset_requires_shares_outstanding(config)
        else None
    )
    benchmark_returns = (
        load_benchmark_returns_from_config(config)
        if _dataset_requires_benchmark_returns(config)
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
        trading_status=trading_status,
        shares_outstanding=shares_outstanding,
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
    trading_status: pd.DataFrame | None = None,
    shares_outstanding: pd.DataFrame | None = None,
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
        trading_status=trading_status,
        shares_outstanding=shares_outstanding,
        benchmark_returns=benchmark_returns,
        include_market_cap=config.dataset.include_market_cap,
        market_cap_bucket_count=config.dataset.market_cap_bucket_count,
        average_true_range_window=config.dataset.average_true_range_window,
        normalized_average_true_range_window=(
            config.dataset.normalized_average_true_range_window
        ),
        amihud_illiquidity_window=config.dataset.amihud_illiquidity_window,
        dollar_volume_shock_window=config.dataset.dollar_volume_shock_window,
        dollar_volume_zscore_window=config.dataset.dollar_volume_zscore_window,
        volume_shock_window=config.dataset.volume_shock_window,
        relative_volume_window=config.dataset.relative_volume_window,
        relative_dollar_volume_window=config.dataset.relative_dollar_volume_window,
        fundamental_metrics=(
            config.dataset.fundamental_metrics if fundamentals is not None else None
        ),
        valuation_metrics=(
            config.dataset.valuation_metrics
            if fundamentals is not None and config.dataset.valuation_metrics
            else None
        ),
        quality_ratio_metrics=(
            config.dataset.quality_ratio_metrics
            if fundamentals is not None and config.dataset.quality_ratio_metrics
            else None
        ),
        growth_metrics=(
            config.dataset.growth_metrics
            if fundamentals is not None and config.dataset.growth_metrics
            else None
        ),
        stability_ratio_metrics=(
            config.dataset.stability_ratio_metrics
            if fundamentals is not None and config.dataset.stability_ratio_metrics
            else None
        ),
        classification_fields=(
            config.dataset.classification_fields if classifications is not None else None
        ),
        membership_indexes=(
            _dataset_membership_indexes(config) if memberships is not None else None
        ),
        universe_required_membership_indexes=(
            universe_config.required_membership_indexes or None
            if universe_config is not None
            else None
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
        yang_zhang_volatility_window=config.dataset.yang_zhang_volatility_window,
        realized_volatility_window=config.dataset.realized_volatility_window,
        higher_moments_window=config.dataset.higher_moments_window,
        benchmark_residual_return_window=(
            config.dataset.benchmark_residual_return_window
        ),
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
        universe_require_tradable=(
            universe_config.require_tradable
            if universe_config is not None
            else False
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
    signaled, signal_column = build_factor_signal(
        dataset,
        name=signal_config.name,
        parameters=_signal_parameters_from_config(signal_config),
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


def _signal_parameters_from_config(signal_config: Any) -> dict[str, int]:
    """Extract explicit factor parameters from a validated signal config."""
    parameters: dict[str, int] = {}
    for parameter_name in ("lookback", "short_window", "long_window"):
        value = getattr(signal_config, parameter_name, None)
        if value is not None:
            parameters[parameter_name] = value
    return parameters


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
            position_cap_column=portfolio_config.position_cap_column,
            group_column=portfolio_config.group_column,
            max_group_weight=portfolio_config.max_group_weight,
            factor_exposure_bounds=_portfolio_factor_exposure_bounds(
                portfolio_config,
            ),
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
        position_cap_column=portfolio_config.position_cap_column,
        group_column=portfolio_config.group_column,
        max_group_weight=portfolio_config.max_group_weight,
        factor_exposure_bounds=_portfolio_factor_exposure_bounds(portfolio_config),
    )


def _portfolio_factor_exposure_bounds(
    portfolio_config: Any,
) -> tuple[tuple[str, float | None, float | None], ...]:
    """Convert config objects into portfolio-builder exposure bound tuples."""
    return tuple(
        (
            bound.column,
            bound.min_exposure,
            bound.max_exposure,
        )
        for bound in portfolio_config.factor_exposure_bounds
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
    report_text = render_report_text(context, config=config)
    workflow_configuration = _build_config_snapshot(config)
    research_metadata = _build_research_metadata_from_config(config)
    metadata = build_report_metadata(
        context,
        config=config,
        config_path=config_path,
        workflow_configuration=workflow_configuration,
        research_metadata=research_metadata,
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
    report_text = render_report_text(context, config=config)
    workflow_configuration = _build_config_snapshot(config)
    research_metadata = _build_research_metadata_from_config(config)
    metadata = build_report_metadata(
        context,
        config=config,
        config_path=config_path,
        workflow_configuration=workflow_configuration,
        research_metadata=research_metadata,
    )
    artifact_paths = write_artifact_bundle(
        artifact_dir,
        results=context["backtest"],
        report_text=report_text,
        metadata=metadata,
    )
    chart_bundle = write_report_chart_bundle_from_context(
        context,
        output_dir=Path(artifact_paths["artifact_dir"]) / "charts",
        workflow_configuration=workflow_configuration,
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
    html_path = write_report_html_page(
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
    return write_report_chart_bundle_from_context(
        context,
        output_dir=output_dir,
        workflow_configuration=_build_config_snapshot(config),
        config_path=config_path,
        command_name="plot-report",
    )


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
    shares_outstanding = (
        load_shares_outstanding_from_config(config)
        if config.shares_outstanding is not None
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
    trading_status = (
        load_trading_status_from_config(config)
        if config.trading_status is not None
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
    if config.shares_outstanding is not None:
        sections.append(describe_shares_outstanding_configuration(config))
        sections.append(
            describe_shares_outstanding_data(
                shares_outstanding,
                config=config,
            )
        )
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
    if config.trading_status is not None:
        sections.append(describe_trading_status_configuration(config))
        sections.append(describe_trading_status_data(trading_status, config=config))
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
                fundamentals if _dataset_requires_fundamentals(config) else None
            ),
            classifications=(
                classifications if config.dataset.classification_fields else None
            ),
            memberships=(
                memberships if _dataset_requires_memberships(config) else None
            ),
            borrow_availability=(
                borrow_availability if config.dataset.borrow_fields else None
            ),
            trading_status=(
                trading_status if _dataset_requires_trading_status(config) else None
            ),
            shares_outstanding=(
                shares_outstanding if _dataset_requires_shares_outstanding(config) else None
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
    require_backtest_config(config)
    candidate_values = _normalize_sweep_values(values)
    train_periods = normalize_positive_int(
        train_periods,
        parameter_name="train_periods",
    )
    test_periods = normalize_positive_int(
        test_periods,
        parameter_name="test_periods",
    )
    selection_metric = normalize_walk_forward_selection_metric(selection_metric)

    dataset = build_dataset_from_config(config)
    unique_dates = extract_unique_dates(dataset)
    folds = build_walk_forward_folds(
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
            train_evaluation = evaluate_walk_forward_slice(
                signaled=candidate["signaled"],
                weighted=candidate["weighted"],
                signal_column=candidate["signal_column"],
                config=config,
                evaluation_dates=fold["train_dates"],
            )
            candidate_score = extract_walk_forward_selection_score(
                train_evaluation,
                selection_metric=selection_metric,
            )
            if candidate_score > best_score:
                best_candidate = candidate
                best_score = candidate_score
                best_train_evaluation = train_evaluation

        if best_candidate is None or best_train_evaluation is None:
            raise WorkflowError("walk-forward selection could not choose a valid candidate.")

        test_evaluation = evaluate_walk_forward_slice(
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
    dataset = build_dataset_from_config(config, market_data=market_data)
    signaled, signal_column = add_signal_from_config(dataset, config)
    _validate_diagnostics_column(signaled, config)
    weighted = build_weights_from_config(
        signaled,
        score_column=signal_column,
        config=config,
    )
    portfolio_diversification_summary = summarize_portfolio_diversification(
        weighted,
        weight_column="portfolio_weight",
    )
    portfolio_group_exposure_summary = _summarize_portfolio_group_exposure(
        weighted,
        config=config,
    )
    portfolio_numeric_exposure_summary = _summarize_portfolio_numeric_exposures(
        weighted,
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
    rolling_ic_series = compute_rolling_ic_series(
        ic_series,
        window=config.diagnostics.rolling_ic_window,
    )
    rolling_ic_summary = summarize_rolling_ic(rolling_ic_series)
    diagnostics_forward_return_columns = _diagnostics_forward_return_columns(config)
    ic_decay_summary = compute_ic_decay_summary(
        signaled,
        signal_column=signal_column,
        forward_return_columns=diagnostics_forward_return_columns,
        method=config.diagnostics.ic_method,
        min_observations=config.diagnostics.min_observations,
    )
    ic_decay_series = compute_ic_decay_series(
        signaled,
        signal_column=signal_column,
        forward_return_columns=diagnostics_forward_return_columns,
        method=config.diagnostics.ic_method,
        min_observations=config.diagnostics.min_observations,
    )
    grouped_ic_series = _compute_grouped_ic_series_from_config(
        signaled,
        signal_column=signal_column,
        config=config,
    )
    grouped_ic_summary = summarize_grouped_ic(grouped_ic_series)
    quantile_summary = compute_quantile_bucket_returns(
        signaled,
        signal_column=signal_column,
        forward_return_column=config.diagnostics.forward_return_column,
        n_quantiles=config.diagnostics.n_quantiles,
        min_observations=config.diagnostics.min_observations,
    )
    quantile_cumulative_returns = compute_quantile_cumulative_returns(
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
    quantile_spread_stability = summarize_quantile_spread_stability(
        quantile_spread_series
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
    grouped_coverage_by_date = _compute_grouped_coverage_by_date_from_config(
        signaled,
        signal_column=signal_column,
        config=config,
    )
    grouped_coverage_summary = _compute_grouped_coverage_summary_from_config(
        signaled,
        signal_column=signal_column,
        config=config,
    )

    return {
        "market_data": market_data,
        "benchmark_data": benchmark_data,
        "dataset": dataset,
        "signaled": signaled,
        "signal_column": signal_column,
        "weighted": weighted,
        "backtest": backtest,
        "portfolio_diversification_summary": portfolio_diversification_summary,
        "portfolio_group_exposure_summary": portfolio_group_exposure_summary,
        "portfolio_numeric_exposure_summary": portfolio_numeric_exposure_summary,
        "rolling_benchmark_risk": rolling_benchmark_risk,
        "performance_summary": performance_summary,
        "relative_performance_summary": relative_performance_summary,
        "risk_summary": risk_summary,
        "benchmark_risk_summary": benchmark_risk_summary,
        "ic_series": ic_series,
        "ic_summary": ic_summary,
        "rolling_ic_series": rolling_ic_series,
        "rolling_ic_summary": rolling_ic_summary,
        "ic_decay_summary": ic_decay_summary,
        "ic_decay_series": ic_decay_series,
        "grouped_ic_series": grouped_ic_series,
        "grouped_ic_summary": grouped_ic_summary,
        "quantile_summary": quantile_summary,
        "quantile_cumulative_returns": quantile_cumulative_returns,
        "quantile_spread_series": quantile_spread_series,
        "quantile_spread_stability": quantile_spread_stability,
        "coverage_summary": coverage_summary,
        "coverage_by_date": coverage_by_date,
        "grouped_coverage_by_date": grouped_coverage_by_date,
        "grouped_coverage_summary": grouped_coverage_summary,
    }


def _summarize_portfolio_group_exposure(
    weighted: pd.DataFrame,
    *,
    config: AlphaForgeConfig,
) -> pd.DataFrame:
    """Summarize target-weight group exposure when a group constraint is configured."""
    portfolio = config.portfolio
    if portfolio is None or portfolio.group_column is None:
        return pd.DataFrame()
    return summarize_group_exposure(
        weighted,
        group_column=portfolio.group_column,
        weight_column="portfolio_weight",
    )


def _summarize_portfolio_numeric_exposures(
    weighted: pd.DataFrame,
    *,
    config: AlphaForgeConfig,
) -> pd.DataFrame:
    """Summarize target-weight numeric exposures when explicitly configured."""
    if not config.diagnostics.exposure_columns:
        return pd.DataFrame()
    return summarize_numeric_exposures(
        weighted,
        exposure_columns=config.diagnostics.exposure_columns,
        weight_column="portfolio_weight",
    )


def build_research_context_metadata(config: AlphaForgeConfig) -> dict[str, Any]:
    """Build lightweight research-context metadata for artifact bundles."""
    market_data = load_market_data_from_config(config)
    benchmark_data = (
        load_benchmark_returns_from_config(config)
        if config.benchmark is not None
        else None
    )
    shares_outstanding = (
        load_shares_outstanding_from_config(config)
        if _dataset_requires_shares_outstanding(config)
        else None
    )
    dataset = (
        build_dataset_from_market_data(
            market_data,
            config=config,
            shares_outstanding=shares_outstanding,
        )
        if config.universe is not None
        else None
    )
    research_metadata = _build_research_metadata_from_config(config)
    return {
        "workflow_configuration": _build_config_snapshot(config),
        **research_metadata,
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
    return _build_walk_forward_artifact_metadata(
        config_path=config_path,
        parameter_name=parameter_name,
        values=values,
        train_periods=train_periods,
        test_periods=test_periods,
        selection_metric=selection_metric,
        fold_results=fold_results,
        overall_summary=overall_summary,
        research_context=build_research_context_metadata(config),
    )


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
            "average_true_range_window": config.dataset.average_true_range_window,
            "normalized_average_true_range_window": (
                config.dataset.normalized_average_true_range_window
            ),
            "amihud_illiquidity_window": config.dataset.amihud_illiquidity_window,
            "dollar_volume_shock_window": config.dataset.dollar_volume_shock_window,
            "dollar_volume_zscore_window": config.dataset.dollar_volume_zscore_window,
            "volume_shock_window": config.dataset.volume_shock_window,
            "relative_volume_window": config.dataset.relative_volume_window,
            "relative_dollar_volume_window": config.dataset.relative_dollar_volume_window,
            "garman_klass_volatility_window": config.dataset.garman_klass_volatility_window,
            "parkinson_volatility_window": config.dataset.parkinson_volatility_window,
            "rogers_satchell_volatility_window": (
                config.dataset.rogers_satchell_volatility_window
            ),
            "yang_zhang_volatility_window": config.dataset.yang_zhang_volatility_window,
            "realized_volatility_window": config.dataset.realized_volatility_window,
            "higher_moments_window": config.dataset.higher_moments_window,
            "benchmark_residual_return_window": (
                config.dataset.benchmark_residual_return_window
            ),
            "benchmark_rolling_window": config.dataset.benchmark_rolling_window,
            "fundamental_metrics": list(config.dataset.fundamental_metrics),
            "valuation_metrics": list(config.dataset.valuation_metrics),
            "quality_ratio_metrics": [
                list(metric_pair) for metric_pair in config.dataset.quality_ratio_metrics
            ],
            "growth_metrics": list(config.dataset.growth_metrics),
            "stability_ratio_metrics": [
                list(metric_pair)
                for metric_pair in config.dataset.stability_ratio_metrics
            ],
            "classification_fields": list(config.dataset.classification_fields),
            "membership_indexes": list(config.dataset.membership_indexes),
            "borrow_fields": list(config.dataset.borrow_fields),
            "include_market_cap": config.dataset.include_market_cap,
            "market_cap_bucket_count": config.dataset.market_cap_bucket_count,
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
            "rolling_ic_window": config.diagnostics.rolling_ic_window,
            "group_columns": list(config.diagnostics.group_columns),
            "exposure_columns": list(config.diagnostics.exposure_columns),
        },
    }

    if config.signal is not None:
        snapshot["signal"] = {
            "name": config.signal.name,
            "lookback": config.signal.lookback,
            "short_window": config.signal.short_window,
            "long_window": config.signal.long_window,
            "winsorize_quantile": config.signal.winsorize_quantile,
            "clip_lower_bound": config.signal.clip_lower_bound,
            "clip_upper_bound": config.signal.clip_upper_bound,
            "cross_sectional_residualize_columns": list(
                config.signal.cross_sectional_residualize_columns
            ),
            "cross_sectional_neutralize_group_column": (
                config.signal.cross_sectional_neutralize_group_column
            ),
            "cross_sectional_normalization": (
                config.signal.cross_sectional_normalization
            ),
            "cross_sectional_group_column": (
                config.signal.cross_sectional_group_column
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
            "position_cap_column": config.portfolio.position_cap_column,
            "group_column": config.portfolio.group_column,
            "max_group_weight": config.portfolio.max_group_weight,
            "factor_exposure_bounds": [
                {
                    "column": bound.column,
                    "min": bound.min_exposure,
                    "max": bound.max_exposure,
                }
                for bound in config.portfolio.factor_exposure_bounds
            ],
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
    if config.shares_outstanding is not None:
        snapshot["shares_outstanding"] = {
            "path": str(config.shares_outstanding.path),
            "effective_date_column": (
                config.shares_outstanding.effective_date_column
            ),
            "shares_outstanding_column": (
                config.shares_outstanding.shares_outstanding_column
            ),
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
    if config.trading_status is not None:
        snapshot["trading_status"] = {
            "path": str(config.trading_status.path),
            "effective_date_column": config.trading_status.effective_date_column,
            "is_tradable_column": config.trading_status.is_tradable_column,
            "status_reason_column": config.trading_status.status_reason_column,
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
            "required_membership_indexes": list(
                config.universe.required_membership_indexes
            ),
            "require_tradable": config.universe.require_tradable,
            "lag": config.universe.lag,
            "average_volume_window": config.universe.average_volume_window,
            "average_dollar_volume_window": config.universe.average_dollar_volume_window,
        }
    return snapshot


def _build_dataset_feature_metadata_from_config(
    config: AlphaForgeConfig,
) -> list[dict[str, Any]]:
    """Build dataset feature provenance metadata from validated config."""
    universe_config = config.universe
    return build_research_dataset_feature_metadata(
        forward_horizons=config.dataset.forward_horizons,
        volatility_window=config.dataset.volatility_window,
        average_volume_window=config.dataset.average_volume_window,
        average_true_range_window=config.dataset.average_true_range_window,
        normalized_average_true_range_window=(
            config.dataset.normalized_average_true_range_window
        ),
        amihud_illiquidity_window=config.dataset.amihud_illiquidity_window,
        dollar_volume_shock_window=config.dataset.dollar_volume_shock_window,
        dollar_volume_zscore_window=config.dataset.dollar_volume_zscore_window,
        volume_shock_window=config.dataset.volume_shock_window,
        relative_volume_window=config.dataset.relative_volume_window,
        relative_dollar_volume_window=config.dataset.relative_dollar_volume_window,
        garman_klass_volatility_window=config.dataset.garman_klass_volatility_window,
        parkinson_volatility_window=config.dataset.parkinson_volatility_window,
        rogers_satchell_volatility_window=(
            config.dataset.rogers_satchell_volatility_window
        ),
        yang_zhang_volatility_window=config.dataset.yang_zhang_volatility_window,
        realized_volatility_window=config.dataset.realized_volatility_window,
        higher_moments_window=config.dataset.higher_moments_window,
        benchmark_residual_return_window=(
            config.dataset.benchmark_residual_return_window
        ),
        benchmark_rolling_window=config.dataset.benchmark_rolling_window,
        fundamental_metrics=config.dataset.fundamental_metrics,
        valuation_metrics=config.dataset.valuation_metrics,
        quality_ratio_metrics=config.dataset.quality_ratio_metrics,
        growth_metrics=config.dataset.growth_metrics,
        stability_ratio_metrics=config.dataset.stability_ratio_metrics,
        classification_fields=config.dataset.classification_fields,
        membership_indexes=_dataset_membership_indexes(config),
        borrow_fields=config.dataset.borrow_fields,
        include_market_cap=config.dataset.include_market_cap,
        market_cap_bucket_count=config.dataset.market_cap_bucket_count,
        universe_enabled=universe_config is not None,
        universe_lag=universe_config.lag if universe_config is not None else None,
        universe_average_volume_window=(
            universe_config.average_volume_window
            if universe_config is not None
            else None
        ),
        universe_average_dollar_volume_window=(
            universe_config.average_dollar_volume_window
            if universe_config is not None
            else None
        ),
        universe_required_membership_indexes=(
            universe_config.required_membership_indexes
            if universe_config is not None
            else ()
        ),
        universe_require_tradable=(
            universe_config.require_tradable
            if universe_config is not None
            else False
        ),
    )


def _build_signal_pipeline_metadata_from_config(
    config: AlphaForgeConfig,
) -> dict[str, Any]:
    """Build configured factor and transform metadata from validated config."""
    signal_config = require_signal_config(config)
    return build_signal_pipeline_metadata(
        factor_name=signal_config.name,
        factor_parameters=_signal_parameters_from_config(signal_config),
        winsorize_quantile=signal_config.winsorize_quantile,
        clip_lower_bound=signal_config.clip_lower_bound,
        clip_upper_bound=signal_config.clip_upper_bound,
        residualize_columns=signal_config.cross_sectional_residualize_columns,
        neutralize_group_column=(
            signal_config.cross_sectional_neutralize_group_column
        ),
        normalization=signal_config.cross_sectional_normalization,
        normalization_group_column=signal_config.cross_sectional_group_column,
    )


def _build_research_metadata_from_config(config: AlphaForgeConfig) -> dict[str, Any]:
    """Build shared research-plan metadata for report and experiment artifacts."""
    dataset_feature_metadata = _build_dataset_feature_metadata_from_config(config)
    signal_pipeline_metadata = _build_signal_pipeline_metadata_from_config(config)
    return {
        "dataset_feature_metadata": dataset_feature_metadata,
        "signal_pipeline_metadata": signal_pipeline_metadata,
        "feature_cache_metadata": build_research_feature_cache_metadata(
            dataset_feature_metadata=dataset_feature_metadata,
            signal_pipeline_metadata=signal_pipeline_metadata,
        ),
    }


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
        and signal_config.clip_lower_bound is None
        and signal_config.clip_upper_bound is None
        and not signal_config.cross_sectional_residualize_columns
        and signal_config.cross_sectional_neutralize_group_column is None
        and signal_config.cross_sectional_normalization == "none"
        and signal_config.cross_sectional_group_column is None
    ):
        return frame, signal_column

    return apply_cross_sectional_signal_transform(
        frame,
        score_column=signal_column,
        winsorize_quantile=signal_config.winsorize_quantile,
        clip_lower_bound=signal_config.clip_lower_bound,
        clip_upper_bound=signal_config.clip_upper_bound,
        residualize_columns=signal_config.cross_sectional_residualize_columns,
        neutralize_group_column=(
            signal_config.cross_sectional_neutralize_group_column
        ),
        normalization=signal_config.cross_sectional_normalization,
        normalization_group_column=signal_config.cross_sectional_group_column,
    )


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
    parameter_value = normalize_positive_int(
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


def _normalize_sweep_values(values: list[int]) -> list[int]:
    """Validate and normalize an ordered list of sweep candidates."""
    if not values:
        raise WorkflowError("sweep values must contain at least one positive integer.")

    normalized: list[int] = []
    seen: set[int] = set()
    for value in values:
        numeric_value = normalize_positive_int(value, parameter_name="sweep value")
        if numeric_value in seen:
            raise WorkflowError(
                f"sweep values must be unique; received duplicate value {numeric_value}."
            )
        normalized.append(numeric_value)
        seen.add(numeric_value)
    return normalized


def _compute_grouped_ic_series_from_config(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    config: AlphaForgeConfig,
) -> pd.DataFrame:
    """Compute grouped IC diagnostics for explicitly configured group columns."""
    frames = [
        compute_grouped_ic_series(
            frame,
            signal_column=signal_column,
            forward_return_column=config.diagnostics.forward_return_column,
            group_column=group_column,
            method=config.diagnostics.ic_method,
            min_observations=config.diagnostics.min_observations,
        )
        for group_column in config.diagnostics.group_columns
    ]
    if not frames:
        return pd.DataFrame(
            columns=[
                "date",
                "group_column",
                "group_value",
                "ic",
                "observations",
                "method",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _compute_grouped_coverage_by_date_from_config(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    config: AlphaForgeConfig,
) -> pd.DataFrame:
    """Compute grouped coverage diagnostics for configured group columns."""
    frames = [
        compute_signal_coverage_by_date_and_group(
            frame,
            signal_column=signal_column,
            forward_return_column=config.diagnostics.forward_return_column,
            group_column=group_column,
        )
        for group_column in config.diagnostics.group_columns
    ]
    if not frames:
        return pd.DataFrame(
            columns=[
                "date",
                "group_column",
                "group_value",
                "total_rows",
                "signal_non_null_rows",
                "forward_return_non_null_rows",
                "usable_rows",
                "signal_coverage_ratio",
                "forward_return_coverage_ratio",
                "joint_coverage_ratio",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _compute_grouped_coverage_summary_from_config(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    config: AlphaForgeConfig,
) -> pd.DataFrame:
    """Summarize grouped coverage diagnostics for configured group columns."""
    frames = [
        summarize_signal_coverage_by_group(
            frame,
            signal_column=signal_column,
            forward_return_column=config.diagnostics.forward_return_column,
            group_column=group_column,
        )
        for group_column in config.diagnostics.group_columns
    ]
    if not frames:
        return pd.DataFrame(
            columns=[
                "group_column",
                "group_value",
                "dates",
                "total_rows",
                "signal_non_null_rows",
                "forward_return_non_null_rows",
                "usable_rows",
                "signal_coverage_ratio",
                "forward_return_coverage_ratio",
                "joint_coverage_ratio",
                "average_daily_usable_rows",
                "minimum_daily_usable_rows",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _diagnostics_forward_return_columns(config: AlphaForgeConfig) -> tuple[str, ...]:
    """Return configured label columns for IC decay summaries."""
    return tuple(
        f"forward_return_{horizon}d"
        for horizon in config.dataset.forward_horizons
    )


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

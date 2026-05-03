"""Config-driven data loading helpers for CLI workflows."""

from __future__ import annotations

import pandas as pd

from alphaforge.cli.errors import WorkflowError
from alphaforge.common import AlphaForgeConfig
from alphaforge.data import (
    apply_split_adjustments,
    load_benchmark_returns,
    load_borrow_availability,
    load_classifications,
    load_corporate_actions,
    load_fundamentals,
    load_memberships,
    load_ohlcv,
    load_shares_outstanding,
    load_symbol_metadata,
    load_trading_calendar,
    load_trading_status,
)

__all__ = [
    "load_market_data_from_config",
    "load_trading_calendar_from_config",
    "load_symbol_metadata_from_config",
    "load_corporate_actions_from_config",
    "load_fundamentals_from_config",
    "load_shares_outstanding_from_config",
    "load_classifications_from_config",
    "load_memberships_from_config",
    "load_borrow_availability_from_config",
    "load_trading_status_from_config",
    "load_benchmark_returns_from_config",
]


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


def load_shares_outstanding_from_config(config: AlphaForgeConfig) -> pd.DataFrame:
    """Load and validate the optional shares-outstanding input from config."""
    shares_outstanding_config = config.shares_outstanding
    if shares_outstanding_config is None:
        raise WorkflowError(
            "The config does not include a [shares_outstanding] section."
        )

    return load_shares_outstanding(
        shares_outstanding_config.path,
        effective_date_column=shares_outstanding_config.effective_date_column,
        shares_outstanding_column=(
            shares_outstanding_config.shares_outstanding_column
        ),
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


def load_trading_status_from_config(config: AlphaForgeConfig) -> pd.DataFrame:
    """Load and validate the optional trading status input from config."""
    trading_status_config = config.trading_status
    if trading_status_config is None:
        raise WorkflowError("The config does not include a [trading_status] section.")

    return load_trading_status(
        trading_status_config.path,
        effective_date_column=trading_status_config.effective_date_column,
        is_tradable_column=trading_status_config.is_tradable_column,
        status_reason_column=trading_status_config.status_reason_column,
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


def dataset_requires_benchmark_returns(config: AlphaForgeConfig) -> bool:
    """Return whether dataset construction needs benchmark returns."""
    return (
        config.dataset.benchmark_rolling_window is not None
        or config.dataset.benchmark_residual_return_window is not None
    )


def dataset_requires_fundamentals(config: AlphaForgeConfig) -> bool:
    """Return whether dataset construction needs fundamentals."""
    return bool(
        config.dataset.fundamental_metrics
        or config.dataset.valuation_metrics
        or config.dataset.quality_ratio_metrics
        or config.dataset.growth_metrics
        or config.dataset.stability_ratio_metrics
    )


def dataset_requires_shares_outstanding(config: AlphaForgeConfig) -> bool:
    """Return whether dataset construction needs shares-outstanding data."""
    return config.dataset.include_market_cap


def dataset_requires_memberships(config: AlphaForgeConfig) -> bool:
    """Return whether dataset construction needs index memberships."""
    return bool(dataset_membership_indexes(config))


def dataset_requires_trading_status(config: AlphaForgeConfig) -> bool:
    """Return whether dataset construction needs trading status data."""
    return bool(
        config.universe is not None and config.universe.require_tradable
    ) or bool(
        config.trading_status is not None
        and config.backtest is not None
        and config.backtest.tradable_column == "trading_is_tradable"
    )


def dataset_membership_indexes(config: AlphaForgeConfig) -> tuple[str, ...]:
    """Return all configured membership indexes needed by dataset construction."""
    merged: list[str] = []
    seen: set[str] = set()
    universe_indexes = (
        config.universe.required_membership_indexes
        if config.universe is not None
        else ()
    )
    for index_name in (*config.dataset.membership_indexes, *universe_indexes):
        if index_name in seen:
            continue
        merged.append(index_name)
        seen.add(index_name)
    return tuple(merged)

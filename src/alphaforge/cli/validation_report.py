"""Validation report assembly for the validate-data CLI command."""

from __future__ import annotations

from alphaforge.cli.data_loading import (
    dataset_requires_fundamentals,
    dataset_requires_memberships,
    dataset_requires_shares_outstanding,
    dataset_requires_trading_status,
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
from alphaforge.cli.pipeline import build_dataset_from_market_data
from alphaforge.cli.reports import (
    describe_benchmark_configuration,
    describe_benchmark_data,
    describe_borrow_availability_configuration,
    describe_borrow_availability_data,
    describe_classifications_configuration,
    describe_classifications_data,
    describe_corporate_actions_configuration,
    describe_corporate_actions_data,
    describe_data_quality,
    describe_fundamentals_configuration,
    describe_fundamentals_data,
    describe_market_data,
    describe_memberships_configuration,
    describe_memberships_data,
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
)
from alphaforge.common import AlphaForgeConfig
from alphaforge.data import ensure_dates_on_trading_calendar


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
                fundamentals if dataset_requires_fundamentals(config) else None
            ),
            classifications=(
                classifications if config.dataset.classification_fields else None
            ),
            memberships=(
                memberships if dataset_requires_memberships(config) else None
            ),
            borrow_availability=(
                borrow_availability if config.dataset.borrow_fields else None
            ),
            trading_status=(
                trading_status if dataset_requires_trading_status(config) else None
            ),
            shares_outstanding=(
                shares_outstanding
                if dataset_requires_shares_outstanding(config)
                else None
            ),
        )
        sections.append(describe_universe_configuration(config))
        sections.append(describe_universe_eligibility(dataset))
    return "\n\n".join(section for section in sections if section)


__all__ = ["build_validate_data_text"]

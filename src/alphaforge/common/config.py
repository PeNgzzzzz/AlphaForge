"""Configuration loading for AlphaForge CLI workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from alphaforge.common.errors import AlphaForgeError
from alphaforge.common.validation import (
    normalize_choice_string as _common_choice_string,
    normalize_finite_float as _common_finite_float,
    normalize_non_negative_float as _common_non_negative_float,
    normalize_non_empty_string as _common_non_empty_string,
    normalize_optional_non_negative_float as _common_optional_non_negative_float,
    normalize_optional_positive_float as _common_optional_positive_float,
    normalize_positive_float as _common_positive_float,
    normalize_positive_int as _common_positive_int,
    normalize_unique_non_empty_string_pair_sequence as _common_string_pair_sequence,
    normalize_unique_non_empty_string_sequence as _common_string_sequence,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.9 fallback
    import tomli as tomllib


class ConfigError(AlphaForgeError):
    """Raised when pipeline configuration files are missing or invalid."""


_SUPPORTED_DATA_SUFFIXES = {".csv", ".parquet"}


@dataclass(frozen=True)
class DataConfig:
    """Raw market data input configuration."""

    path: Path
    price_adjustment: str = "raw"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Optional benchmark return-series configuration."""

    path: Path
    name: str = "Benchmark"
    return_column: str = "benchmark_return"
    rolling_window: int = 20


@dataclass(frozen=True)
class SymbolMetadataConfig:
    """Optional symbol metadata input configuration."""

    path: Path
    listing_date_column: str = "listing_date"
    delisting_date_column: str = "delisting_date"


@dataclass(frozen=True)
class CorporateActionsConfig:
    """Optional corporate-actions input configuration."""

    path: Path
    ex_date_column: str = "ex_date"
    action_type_column: str = "action_type"
    split_ratio_column: str = "split_ratio"
    cash_amount_column: str = "cash_amount"


@dataclass(frozen=True)
class FundamentalsConfig:
    """Optional long-form fundamentals input configuration."""

    path: Path
    period_end_column: str = "period_end_date"
    release_date_column: str = "release_date"
    metric_name_column: str = "metric_name"
    metric_value_column: str = "metric_value"


@dataclass(frozen=True)
class SharesOutstandingConfig:
    """Optional shares-outstanding input configuration."""

    path: Path
    effective_date_column: str = "effective_date"
    shares_outstanding_column: str = "shares_outstanding"


@dataclass(frozen=True)
class ClassificationsConfig:
    """Optional sector/industry classifications input configuration."""

    path: Path
    effective_date_column: str = "effective_date"
    sector_column: str = "sector"
    industry_column: str = "industry"


@dataclass(frozen=True)
class MembershipsConfig:
    """Optional index membership input configuration."""

    path: Path
    effective_date_column: str = "effective_date"
    index_column: str = "index_name"
    is_member_column: str = "is_member"


@dataclass(frozen=True)
class BorrowAvailabilityConfig:
    """Optional borrow availability input configuration."""

    path: Path
    effective_date_column: str = "effective_date"
    is_borrowable_column: str = "is_borrowable"
    borrow_fee_bps_column: str = "borrow_fee_bps"


@dataclass(frozen=True)
class TradingStatusConfig:
    """Optional trading status input configuration."""

    path: Path
    effective_date_column: str = "effective_date"
    is_tradable_column: str = "is_tradable"
    status_reason_column: str = "status_reason"


@dataclass(frozen=True)
class CalendarConfig:
    """Optional trading calendar input configuration."""

    path: Path
    name: str = "Trading Calendar"
    date_column: str = "date"


@dataclass(frozen=True)
class DatasetConfig:
    """Research dataset construction settings."""

    forward_horizons: tuple[int, ...] = (1,)
    volatility_window: int = 20
    average_volume_window: int = 20
    average_true_range_window: int | None = None
    normalized_average_true_range_window: int | None = None
    amihud_illiquidity_window: int | None = None
    dollar_volume_shock_window: int | None = None
    dollar_volume_zscore_window: int | None = None
    volume_shock_window: int | None = None
    relative_volume_window: int | None = None
    relative_dollar_volume_window: int | None = None
    garman_klass_volatility_window: int | None = None
    parkinson_volatility_window: int | None = None
    rogers_satchell_volatility_window: int | None = None
    yang_zhang_volatility_window: int | None = None
    realized_volatility_window: int | None = None
    higher_moments_window: int | None = None
    benchmark_residual_return_window: int | None = None
    benchmark_rolling_window: int | None = None
    fundamental_metrics: tuple[str, ...] = ()
    valuation_metrics: tuple[str, ...] = ()
    quality_ratio_metrics: tuple[tuple[str, str], ...] = ()
    growth_metrics: tuple[str, ...] = ()
    stability_ratio_metrics: tuple[tuple[str, str], ...] = ()
    classification_fields: tuple[str, ...] = ()
    membership_indexes: tuple[str, ...] = ()
    borrow_fields: tuple[str, ...] = ()
    include_market_cap: bool = False
    market_cap_bucket_count: int | None = None


@dataclass(frozen=True)
class UniverseConfig:
    """Optional tradability-aware universe filtering settings."""

    min_price: float | None = None
    min_average_volume: float | None = None
    min_average_dollar_volume: float | None = None
    min_listing_history_days: int | None = None
    required_membership_indexes: tuple[str, ...] = ()
    require_tradable: bool = False
    lag: int = 1
    average_volume_window: int | None = None
    average_dollar_volume_window: int | None = None


@dataclass(frozen=True)
class SignalConfig:
    """Signal generation settings."""

    name: str
    lookback: int | None = None
    short_window: int | None = None
    long_window: int | None = None
    winsorize_quantile: float | None = None
    clip_lower_bound: float | None = None
    clip_upper_bound: float | None = None
    cross_sectional_residualize_columns: tuple[str, ...] = ()
    cross_sectional_neutralize_group_column: str | None = None
    cross_sectional_normalization: str = "none"
    cross_sectional_group_column: str | None = None


@dataclass(frozen=True)
class FactorExposureBoundConfig:
    """Shrink-only target-weight factor exposure bound configuration."""

    column: str
    min_exposure: float | None = None
    max_exposure: float | None = None


@dataclass(frozen=True)
class PortfolioConfig:
    """Portfolio construction settings."""

    construction: str = "long_only"
    top_n: int = 1
    bottom_n: int | None = None
    weighting: str = "equal"
    exposure: float = 1.0
    long_exposure: float = 1.0
    short_exposure: float = 1.0
    max_position_weight: float | None = None
    position_cap_column: str | None = None
    group_column: str | None = None
    max_group_weight: float | None = None
    factor_exposure_bounds: tuple[FactorExposureBoundConfig, ...] = ()


@dataclass(frozen=True)
class BacktestConfig:
    """Daily backtest settings."""

    signal_delay: int = 1
    fill_timing: str = "close"
    rebalance_frequency: str = "daily"
    transaction_cost_bps: float | None = None
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    commission_bps_column: str | None = None
    slippage_bps_column: str | None = None
    max_trade_weight_column: str | None = None
    max_turnover: float | None = None
    initial_nav: float = 1.0


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Factor diagnostics settings."""

    forward_return_column: str = "forward_return_1d"
    ic_method: str = "pearson"
    n_quantiles: int = 5
    min_observations: int = 5
    rolling_ic_window: int = 20
    group_columns: tuple[str, ...] = ()
    exposure_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class AlphaForgeConfig:
    """Top-level pipeline configuration."""

    data: DataConfig
    calendar: CalendarConfig | None
    symbol_metadata: SymbolMetadataConfig | None
    corporate_actions: CorporateActionsConfig | None
    fundamentals: FundamentalsConfig | None
    shares_outstanding: SharesOutstandingConfig | None
    classifications: ClassificationsConfig | None
    memberships: MembershipsConfig | None
    borrow_availability: BorrowAvailabilityConfig | None
    trading_status: TradingStatusConfig | None
    benchmark: BenchmarkConfig | None
    dataset: DatasetConfig
    universe: UniverseConfig | None
    signal: SignalConfig | None
    portfolio: PortfolioConfig | None
    backtest: BacktestConfig | None
    diagnostics: DiagnosticsConfig


def load_pipeline_config(path: str | Path) -> AlphaForgeConfig:
    """Load and validate an AlphaForge pipeline TOML config file."""
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")
    if not config_path.is_file():
        raise ConfigError(f"Config path is not a file: {config_path}")

    with config_path.open("rb") as handle:
        try:
            raw = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"Invalid TOML in config file: {config_path}") from exc

    if not isinstance(raw, Mapping):
        raise ConfigError("Config file must parse to a mapping.")

    data_section = _require_mapping(raw, section_name="data")
    calendar_section = _optional_mapping(raw, section_name="calendar")
    symbol_metadata_section = _optional_mapping(raw, section_name="symbol_metadata")
    corporate_actions_section = _optional_mapping(raw, section_name="corporate_actions")
    fundamentals_section = _optional_mapping(raw, section_name="fundamentals")
    shares_outstanding_section = _optional_mapping(
        raw,
        section_name="shares_outstanding",
    )
    classifications_section = _optional_mapping(raw, section_name="classifications")
    memberships_section = _optional_mapping(raw, section_name="memberships")
    borrow_availability_section = _optional_mapping(
        raw,
        section_name="borrow_availability",
    )
    trading_status_section = _optional_mapping(raw, section_name="trading_status")
    benchmark_section = _optional_mapping(raw, section_name="benchmark")
    dataset_section = _optional_mapping(raw, section_name="dataset")
    universe_section = _optional_mapping(raw, section_name="universe")
    signal_section = _optional_mapping(raw, section_name="signal")
    portfolio_section = _optional_mapping(raw, section_name="portfolio")
    backtest_section = _optional_mapping(raw, section_name="backtest")
    diagnostics_section = _optional_mapping(raw, section_name="diagnostics")

    config = AlphaForgeConfig(
        data=_parse_data_config(data_section, config_path=config_path),
        calendar=_parse_calendar_config(
            calendar_section,
            config_path=config_path,
        ),
        symbol_metadata=_parse_symbol_metadata_config(
            symbol_metadata_section,
            config_path=config_path,
        ),
        corporate_actions=_parse_corporate_actions_config(
            corporate_actions_section,
            config_path=config_path,
        ),
        fundamentals=_parse_fundamentals_config(
            fundamentals_section,
            config_path=config_path,
        ),
        shares_outstanding=_parse_shares_outstanding_config(
            shares_outstanding_section,
            config_path=config_path,
        ),
        classifications=_parse_classifications_config(
            classifications_section,
            config_path=config_path,
        ),
        memberships=_parse_memberships_config(
            memberships_section,
            config_path=config_path,
        ),
        borrow_availability=_parse_borrow_availability_config(
            borrow_availability_section,
            config_path=config_path,
        ),
        trading_status=_parse_trading_status_config(
            trading_status_section,
            config_path=config_path,
        ),
        benchmark=_parse_benchmark_config(
            benchmark_section,
            config_path=config_path,
        ),
        dataset=_parse_dataset_config(dataset_section),
        universe=_parse_universe_config(universe_section),
        signal=_parse_signal_config(signal_section),
        portfolio=_parse_portfolio_config(portfolio_section),
        backtest=_parse_backtest_config(backtest_section),
        diagnostics=_parse_diagnostics_config(diagnostics_section),
    )
    _validate_cross_section_settings(config)
    return config


def _parse_data_config(
    section: Mapping[str, Any], *, config_path: Path
) -> DataConfig:
    """Parse the required data input section."""
    return DataConfig(
        path=_parse_supported_input_path(
            section.get("path"),
            field_name="data.path",
            config_path=config_path,
        ),
        price_adjustment=_normalize_choice_string(
            section.get("price_adjustment", "raw"),
            "data.price_adjustment",
            choices={"raw", "split_adjusted"},
        ),
    )


def _parse_benchmark_config(
    section: Mapping[str, Any] | None,
    *,
    config_path: Path,
) -> BenchmarkConfig | None:
    """Parse the optional benchmark return-series section."""
    if section is None:
        return None

    return BenchmarkConfig(
        path=_parse_supported_input_path(
            section.get("path"),
            field_name="benchmark.path",
            config_path=config_path,
        ),
        name=_normalize_non_empty_string(
            section.get("name", "Benchmark"),
            "benchmark.name",
        ),
        return_column=_normalize_non_empty_string(
            section.get("return_column", "benchmark_return"),
            "benchmark.return_column",
        ),
        rolling_window=_normalize_positive_int(
            section.get("rolling_window", 20),
            "benchmark.rolling_window",
        ),
    )


def _parse_symbol_metadata_config(
    section: Mapping[str, Any] | None,
    *,
    config_path: Path,
) -> SymbolMetadataConfig | None:
    """Parse the optional symbol metadata section."""
    if section is None:
        return None

    return SymbolMetadataConfig(
        path=_parse_supported_input_path(
            section.get("path"),
            field_name="symbol_metadata.path",
            config_path=config_path,
        ),
        listing_date_column=_normalize_non_empty_string(
            section.get("listing_date_column", "listing_date"),
            "symbol_metadata.listing_date_column",
        ),
        delisting_date_column=_normalize_non_empty_string(
            section.get("delisting_date_column", "delisting_date"),
            "symbol_metadata.delisting_date_column",
        ),
    )


def _parse_corporate_actions_config(
    section: Mapping[str, Any] | None,
    *,
    config_path: Path,
) -> CorporateActionsConfig | None:
    """Parse the optional corporate-actions section."""
    if section is None:
        return None

    return CorporateActionsConfig(
        path=_parse_supported_input_path(
            section.get("path"),
            field_name="corporate_actions.path",
            config_path=config_path,
        ),
        ex_date_column=_normalize_non_empty_string(
            section.get("ex_date_column", "ex_date"),
            "corporate_actions.ex_date_column",
        ),
        action_type_column=_normalize_non_empty_string(
            section.get("action_type_column", "action_type"),
            "corporate_actions.action_type_column",
        ),
        split_ratio_column=_normalize_non_empty_string(
            section.get("split_ratio_column", "split_ratio"),
            "corporate_actions.split_ratio_column",
        ),
        cash_amount_column=_normalize_non_empty_string(
            section.get("cash_amount_column", "cash_amount"),
            "corporate_actions.cash_amount_column",
        ),
    )


def _parse_fundamentals_config(
    section: Mapping[str, Any] | None,
    *,
    config_path: Path,
) -> FundamentalsConfig | None:
    """Parse the optional fundamentals section."""
    if section is None:
        return None

    return FundamentalsConfig(
        path=_parse_supported_input_path(
            section.get("path"),
            field_name="fundamentals.path",
            config_path=config_path,
        ),
        period_end_column=_normalize_non_empty_string(
            section.get("period_end_column", "period_end_date"),
            "fundamentals.period_end_column",
        ),
        release_date_column=_normalize_non_empty_string(
            section.get("release_date_column", "release_date"),
            "fundamentals.release_date_column",
        ),
        metric_name_column=_normalize_non_empty_string(
            section.get("metric_name_column", "metric_name"),
            "fundamentals.metric_name_column",
        ),
        metric_value_column=_normalize_non_empty_string(
            section.get("metric_value_column", "metric_value"),
            "fundamentals.metric_value_column",
        ),
    )


def _parse_shares_outstanding_config(
    section: Mapping[str, Any] | None,
    *,
    config_path: Path,
) -> SharesOutstandingConfig | None:
    """Parse the optional shares-outstanding section."""
    if section is None:
        return None

    return SharesOutstandingConfig(
        path=_parse_supported_input_path(
            section.get("path"),
            field_name="shares_outstanding.path",
            config_path=config_path,
        ),
        effective_date_column=_normalize_non_empty_string(
            section.get("effective_date_column", "effective_date"),
            "shares_outstanding.effective_date_column",
        ),
        shares_outstanding_column=_normalize_non_empty_string(
            section.get("shares_outstanding_column", "shares_outstanding"),
            "shares_outstanding.shares_outstanding_column",
        ),
    )


def _parse_classifications_config(
    section: Mapping[str, Any] | None,
    *,
    config_path: Path,
) -> ClassificationsConfig | None:
    """Parse the optional classifications section."""
    if section is None:
        return None

    return ClassificationsConfig(
        path=_parse_supported_input_path(
            section.get("path"),
            field_name="classifications.path",
            config_path=config_path,
        ),
        effective_date_column=_normalize_non_empty_string(
            section.get("effective_date_column", "effective_date"),
            "classifications.effective_date_column",
        ),
        sector_column=_normalize_non_empty_string(
            section.get("sector_column", "sector"),
            "classifications.sector_column",
        ),
        industry_column=_normalize_non_empty_string(
            section.get("industry_column", "industry"),
            "classifications.industry_column",
        ),
    )


def _parse_memberships_config(
    section: Mapping[str, Any] | None,
    *,
    config_path: Path,
) -> MembershipsConfig | None:
    """Parse the optional memberships section."""
    if section is None:
        return None

    return MembershipsConfig(
        path=_parse_supported_input_path(
            section.get("path"),
            field_name="memberships.path",
            config_path=config_path,
        ),
        effective_date_column=_normalize_non_empty_string(
            section.get("effective_date_column", "effective_date"),
            "memberships.effective_date_column",
        ),
        index_column=_normalize_non_empty_string(
            section.get("index_column", "index_name"),
            "memberships.index_column",
        ),
        is_member_column=_normalize_non_empty_string(
            section.get("is_member_column", "is_member"),
            "memberships.is_member_column",
        ),
    )


def _parse_borrow_availability_config(
    section: Mapping[str, Any] | None,
    *,
    config_path: Path,
) -> BorrowAvailabilityConfig | None:
    """Parse the optional borrow availability section."""
    if section is None:
        return None

    return BorrowAvailabilityConfig(
        path=_parse_supported_input_path(
            section.get("path"),
            field_name="borrow_availability.path",
            config_path=config_path,
        ),
        effective_date_column=_normalize_non_empty_string(
            section.get("effective_date_column", "effective_date"),
            "borrow_availability.effective_date_column",
        ),
        is_borrowable_column=_normalize_non_empty_string(
            section.get("is_borrowable_column", "is_borrowable"),
            "borrow_availability.is_borrowable_column",
        ),
        borrow_fee_bps_column=_normalize_non_empty_string(
            section.get("borrow_fee_bps_column", "borrow_fee_bps"),
            "borrow_availability.borrow_fee_bps_column",
        ),
    )


def _parse_trading_status_config(
    section: Mapping[str, Any] | None,
    *,
    config_path: Path,
) -> TradingStatusConfig | None:
    """Parse the optional trading status section."""
    if section is None:
        return None

    return TradingStatusConfig(
        path=_parse_supported_input_path(
            section.get("path"),
            field_name="trading_status.path",
            config_path=config_path,
        ),
        effective_date_column=_normalize_non_empty_string(
            section.get("effective_date_column", "effective_date"),
            "trading_status.effective_date_column",
        ),
        is_tradable_column=_normalize_non_empty_string(
            section.get("is_tradable_column", "is_tradable"),
            "trading_status.is_tradable_column",
        ),
        status_reason_column=_normalize_non_empty_string(
            section.get("status_reason_column", "status_reason"),
            "trading_status.status_reason_column",
        ),
    )


def _parse_calendar_config(
    section: Mapping[str, Any] | None,
    *,
    config_path: Path,
) -> CalendarConfig | None:
    """Parse the optional trading calendar section."""
    if section is None:
        return None

    return CalendarConfig(
        path=_parse_supported_input_path(
            section.get("path"),
            field_name="calendar.path",
            config_path=config_path,
        ),
        name=_normalize_non_empty_string(
            section.get("name", "Trading Calendar"),
            "calendar.name",
        ),
        date_column=_normalize_non_empty_string(
            section.get("date_column", "date"),
            "calendar.date_column",
        ),
    )


def _parse_dataset_config(section: Mapping[str, Any] | None) -> DatasetConfig:
    """Parse the optional dataset section."""
    if section is None:
        return DatasetConfig()

    forward_horizons_raw = section.get("forward_horizons", [1])
    if isinstance(forward_horizons_raw, int):
        forward_horizons = (_normalize_positive_int(forward_horizons_raw, "dataset.forward_horizons"),)
    elif isinstance(forward_horizons_raw, list):
        forward_horizons = tuple(
            _normalize_positive_int(value, "dataset.forward_horizons")
            for value in forward_horizons_raw
        )
        if not forward_horizons:
            raise ConfigError("dataset.forward_horizons must contain at least one value.")
    else:
        raise ConfigError("dataset.forward_horizons must be an integer or a list of integers.")

    fundamental_metrics = _normalize_string_list(
        section.get("fundamental_metrics", []),
        "dataset.fundamental_metrics",
    )

    valuation_metrics = _normalize_string_list(
        section.get("valuation_metrics", []),
        "dataset.valuation_metrics",
    )

    quality_ratio_metrics = _normalize_metric_pair_list(
        section.get("quality_ratio_metrics", []),
        "dataset.quality_ratio_metrics",
    )

    growth_metrics = _normalize_string_list(
        section.get("growth_metrics", []),
        "dataset.growth_metrics",
    )

    stability_ratio_metrics = _normalize_metric_pair_list(
        section.get("stability_ratio_metrics", []),
        "dataset.stability_ratio_metrics",
    )

    classification_fields = _normalize_choice_string_list(
        section.get("classification_fields", []),
        "dataset.classification_fields",
        choices={"sector", "industry"},
    )

    membership_indexes = _normalize_string_list(
        section.get("membership_indexes", []),
        "dataset.membership_indexes",
    )

    borrow_fields = _normalize_choice_string_list(
        section.get("borrow_fields", []),
        "dataset.borrow_fields",
        choices={"is_borrowable", "borrow_fee_bps"},
    )

    yang_zhang_volatility_window = _normalize_optional_positive_int(
        section.get("yang_zhang_volatility_window"),
        "dataset.yang_zhang_volatility_window",
    )
    if yang_zhang_volatility_window is not None and yang_zhang_volatility_window < 2:
        raise ConfigError("dataset.yang_zhang_volatility_window must be at least 2.")
    dollar_volume_zscore_window = _normalize_optional_positive_int(
        section.get("dollar_volume_zscore_window"),
        "dataset.dollar_volume_zscore_window",
    )
    if (
        dollar_volume_zscore_window is not None
        and dollar_volume_zscore_window < 2
    ):
        raise ConfigError("dataset.dollar_volume_zscore_window must be at least 2.")
    benchmark_residual_return_window = _normalize_optional_positive_int(
        section.get("benchmark_residual_return_window"),
        "dataset.benchmark_residual_return_window",
    )
    if (
        benchmark_residual_return_window is not None
        and benchmark_residual_return_window < 2
    ):
        raise ConfigError(
            "dataset.benchmark_residual_return_window must be at least 2."
        )

    return DatasetConfig(
        forward_horizons=forward_horizons,
        volatility_window=_normalize_positive_int(
            section.get("volatility_window", 20),
            "dataset.volatility_window",
        ),
        average_volume_window=_normalize_positive_int(
            section.get("average_volume_window", 20),
            "dataset.average_volume_window",
        ),
        average_true_range_window=_normalize_optional_positive_int(
            section.get("average_true_range_window"),
            "dataset.average_true_range_window",
        ),
        normalized_average_true_range_window=_normalize_optional_positive_int(
            section.get("normalized_average_true_range_window"),
            "dataset.normalized_average_true_range_window",
        ),
        amihud_illiquidity_window=_normalize_optional_positive_int(
            section.get("amihud_illiquidity_window"),
            "dataset.amihud_illiquidity_window",
        ),
        dollar_volume_shock_window=_normalize_optional_positive_int(
            section.get("dollar_volume_shock_window"),
            "dataset.dollar_volume_shock_window",
        ),
        dollar_volume_zscore_window=dollar_volume_zscore_window,
        volume_shock_window=_normalize_optional_positive_int(
            section.get("volume_shock_window"),
            "dataset.volume_shock_window",
        ),
        relative_volume_window=_normalize_optional_positive_int(
            section.get("relative_volume_window"),
            "dataset.relative_volume_window",
        ),
        relative_dollar_volume_window=_normalize_optional_positive_int(
            section.get("relative_dollar_volume_window"),
            "dataset.relative_dollar_volume_window",
        ),
        garman_klass_volatility_window=_normalize_optional_positive_int(
            section.get("garman_klass_volatility_window"),
            "dataset.garman_klass_volatility_window",
        ),
        parkinson_volatility_window=_normalize_optional_positive_int(
            section.get("parkinson_volatility_window"),
            "dataset.parkinson_volatility_window",
        ),
        rogers_satchell_volatility_window=_normalize_optional_positive_int(
            section.get("rogers_satchell_volatility_window"),
            "dataset.rogers_satchell_volatility_window",
        ),
        yang_zhang_volatility_window=yang_zhang_volatility_window,
        realized_volatility_window=_normalize_optional_positive_int(
            section.get("realized_volatility_window"),
            "dataset.realized_volatility_window",
        ),
        higher_moments_window=_normalize_optional_positive_int(
            section.get("higher_moments_window"),
            "dataset.higher_moments_window",
        ),
        benchmark_residual_return_window=benchmark_residual_return_window,
        benchmark_rolling_window=_normalize_optional_positive_int(
            section.get("benchmark_rolling_window"),
            "dataset.benchmark_rolling_window",
        ),
        fundamental_metrics=fundamental_metrics,
        valuation_metrics=valuation_metrics,
        quality_ratio_metrics=quality_ratio_metrics,
        growth_metrics=growth_metrics,
        stability_ratio_metrics=stability_ratio_metrics,
        classification_fields=classification_fields,
        membership_indexes=membership_indexes,
        borrow_fields=borrow_fields,
        include_market_cap=_normalize_bool(
            section.get("include_market_cap", False),
            "dataset.include_market_cap",
        ),
        market_cap_bucket_count=_normalize_optional_positive_int(
            section.get("market_cap_bucket_count"),
            "dataset.market_cap_bucket_count",
        ),
    )


def _parse_universe_config(section: Mapping[str, Any] | None) -> UniverseConfig | None:
    """Parse the optional universe filtering section."""
    if section is None:
        return None

    return UniverseConfig(
        min_price=_normalize_optional_positive_float(
            section.get("min_price"),
            "universe.min_price",
        ),
        min_average_volume=_normalize_optional_positive_float(
            section.get("min_average_volume"),
            "universe.min_average_volume",
        ),
        min_average_dollar_volume=_normalize_optional_positive_float(
            section.get("min_average_dollar_volume"),
            "universe.min_average_dollar_volume",
        ),
        min_listing_history_days=_normalize_optional_positive_int(
            section.get("min_listing_history_days"),
            "universe.min_listing_history_days",
        ),
        required_membership_indexes=_normalize_string_list(
            section.get("required_membership_indexes", []),
            "universe.required_membership_indexes",
        ),
        require_tradable=_normalize_bool(
            section.get("require_tradable", False),
            "universe.require_tradable",
        ),
        lag=_normalize_positive_int(section.get("lag", 1), "universe.lag"),
        average_volume_window=_normalize_optional_positive_int(
            section.get("average_volume_window"),
            "universe.average_volume_window",
        ),
        average_dollar_volume_window=_normalize_optional_positive_int(
            section.get("average_dollar_volume_window"),
            "universe.average_dollar_volume_window",
        ),
    )


def _parse_signal_config(section: Mapping[str, Any] | None) -> SignalConfig | None:
    """Parse the optional signal section."""
    if section is None:
        return None

    name = _normalize_choice_string(
        section.get("name"),
        "signal.name",
        choices={"momentum", "mean_reversion", "trend"},
    )

    winsorize_quantile = _normalize_optional_half_open_probability(
        section.get("winsorize_quantile"),
        "signal.winsorize_quantile",
    )
    clip_lower_bound, clip_upper_bound = _normalize_signal_clip_bounds(section)
    cross_sectional_residualize_columns = _normalize_string_list(
        section.get("cross_sectional_residualize_columns", []),
        "signal.cross_sectional_residualize_columns",
    )
    cross_sectional_normalization = _normalize_choice_string(
        section.get("cross_sectional_normalization", "none"),
        "signal.cross_sectional_normalization",
        choices={"none", "rank", "robust_zscore", "zscore"},
    )
    cross_sectional_group_column = (
        _normalize_non_empty_string(
            section["cross_sectional_group_column"],
            "signal.cross_sectional_group_column",
        )
        if "cross_sectional_group_column" in section
        else None
    )
    if (
        cross_sectional_group_column is not None
        and cross_sectional_normalization == "none"
    ):
        raise ConfigError(
            "signal.cross_sectional_group_column requires "
            "signal.cross_sectional_normalization to be one of "
            "{'rank', 'robust_zscore', 'zscore'}."
        )
    cross_sectional_neutralize_group_column = (
        _normalize_non_empty_string(
            section["cross_sectional_neutralize_group_column"],
            "signal.cross_sectional_neutralize_group_column",
        )
        if "cross_sectional_neutralize_group_column" in section
        else None
    )

    if name in {"momentum", "mean_reversion"}:
        return SignalConfig(
            name=name,
            lookback=_normalize_positive_int(section.get("lookback", 1), "signal.lookback"),
            winsorize_quantile=winsorize_quantile,
            clip_lower_bound=clip_lower_bound,
            clip_upper_bound=clip_upper_bound,
            cross_sectional_residualize_columns=(
                cross_sectional_residualize_columns
            ),
            cross_sectional_neutralize_group_column=(
                cross_sectional_neutralize_group_column
            ),
            cross_sectional_normalization=cross_sectional_normalization,
            cross_sectional_group_column=cross_sectional_group_column,
        )

    return SignalConfig(
        name=name,
        short_window=_normalize_positive_int(
            section.get("short_window", 20),
            "signal.short_window",
        ),
        long_window=_normalize_positive_int(
            section.get("long_window", 60),
            "signal.long_window",
        ),
        winsorize_quantile=winsorize_quantile,
        clip_lower_bound=clip_lower_bound,
        clip_upper_bound=clip_upper_bound,
        cross_sectional_residualize_columns=cross_sectional_residualize_columns,
        cross_sectional_neutralize_group_column=(
            cross_sectional_neutralize_group_column
        ),
        cross_sectional_normalization=cross_sectional_normalization,
        cross_sectional_group_column=cross_sectional_group_column,
    )


def _parse_portfolio_config(
    section: Mapping[str, Any] | None,
) -> PortfolioConfig | None:
    """Parse the optional portfolio section."""
    if section is None:
        return None

    construction = _normalize_choice_string(
        section.get("construction", "long_only"),
        "portfolio.construction",
        choices={"long_only", "long_short"},
    )
    weighting = _normalize_choice_string(
        section.get("weighting", "equal"),
        "portfolio.weighting",
        choices={"equal", "score"},
    )

    group_column = (
        _normalize_non_empty_string(
            section["group_column"],
            "portfolio.group_column",
        )
        if "group_column" in section
        else None
    )
    max_group_weight = _normalize_optional_positive_float(
        section.get("max_group_weight"),
        "portfolio.max_group_weight",
    )
    position_cap_column = (
        _normalize_non_empty_string(
            section["position_cap_column"],
            "portfolio.position_cap_column",
        )
        if "position_cap_column" in section
        else None
    )
    if (group_column is None) != (max_group_weight is None):
        raise ConfigError(
            "portfolio.group_column and portfolio.max_group_weight must be "
            "configured together."
        )
    factor_exposure_bounds = _parse_factor_exposure_bounds(
        section.get("factor_exposure_bounds", []),
    )

    return PortfolioConfig(
        construction=construction,
        top_n=_normalize_positive_int(section.get("top_n", 1), "portfolio.top_n"),
        bottom_n=(
            _normalize_positive_int(section["bottom_n"], "portfolio.bottom_n")
            if "bottom_n" in section
            else None
        ),
        weighting=weighting,
        exposure=_normalize_non_negative_float(section.get("exposure", 1.0), "portfolio.exposure"),
        long_exposure=_normalize_non_negative_float(
            section.get("long_exposure", 1.0),
            "portfolio.long_exposure",
        ),
        short_exposure=_normalize_non_negative_float(
            section.get("short_exposure", 1.0),
            "portfolio.short_exposure",
        ),
        max_position_weight=_normalize_optional_positive_float(
            section.get("max_position_weight"),
            "portfolio.max_position_weight",
        ),
        position_cap_column=position_cap_column,
        group_column=group_column,
        max_group_weight=max_group_weight,
        factor_exposure_bounds=factor_exposure_bounds,
    )


def _parse_factor_exposure_bounds(
    value: Any,
) -> tuple[FactorExposureBoundConfig, ...]:
    """Parse shrink-only target-weight factor exposure bounds."""
    if not isinstance(value, list):
        raise ConfigError("portfolio.factor_exposure_bounds must be a list of tables.")

    bounds: list[FactorExposureBoundConfig] = []
    seen_columns: set[str] = set()
    for index, raw_bound in enumerate(value, start=1):
        field_prefix = f"portfolio.factor_exposure_bounds[{index}]"
        if not isinstance(raw_bound, Mapping):
            raise ConfigError(
                "portfolio.factor_exposure_bounds entries must be tables."
            )

        column = _normalize_non_empty_string(
            raw_bound.get("column"),
            f"{field_prefix}.column",
        )
        if column in seen_columns:
            raise ConfigError(
                "portfolio.factor_exposure_bounds must not contain duplicate columns."
            )
        seen_columns.add(column)

        min_exposure = (
            _normalize_finite_float(raw_bound["min"], f"{field_prefix}.min")
            if "min" in raw_bound
            else None
        )
        max_exposure = (
            _normalize_finite_float(raw_bound["max"], f"{field_prefix}.max")
            if "max" in raw_bound
            else None
        )
        if min_exposure is None and max_exposure is None:
            raise ConfigError(
                "portfolio.factor_exposure_bounds entries must include min or max."
            )
        if (
            min_exposure is not None
            and max_exposure is not None
            and min_exposure > max_exposure
        ):
            raise ConfigError(
                "portfolio.factor_exposure_bounds min cannot exceed max."
            )
        if min_exposure is not None and min_exposure > 0.0:
            raise ConfigError(
                "portfolio.factor_exposure_bounds min must be <= 0."
            )
        if max_exposure is not None and max_exposure < 0.0:
            raise ConfigError(
                "portfolio.factor_exposure_bounds max must be >= 0."
            )

        bounds.append(
            FactorExposureBoundConfig(
                column=column,
                min_exposure=min_exposure,
                max_exposure=max_exposure,
            )
        )

    return tuple(bounds)


def _parse_backtest_config(
    section: Mapping[str, Any] | None,
) -> BacktestConfig | None:
    """Parse the optional backtest section."""
    if section is None:
        return None

    has_legacy_transaction_cost = "transaction_cost_bps" in section
    has_split_costs = "commission_bps" in section or "slippage_bps" in section
    has_row_costs = (
        "commission_bps_column" in section or "slippage_bps_column" in section
    )
    if has_legacy_transaction_cost and (has_split_costs or has_row_costs):
        raise ConfigError(
            "backtest.transaction_cost_bps cannot be combined with "
            "backtest.commission_bps, backtest.slippage_bps, "
            "backtest.commission_bps_column, or backtest.slippage_bps_column."
        )
    if "commission_bps" in section and "commission_bps_column" in section:
        raise ConfigError(
            "backtest.commission_bps cannot be combined with "
            "backtest.commission_bps_column."
        )
    if "slippage_bps" in section and "slippage_bps_column" in section:
        raise ConfigError(
            "backtest.slippage_bps cannot be combined with "
            "backtest.slippage_bps_column."
        )

    return BacktestConfig(
        signal_delay=_normalize_positive_int(
            section.get("signal_delay", 1),
            "backtest.signal_delay",
        ),
        fill_timing=_normalize_choice_string(
            section.get("fill_timing", "close"),
            "backtest.fill_timing",
            choices={"close", "next_close"},
        ),
        rebalance_frequency=_normalize_choice_string(
            section.get("rebalance_frequency", "daily"),
            "backtest.rebalance_frequency",
            choices={"daily", "weekly", "monthly"},
        ),
        transaction_cost_bps=_normalize_optional_non_negative_float(
            section.get("transaction_cost_bps"),
            "backtest.transaction_cost_bps",
        ),
        commission_bps=_normalize_non_negative_float(
            section.get("commission_bps", 0.0),
            "backtest.commission_bps",
        ),
        slippage_bps=_normalize_non_negative_float(
            section.get("slippage_bps", 0.0),
            "backtest.slippage_bps",
        ),
        commission_bps_column=(
            _normalize_non_empty_string(
                section["commission_bps_column"],
                "backtest.commission_bps_column",
            )
            if "commission_bps_column" in section
            else None
        ),
        slippage_bps_column=(
            _normalize_non_empty_string(
                section["slippage_bps_column"],
                "backtest.slippage_bps_column",
            )
            if "slippage_bps_column" in section
            else None
        ),
        max_trade_weight_column=(
            _normalize_non_empty_string(
                section["max_trade_weight_column"],
                "backtest.max_trade_weight_column",
            )
            if "max_trade_weight_column" in section
            else None
        ),
        max_turnover=_normalize_optional_non_negative_float(
            section.get("max_turnover"),
            "backtest.max_turnover",
        ),
        initial_nav=_normalize_positive_float(
            section.get("initial_nav", 1.0),
            "backtest.initial_nav",
        ),
    )


def _parse_diagnostics_config(section: Mapping[str, Any] | None) -> DiagnosticsConfig:
    """Parse the optional diagnostics section."""
    if section is None:
        return DiagnosticsConfig()

    return DiagnosticsConfig(
        forward_return_column=_normalize_non_empty_string(
            section.get("forward_return_column", "forward_return_1d"),
            "diagnostics.forward_return_column",
        ),
        ic_method=_normalize_choice_string(
            section.get("ic_method", "pearson"),
            "diagnostics.ic_method",
            choices={"pearson", "spearman"},
        ),
        n_quantiles=_normalize_positive_int(
            section.get("n_quantiles", 5),
            "diagnostics.n_quantiles",
        ),
        min_observations=_normalize_positive_int(
            section.get("min_observations", 5),
            "diagnostics.min_observations",
        ),
        rolling_ic_window=_normalize_positive_int(
            section.get("rolling_ic_window", 20),
            "diagnostics.rolling_ic_window",
        ),
        group_columns=_normalize_string_list(
            section.get("group_columns", []),
            "diagnostics.group_columns",
        ),
        exposure_columns=_normalize_string_list(
            section.get("exposure_columns", []),
            "diagnostics.exposure_columns",
        ),
    )


def _validate_cross_section_settings(config: AlphaForgeConfig) -> None:
    """Validate config relationships that span multiple sections."""
    if (
        config.data.price_adjustment == "split_adjusted"
        and config.corporate_actions is None
    ):
        raise ConfigError(
            "data.price_adjustment='split_adjusted' requires a [corporate_actions] section."
        )

    if config.dataset.fundamental_metrics and config.fundamentals is None:
        raise ConfigError(
            "dataset.fundamental_metrics requires a [fundamentals] section."
        )
    if config.dataset.valuation_metrics and config.fundamentals is None:
        raise ConfigError(
            "dataset.valuation_metrics requires a [fundamentals] section."
        )
    if config.dataset.quality_ratio_metrics and config.fundamentals is None:
        raise ConfigError(
            "dataset.quality_ratio_metrics requires a [fundamentals] section."
        )
    if config.dataset.growth_metrics and config.fundamentals is None:
        raise ConfigError(
            "dataset.growth_metrics requires a [fundamentals] section."
        )
    if config.dataset.stability_ratio_metrics and config.fundamentals is None:
        raise ConfigError(
            "dataset.stability_ratio_metrics requires a [fundamentals] section."
        )
    if config.dataset.classification_fields and config.classifications is None:
        raise ConfigError(
            "dataset.classification_fields requires a [classifications] section."
        )
    if config.dataset.membership_indexes and config.memberships is None:
        raise ConfigError(
            "dataset.membership_indexes requires a [memberships] section."
        )
    if (
        config.universe is not None
        and config.universe.required_membership_indexes
        and config.memberships is None
    ):
        raise ConfigError(
            "universe.required_membership_indexes requires a [memberships] section."
        )
    if (
        config.universe is not None
        and config.universe.require_tradable
        and config.trading_status is None
    ):
        raise ConfigError(
            "universe.require_tradable requires a [trading_status] section."
        )
    if config.dataset.borrow_fields and config.borrow_availability is None:
        raise ConfigError(
            "dataset.borrow_fields requires a [borrow_availability] section."
        )
    if config.dataset.include_market_cap and config.shares_outstanding is None:
        raise ConfigError(
            "dataset.include_market_cap requires a [shares_outstanding] section."
        )
    if (
        config.dataset.market_cap_bucket_count is not None
        and config.dataset.market_cap_bucket_count < 2
    ):
        raise ConfigError("dataset.market_cap_bucket_count must be at least 2.")
    if (
        config.dataset.market_cap_bucket_count is not None
        and not config.dataset.include_market_cap
    ):
        raise ConfigError(
            "dataset.market_cap_bucket_count requires "
            "dataset.include_market_cap=true."
        )
    if (
        config.dataset.higher_moments_window is not None
        and config.dataset.higher_moments_window < 4
    ):
        raise ConfigError("dataset.higher_moments_window must be at least 4.")
    if (
        config.dataset.benchmark_rolling_window is not None
        and config.benchmark is None
    ):
        raise ConfigError(
            "dataset.benchmark_rolling_window requires a [benchmark] section."
        )
    if (
        config.dataset.benchmark_residual_return_window is not None
        and config.benchmark is None
    ):
        raise ConfigError(
            "dataset.benchmark_residual_return_window requires a [benchmark] section."
        )

    if config.signal is not None and config.signal.name == "trend":
        short_window = config.signal.short_window or 20
        long_window = config.signal.long_window or 60
        if short_window >= long_window:
            raise ConfigError("signal.short_window must be smaller than signal.long_window for trend signals.")

    if config.universe is not None:
        universe = config.universe
        if (
            universe.min_price is None
            and universe.min_average_volume is None
            and universe.min_average_dollar_volume is None
            and universe.min_listing_history_days is None
            and not universe.required_membership_indexes
            and not universe.require_tradable
        ):
            raise ConfigError(
                "[universe] must configure at least one filtering rule."
            )
        if (
            universe.average_volume_window is not None
            and universe.min_average_volume is None
        ):
            raise ConfigError(
                "universe.average_volume_window requires universe.min_average_volume."
            )
        if (
            universe.average_dollar_volume_window is not None
            and universe.min_average_dollar_volume is None
        ):
            raise ConfigError(
                "universe.average_dollar_volume_window requires "
                "universe.min_average_dollar_volume."
            )

    if config.portfolio is not None:
        portfolio = config.portfolio
        if (
            portfolio.max_position_weight is not None
            and portfolio.construction == "long_only"
            and portfolio.max_position_weight > portfolio.exposure
        ):
            raise ConfigError(
                "portfolio.max_position_weight cannot exceed portfolio.exposure "
                "for long-only construction."
            )
        if (
            portfolio.max_position_weight is not None
            and portfolio.construction == "long_short"
            and (
                portfolio.max_position_weight > portfolio.long_exposure
                or portfolio.max_position_weight > portfolio.short_exposure
            )
        ):
            raise ConfigError(
                "portfolio.max_position_weight cannot exceed long_exposure or "
                "short_exposure for long-short construction."
            )
        if (
            portfolio.max_group_weight is not None
            and portfolio.construction == "long_only"
            and portfolio.max_group_weight > portfolio.exposure
        ):
            raise ConfigError(
                "portfolio.max_group_weight cannot exceed portfolio.exposure "
                "for long-only construction."
            )
        if (
            portfolio.max_group_weight is not None
            and portfolio.construction == "long_short"
            and (
                portfolio.max_group_weight > portfolio.long_exposure
                or portfolio.max_group_weight > portfolio.short_exposure
            )
        ):
            raise ConfigError(
                "portfolio.max_group_weight cannot exceed long_exposure or "
                "short_exposure for long-short construction."
            )

    _normalize_choice_string(
        config.diagnostics.ic_method,
        "diagnostics.ic_method",
        choices={"pearson", "spearman"},
    )

    if config.diagnostics.n_quantiles < 2:
        raise ConfigError("diagnostics.n_quantiles must be greater than or equal to 2.")

    if config.diagnostics.min_observations < config.diagnostics.n_quantiles:
        raise ConfigError(
            "diagnostics.min_observations must be greater than or equal to diagnostics.n_quantiles."
        )

    if config.diagnostics.rolling_ic_window < 2:
        raise ConfigError(
            "diagnostics.rolling_ic_window must be greater than or equal to 2."
        )


def _require_mapping(raw: Mapping[str, Any], *, section_name: str) -> Mapping[str, Any]:
    """Read a required table section from the parsed TOML document."""
    section = raw.get(section_name)
    if not isinstance(section, Mapping):
        raise ConfigError(f"Config file is missing required section [{section_name}].")
    return section


def _parse_supported_input_path(
    value: Any,
    *,
    field_name: str,
    config_path: Path,
) -> Path:
    """Resolve and validate a config-driven local input file path."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field_name} must be a non-empty string.")

    resolved_path = Path(value)
    if not resolved_path.is_absolute():
        resolved_path = (config_path.parent / resolved_path).resolve()

    if not resolved_path.exists():
        raise ConfigError(f"Configured path does not exist for {field_name}: {resolved_path}")
    if not resolved_path.is_file():
        raise ConfigError(f"Configured path must point to a file for {field_name}: {resolved_path}")
    if resolved_path.suffix.lower() not in _SUPPORTED_DATA_SUFFIXES:
        supported_suffixes = ", ".join(sorted(_SUPPORTED_DATA_SUFFIXES))
        raise ConfigError(
            f"{field_name} must use one of the supported file types "
            f"({supported_suffixes}): {resolved_path}"
        )
    return resolved_path


def _optional_mapping(
    raw: Mapping[str, Any], *, section_name: str
) -> Mapping[str, Any] | None:
    """Read an optional table section from the parsed TOML document."""
    section = raw.get(section_name)
    if section is None:
        return None
    if not isinstance(section, Mapping):
        raise ConfigError(f"Config section [{section_name}] must be a table.")
    return section


def _normalize_non_empty_string(value: Any, field_name: str) -> str:
    """Validate non-empty string config fields."""
    return _common_non_empty_string(
        value,
        parameter_name=field_name,
        error_factory=ConfigError,
    )


def _normalize_string_list(value: Any, field_name: str) -> tuple[str, ...]:
    """Validate a list of unique non-empty strings."""
    if not isinstance(value, list):
        raise ConfigError(f"{field_name} must be a list of strings.")
    return _common_string_sequence(
        value,
        parameter_name=field_name,
        error_factory=ConfigError,
        item_error_message=f"{field_name} must be a non-empty string.",
        duplicate_error_message=f"{field_name} must not contain duplicates.",
    )


def _normalize_metric_pair_list(
    value: Any,
    field_name: str,
) -> tuple[tuple[str, str], ...]:
    """Validate a list of numerator/denominator metric-name pairs."""
    if not isinstance(value, list):
        raise ConfigError(
            f"{field_name} must be a list of [numerator, denominator] string pairs."
        )
    pair_error_message = (
        f"{field_name} must be a list of [numerator, denominator] string pairs."
    )
    return _common_string_pair_sequence(
        value,
        parameter_name=field_name,
        error_factory=ConfigError,
        pair_type=list,
        pair_error_message=pair_error_message,
        item_error_message=f"{field_name} must be a non-empty string.",
        duplicate_error_message=f"{field_name} must not contain duplicate metric pairs.",
        equal_items_error_message=(
            f"{field_name} numerator and denominator must be different."
        ),
    )


def _normalize_choice_string(
    value: Any,
    field_name: str,
    *,
    choices: set[str],
) -> str:
    """Validate string fields against a fixed choice set."""
    return _common_choice_string(
        value,
        parameter_name=field_name,
        choices=choices,
        error_factory=ConfigError,
    )


def _normalize_choice_string_list(
    value: Any,
    field_name: str,
    *,
    choices: set[str],
) -> tuple[str, ...]:
    """Validate a list of unique strings against a fixed choice set."""
    return tuple(
        _normalize_choice_string(item, field_name, choices=choices)
        for item in _normalize_string_list(value, field_name)
    )


def _normalize_bool(value: Any, field_name: str) -> bool:
    """Validate boolean config fields."""
    if not isinstance(value, bool):
        raise ConfigError(f"{field_name} must be a boolean.")
    return value


def _normalize_positive_int(value: Any, field_name: str) -> int:
    """Validate positive integer config fields."""
    return _common_positive_int(
        value,
        parameter_name=field_name,
        error_factory=ConfigError,
    )


def _normalize_non_negative_float(value: Any, field_name: str) -> float:
    """Validate non-negative float config fields."""
    return _common_non_negative_float(
        value,
        parameter_name=field_name,
        error_factory=ConfigError,
    )


def _normalize_positive_float(value: Any, field_name: str) -> float:
    """Validate strictly positive float config fields."""
    return _common_positive_float(
        value,
        parameter_name=field_name,
        error_factory=ConfigError,
    )


def _normalize_optional_positive_float(value: Any, field_name: str) -> float | None:
    """Validate optional positive float config fields."""
    return _common_optional_positive_float(
        value,
        parameter_name=field_name,
        error_factory=ConfigError,
    )


def _normalize_optional_positive_int(value: Any, field_name: str) -> int | None:
    """Validate optional positive integer config fields."""
    if value is None:
        return None
    return _normalize_positive_int(value, field_name)


def _normalize_optional_non_negative_float(value: Any, field_name: str) -> float | None:
    """Validate optional non-negative float config fields."""
    return _common_optional_non_negative_float(
        value,
        parameter_name=field_name,
        error_factory=ConfigError,
    )


def _normalize_optional_half_open_probability(
    value: Any,
    field_name: str,
) -> float | None:
    """Validate optional probability-like fields constrained to [0.0, 0.5)."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be a float in [0.0, 0.5).")

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be a float in [0.0, 0.5).") from exc

    if numeric_value < 0.0 or numeric_value >= 0.5:
        raise ConfigError(f"{field_name} must be a float in [0.0, 0.5).")
    return numeric_value


def _normalize_signal_clip_bounds(
    section: Mapping[str, Any],
) -> tuple[float | None, float | None]:
    """Validate optional explicit signal-clipping bounds."""
    lower_raw = section.get("clip_lower_bound")
    upper_raw = section.get("clip_upper_bound")
    if lower_raw is None and upper_raw is None:
        return None, None
    if lower_raw is None or upper_raw is None:
        raise ConfigError(
            "signal.clip_lower_bound and signal.clip_upper_bound must be "
            "configured together."
        )

    lower_bound = _normalize_finite_float(
        lower_raw,
        "signal.clip_lower_bound",
    )
    upper_bound = _normalize_finite_float(
        upper_raw,
        "signal.clip_upper_bound",
    )
    if lower_bound >= upper_bound:
        raise ConfigError(
            "signal.clip_lower_bound must be smaller than "
            "signal.clip_upper_bound."
        )
    return lower_bound, upper_bound


def _normalize_finite_float(value: Any, field_name: str) -> float:
    """Validate finite float config fields."""
    return _common_finite_float(
        value,
        parameter_name=field_name,
        error_factory=ConfigError,
    )

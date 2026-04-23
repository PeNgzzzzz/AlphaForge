"""Configuration loading for AlphaForge CLI workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.9 fallback
    import tomli as tomllib


class ConfigError(ValueError):
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
    parkinson_volatility_window: int | None = None
    realized_volatility_window: int | None = None
    higher_moments_window: int | None = None
    benchmark_rolling_window: int | None = None
    fundamental_metrics: tuple[str, ...] = ()
    classification_fields: tuple[str, ...] = ()
    membership_indexes: tuple[str, ...] = ()
    borrow_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class UniverseConfig:
    """Optional tradability-aware universe filtering settings."""

    min_price: float | None = None
    min_average_volume: float | None = None
    min_average_dollar_volume: float | None = None
    min_listing_history_days: int | None = None
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
    cross_sectional_normalization: str = "none"


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


@dataclass(frozen=True)
class BacktestConfig:
    """Daily backtest settings."""

    signal_delay: int = 1
    rebalance_frequency: str = "daily"
    transaction_cost_bps: float | None = None
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    max_turnover: float | None = None
    initial_nav: float = 1.0


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Factor diagnostics settings."""

    forward_return_column: str = "forward_return_1d"
    ic_method: str = "pearson"
    n_quantiles: int = 5
    min_observations: int = 5


@dataclass(frozen=True)
class AlphaForgeConfig:
    """Top-level pipeline configuration."""

    data: DataConfig
    calendar: CalendarConfig | None
    symbol_metadata: SymbolMetadataConfig | None
    corporate_actions: CorporateActionsConfig | None
    fundamentals: FundamentalsConfig | None
    classifications: ClassificationsConfig | None
    memberships: MembershipsConfig | None
    borrow_availability: BorrowAvailabilityConfig | None
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
    classifications_section = _optional_mapping(raw, section_name="classifications")
    memberships_section = _optional_mapping(raw, section_name="memberships")
    borrow_availability_section = _optional_mapping(
        raw,
        section_name="borrow_availability",
    )
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

    fundamental_metrics_raw = section.get("fundamental_metrics", [])
    if not isinstance(fundamental_metrics_raw, list):
        raise ConfigError("dataset.fundamental_metrics must be a list of strings.")
    fundamental_metrics = tuple(
        _normalize_non_empty_string(value, "dataset.fundamental_metrics")
        for value in fundamental_metrics_raw
    )
    if len(set(fundamental_metrics)) != len(fundamental_metrics):
        raise ConfigError("dataset.fundamental_metrics must not contain duplicates.")

    classification_fields_raw = section.get("classification_fields", [])
    if not isinstance(classification_fields_raw, list):
        raise ConfigError("dataset.classification_fields must be a list of strings.")
    classification_fields = tuple(
        _normalize_choice_string(
            value,
            "dataset.classification_fields",
            choices={"sector", "industry"},
        )
        for value in classification_fields_raw
    )
    if len(set(classification_fields)) != len(classification_fields):
        raise ConfigError("dataset.classification_fields must not contain duplicates.")

    membership_indexes_raw = section.get("membership_indexes", [])
    if not isinstance(membership_indexes_raw, list):
        raise ConfigError("dataset.membership_indexes must be a list of strings.")
    membership_indexes = tuple(
        _normalize_non_empty_string(value, "dataset.membership_indexes")
        for value in membership_indexes_raw
    )
    if len(set(membership_indexes)) != len(membership_indexes):
        raise ConfigError("dataset.membership_indexes must not contain duplicates.")

    borrow_fields_raw = section.get("borrow_fields", [])
    if not isinstance(borrow_fields_raw, list):
        raise ConfigError("dataset.borrow_fields must be a list of strings.")
    borrow_fields = tuple(
        _normalize_choice_string(
            value,
            "dataset.borrow_fields",
            choices={"is_borrowable", "borrow_fee_bps"},
        )
        for value in borrow_fields_raw
    )
    if len(set(borrow_fields)) != len(borrow_fields):
        raise ConfigError("dataset.borrow_fields must not contain duplicates.")

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
        parkinson_volatility_window=_normalize_optional_positive_int(
            section.get("parkinson_volatility_window"),
            "dataset.parkinson_volatility_window",
        ),
        realized_volatility_window=_normalize_optional_positive_int(
            section.get("realized_volatility_window"),
            "dataset.realized_volatility_window",
        ),
        higher_moments_window=_normalize_optional_positive_int(
            section.get("higher_moments_window"),
            "dataset.higher_moments_window",
        ),
        benchmark_rolling_window=_normalize_optional_positive_int(
            section.get("benchmark_rolling_window"),
            "dataset.benchmark_rolling_window",
        ),
        fundamental_metrics=fundamental_metrics,
        classification_fields=classification_fields,
        membership_indexes=membership_indexes,
        borrow_fields=borrow_fields,
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

    name = section.get("name")
    if name not in {"momentum", "mean_reversion", "trend"}:
        raise ConfigError("signal.name must be one of {'momentum', 'mean_reversion', 'trend'}.")

    winsorize_quantile = _normalize_optional_half_open_probability(
        section.get("winsorize_quantile"),
        "signal.winsorize_quantile",
    )
    cross_sectional_normalization = _normalize_choice_string(
        section.get("cross_sectional_normalization", "none"),
        "signal.cross_sectional_normalization",
        choices={"none", "rank", "zscore"},
    )

    if name in {"momentum", "mean_reversion"}:
        return SignalConfig(
            name=name,
            lookback=_normalize_positive_int(section.get("lookback", 1), "signal.lookback"),
            winsorize_quantile=winsorize_quantile,
            cross_sectional_normalization=cross_sectional_normalization,
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
        cross_sectional_normalization=cross_sectional_normalization,
    )


def _parse_portfolio_config(
    section: Mapping[str, Any] | None,
) -> PortfolioConfig | None:
    """Parse the optional portfolio section."""
    if section is None:
        return None

    construction = section.get("construction", "long_only")
    if construction not in {"long_only", "long_short"}:
        raise ConfigError("portfolio.construction must be one of {'long_only', 'long_short'}.")

    weighting = section.get("weighting", "equal")
    if weighting not in {"equal", "score"}:
        raise ConfigError("portfolio.weighting must be one of {'equal', 'score'}.")

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
    )


def _parse_backtest_config(
    section: Mapping[str, Any] | None,
) -> BacktestConfig | None:
    """Parse the optional backtest section."""
    if section is None:
        return None

    has_legacy_transaction_cost = "transaction_cost_bps" in section
    has_split_costs = "commission_bps" in section or "slippage_bps" in section
    if has_legacy_transaction_cost and has_split_costs:
        raise ConfigError(
            "backtest.transaction_cost_bps cannot be combined with "
            "backtest.commission_bps or backtest.slippage_bps."
        )

    return BacktestConfig(
        signal_delay=_normalize_positive_int(
            section.get("signal_delay", 1),
            "backtest.signal_delay",
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
        ic_method=_normalize_non_empty_string(section.get("ic_method", "pearson"), "diagnostics.ic_method"),
        n_quantiles=_normalize_positive_int(
            section.get("n_quantiles", 5),
            "diagnostics.n_quantiles",
        ),
        min_observations=_normalize_positive_int(
            section.get("min_observations", 5),
            "diagnostics.min_observations",
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
    if config.dataset.classification_fields and config.classifications is None:
        raise ConfigError(
            "dataset.classification_fields requires a [classifications] section."
        )
    if config.dataset.membership_indexes and config.memberships is None:
        raise ConfigError(
            "dataset.membership_indexes requires a [memberships] section."
        )
    if config.dataset.borrow_fields and config.borrow_availability is None:
        raise ConfigError(
            "dataset.borrow_fields requires a [borrow_availability] section."
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
        ):
            raise ConfigError(
                "[universe] must configure at least one filtering threshold."
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

    if config.diagnostics.ic_method not in {"pearson", "spearman"}:
        raise ConfigError("diagnostics.ic_method must be one of {'pearson', 'spearman'}.")

    if config.diagnostics.n_quantiles < 2:
        raise ConfigError("diagnostics.n_quantiles must be greater than or equal to 2.")

    if config.diagnostics.min_observations < config.diagnostics.n_quantiles:
        raise ConfigError(
            "diagnostics.min_observations must be greater than or equal to diagnostics.n_quantiles."
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
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _normalize_choice_string(
    value: Any,
    field_name: str,
    *,
    choices: set[str],
) -> str:
    """Validate string fields against a fixed choice set."""
    normalized = _normalize_non_empty_string(value, field_name)
    if normalized not in choices:
        allowed_text = ", ".join(repr(choice) for choice in sorted(choices))
        raise ConfigError(f"{field_name} must be one of {{{allowed_text}}}.")
    return normalized


def _normalize_positive_int(value: Any, field_name: str) -> int:
    """Validate positive integer config fields."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{field_name} must be a positive integer.")
    return value


def _normalize_non_negative_float(value: Any, field_name: str) -> float:
    """Validate non-negative float config fields."""
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be a non-negative float.")

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be a non-negative float.") from exc

    if numeric_value < 0.0:
        raise ConfigError(f"{field_name} must be a non-negative float.")
    return numeric_value


def _normalize_positive_float(value: Any, field_name: str) -> float:
    """Validate strictly positive float config fields."""
    numeric_value = _normalize_non_negative_float(value, field_name)
    if numeric_value <= 0.0:
        raise ConfigError(f"{field_name} must be a positive float.")
    return numeric_value


def _normalize_optional_positive_float(value: Any, field_name: str) -> float | None:
    """Validate optional positive float config fields."""
    if value is None:
        return None
    return _normalize_positive_float(value, field_name)


def _normalize_optional_positive_int(value: Any, field_name: str) -> int | None:
    """Validate optional positive integer config fields."""
    if value is None:
        return None
    return _normalize_positive_int(value, field_name)


def _normalize_optional_non_negative_float(value: Any, field_name: str) -> float | None:
    """Validate optional non-negative float config fields."""
    if value is None:
        return None
    return _normalize_non_negative_float(value, field_name)


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

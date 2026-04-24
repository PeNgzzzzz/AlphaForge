"""Research dataset construction utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np
import pandas as pd

from alphaforge.data import (
    DataValidationError,
    ensure_dates_on_trading_calendar,
    validate_ohlcv,
    validate_symbol_metadata,
    validate_trading_calendar,
)
from alphaforge.features.classifications_join import attach_classifications_asof
from alphaforge.features.borrow_join import attach_borrow_availability_asof
from alphaforge.features.fundamentals_join import attach_fundamentals_asof
from alphaforge.features.membership_join import attach_memberships_asof
from alphaforge.features.rolling_statistics import (
    attach_average_true_range,
    attach_amihud_illiquidity,
    attach_dollar_volume_zscore,
    attach_garman_klass_volatility,
    attach_normalized_average_true_range,
    attach_relative_volume,
    attach_relative_dollar_volume,
    attach_volume_shock,
    attach_parkinson_volatility,
    attach_realized_volatility_family,
    attach_rogers_satchell_volatility,
    attach_yang_zhang_volatility,
    attach_rolling_higher_moments,
    attach_rolling_benchmark_statistics,
)

ForwardHorizonInput = Union[int, Sequence[int]]


def build_research_dataset(
    frame: pd.DataFrame,
    *,
    trading_calendar: pd.DataFrame | None = None,
    symbol_metadata: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    classifications: pd.DataFrame | None = None,
    memberships: pd.DataFrame | None = None,
    borrow_availability: pd.DataFrame | None = None,
    benchmark_returns: pd.DataFrame | None = None,
    average_true_range_window: int | None = None,
    normalized_average_true_range_window: int | None = None,
    amihud_illiquidity_window: int | None = None,
    dollar_volume_zscore_window: int | None = None,
    volume_shock_window: int | None = None,
    relative_volume_window: int | None = None,
    relative_dollar_volume_window: int | None = None,
    garman_klass_volatility_window: int | None = None,
    parkinson_volatility_window: int | None = None,
    rogers_satchell_volatility_window: int | None = None,
    yang_zhang_volatility_window: int | None = None,
    realized_volatility_window: int | None = None,
    higher_moments_window: int | None = None,
    fundamental_metrics: Sequence[str] | None = None,
    classification_fields: Sequence[str] | None = None,
    membership_indexes: Sequence[str] | None = None,
    borrow_fields: Sequence[str] | None = None,
    benchmark_rolling_window: int | None = None,
    forward_horizons: ForwardHorizonInput = (1,),
    volatility_window: int = 20,
    average_volume_window: int = 20,
    minimum_price: float | None = None,
    minimum_average_volume: float | None = None,
    minimum_average_dollar_volume: float | None = None,
    minimum_listing_history_days: int | None = None,
    universe_lag: int = 1,
    universe_average_volume_window: int | None = None,
    universe_average_dollar_volume_window: int | None = None,
) -> pd.DataFrame:
    """Build a daily research dataset with explicit close-to-future alignment.

    Timing convention:
    - each row is anchored at the close of ``date``
    - feature columns use information available through that close
    - ``forward_return_{h}d`` labels start after that close and end ``h`` bars later
    - date-only fundamentals releases become usable on the next market session
    - date-only classification effective dates become active on the first
      market session not earlier than ``effective_date``
    - date-only membership effective dates become active on the first
      market session not earlier than ``effective_date``
    - date-only borrow availability effective dates become active on the first
      market session not earlier than ``effective_date``
    - optional average true range uses trailing ``high`` / ``low`` plus
      ``close_{t-1}``, all available by that same close
    - optional normalized average true range uses that same trailing ATR
      definition divided by ``close_t``, which is also known by that close
    - optional Amihud illiquidity uses trailing ``abs(daily_return) / (close * volume)``
      observations available through that same close
    - optional dollar volume z-score uses same-day ``log(close * volume)``
      against prior rolling log dollar-volume observations
    - optional volume shock uses same-day ``log(volume)`` against prior rolling
      log-volume observations
    - optional relative volume uses same-day ``volume`` divided by the trailing
      average of prior daily volume observations
    - optional relative dollar volume uses same-day ``close * volume`` divided
      by the trailing average of prior daily dollar volume observations
    - optional Garman-Klass volatility uses only trailing ``open`` / ``high`` /
      ``low`` / ``close`` observations available through that same close
    - optional Parkinson volatility uses only trailing ``high`` / ``low``
      observations available through that same close
    - optional Rogers-Satchell volatility uses only trailing ``open`` /
      ``high`` / ``low`` / ``close`` observations available through that same
      close
    - optional Yang-Zhang volatility uses trailing ``open`` / ``high`` /
      ``low`` / ``close`` observations, including ``close_{t-1}`` for the
      overnight component, all available by that same close
    - optional realized volatility family uses only trailing ``daily_return``
      observations available through that same close
    - optional rolling skew / kurtosis use only trailing ``daily_return``
      observations available through that same close
    - optional universe filters use lagged per-symbol observations from
      ``universe_filter_date`` so the filter itself stays explicit
    """
    if fundamental_metrics is not None and fundamentals is None:
        raise ValueError(
            "fundamental_metrics requires fundamentals to be provided."
        )
    if classification_fields is not None and classifications is None:
        raise ValueError(
            "classification_fields requires classifications to be provided."
        )
    if membership_indexes is not None and memberships is None:
        raise ValueError("membership_indexes requires memberships to be provided.")
    if borrow_fields is not None and borrow_availability is None:
        raise ValueError("borrow_fields requires borrow_availability to be provided.")
    if benchmark_rolling_window is not None and benchmark_returns is None:
        raise ValueError(
            "benchmark_rolling_window requires benchmark_returns to be provided."
        )
    average_true_range_window = _normalize_optional_positive_int(
        average_true_range_window,
        parameter_name="average_true_range_window",
    )
    normalized_average_true_range_window = _normalize_optional_positive_int(
        normalized_average_true_range_window,
        parameter_name="normalized_average_true_range_window",
    )
    amihud_illiquidity_window = _normalize_optional_positive_int(
        amihud_illiquidity_window,
        parameter_name="amihud_illiquidity_window",
    )
    dollar_volume_zscore_window = _normalize_optional_positive_int(
        dollar_volume_zscore_window,
        parameter_name="dollar_volume_zscore_window",
    )
    if dollar_volume_zscore_window is not None and dollar_volume_zscore_window < 2:
        raise ValueError("dollar_volume_zscore_window must be at least 2.")
    volume_shock_window = _normalize_optional_positive_int(
        volume_shock_window,
        parameter_name="volume_shock_window",
    )
    relative_volume_window = _normalize_optional_positive_int(
        relative_volume_window,
        parameter_name="relative_volume_window",
    )
    relative_dollar_volume_window = _normalize_optional_positive_int(
        relative_dollar_volume_window,
        parameter_name="relative_dollar_volume_window",
    )
    garman_klass_volatility_window = _normalize_optional_positive_int(
        garman_klass_volatility_window,
        parameter_name="garman_klass_volatility_window",
    )
    parkinson_volatility_window = _normalize_optional_positive_int(
        parkinson_volatility_window,
        parameter_name="parkinson_volatility_window",
    )
    rogers_satchell_volatility_window = _normalize_optional_positive_int(
        rogers_satchell_volatility_window,
        parameter_name="rogers_satchell_volatility_window",
    )
    yang_zhang_volatility_window = _normalize_optional_positive_int(
        yang_zhang_volatility_window,
        parameter_name="yang_zhang_volatility_window",
    )
    if yang_zhang_volatility_window is not None and yang_zhang_volatility_window < 2:
        raise ValueError("yang_zhang_volatility_window must be at least 2.")
    realized_volatility_window = _normalize_optional_positive_int(
        realized_volatility_window,
        parameter_name="realized_volatility_window",
    )
    higher_moments_window = _normalize_optional_positive_int(
        higher_moments_window,
        parameter_name="higher_moments_window",
    )
    if higher_moments_window is not None and higher_moments_window < 4:
        raise ValueError("higher_moments_window must be at least 4.")
    normalized_horizons = _normalize_forward_horizons(forward_horizons)
    volatility_window = _normalize_window(
        volatility_window, parameter_name="volatility_window"
    )
    average_volume_window = _normalize_window(
        average_volume_window, parameter_name="average_volume_window"
    )
    minimum_price = _normalize_optional_positive_float(
        minimum_price,
        parameter_name="minimum_price",
    )
    minimum_average_volume = _normalize_optional_positive_float(
        minimum_average_volume,
        parameter_name="minimum_average_volume",
    )
    minimum_average_dollar_volume = _normalize_optional_positive_float(
        minimum_average_dollar_volume,
        parameter_name="minimum_average_dollar_volume",
    )
    minimum_listing_history_days = _normalize_optional_positive_int(
        minimum_listing_history_days,
        parameter_name="minimum_listing_history_days",
    )

    dataset = validate_ohlcv(frame, source="research dataset input").copy()
    validated_trading_calendar = (
        validate_trading_calendar(
            trading_calendar,
            source="trading calendar input",
        )
        if trading_calendar is not None
        else None
    )
    if validated_trading_calendar is not None:
        ensure_dates_on_trading_calendar(
            dataset["date"],
            validated_trading_calendar,
            source="research dataset input",
        )
    if symbol_metadata is not None:
        dataset = _attach_symbol_metadata(
            dataset,
            symbol_metadata,
        )
    close_by_symbol = dataset.groupby("symbol", sort=False)["close"]
    volume_by_symbol = dataset.groupby("symbol", sort=False)["volume"]

    previous_close = close_by_symbol.shift(1)
    daily_return = dataset["close"].div(previous_close).sub(1.0)

    dataset["daily_return"] = daily_return
    dataset["log_return"] = pd.Series(np.log1p(daily_return), index=dataset.index)

    dataset[f"rolling_volatility_{volatility_window}d"] = (
        daily_return.groupby(dataset["symbol"], sort=False)
        .transform(
            lambda values: values.rolling(
                window=volatility_window,
                min_periods=volatility_window,
            ).std()
        )
    )
    dataset[f"rolling_average_volume_{average_volume_window}d"] = (
        volume_by_symbol.transform(
            lambda values: values.rolling(
                window=average_volume_window,
                min_periods=average_volume_window,
            ).mean()
        )
    )

    # Forward returns are the label side of the research dataset.
    for horizon in normalized_horizons:
        future_close = close_by_symbol.shift(-horizon)
        dataset[f"forward_return_{horizon}d"] = future_close.div(dataset["close"]).sub(
            1.0
        )

    if average_true_range_window is not None:
        dataset = attach_average_true_range(
            dataset,
            window=average_true_range_window,
        )
    if normalized_average_true_range_window is not None:
        dataset = attach_normalized_average_true_range(
            dataset,
            window=normalized_average_true_range_window,
        )
    if amihud_illiquidity_window is not None:
        dataset = attach_amihud_illiquidity(
            dataset,
            window=amihud_illiquidity_window,
        )
    if dollar_volume_zscore_window is not None:
        dataset = attach_dollar_volume_zscore(
            dataset,
            window=dollar_volume_zscore_window,
        )
    if volume_shock_window is not None:
        dataset = attach_volume_shock(
            dataset,
            window=volume_shock_window,
        )
    if relative_volume_window is not None:
        dataset = attach_relative_volume(
            dataset,
            window=relative_volume_window,
        )
    if relative_dollar_volume_window is not None:
        dataset = attach_relative_dollar_volume(
            dataset,
            window=relative_dollar_volume_window,
        )
    if garman_klass_volatility_window is not None:
        dataset = attach_garman_klass_volatility(
            dataset,
            window=garman_klass_volatility_window,
        )
    if parkinson_volatility_window is not None:
        dataset = attach_parkinson_volatility(
            dataset,
            window=parkinson_volatility_window,
        )
    if rogers_satchell_volatility_window is not None:
        dataset = attach_rogers_satchell_volatility(
            dataset,
            window=rogers_satchell_volatility_window,
        )
    if yang_zhang_volatility_window is not None:
        dataset = attach_yang_zhang_volatility(
            dataset,
            window=yang_zhang_volatility_window,
        )
    if realized_volatility_window is not None:
        dataset = attach_realized_volatility_family(
            dataset,
            window=realized_volatility_window,
        )
    if higher_moments_window is not None:
        dataset = attach_rolling_higher_moments(
            dataset,
            window=higher_moments_window,
        )
    if fundamentals is not None:
        dataset = attach_fundamentals_asof(
            dataset,
            fundamentals,
            trading_calendar=validated_trading_calendar,
            metrics=fundamental_metrics,
        )
    if classifications is not None:
        dataset = attach_classifications_asof(
            dataset,
            classifications,
            trading_calendar=validated_trading_calendar,
            fields=classification_fields,
        )
    if memberships is not None:
        dataset = attach_memberships_asof(
            dataset,
            memberships,
            trading_calendar=validated_trading_calendar,
            indexes=membership_indexes,
        )
    if borrow_availability is not None:
        dataset = attach_borrow_availability_asof(
            dataset,
            borrow_availability,
            trading_calendar=validated_trading_calendar,
            fields=borrow_fields,
        )
    if benchmark_returns is not None:
        dataset = attach_rolling_benchmark_statistics(
            dataset,
            benchmark_returns,
            window=(
                benchmark_rolling_window
                if benchmark_rolling_window is not None
                else 20
            ),
        )

    if _universe_filters_enabled(
        minimum_price=minimum_price,
        minimum_average_volume=minimum_average_volume,
        minimum_average_dollar_volume=minimum_average_dollar_volume,
        minimum_listing_history_days=minimum_listing_history_days,
    ):
        universe_lag = _normalize_window(
            universe_lag,
            parameter_name="universe_lag",
        )
        universe_average_volume_window = _resolve_optional_window(
            universe_average_volume_window,
            fallback_window=average_volume_window,
            parameter_name="universe_average_volume_window",
        )
        universe_average_dollar_volume_window = _resolve_optional_window(
            universe_average_dollar_volume_window,
            fallback_window=average_volume_window,
            parameter_name="universe_average_dollar_volume_window",
        )
        dataset = _apply_universe_filters(
            dataset,
            trading_calendar=validated_trading_calendar,
            minimum_price=minimum_price,
            minimum_average_volume=minimum_average_volume,
            minimum_average_dollar_volume=minimum_average_dollar_volume,
            minimum_listing_history_days=minimum_listing_history_days,
            universe_lag=universe_lag,
            universe_average_volume_window=universe_average_volume_window,
            universe_average_dollar_volume_window=universe_average_dollar_volume_window,
        )

    return dataset


def _normalize_forward_horizons(forward_horizons: ForwardHorizonInput) -> tuple[int, ...]:
    """Validate and normalize forward return horizons."""
    if isinstance(forward_horizons, int) and not isinstance(forward_horizons, bool):
        normalized = (forward_horizons,)
    else:
        normalized = tuple(forward_horizons)

    if not normalized:
        raise ValueError("forward_horizons must contain at least one positive integer.")

    for horizon in normalized:
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError(
                "forward_horizons must contain only positive integer horizons."
            )

    if len(set(normalized)) != len(normalized):
        raise ValueError("forward_horizons must not contain duplicate horizons.")

    return normalized


def _normalize_window(window: int, *, parameter_name: str) -> int:
    """Validate rolling window inputs."""
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError(f"{parameter_name} must be a positive integer.")
    return window


def _normalize_optional_positive_float(
    value: float | None, *, parameter_name: str
) -> float | None:
    """Validate optional positive float inputs."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{parameter_name} must be a positive float.")

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{parameter_name} must be a positive float.") from exc

    if numeric_value <= 0.0:
        raise ValueError(f"{parameter_name} must be a positive float.")
    return numeric_value


def _normalize_optional_positive_int(
    value: int | None, *, parameter_name: str
) -> int | None:
    """Validate optional positive integer inputs."""
    if value is None:
        return None
    return _normalize_window(value, parameter_name=parameter_name)


def _universe_filters_enabled(
    *,
    minimum_price: float | None,
    minimum_average_volume: float | None,
    minimum_average_dollar_volume: float | None,
    minimum_listing_history_days: int | None,
) -> bool:
    """Return whether any tradability filter is configured."""
    return any(
        value is not None
        for value in (
            minimum_price,
            minimum_average_volume,
            minimum_average_dollar_volume,
            minimum_listing_history_days,
        )
    )


def _apply_universe_filters(
    dataset: pd.DataFrame,
    *,
    trading_calendar: pd.DataFrame | None,
    minimum_price: float | None,
    minimum_average_volume: float | None,
    minimum_average_dollar_volume: float | None,
    minimum_listing_history_days: int | None,
    universe_lag: int,
    universe_average_volume_window: int,
    universe_average_dollar_volume_window: int,
) -> pd.DataFrame:
    """Add lagged tradability-aware universe filter columns to a dataset."""
    filtered = dataset.copy()
    symbol_groups = filtered.groupby("symbol", sort=False)
    lagged_filter_date = symbol_groups["date"].shift(universe_lag)
    has_universe_history = lagged_filter_date.notna()
    exclusion_reasons = pd.Series("", index=filtered.index, dtype="object")

    filtered["universe_filter_date"] = lagged_filter_date
    filtered["has_universe_history"] = has_universe_history
    filtered["listing_history_days"] = _compute_listing_history_days(
        filtered,
        trading_calendar=trading_calendar,
    )
    filtered["universe_lagged_listing_history_days"] = filtered.groupby(
        "symbol", sort=False
    )["listing_history_days"].shift(universe_lag)
    filtered["universe_lagged_close"] = symbol_groups["close"].shift(universe_lag)

    exclusion_reasons = _append_exclusion_reason(
        exclusion_reasons,
        ~has_universe_history,
        "insufficient_universe_history",
    )

    if minimum_listing_history_days is not None:
        passes_listing_history = filtered["universe_lagged_listing_history_days"].ge(
            minimum_listing_history_days
        )
        filtered["passes_universe_min_listing_history"] = passes_listing_history
        insufficient_listing_history = has_universe_history & ~passes_listing_history
        exclusion_reasons = _append_exclusion_reason(
            exclusion_reasons,
            insufficient_listing_history,
            "insufficient_listing_history",
        )

    if minimum_price is not None:
        passes_min_price = filtered["universe_lagged_close"].ge(minimum_price)
        filtered["passes_universe_min_price"] = passes_min_price
        below_min_price = has_universe_history & ~passes_min_price
        exclusion_reasons = _append_exclusion_reason(
            exclusion_reasons,
            below_min_price,
            "below_min_price",
        )

    if minimum_average_volume is not None:
        filtered, average_volume_column = _ensure_rolling_mean_column(
            filtered,
            source_column="volume",
            window=universe_average_volume_window,
            column_prefix="rolling_average_volume",
        )
        lagged_average_volume_column = (
            f"universe_lagged_average_volume_{universe_average_volume_window}d"
        )
        filtered[lagged_average_volume_column] = filtered.groupby(
            "symbol", sort=False
        )[average_volume_column].shift(universe_lag)
        passes_min_average_volume = filtered[lagged_average_volume_column].ge(
            minimum_average_volume
        )
        filtered["passes_universe_min_average_volume"] = passes_min_average_volume
        insufficient_average_volume_history = (
            has_universe_history & filtered[lagged_average_volume_column].isna()
        )
        below_min_average_volume = (
            has_universe_history
            & filtered[lagged_average_volume_column].notna()
            & ~passes_min_average_volume
        )
        exclusion_reasons = _append_exclusion_reason(
            exclusion_reasons,
            insufficient_average_volume_history,
            "insufficient_average_volume_history",
        )
        exclusion_reasons = _append_exclusion_reason(
            exclusion_reasons,
            below_min_average_volume,
            "below_min_average_volume",
        )

    if minimum_average_dollar_volume is not None:
        filtered["daily_dollar_volume"] = filtered["close"] * filtered["volume"]
        filtered, rolling_adv_column = _ensure_rolling_mean_column(
            filtered,
            source_column="daily_dollar_volume",
            window=universe_average_dollar_volume_window,
            column_prefix="rolling_average_dollar_volume",
        )
        lagged_average_dollar_volume_column = (
            "universe_lagged_average_dollar_volume_"
            f"{universe_average_dollar_volume_window}d"
        )
        filtered[lagged_average_dollar_volume_column] = filtered.groupby(
            "symbol", sort=False
        )[rolling_adv_column].shift(universe_lag)
        passes_min_average_dollar_volume = filtered[
            lagged_average_dollar_volume_column
        ].ge(minimum_average_dollar_volume)
        filtered["passes_universe_min_average_dollar_volume"] = (
            passes_min_average_dollar_volume
        )
        insufficient_adv_history = (
            has_universe_history
            & filtered[lagged_average_dollar_volume_column].isna()
        )
        below_min_average_dollar_volume = (
            has_universe_history
            & filtered[lagged_average_dollar_volume_column].notna()
            & ~passes_min_average_dollar_volume
        )
        exclusion_reasons = _append_exclusion_reason(
            exclusion_reasons,
            insufficient_adv_history,
            "insufficient_adv_history",
        )
        exclusion_reasons = _append_exclusion_reason(
            exclusion_reasons,
            below_min_average_dollar_volume,
            "below_min_average_dollar_volume",
        )

    filtered["is_universe_eligible"] = exclusion_reasons.eq("")
    filtered["universe_exclusion_reason"] = exclusion_reasons.astype("string")
    return filtered


def _attach_symbol_metadata(
    dataset: pd.DataFrame,
    symbol_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Attach validated symbol metadata and fail on listing-window violations."""
    validated_metadata = validate_symbol_metadata(
        symbol_metadata,
        source="symbol metadata input",
    )
    merged = dataset.merge(
        validated_metadata,
        on="symbol",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    missing_metadata = merged["listing_date"].isna()
    if missing_metadata.any():
        missing_symbols = (
            merged.loc[missing_metadata, "symbol"].drop_duplicates().sort_values().tolist()
        )
        missing_text = ", ".join(str(symbol) for symbol in missing_symbols)
        raise DataValidationError(
            "symbol metadata input is missing rows for market-data symbols: "
            f"{missing_text}."
        )

    before_listing = merged["date"].lt(merged["listing_date"])
    if before_listing.any():
        invalid_rows = (
            merged.loc[before_listing, ["symbol", "date", "listing_date"]]
            .drop_duplicates()
            .sort_values(["symbol", "date"])
        )
        sample = ", ".join(
            f"({row.symbol}, {row.date.date().isoformat()} < {row.listing_date.date().isoformat()})"
            for row in invalid_rows.itertuples(index=False)
        )
        raise DataValidationError(
            "research dataset input contains rows before symbol listing_date: "
            f"{sample}."
        )

    after_delisting = merged["delisting_date"].notna() & merged["date"].gt(
        merged["delisting_date"]
    )
    if after_delisting.any():
        invalid_rows = (
            merged.loc[after_delisting, ["symbol", "date", "delisting_date"]]
            .drop_duplicates()
            .sort_values(["symbol", "date"])
        )
        sample = ", ".join(
            f"({row.symbol}, {row.date.date().isoformat()} > {row.delisting_date.date().isoformat()})"
            for row in invalid_rows.itertuples(index=False)
        )
        raise DataValidationError(
            "research dataset input contains rows after symbol delisting_date: "
            f"{sample}."
        )

    return merged

def _compute_listing_history_days(
    dataset: pd.DataFrame,
    *,
    trading_calendar: pd.DataFrame | None,
) -> pd.Series:
    """Compute listing history using a calendar when one is configured."""
    if "listing_date" not in dataset.columns:
        return dataset.groupby("symbol", sort=False).cumcount().add(1)

    if trading_calendar is None:
        calendar_dates = pd.Index(dataset["date"].drop_duplicates().sort_values())
    else:
        calendar_dates = pd.Index(trading_calendar["date"])

    date_positions = pd.Series(
        np.arange(len(calendar_dates), dtype=float),
        index=calendar_dates,
    )
    current_positions = dataset["date"].map(date_positions)
    listing_start_positions = pd.Series(
        calendar_dates.searchsorted(dataset["listing_date"], side="left"),
        index=dataset.index,
        dtype=float,
    )
    return current_positions.sub(listing_start_positions).add(1.0)


def _ensure_rolling_mean_column(
    dataset: pd.DataFrame,
    *,
    source_column: str,
    window: int,
    column_prefix: str,
) -> tuple[pd.DataFrame, str]:
    """Ensure a per-symbol rolling mean column exists and return its name."""
    column_name = f"{column_prefix}_{window}d"
    if column_name in dataset.columns:
        return dataset, column_name

    updated = dataset.copy()
    source_by_symbol = updated.groupby("symbol", sort=False)[source_column]
    updated[column_name] = source_by_symbol.transform(
        lambda values: values.rolling(
            window=window,
            min_periods=window,
        ).mean()
    )
    return updated, column_name


def _resolve_optional_window(
    value: int | None,
    *,
    fallback_window: int,
    parameter_name: str,
) -> int:
    """Resolve an optional rolling window with a validated fallback."""
    if value is None:
        return fallback_window
    return _normalize_window(value, parameter_name=parameter_name)


def _append_exclusion_reason(
    exclusion_reasons: pd.Series,
    mask: pd.Series,
    reason: str,
) -> pd.Series:
    """Append a symbolic exclusion reason to the selected rows."""
    if not bool(mask.any()):
        return exclusion_reasons

    updated = exclusion_reasons.copy()
    updated.loc[mask] = updated.loc[mask].map(
        lambda current: reason if current == "" else f"{current};{reason}"
    )
    return updated

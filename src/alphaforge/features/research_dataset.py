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
from alphaforge.features.fundamentals_join import attach_fundamentals_asof

ForwardHorizonInput = Union[int, Sequence[int]]


def build_research_dataset(
    frame: pd.DataFrame,
    *,
    trading_calendar: pd.DataFrame | None = None,
    symbol_metadata: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    fundamental_metrics: Sequence[str] | None = None,
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
    - optional universe filters use lagged per-symbol observations from
      ``universe_filter_date`` so the filter itself stays explicit
    """
    if fundamental_metrics is not None and fundamentals is None:
        raise ValueError(
            "fundamental_metrics requires fundamentals to be provided."
        )
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

    if fundamentals is not None:
        dataset = attach_fundamentals_asof(
            dataset,
            fundamentals,
            trading_calendar=validated_trading_calendar,
            metrics=fundamental_metrics,
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

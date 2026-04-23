"""Adjusted-price utilities for daily OHLCV data."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from alphaforge.data.corporate_actions import validate_corporate_actions
from alphaforge.data.market_data import CANONICAL_OHLCV_COLUMNS, validate_ohlcv

_PRICE_COLUMNS = ("open", "high", "low", "close")
_FACTOR_COLUMNS = ("price_adjustment_factor", "volume_adjustment_factor")


def apply_split_adjustments(
    ohlcv: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    *,
    ohlcv_source: str = "ohlcv",
    corporate_actions_source: str = "corporate actions",
) -> pd.DataFrame:
    """Return a backward split-adjusted OHLCV panel.

    The adjustment is purely representational: rows dated strictly before a split
    ex-date are rescaled so split events do not create artificial price jumps.
    Ex-date rows themselves are left unchanged because daily prices on the ex-date
    are already assumed to trade on the post-split basis.
    """

    validated_ohlcv = validate_ohlcv(ohlcv, source=ohlcv_source)
    validated_actions = validate_corporate_actions(
        corporate_actions,
        source=corporate_actions_source,
    )

    adjusted = validated_ohlcv.copy()
    adjusted["price_adjustment_factor"] = 1.0
    adjusted["volume_adjustment_factor"] = 1.0

    split_actions = validated_actions.loc[
        validated_actions["action_type"].eq("split"),
        ["symbol", "ex_date", "split_ratio"],
    ].copy()
    if split_actions.empty:
        return adjusted

    adjusted_groups: list[pd.DataFrame] = []
    split_groups = {
        symbol: frame.sort_values("ex_date", ascending=False, kind="mergesort")
        for symbol, frame in split_actions.groupby("symbol", sort=False)
    }

    for symbol, symbol_frame in adjusted.groupby("symbol", sort=False):
        split_frame = split_groups.get(symbol)
        if split_frame is None or split_frame.empty:
            adjusted_groups.append(symbol_frame)
            continue

        symbol_desc = symbol_frame.sort_values("date", ascending=False, kind="mergesort")
        volume_factors = _compute_future_split_volume_factors(
            symbol_desc["date"],
            split_frame["ex_date"],
            split_frame["split_ratio"],
        )
        price_factors = 1.0 / volume_factors

        symbol_desc = symbol_desc.copy()
        symbol_desc["price_adjustment_factor"] = price_factors
        symbol_desc["volume_adjustment_factor"] = volume_factors
        for column in _PRICE_COLUMNS:
            symbol_desc[column] = symbol_desc[column] * price_factors
        symbol_desc["volume"] = symbol_desc["volume"] * volume_factors
        adjusted_groups.append(symbol_desc.sort_values("date", kind="mergesort"))

    adjusted = pd.concat(adjusted_groups, ignore_index=True)
    extra_columns = [
        column
        for column in adjusted.columns
        if column not in {*CANONICAL_OHLCV_COLUMNS, *_FACTOR_COLUMNS}
    ]
    adjusted = adjusted.loc[
        :, [*CANONICAL_OHLCV_COLUMNS, *extra_columns, *_FACTOR_COLUMNS]
    ]
    adjusted = adjusted.sort_values(["symbol", "date"], kind="mergesort")
    return adjusted.reset_index(drop=True)


def _compute_future_split_volume_factors(
    market_dates: pd.Series,
    split_dates: Iterable[pd.Timestamp],
    split_ratios: Iterable[float],
) -> np.ndarray:
    """Compute backward volume factors from future split events."""

    ordered_split_dates = list(split_dates)
    ordered_split_ratios = list(split_ratios)
    next_split_index = 0
    cumulative_ratio = 1.0
    volume_factors: list[float] = []

    for market_date in market_dates.tolist():
        while (
            next_split_index < len(ordered_split_dates)
            and ordered_split_dates[next_split_index] > market_date
        ):
            cumulative_ratio *= float(ordered_split_ratios[next_split_index])
            next_split_index += 1
        volume_factors.append(cumulative_ratio)

    return np.asarray(volume_factors, dtype="float64")

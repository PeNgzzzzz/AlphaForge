"""Point-in-time-safe shares-outstanding joins for research datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.data import DataValidationError, validate_shares_outstanding


def attach_shares_outstanding_asof(
    dataset: pd.DataFrame,
    shares_outstanding: pd.DataFrame,
    *,
    trading_calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach effective-date shares outstanding and same-row market cap.

    Timing convention:
    - ``effective_date`` is the first date the share count can apply
    - if ``effective_date`` is not a market session, the first later session is used
    - when a trading calendar is configured, session mapping comes from that calendar
    - otherwise, the next observed market date in the dataset is used
    """
    validated_shares = validate_shares_outstanding(
        shares_outstanding,
        source="shares outstanding input",
    )
    conflicting_columns = [
        column_name
        for column_name in ("shares_outstanding", "market_cap")
        if column_name in dataset.columns
    ]
    if conflicting_columns:
        conflict_text = ", ".join(conflicting_columns)
        raise DataValidationError(
            "research dataset already contains shares-outstanding output columns: "
            f"{conflict_text}."
        )

    prepared_shares = _prepare_shares_outstanding_for_join(
        validated_shares,
        dataset_dates=dataset["date"],
        trading_calendar=trading_calendar,
    )
    attached = dataset.copy()
    attached["shares_outstanding"] = np.nan

    grouped_shares = {
        symbol: symbol_frame
        for symbol, symbol_frame in prepared_shares.groupby("symbol", sort=False)
    }
    for symbol, row_indexes in attached.groupby("symbol", sort=False).groups.items():
        symbol_shares = grouped_shares.get(symbol)
        if symbol_shares is None or symbol_shares.empty:
            continue

        row_index = pd.Index(row_indexes)
        symbol_dates = attached.loc[row_index, "date"].to_numpy()
        availability_dates = symbol_shares["availability_date"].to_numpy()
        positions = availability_dates.searchsorted(symbol_dates, side="right") - 1
        valid_positions = positions >= 0
        if not bool(valid_positions.any()):
            continue

        joined_shares = np.full(len(row_index), np.nan, dtype=float)
        share_values = symbol_shares["shares_outstanding"].to_numpy(dtype=float)
        joined_shares[valid_positions] = share_values[positions[valid_positions]]
        attached.loc[row_index, "shares_outstanding"] = joined_shares

    attached["market_cap"] = attached["close"] * attached["shares_outstanding"]
    return attached


def _prepare_shares_outstanding_for_join(
    shares_outstanding: pd.DataFrame,
    *,
    dataset_dates: pd.Series,
    trading_calendar: pd.DataFrame | None,
) -> pd.DataFrame:
    """Assign first-session availability dates to shares-outstanding rows."""
    reference_dates = _resolve_reference_dates(
        dataset_dates,
        trading_calendar=trading_calendar,
    )
    availability_positions = reference_dates.searchsorted(
        shares_outstanding["effective_date"],
        side="left",
    )
    has_availability_date = availability_positions < len(reference_dates)

    prepared = shares_outstanding.copy()
    prepared["availability_date"] = pd.NaT
    if bool(has_availability_date.any()):
        prepared.loc[has_availability_date, "availability_date"] = (
            reference_dates.take(availability_positions[has_availability_date])
            .to_numpy(dtype="datetime64[ns]")
        )

    usable = prepared.loc[prepared["availability_date"].notna()].copy()
    duplicated = usable.duplicated(
        subset=["symbol", "availability_date"],
        keep=False,
    )
    if duplicated.any():
        conflicting_rows = (
            usable.loc[duplicated, ["symbol", "availability_date"]]
            .drop_duplicates()
            .sort_values(["symbol", "availability_date"])
        )
        sample = ", ".join(
            f"({row.symbol}, {row.availability_date.date().isoformat()})"
            for row in conflicting_rows.itertuples(index=False)
        )
        raise DataValidationError(
            "shares outstanding input contains multiple rows that become active on "
            "the same market session for one symbol: "
            f"{sample}."
        )

    return usable.sort_values(
        ["symbol", "availability_date", "effective_date"],
        kind="mergesort",
    ).reset_index(drop=True)


def _resolve_reference_dates(
    dataset_dates: pd.Series,
    *,
    trading_calendar: pd.DataFrame | None,
) -> pd.Index:
    """Resolve the ordered session dates used for effective-date mapping."""
    if trading_calendar is None:
        return pd.Index(dataset_dates.drop_duplicates().sort_values())
    return pd.Index(trading_calendar["date"])

"""Point-in-time-safe index membership joins for research datasets."""

from __future__ import annotations

from collections.abc import Sequence
import re

import numpy as np
import pandas as pd

from alphaforge.data import DataValidationError, validate_memberships

_NON_IDENTIFIER_PATTERN = re.compile(r"[^0-9A-Za-z]+")


def attach_memberships_asof(
    dataset: pd.DataFrame,
    memberships: pd.DataFrame,
    *,
    trading_calendar: pd.DataFrame | None = None,
    indexes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Attach the latest effective index membership status to each row.

    Timing convention:
    - ``effective_date`` is the first date a membership status can apply
    - if ``effective_date`` is not a market session, the first later session is used
    - when a trading calendar is configured, session mapping comes from that calendar
    - otherwise, the next observed market date in the dataset is used
    """
    validated_memberships = validate_memberships(
        memberships,
        source="memberships input",
    )
    selected_indexes = _normalize_selected_indexes(
        indexes,
        available_indexes=validated_memberships["index_name"],
    )
    output_columns = {
        index_name: _membership_column_name(index_name)
        for index_name in selected_indexes
    }
    if len(set(output_columns.values())) != len(output_columns):
        raise DataValidationError(
            "memberships input contains index_name values that normalize to the "
            "same dataset column name."
        )
    conflicting_columns = [
        column_name
        for column_name in output_columns.values()
        if column_name in dataset.columns
    ]
    if conflicting_columns:
        conflict_text = ", ".join(conflicting_columns)
        raise DataValidationError(
            "research dataset already contains membership output columns: "
            f"{conflict_text}."
        )

    prepared_memberships = _prepare_memberships_for_join(
        validated_memberships,
        dataset_dates=dataset["date"],
        trading_calendar=trading_calendar,
        selected_indexes=selected_indexes,
    )
    attached = dataset.copy()
    for column_name in output_columns.values():
        attached[column_name] = pd.Series(pd.NA, index=attached.index, dtype="boolean")

    grouped_memberships = {
        symbol: symbol_frame
        for symbol, symbol_frame in prepared_memberships.groupby("symbol", sort=False)
    }
    for symbol, row_indexes in attached.groupby("symbol", sort=False).groups.items():
        symbol_memberships = grouped_memberships.get(symbol)
        if symbol_memberships is None or symbol_memberships.empty:
            continue

        row_index = pd.Index(row_indexes)
        symbol_dates = attached.loc[row_index, "date"].to_numpy()
        for index_name, column_name in output_columns.items():
            membership_rows = symbol_memberships.loc[
                symbol_memberships["index_name"] == index_name,
                ["availability_date", "is_member"],
            ]
            if membership_rows.empty:
                continue

            availability_dates = membership_rows["availability_date"].to_numpy()
            membership_values = membership_rows["is_member"].to_numpy(dtype=bool)
            positions = availability_dates.searchsorted(symbol_dates, side="right") - 1
            joined_values = np.empty(len(row_index), dtype=object)
            joined_values[:] = pd.NA
            valid_positions = positions >= 0
            joined_values[valid_positions] = membership_values[
                positions[valid_positions]
            ]
            attached.loc[row_index, column_name] = pd.array(
                joined_values,
                dtype="boolean",
            )

    return attached


def _prepare_memberships_for_join(
    memberships: pd.DataFrame,
    *,
    dataset_dates: pd.Series,
    trading_calendar: pd.DataFrame | None,
    selected_indexes: tuple[str, ...],
) -> pd.DataFrame:
    """Filter memberships and assign first-session availability dates."""
    filtered = memberships.loc[
        memberships["index_name"].isin(selected_indexes)
    ].copy()
    reference_dates = _resolve_reference_dates(
        dataset_dates,
        trading_calendar=trading_calendar,
    )
    availability_positions = reference_dates.searchsorted(
        filtered["effective_date"],
        side="left",
    )
    has_availability_date = availability_positions < len(reference_dates)
    filtered["availability_date"] = pd.NaT
    if bool(has_availability_date.any()):
        filtered.loc[has_availability_date, "availability_date"] = (
            reference_dates.take(availability_positions[has_availability_date])
            .to_numpy(dtype="datetime64[ns]")
        )

    usable = filtered.loc[filtered["availability_date"].notna()].copy()
    duplicated = usable.duplicated(
        subset=["symbol", "index_name", "availability_date"],
        keep=False,
    )
    if duplicated.any():
        conflicting_rows = (
            usable.loc[
                duplicated,
                ["symbol", "index_name", "availability_date"],
            ]
            .drop_duplicates()
            .sort_values(["symbol", "index_name", "availability_date"])
        )
        sample = ", ".join(
            (
                f"({row.symbol}, {row.index_name}, "
                f"{row.availability_date.date().isoformat()})"
            )
            for row in conflicting_rows.itertuples(index=False)
        )
        raise DataValidationError(
            "memberships input contains multiple rows that become active on "
            "the same market session for one symbol/index: "
            f"{sample}."
        )

    return usable.sort_values(
        ["symbol", "index_name", "availability_date", "effective_date"],
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


def _normalize_selected_indexes(
    indexes: Sequence[str] | None,
    *,
    available_indexes: pd.Series,
) -> tuple[str, ...]:
    """Normalize and validate the selected index-name list."""
    if indexes is None:
        normalized_indexes = tuple(
            pd.Index(available_indexes).drop_duplicates().sort_values().tolist()
        )
    else:
        normalized_indexes = tuple(
            _normalize_index_name(index_name) for index_name in indexes
        )

    if not normalized_indexes:
        raise ValueError("membership_indexes must contain at least one index name.")
    if len(set(normalized_indexes)) != len(normalized_indexes):
        raise ValueError("membership_indexes must not contain duplicate index names.")

    available = set(available_indexes.tolist())
    missing_indexes = sorted(
        index_name for index_name in normalized_indexes if index_name not in available
    )
    if missing_indexes:
        missing_text = ", ".join(missing_indexes)
        raise DataValidationError(
            "memberships input is missing configured index_name values: "
            f"{missing_text}."
        )
    return normalized_indexes


def _normalize_index_name(index_name: str) -> str:
    """Normalize one selected index name."""
    if not isinstance(index_name, str):
        raise ValueError("membership_indexes must contain only non-empty strings.")
    normalized = index_name.strip()
    if normalized == "":
        raise ValueError("membership_indexes must contain only non-empty strings.")
    return normalized


def _membership_column_name(index_name: str) -> str:
    """Normalize an index name into a deterministic dataset column name."""
    normalized = _NON_IDENTIFIER_PATTERN.sub("_", index_name.strip().lower()).strip(
        "_"
    )
    if normalized == "":
        raise DataValidationError(
            "memberships input contains an index_name that cannot be converted "
            "into a dataset column name."
        )
    return f"membership_{normalized}"

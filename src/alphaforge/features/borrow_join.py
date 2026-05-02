"""Point-in-time-safe borrow availability joins for research datasets."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from alphaforge.common.validation import (
    normalize_unique_non_empty_string_sequence as _common_string_sequence,
)
from alphaforge.data import DataValidationError, validate_borrow_availability

_BORROW_OUTPUT_COLUMNS = {
    "is_borrowable": "borrow_is_borrowable",
    "borrow_fee_bps": "borrow_fee_bps",
}


def attach_borrow_availability_asof(
    dataset: pd.DataFrame,
    borrow_availability: pd.DataFrame,
    *,
    trading_calendar: pd.DataFrame | None = None,
    fields: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Attach the latest effective borrow availability state to each row.

    Timing convention:
    - ``effective_date`` is the first date a borrow state can apply
    - if ``effective_date`` is not a market session, the first later session is used
    - when a trading calendar is configured, session mapping comes from that calendar
    - otherwise, the next observed market date in the dataset is used
    """
    validated_borrow = validate_borrow_availability(
        borrow_availability,
        source="borrow availability input",
    )
    selected_fields = _normalize_selected_fields(fields)
    output_columns = {
        field_name: _BORROW_OUTPUT_COLUMNS[field_name] for field_name in selected_fields
    }
    conflicting_columns = [
        column_name
        for column_name in output_columns.values()
        if column_name in dataset.columns
    ]
    if conflicting_columns:
        conflict_text = ", ".join(conflicting_columns)
        raise DataValidationError(
            "research dataset already contains borrow availability output columns: "
            f"{conflict_text}."
        )

    prepared_borrow = _prepare_borrow_availability_for_join(
        validated_borrow,
        dataset_dates=dataset["date"],
        trading_calendar=trading_calendar,
    )
    attached = dataset.copy()
    for field_name, column_name in output_columns.items():
        if field_name == "is_borrowable":
            attached[column_name] = pd.Series(
                pd.NA,
                index=attached.index,
                dtype="boolean",
            )
        else:
            attached[column_name] = np.nan

    grouped_borrow = {
        symbol: symbol_frame
        for symbol, symbol_frame in prepared_borrow.groupby("symbol", sort=False)
    }
    for symbol, row_indexes in attached.groupby("symbol", sort=False).groups.items():
        symbol_borrow = grouped_borrow.get(symbol)
        if symbol_borrow is None or symbol_borrow.empty:
            continue

        row_index = pd.Index(row_indexes)
        symbol_dates = attached.loc[row_index, "date"].to_numpy()
        availability_dates = symbol_borrow["availability_date"].to_numpy()
        positions = availability_dates.searchsorted(symbol_dates, side="right") - 1
        valid_positions = positions >= 0
        if not bool(valid_positions.any()):
            continue

        valid_row_index = row_index[valid_positions]
        selected_positions = positions[valid_positions]
        if "is_borrowable" in output_columns:
            borrowable_values = symbol_borrow["is_borrowable"].to_numpy(dtype=bool)
            joined_borrowable = np.empty(len(valid_row_index), dtype=object)
            joined_borrowable[:] = pd.NA
            joined_borrowable[:] = borrowable_values[selected_positions]
            attached.loc[valid_row_index, output_columns["is_borrowable"]] = pd.array(
                joined_borrowable,
                dtype="boolean",
            )
        if "borrow_fee_bps" in output_columns:
            fee_values = symbol_borrow["borrow_fee_bps"].to_numpy(dtype=float)
            attached.loc[
                valid_row_index,
                output_columns["borrow_fee_bps"],
            ] = fee_values[selected_positions]

    return attached


def _prepare_borrow_availability_for_join(
    borrow_availability: pd.DataFrame,
    *,
    dataset_dates: pd.Series,
    trading_calendar: pd.DataFrame | None,
) -> pd.DataFrame:
    """Assign first-session availability dates to borrow rows."""
    reference_dates = _resolve_reference_dates(
        dataset_dates,
        trading_calendar=trading_calendar,
    )
    availability_positions = reference_dates.searchsorted(
        borrow_availability["effective_date"],
        side="left",
    )
    has_availability_date = availability_positions < len(reference_dates)

    prepared = borrow_availability.copy()
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
            "borrow availability input contains multiple rows that become active on "
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


def _normalize_selected_fields(fields: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize and validate selected borrow fields."""
    if fields is None:
        normalized_fields = tuple(_BORROW_OUTPUT_COLUMNS)
    else:
        normalized_fields = tuple(
            field_name.lower()
            for field_name in _common_string_sequence(
                fields,
                parameter_name="borrow_fields",
                item_error_message="borrow_fields must contain only non-empty strings.",
                duplicate_error_message=(
                    "borrow_fields must not contain duplicate fields."
                ),
            )
        )

    if not normalized_fields:
        raise ValueError("borrow_fields must contain at least one field.")
    if len(set(normalized_fields)) != len(normalized_fields):
        raise ValueError("borrow_fields must not contain duplicate fields.")
    invalid_fields = [
        field_name
        for field_name in normalized_fields
        if field_name not in _BORROW_OUTPUT_COLUMNS
    ]
    if invalid_fields:
        invalid_text = ", ".join(invalid_fields)
        raise ValueError(
            "borrow_fields must contain only 'is_borrowable' or 'borrow_fee_bps': "
            f"{invalid_text}."
        )
    return normalized_fields

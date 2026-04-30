"""Point-in-time-safe trading status joins for research datasets."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from alphaforge.data import DataValidationError, validate_trading_status

_TRADING_STATUS_OUTPUT_COLUMNS = {
    "is_tradable": "trading_is_tradable",
    "status_reason": "trading_status_reason",
}


def attach_trading_status_asof(
    dataset: pd.DataFrame,
    trading_status: pd.DataFrame,
    *,
    trading_calendar: pd.DataFrame | None = None,
    fields: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Attach the latest effective trading status to each row.

    Timing convention:
    - ``effective_date`` is the first date a trading status can apply
    - if ``effective_date`` is not a market session, the first later session is used
    - when a trading calendar is configured, session mapping comes from that calendar
    - otherwise, the next observed market date in the dataset is used
    """
    validated_status = validate_trading_status(
        trading_status,
        source="trading status input",
    )
    selected_fields = _normalize_selected_fields(fields)
    output_columns = {
        field_name: _TRADING_STATUS_OUTPUT_COLUMNS[field_name]
        for field_name in selected_fields
    }
    conflicting_columns = [
        column_name
        for column_name in output_columns.values()
        if column_name in dataset.columns
    ]
    if conflicting_columns:
        conflict_text = ", ".join(conflicting_columns)
        raise DataValidationError(
            "research dataset already contains trading status output columns: "
            f"{conflict_text}."
        )

    prepared_status = _prepare_trading_status_for_join(
        validated_status,
        dataset_dates=dataset["date"],
        trading_calendar=trading_calendar,
    )
    attached = dataset.copy()
    for field_name, column_name in output_columns.items():
        if field_name == "is_tradable":
            attached[column_name] = pd.Series(
                pd.NA,
                index=attached.index,
                dtype="boolean",
            )
        else:
            attached[column_name] = pd.Series(
                pd.NA,
                index=attached.index,
                dtype="string",
            )

    grouped_status = {
        symbol: symbol_frame
        for symbol, symbol_frame in prepared_status.groupby("symbol", sort=False)
    }
    for symbol, row_indexes in attached.groupby("symbol", sort=False).groups.items():
        symbol_status = grouped_status.get(symbol)
        if symbol_status is None or symbol_status.empty:
            continue

        row_index = pd.Index(row_indexes)
        symbol_dates = attached.loc[row_index, "date"].to_numpy()
        availability_dates = symbol_status["availability_date"].to_numpy()
        positions = availability_dates.searchsorted(symbol_dates, side="right") - 1
        valid_positions = positions >= 0
        if not bool(valid_positions.any()):
            continue

        valid_row_index = row_index[valid_positions]
        selected_positions = positions[valid_positions]
        if "is_tradable" in output_columns:
            tradable_values = symbol_status["is_tradable"].to_numpy(dtype=bool)
            joined_tradable = np.empty(len(valid_row_index), dtype=object)
            joined_tradable[:] = pd.NA
            joined_tradable[:] = tradable_values[selected_positions]
            attached.loc[
                valid_row_index,
                output_columns["is_tradable"],
            ] = pd.array(joined_tradable, dtype="boolean")
        if "status_reason" in output_columns:
            reason_values = symbol_status["status_reason"].astype("string").to_numpy()
            attached.loc[
                valid_row_index,
                output_columns["status_reason"],
            ] = pd.array(reason_values[selected_positions], dtype="string")

    return attached


def _prepare_trading_status_for_join(
    trading_status: pd.DataFrame,
    *,
    dataset_dates: pd.Series,
    trading_calendar: pd.DataFrame | None,
) -> pd.DataFrame:
    """Assign first-session availability dates to trading status rows."""
    reference_dates = _resolve_reference_dates(
        dataset_dates,
        trading_calendar=trading_calendar,
    )
    availability_positions = reference_dates.searchsorted(
        trading_status["effective_date"],
        side="left",
    )
    has_availability_date = availability_positions < len(reference_dates)

    prepared = trading_status.copy()
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
            "trading status input contains multiple rows that become active on "
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
    """Normalize and validate selected trading status fields."""
    if fields is None:
        normalized_fields = tuple(_TRADING_STATUS_OUTPUT_COLUMNS)
    else:
        normalized_fields = tuple(_normalize_field_name(field_name) for field_name in fields)

    if not normalized_fields:
        raise ValueError("trading_status_fields must contain at least one field.")
    if len(set(normalized_fields)) != len(normalized_fields):
        raise ValueError("trading_status_fields must not contain duplicate fields.")
    invalid_fields = [
        field_name
        for field_name in normalized_fields
        if field_name not in _TRADING_STATUS_OUTPUT_COLUMNS
    ]
    if invalid_fields:
        invalid_text = ", ".join(invalid_fields)
        raise ValueError(
            "trading_status_fields must contain only 'is_tradable' or "
            f"'status_reason': {invalid_text}."
        )
    return normalized_fields


def _normalize_field_name(field_name: str) -> str:
    """Normalize one selected trading status field."""
    if not isinstance(field_name, str):
        raise ValueError("trading_status_fields must contain only non-empty strings.")
    normalized = field_name.strip().lower()
    if normalized == "":
        raise ValueError("trading_status_fields must contain only non-empty strings.")
    return normalized

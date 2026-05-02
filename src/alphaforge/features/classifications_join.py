"""Point-in-time-safe sector and industry classification joins."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from alphaforge.common.validation import (
    normalize_unique_non_empty_string_sequence as _common_string_sequence,
)
from alphaforge.data import DataValidationError, validate_classifications


def attach_classifications_asof(
    dataset: pd.DataFrame,
    classifications: pd.DataFrame,
    *,
    trading_calendar: pd.DataFrame | None = None,
    fields: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Attach the latest effective sector/industry classifications to each row.

    Timing convention:
    - ``effective_date`` is treated as the first date the classification can apply
    - if ``effective_date`` is not a market session, the first later session is used
    - when a trading calendar is configured, session mapping comes from that calendar
    - otherwise, the next observed market date in the dataset is used
    """
    validated_classifications = validate_classifications(
        classifications,
        source="classifications input",
    )
    selected_fields = _normalize_selected_fields(fields)
    output_columns = {
        field_name: f"classification_{field_name}" for field_name in selected_fields
    }
    conflicting_columns = [
        column_name
        for column_name in output_columns.values()
        if column_name in dataset.columns
    ]
    if conflicting_columns:
        conflict_text = ", ".join(conflicting_columns)
        raise DataValidationError(
            "research dataset already contains classification output columns: "
            f"{conflict_text}."
        )

    prepared_classifications = _prepare_classifications_for_join(
        validated_classifications,
        dataset_dates=dataset["date"],
        trading_calendar=trading_calendar,
    )
    attached = dataset.copy()
    for column_name in output_columns.values():
        attached[column_name] = pd.Series(pd.NA, index=attached.index, dtype="string")

    grouped_classifications = {
        symbol: symbol_frame
        for symbol, symbol_frame in prepared_classifications.groupby("symbol", sort=False)
    }
    for symbol, row_indexes in attached.groupby("symbol", sort=False).groups.items():
        symbol_classifications = grouped_classifications.get(symbol)
        if symbol_classifications is None or symbol_classifications.empty:
            continue

        row_index = pd.Index(row_indexes)
        symbol_dates = attached.loc[row_index, "date"].to_numpy()
        availability_dates = symbol_classifications["availability_date"].to_numpy()
        positions = availability_dates.searchsorted(symbol_dates, side="right") - 1
        valid_positions = positions >= 0
        if not bool(valid_positions.any()):
            continue

        valid_row_index = row_index[valid_positions]
        selected_positions = positions[valid_positions]
        for field_name, column_name in output_columns.items():
            field_values = symbol_classifications[field_name].to_numpy(dtype=object)
            attached.loc[valid_row_index, column_name] = field_values[selected_positions]

    return attached


def _prepare_classifications_for_join(
    classifications: pd.DataFrame,
    *,
    dataset_dates: pd.Series,
    trading_calendar: pd.DataFrame | None,
) -> pd.DataFrame:
    """Assign first-session availability dates to classification rows."""
    reference_dates = _resolve_reference_dates(
        dataset_dates,
        trading_calendar=trading_calendar,
    )
    availability_positions = reference_dates.searchsorted(
        classifications["effective_date"],
        side="left",
    )
    has_availability_date = availability_positions < len(reference_dates)

    prepared = classifications.copy()
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
            "classifications input contains multiple rows that become active on "
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
    """Normalize the selected classification fields."""
    if fields is None:
        normalized_fields = ("sector", "industry")
    else:
        normalized_fields = tuple(
            field_name.lower()
            for field_name in _common_string_sequence(
                fields,
                parameter_name="classification_fields",
                item_error_message=(
                    "classification_fields must contain only non-empty strings."
                ),
                duplicate_error_message=(
                    "classification_fields must not contain duplicate fields."
                ),
            )
        )

    if not normalized_fields:
        raise ValueError("classification_fields must contain at least one field.")
    if len(set(normalized_fields)) != len(normalized_fields):
        raise ValueError("classification_fields must not contain duplicate fields.")
    invalid_fields = [
        field_name
        for field_name in normalized_fields
        if field_name not in {"sector", "industry"}
    ]
    if invalid_fields:
        invalid_text = ", ".join(invalid_fields)
        raise ValueError(
            "classification_fields must contain only 'sector' or 'industry': "
            f"{invalid_text}."
        )
    return normalized_fields

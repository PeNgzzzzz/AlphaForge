"""Market-cap bucket features for research datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.common.validation import (
    normalize_non_empty_string as _common_non_empty_string,
    require_columns as _common_require_columns,
)
from alphaforge.data import DataValidationError


def attach_market_cap_buckets(
    dataset: pd.DataFrame,
    *,
    n_buckets: int,
    market_cap_column: str = "market_cap",
    output_column: str = "market_cap_bucket",
) -> pd.DataFrame:
    """Attach same-date cross-sectional market-cap quantile buckets.

    Bucket labels are one-based integers where ``1`` is the smallest market-cap
    bucket for that date. Dates with too few usable names, too few distinct
    market caps, or duplicate quantile edges are left missing rather than forced
    into arbitrary buckets.
    """
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("attach_market_cap_buckets expects a pandas DataFrame.")
    n_buckets = _normalize_bucket_count(n_buckets)
    market_cap_column = _normalize_column_name(
        market_cap_column,
        field_name="market_cap_column",
    )
    output_column = _normalize_column_name(output_column, field_name="output_column")

    _validate_required_columns(
        dataset,
        required_columns=("date", market_cap_column),
    )
    if output_column in dataset.columns:
        raise DataValidationError(
            "research dataset already contains market-cap bucket output column: "
            f"{output_column}."
        )
    if dataset["date"].isna().any():
        raise DataValidationError(
            "market-cap bucket input contains missing date values."
        )

    numeric_market_cap = pd.to_numeric(dataset[market_cap_column], errors="coerce")
    raw_market_cap = dataset[market_cap_column]
    market_cap_values = numeric_market_cap.to_numpy(dtype=float, na_value=np.nan)
    present_market_cap = raw_market_cap.notna()
    finite_market_cap = pd.Series(
        np.isfinite(market_cap_values),
        index=dataset.index,
    )

    if (present_market_cap & ~finite_market_cap).any():
        raise DataValidationError(
            f"{market_cap_column} must be finite when present."
        )
    if (finite_market_cap & numeric_market_cap.le(0.0)).any():
        raise DataValidationError(
            f"{market_cap_column} must be strictly positive when present."
        )

    buckets = pd.Series(
        pd.array([pd.NA] * len(dataset), dtype="Int64"),
        index=dataset.index,
        name=output_column,
    )
    labels = list(range(1, n_buckets + 1))

    for _, row_indexes in dataset.groupby("date", sort=False).groups.items():
        row_index = pd.Index(row_indexes)
        usable_index = row_index[finite_market_cap.loc[row_index].to_numpy()]
        if len(usable_index) < n_buckets:
            continue

        date_market_cap = numeric_market_cap.loc[usable_index]
        if date_market_cap.nunique(dropna=True) < n_buckets:
            continue

        try:
            assigned = pd.qcut(
                date_market_cap,
                q=n_buckets,
                labels=labels,
                duplicates="raise",
            )
        except ValueError:
            continue

        buckets.loc[usable_index] = pd.Series(
            assigned,
            index=usable_index,
        ).astype("Int64")

    attached = dataset.copy()
    attached[output_column] = buckets
    return attached


def _normalize_bucket_count(value: int) -> int:
    """Validate the configured market-cap bucket count."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError("n_buckets must be an integer greater than or equal to 2.")
    return value


def _normalize_column_name(value: str, *, field_name: str) -> str:
    """Validate user-facing column-name parameters."""
    return _common_non_empty_string(value, parameter_name=field_name)


def _validate_required_columns(
    dataset: pd.DataFrame,
    *,
    required_columns: tuple[str, ...],
) -> None:
    """Fail loudly when required input columns are absent."""
    _common_require_columns(
        dataset.columns,
        required_columns,
        source="market-cap bucket input",
        error_factory=DataValidationError,
    )

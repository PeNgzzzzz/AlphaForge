"""Corporate-action loading and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphaforge.data._validation import (
    DataValidationError,
    PathLike,
    parse_daily_dates,
    parse_optional_numeric_column,
    parse_symbols,
)

CANONICAL_CORPORATE_ACTION_COLUMNS = (
    "symbol",
    "ex_date",
    "action_type",
    "split_ratio",
    "cash_amount",
)
_SUPPORTED_ACTION_TYPES = {"cash_dividend", "split"}


def load_corporate_actions(
    path: PathLike,
    *,
    ex_date_column: str = "ex_date",
    action_type_column: str = "action_type",
    split_ratio_column: str = "split_ratio",
    cash_amount_column: str = "cash_amount",
) -> pd.DataFrame:
    """Load and validate corporate actions from CSV or Parquet."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        frame = pd.read_csv(file_path)
    elif suffix == ".parquet":
        frame = pd.read_parquet(file_path)
    else:
        raise ValueError(
            f"Unsupported file format for {file_path}. Expected .csv or .parquet."
        )

    return validate_corporate_actions(
        frame,
        ex_date_column=ex_date_column,
        action_type_column=action_type_column,
        split_ratio_column=split_ratio_column,
        cash_amount_column=cash_amount_column,
        source=str(file_path),
    )


def validate_corporate_actions(
    frame: pd.DataFrame,
    *,
    ex_date_column: str = "ex_date",
    action_type_column: str = "action_type",
    split_ratio_column: str = "split_ratio",
    cash_amount_column: str = "cash_amount",
    source: str = "dataframe",
) -> pd.DataFrame:
    """Validate a corporate-actions frame and return a normalized copy."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("validate_corporate_actions expects a pandas DataFrame.")

    for field_name, column_name in {
        "ex_date_column": ex_date_column,
        "action_type_column": action_type_column,
        "split_ratio_column": split_ratio_column,
        "cash_amount_column": cash_amount_column,
    }.items():
        if not isinstance(column_name, str) or not column_name.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")

    required_columns = ["symbol", ex_date_column, action_type_column]
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise DataValidationError(
            f"{source} is missing required corporate action columns: {missing_text}."
        )

    validated = frame.copy()
    validated["symbol"] = parse_symbols(validated["symbol"], source=source)
    validated["ex_date"] = parse_daily_dates(
        validated[ex_date_column],
        source=source,
        dataset_label="corporate action data",
    )

    action_types = validated[action_type_column].astype("string").str.strip().str.lower()
    if action_types.isna().any() or action_types.eq("").any():
        raise DataValidationError(
            f"{source} contains missing or empty corporate action type values."
        )
    invalid_action_types = action_types[~action_types.isin(_SUPPORTED_ACTION_TYPES)]
    if not invalid_action_types.empty:
        sample = ", ".join(sorted(set(invalid_action_types.tolist())))
        raise DataValidationError(
            f"{source} contains unsupported corporate action types: {sample}."
        )
    validated["action_type"] = action_types

    has_split_ratio_column = split_ratio_column in frame.columns
    has_cash_amount_column = cash_amount_column in frame.columns

    if has_split_ratio_column:
        validated["split_ratio"] = parse_optional_numeric_column(
            validated[split_ratio_column],
            column_name=split_ratio_column,
            source=source,
        )
    else:
        validated["split_ratio"] = pd.Series(
            float("nan"),
            index=validated.index,
            dtype="float64",
        )

    if has_cash_amount_column:
        validated["cash_amount"] = parse_optional_numeric_column(
            validated[cash_amount_column],
            column_name=cash_amount_column,
            source=source,
        )
    else:
        validated["cash_amount"] = pd.Series(
            float("nan"),
            index=validated.index,
            dtype="float64",
        )

    duplicate_rows = validated.duplicated(
        subset=["symbol", "ex_date", "action_type"],
        keep=False,
    )
    if duplicate_rows.any():
        duplicates = (
            validated.loc[duplicate_rows, ["symbol", "ex_date", "action_type"]]
            .drop_duplicates()
            .sort_values(["symbol", "ex_date", "action_type"], kind="mergesort")
        )
        sample = ", ".join(
            f"{row.symbol}@{row.ex_date.date().isoformat()}:{row.action_type}"
            for row in duplicates.itertuples(index=False)
        )
        raise DataValidationError(
            f"{source} contains duplicate corporate action keys: {sample}."
        )

    _validate_split_actions(
        validated,
        has_split_ratio_column=has_split_ratio_column,
        cash_amount_column=cash_amount_column,
        split_ratio_column=split_ratio_column,
        source=source,
    )
    _validate_cash_dividend_actions(
        validated,
        cash_amount_column=cash_amount_column,
        has_cash_amount_column=has_cash_amount_column,
        split_ratio_column=split_ratio_column,
        source=source,
    )

    extra_columns = [
        column
        for column in validated.columns
        if column
        not in {
            "symbol",
            ex_date_column,
            action_type_column,
            split_ratio_column,
            cash_amount_column,
            "ex_date",
            "action_type",
            "split_ratio",
            "cash_amount",
        }
    ]
    validated = validated.loc[
        :, [*CANONICAL_CORPORATE_ACTION_COLUMNS, *extra_columns]
    ]
    if validated.empty:
        raise DataValidationError("Corporate actions must contain at least one row.")

    validated = validated.sort_values(
        ["symbol", "ex_date", "action_type"],
        kind="mergesort",
    )
    return validated.reset_index(drop=True)


def _validate_split_actions(
    frame: pd.DataFrame,
    *,
    has_split_ratio_column: bool,
    split_ratio_column: str,
    cash_amount_column: str,
    source: str,
) -> None:
    """Require split actions to use only positive split ratios."""
    split_rows = frame["action_type"].eq("split")
    if not bool(split_rows.any()):
        return

    if not has_split_ratio_column:
        raise DataValidationError(
            f"{source} contains split actions but is missing the '{split_ratio_column}' column."
        )

    split_ratios = frame.loc[split_rows, "split_ratio"]
    invalid_split_ratios = split_ratios.isna() | (~np.isfinite(split_ratios)) | (
        split_ratios <= 0.0
    )
    if invalid_split_ratios.any():
        raise DataValidationError(
            f"{source} contains split actions whose '{split_ratio_column}' values must be finite and greater than 0.0."
        )

    if frame.loc[split_rows, "cash_amount"].notna().any():
        raise DataValidationError(
            f"{source} contains split actions that also populate '{cash_amount_column}'."
        )


def _validate_cash_dividend_actions(
    frame: pd.DataFrame,
    *,
    has_cash_amount_column: bool,
    split_ratio_column: str,
    cash_amount_column: str,
    source: str,
) -> None:
    """Require cash-dividend actions to use only positive cash amounts."""
    cash_dividend_rows = frame["action_type"].eq("cash_dividend")
    if not bool(cash_dividend_rows.any()):
        return

    if not has_cash_amount_column:
        raise DataValidationError(
            f"{source} contains cash_dividend actions but is missing the '{cash_amount_column}' column."
        )

    cash_amounts = frame.loc[cash_dividend_rows, "cash_amount"]
    invalid_cash_amounts = cash_amounts.isna() | (~np.isfinite(cash_amounts)) | (
        cash_amounts <= 0.0
    )
    if invalid_cash_amounts.any():
        raise DataValidationError(
            f"{source} contains cash_dividend actions whose '{cash_amount_column}' values must be finite and greater than 0.0."
        )

    if frame.loc[cash_dividend_rows, "split_ratio"].notna().any():
        raise DataValidationError(
            f"{source} contains cash_dividend actions that also populate '{split_ratio_column}'."
        )

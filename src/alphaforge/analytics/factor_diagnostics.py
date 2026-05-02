"""Cross-sectional factor diagnostics for daily research panels."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd

from alphaforge.common.validation import normalize_positive_int as _common_positive_int


class FactorDiagnosticsError(ValueError):
    """Raised when factor diagnostics inputs or settings are invalid."""


def compute_ic_series(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    forward_return_column: str,
    method: str = "pearson",
    min_observations: int = 2,
) -> pd.DataFrame:
    """Compute per-date cross-sectional IC or rank IC."""
    method = _normalize_ic_method(method)
    min_observations = _normalize_positive_int(
        min_observations,
        parameter_name="min_observations",
    )

    dataset = _prepare_factor_frame(
        frame,
        signal_column=signal_column,
        forward_return_column=forward_return_column,
    )

    rows = []
    for date, group in dataset.groupby("date", sort=True):
        usable = group.loc[:, [signal_column, forward_return_column]].dropna()
        observations = len(usable)
        if observations < min_observations:
            ic = math.nan
        else:
            ic = _compute_ic_from_usable(
                usable,
                signal_column=signal_column,
                forward_return_column=forward_return_column,
                method=method,
            )
        rows.append(
            {
                "date": date,
                "ic": ic,
                "observations": float(observations),
                "method": method,
            }
        )

    return pd.DataFrame(rows)


def summarize_ic(ic_frame: pd.DataFrame) -> pd.Series:
    """Summarize an IC or rank IC time series."""
    required_columns = ["date", "ic", "observations"]
    missing_columns = [column for column in required_columns if column not in ic_frame.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise FactorDiagnosticsError(
            f"ic_frame is missing required columns: {missing_text}."
        )

    dataset = ic_frame.loc[:, ["date", "ic", "observations"]].copy()
    if dataset.empty:
        raise FactorDiagnosticsError("ic_frame must contain at least one row.")

    dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
    if dataset["date"].isna().any():
        raise FactorDiagnosticsError("ic_frame contains invalid date values.")

    dataset["ic"] = _parse_numeric_column(
        dataset["ic"],
        column_name="ic",
        allow_na=True,
    )
    dataset["observations"] = _parse_numeric_column(
        dataset["observations"],
        column_name="observations",
    )

    valid_ic = dataset["ic"].dropna()
    ic_std = valid_ic.std(ddof=1)
    if valid_ic.empty or pd.isna(ic_std) or ic_std == 0.0:
        ic_ir = math.nan
    else:
        ic_ir = valid_ic.mean() / ic_std

    return pd.Series(
        {
            "periods": float(len(dataset)),
            "valid_periods": float(valid_ic.count()),
            "mean_ic": valid_ic.mean(),
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "positive_ic_ratio": valid_ic.gt(0.0).mean(),
            "average_observations": dataset["observations"].mean(),
        },
        name="ic_summary",
    )


def compute_rolling_ic_series(
    ic_frame: pd.DataFrame,
    *,
    window: int = 20,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute trailing rolling diagnostics from a dated IC series.

    Each rolling value at date ``t`` uses only IC observations dated ``<= t``.
    Missing IC values are ignored, and rolling statistics remain missing until
    at least ``min_periods`` valid IC observations are available.
    """
    window = _normalize_positive_int(window, parameter_name="window")
    if window < 2:
        raise FactorDiagnosticsError("window must be greater than or equal to 2.")
    if min_periods is None:
        min_periods = window
    min_periods = _normalize_positive_int(
        min_periods,
        parameter_name="min_periods",
    )
    if min_periods > window:
        raise FactorDiagnosticsError(
            "min_periods must be less than or equal to window."
        )

    dataset = _prepare_ic_frame(ic_frame, required_columns=("date", "ic"))
    rolling_ic = dataset["ic"].rolling(window=window, min_periods=min_periods)
    rolling_mean_ic = rolling_ic.mean()
    rolling_ic_std = rolling_ic.std(ddof=1)
    positive_ic = (
        dataset["ic"].gt(0.0).astype("float64").where(dataset["ic"].notna())
    )

    result = pd.DataFrame(
        {
            "date": dataset["date"],
            "rolling_mean_ic": rolling_mean_ic,
            "rolling_ic_std": rolling_ic_std,
            "rolling_positive_ic_ratio": positive_ic.rolling(
                window=window,
                min_periods=min_periods,
            ).mean(),
            "rolling_valid_periods": rolling_ic.count(),
            "window": float(window),
            "min_periods": float(min_periods),
        }
    )
    result["rolling_ic_ir"] = result["rolling_mean_ic"].div(
        result["rolling_ic_std"]
    )
    invalid_ir = (
        result["rolling_ic_std"].isna()
        | result["rolling_ic_std"].eq(0.0)
    )
    result.loc[invalid_ir, "rolling_ic_ir"] = math.nan
    return result.loc[
        :,
        [
            "date",
            "rolling_mean_ic",
            "rolling_ic_std",
            "rolling_ic_ir",
            "rolling_positive_ic_ratio",
            "rolling_valid_periods",
            "window",
            "min_periods",
        ],
    ]


def summarize_rolling_ic(rolling_ic_frame: pd.DataFrame) -> pd.Series:
    """Summarize a trailing rolling IC diagnostic series."""
    required_columns = [
        "date",
        "rolling_mean_ic",
        "rolling_ic_ir",
        "rolling_positive_ic_ratio",
        "rolling_valid_periods",
        "window",
        "min_periods",
    ]
    dataset = _prepare_ic_frame(
        rolling_ic_frame,
        required_columns=tuple(required_columns),
    )
    for column in required_columns:
        if column == "date":
            continue
        dataset[column] = _parse_numeric_column(
            dataset[column],
            column_name=column,
            allow_na=column.startswith("rolling_"),
        )

    valid_rows = dataset.loc[dataset["rolling_mean_ic"].notna()]
    latest_row = valid_rows.iloc[-1] if not valid_rows.empty else None

    return pd.Series(
        {
            "periods": float(len(dataset)),
            "valid_periods": float(len(valid_rows)),
            "window": float(dataset["window"].iloc[0]),
            "min_periods": float(dataset["min_periods"].iloc[0]),
            "latest_date": (
                latest_row["date"] if latest_row is not None else pd.NaT
            ),
            "latest_rolling_mean_ic": (
                latest_row["rolling_mean_ic"] if latest_row is not None else math.nan
            ),
            "latest_rolling_ic_ir": (
                latest_row["rolling_ic_ir"] if latest_row is not None else math.nan
            ),
            "latest_rolling_positive_ic_ratio": (
                latest_row["rolling_positive_ic_ratio"]
                if latest_row is not None
                else math.nan
            ),
            "average_rolling_mean_ic": valid_rows["rolling_mean_ic"].mean(),
            "minimum_rolling_mean_ic": valid_rows["rolling_mean_ic"].min(),
            "maximum_rolling_mean_ic": valid_rows["rolling_mean_ic"].max(),
        },
        name="rolling_ic_summary",
    )


def compute_ic_decay_summary(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    forward_return_columns: Sequence[str],
    method: str = "pearson",
    min_observations: int = 2,
) -> pd.DataFrame:
    """Summarize IC behavior across configured forward-return horizons.

    The function reuses the same per-date cross-sectional IC logic for each
    label column. It does not create labels or shift data; timing safety comes
    from the supplied forward-return columns.
    """
    method = _normalize_ic_method(method)
    min_observations = _normalize_positive_int(
        min_observations,
        parameter_name="min_observations",
    )
    label_columns = _normalize_forward_return_columns(forward_return_columns)
    ic_decay_series = compute_ic_decay_series(
        frame,
        signal_column=signal_column,
        forward_return_columns=label_columns,
        method=method,
        min_observations=min_observations,
    )

    rows = []
    for order, forward_return_column in enumerate(label_columns):
        ic_series = ic_decay_series.loc[
            ic_decay_series["forward_return_column"].eq(forward_return_column),
            ["date", "ic", "observations", "method"],
        ]
        summary = summarize_ic(
            ic_series.loc[:, ["date", "ic", "observations"]]
        )
        rows.append(
            {
                "horizon": _infer_forward_return_horizon(forward_return_column),
                "forward_return_column": forward_return_column,
                "order": float(order),
                "periods": summary["periods"],
                "valid_periods": summary["valid_periods"],
                "mean_ic": summary["mean_ic"],
                "ic_std": summary["ic_std"],
                "ic_ir": summary["ic_ir"],
                "positive_ic_ratio": summary["positive_ic_ratio"],
                "average_observations": summary["average_observations"],
                "method": method,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "horizon",
            "forward_return_column",
            "order",
            "periods",
            "valid_periods",
            "mean_ic",
            "ic_std",
            "ic_ir",
            "positive_ic_ratio",
            "average_observations",
            "method",
        ],
    )


def compute_ic_decay_series(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    forward_return_columns: Sequence[str],
    method: str = "pearson",
    min_observations: int = 2,
) -> pd.DataFrame:
    """Compute per-date IC across configured forward-return horizons.

    The output is a long-form horizon-by-date series. It does not create labels
    or shift data; timing safety comes from the supplied forward-return columns.
    """
    method = _normalize_ic_method(method)
    min_observations = _normalize_positive_int(
        min_observations,
        parameter_name="min_observations",
    )
    label_columns = _normalize_forward_return_columns(forward_return_columns)

    series_frames = []
    for order, forward_return_column in enumerate(label_columns):
        ic_series = compute_ic_series(
            frame,
            signal_column=signal_column,
            forward_return_column=forward_return_column,
            method=method,
            min_observations=min_observations,
        )
        horizon_series = ic_series.assign(
            horizon=_infer_forward_return_horizon(forward_return_column),
            forward_return_column=forward_return_column,
            order=float(order),
        )
        series_frames.append(
            horizon_series.loc[
                :,
                [
                    "date",
                    "horizon",
                    "forward_return_column",
                    "order",
                    "ic",
                    "observations",
                    "method",
                ],
            ]
        )

    result = pd.concat(series_frames, ignore_index=True)
    return result.sort_values(["date", "order"], kind="mergesort").reset_index(
        drop=True
    )


def compute_grouped_ic_series(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    forward_return_column: str,
    group_column: str,
    method: str = "pearson",
    min_observations: int = 2,
) -> pd.DataFrame:
    """Compute per-date cross-sectional IC within an explicit group column.

    Missing group values are excluded rather than assigned to a fallback bucket.
    The function does not create labels or shift data; timing safety comes from
    the supplied frame and forward-return column.
    """
    method = _normalize_ic_method(method)
    min_observations = _normalize_positive_int(
        min_observations,
        parameter_name="min_observations",
    )
    dataset = _prepare_grouped_factor_frame(
        frame,
        signal_column=signal_column,
        forward_return_column=forward_return_column,
        group_column=group_column,
    )

    rows = []
    for (date, group_value), group in dataset.dropna(
        subset=[group_column]
    ).groupby(["date", group_column], sort=True):
        usable = group.loc[:, [signal_column, forward_return_column]].dropna()
        observations = len(usable)
        if observations < min_observations:
            ic = math.nan
        else:
            ic = _compute_ic_from_usable(
                usable,
                signal_column=signal_column,
                forward_return_column=forward_return_column,
                method=method,
            )
        rows.append(
            {
                "date": date,
                "group_column": group_column,
                "group_value": group_value,
                "ic": ic,
                "observations": float(observations),
                "method": method,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "group_column",
            "group_value",
            "ic",
            "observations",
            "method",
        ],
    )


def summarize_grouped_ic(grouped_ic_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize grouped IC series by group column and group value."""
    dataset = _prepare_grouped_ic_frame(grouped_ic_frame)

    rows = []
    for (group_column, group_value), group in dataset.groupby(
        ["group_column", "group_value"], sort=True
    ):
        valid_ic = group["ic"].dropna()
        ic_std = valid_ic.std(ddof=1)
        if valid_ic.empty or pd.isna(ic_std) or ic_std == 0.0:
            ic_ir = math.nan
        else:
            ic_ir = valid_ic.mean() / ic_std
        rows.append(
            {
                "group_column": group_column,
                "group_value": group_value,
                "periods": float(len(group)),
                "valid_periods": float(valid_ic.count()),
                "mean_ic": valid_ic.mean(),
                "ic_std": ic_std,
                "ic_ir": ic_ir,
                "positive_ic_ratio": valid_ic.gt(0.0).mean(),
                "average_observations": group["observations"].mean(),
                "method": group["method"].iloc[0],
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "group_column",
            "group_value",
            "periods",
            "valid_periods",
            "mean_ic",
            "ic_std",
            "ic_ir",
            "positive_ic_ratio",
            "average_observations",
            "method",
        ],
    )


def compute_quantile_bucket_returns(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    forward_return_column: str,
    n_quantiles: int = 5,
    min_observations: int | None = None,
) -> pd.DataFrame:
    """Compute mean forward returns by signal quantile across dates.

    Quantiles are assigned within each date after ranking the signal cross-section.
    This keeps bucket sizes stable when raw signal values contain ties.
    """
    _prepare_factor_frame(
        frame,
        signal_column=signal_column,
        forward_return_column=forward_return_column,
    )
    per_date_quantiles = _compute_per_date_quantile_bucket_rows(
        frame,
        signal_column=signal_column,
        forward_return_column=forward_return_column,
        n_quantiles=n_quantiles,
        min_observations=min_observations,
    )
    if per_date_quantiles.empty:
        return pd.DataFrame(
            columns=[
                "quantile",
                "mean_forward_return",
                "mean_signal",
                "average_count",
                "periods",
            ]
        )

    summary = (
        per_date_quantiles.groupby("quantile", sort=True)
        .agg(
            mean_forward_return=("mean_forward_return", "mean"),
            mean_signal=("mean_signal", "mean"),
            average_count=("count", "mean"),
            periods=("date", "nunique"),
        )
        .reset_index()
    )
    return summary


def compute_quantile_cumulative_returns(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    forward_return_column: str,
    n_quantiles: int = 5,
    min_observations: int | None = None,
) -> pd.DataFrame:
    """Compute cumulative mean forward-return paths by signal quantile.

    This compounds per-date quantile mean forward returns as a diagnostic path.
    It is not a portfolio backtest or execution simulation.
    """
    per_date_quantiles = _compute_per_date_quantile_bucket_rows(
        frame,
        signal_column=signal_column,
        forward_return_column=forward_return_column,
        n_quantiles=n_quantiles,
        min_observations=min_observations,
    )
    columns = [
        "date",
        "quantile",
        "mean_forward_return",
        "mean_signal",
        "count",
        "cumulative_growth",
        "cumulative_forward_return",
    ]
    if per_date_quantiles.empty:
        return pd.DataFrame(columns=columns)

    dataset = per_date_quantiles.sort_values(
        ["quantile", "date"],
        kind="mergesort",
    ).reset_index(drop=True)
    if dataset["mean_forward_return"].le(-1.0).any():
        raise FactorDiagnosticsError(
            "quantile mean_forward_return values must be greater than -1.0 "
            "to compute cumulative returns."
        )
    dataset["cumulative_growth"] = (
        1.0 + dataset["mean_forward_return"]
    ).groupby(dataset["quantile"], sort=True).cumprod()
    dataset["cumulative_forward_return"] = dataset["cumulative_growth"] - 1.0
    return dataset.loc[:, columns].sort_values(
        ["date", "quantile"],
        kind="mergesort",
    ).reset_index(drop=True)


def compute_signal_coverage_by_date(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    forward_return_column: str,
) -> pd.DataFrame:
    """Compute per-date signal/label coverage diagnostics."""
    dataset = _prepare_factor_frame(
        frame,
        signal_column=signal_column,
        forward_return_column=forward_return_column,
    )
    signal_available = dataset[signal_column].notna()
    forward_available = dataset[forward_return_column].notna()
    jointly_available = signal_available & forward_available

    per_date = (
        dataset.assign(
            signal_available=signal_available,
            forward_available=forward_available,
            jointly_available=jointly_available,
        )
        .groupby("date", sort=True)
        .agg(
            total_rows=(signal_column, "size"),
            signal_non_null_rows=("signal_available", "sum"),
            forward_return_non_null_rows=("forward_available", "sum"),
            usable_rows=("jointly_available", "sum"),
        )
        .reset_index()
    )
    per_date["signal_coverage_ratio"] = (
        per_date["signal_non_null_rows"] / per_date["total_rows"]
    )
    per_date["forward_return_coverage_ratio"] = (
        per_date["forward_return_non_null_rows"] / per_date["total_rows"]
    )
    per_date["joint_coverage_ratio"] = per_date["usable_rows"] / per_date["total_rows"]
    return per_date


def compute_signal_coverage_by_date_and_group(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    forward_return_column: str,
    group_column: str,
) -> pd.DataFrame:
    """Compute per-date signal/label coverage inside an explicit group column.

    Missing group values are excluded rather than assigned to a fallback bucket.
    """
    dataset = _prepare_grouped_factor_frame(
        frame,
        signal_column=signal_column,
        forward_return_column=forward_return_column,
        group_column=group_column,
    ).dropna(subset=[group_column])
    columns = [
        "date",
        "group_column",
        "group_value",
        "total_rows",
        "signal_non_null_rows",
        "forward_return_non_null_rows",
        "usable_rows",
        "signal_coverage_ratio",
        "forward_return_coverage_ratio",
        "joint_coverage_ratio",
    ]
    if dataset.empty:
        return pd.DataFrame(columns=columns)

    signal_available = dataset[signal_column].notna()
    forward_available = dataset[forward_return_column].notna()
    jointly_available = signal_available & forward_available
    per_date_group = (
        dataset.assign(
            signal_available=signal_available,
            forward_available=forward_available,
            jointly_available=jointly_available,
        )
        .groupby(["date", group_column], sort=True)
        .agg(
            total_rows=(signal_column, "size"),
            signal_non_null_rows=("signal_available", "sum"),
            forward_return_non_null_rows=("forward_available", "sum"),
            usable_rows=("jointly_available", "sum"),
        )
        .reset_index()
        .rename(columns={group_column: "group_value"})
    )
    per_date_group.insert(1, "group_column", group_column)
    per_date_group["signal_coverage_ratio"] = (
        per_date_group["signal_non_null_rows"] / per_date_group["total_rows"]
    )
    per_date_group["forward_return_coverage_ratio"] = (
        per_date_group["forward_return_non_null_rows"]
        / per_date_group["total_rows"]
    )
    per_date_group["joint_coverage_ratio"] = (
        per_date_group["usable_rows"] / per_date_group["total_rows"]
    )
    return per_date_group.loc[:, columns]


def summarize_signal_coverage(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    forward_return_column: str,
) -> pd.Series:
    """Summarize usable signal/label coverage for factor diagnostics."""
    dataset = _prepare_factor_frame(
        frame,
        signal_column=signal_column,
        forward_return_column=forward_return_column,
    )

    signal_available = dataset[signal_column].notna()
    forward_available = dataset[forward_return_column].notna()
    jointly_available = signal_available & forward_available
    daily_usable = jointly_available.groupby(dataset["date"], sort=True).sum()

    return pd.Series(
        {
            "dates": float(dataset["date"].nunique()),
            "total_rows": float(len(dataset)),
            "signal_non_null_rows": float(signal_available.sum()),
            "forward_return_non_null_rows": float(forward_available.sum()),
            "usable_rows": float(jointly_available.sum()),
            "signal_coverage_ratio": signal_available.mean(),
            "forward_return_coverage_ratio": forward_available.mean(),
            "joint_coverage_ratio": jointly_available.mean(),
            "average_daily_usable_rows": daily_usable.mean(),
            "minimum_daily_usable_rows": float(daily_usable.min()),
        },
        name="signal_coverage_summary",
    )


def summarize_signal_coverage_by_group(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    forward_return_column: str,
    group_column: str,
) -> pd.DataFrame:
    """Summarize signal/label coverage by explicit group column and value."""
    coverage_by_date = compute_signal_coverage_by_date_and_group(
        frame,
        signal_column=signal_column,
        forward_return_column=forward_return_column,
        group_column=group_column,
    )
    columns = [
        "group_column",
        "group_value",
        "dates",
        "total_rows",
        "signal_non_null_rows",
        "forward_return_non_null_rows",
        "usable_rows",
        "signal_coverage_ratio",
        "forward_return_coverage_ratio",
        "joint_coverage_ratio",
        "average_daily_usable_rows",
        "minimum_daily_usable_rows",
    ]
    if coverage_by_date.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        coverage_by_date.groupby(["group_column", "group_value"], sort=True)
        .agg(
            dates=("date", "nunique"),
            total_rows=("total_rows", "sum"),
            signal_non_null_rows=("signal_non_null_rows", "sum"),
            forward_return_non_null_rows=("forward_return_non_null_rows", "sum"),
            usable_rows=("usable_rows", "sum"),
            average_daily_usable_rows=("usable_rows", "mean"),
            minimum_daily_usable_rows=("usable_rows", "min"),
        )
        .reset_index()
    )
    summary["signal_coverage_ratio"] = (
        summary["signal_non_null_rows"] / summary["total_rows"]
    )
    summary["forward_return_coverage_ratio"] = (
        summary["forward_return_non_null_rows"] / summary["total_rows"]
    )
    summary["joint_coverage_ratio"] = summary["usable_rows"] / summary["total_rows"]
    return summary.loc[:, columns]


def compute_quantile_spread_series(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    forward_return_column: str,
    n_quantiles: int = 5,
    min_observations: int | None = None,
) -> pd.DataFrame:
    """Compute per-date top-minus-bottom quantile forward-return spreads."""
    per_date_quantiles = _compute_per_date_quantile_bucket_rows(
        frame,
        signal_column=signal_column,
        forward_return_column=forward_return_column,
        n_quantiles=n_quantiles,
        min_observations=min_observations,
    )
    if per_date_quantiles.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "bottom_quantile",
                "top_quantile",
                "bottom_mean_forward_return",
                "top_mean_forward_return",
                "top_bottom_spread",
            ]
        )

    spread_rows = []
    for date, group in per_date_quantiles.groupby("date", sort=True):
        sorted_group = group.sort_values("quantile", kind="mergesort").reset_index(drop=True)
        if len(sorted_group) < 2:
            continue
        bottom_row = sorted_group.iloc[0]
        top_row = sorted_group.iloc[-1]
        spread_rows.append(
            {
                "date": date,
                "bottom_quantile": float(bottom_row["quantile"]),
                "top_quantile": float(top_row["quantile"]),
                "bottom_mean_forward_return": float(bottom_row["mean_forward_return"]),
                "top_mean_forward_return": float(top_row["mean_forward_return"]),
                "top_bottom_spread": float(
                    top_row["mean_forward_return"] - bottom_row["mean_forward_return"]
                ),
            }
        )

    return pd.DataFrame(spread_rows)


def summarize_quantile_spread_stability(spread_frame: pd.DataFrame) -> pd.Series:
    """Summarize top-minus-bottom quantile spread consistency through time."""
    dataset = _prepare_quantile_spread_frame(spread_frame)
    if dataset.empty:
        return pd.Series(
            {
                "periods": 0.0,
                "valid_periods": 0.0,
                "mean_spread": math.nan,
                "spread_std": math.nan,
                "spread_stability_ratio": math.nan,
                "positive_spread_ratio": math.nan,
                "negative_spread_ratio": math.nan,
                "latest_date": pd.NaT,
                "latest_spread": math.nan,
                "average_bottom_quantile": math.nan,
                "average_top_quantile": math.nan,
            },
            name="quantile_spread_stability",
        )

    valid_spread = dataset["top_bottom_spread"].dropna()
    spread_std = valid_spread.std(ddof=1)
    if valid_spread.empty or pd.isna(spread_std) or spread_std == 0.0:
        spread_stability_ratio = math.nan
    else:
        spread_stability_ratio = valid_spread.mean() / spread_std

    latest_row = (
        dataset.loc[dataset["top_bottom_spread"].notna()].iloc[-1]
        if not valid_spread.empty
        else None
    )
    return pd.Series(
        {
            "periods": float(len(dataset)),
            "valid_periods": float(valid_spread.count()),
            "mean_spread": valid_spread.mean(),
            "spread_std": spread_std,
            "spread_stability_ratio": spread_stability_ratio,
            "positive_spread_ratio": valid_spread.gt(0.0).mean(),
            "negative_spread_ratio": valid_spread.lt(0.0).mean(),
            "latest_date": latest_row["date"] if latest_row is not None else pd.NaT,
            "latest_spread": (
                latest_row["top_bottom_spread"] if latest_row is not None else math.nan
            ),
            "average_bottom_quantile": dataset["bottom_quantile"].mean(),
            "average_top_quantile": dataset["top_quantile"].mean(),
        },
        name="quantile_spread_stability",
    )


def _prepare_quantile_spread_frame(spread_frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a dated top-bottom quantile spread series."""
    required_columns = [
        "date",
        "bottom_quantile",
        "top_quantile",
        "top_bottom_spread",
    ]
    missing_columns = [
        column for column in required_columns if column not in spread_frame.columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise FactorDiagnosticsError(
            f"quantile spread frame is missing required columns: {missing_text}."
        )

    dataset = spread_frame.loc[:, required_columns].copy()
    if dataset.empty:
        return dataset

    dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
    if dataset["date"].isna().any():
        raise FactorDiagnosticsError(
            "quantile spread frame contains invalid date values."
        )
    for column in ("bottom_quantile", "top_quantile", "top_bottom_spread"):
        dataset[column] = _parse_numeric_column(
            dataset[column],
            column_name=column,
            allow_na=True,
        )
    return dataset.sort_values("date", kind="mergesort").reset_index(drop=True)


def _prepare_factor_frame(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    forward_return_column: str,
) -> pd.DataFrame:
    """Validate a dated panel used for factor diagnostics."""
    required_columns = ["date", signal_column, forward_return_column]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise FactorDiagnosticsError(
            f"factor frame is missing required columns: {missing_text}."
        )

    dataset = frame.loc[:, required_columns].copy()
    dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
    if dataset["date"].isna().any():
        raise FactorDiagnosticsError("factor frame contains invalid date values.")
    if dataset.empty:
        raise FactorDiagnosticsError("factor frame must contain at least one row.")

    dataset[signal_column] = _parse_numeric_column(
        dataset[signal_column],
        column_name=signal_column,
        allow_na=True,
    )
    dataset[forward_return_column] = _parse_numeric_column(
        dataset[forward_return_column],
        column_name=forward_return_column,
        allow_na=True,
    )
    return dataset.sort_values("date", kind="mergesort").reset_index(drop=True)


def _prepare_grouped_factor_frame(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    forward_return_column: str,
    group_column: str,
) -> pd.DataFrame:
    """Validate a dated panel used for grouped factor diagnostics."""
    group_column = _normalize_non_empty_column_name(
        group_column,
        parameter_name="group_column",
    )
    if group_column in {"date", signal_column, forward_return_column}:
        raise FactorDiagnosticsError(
            "group_column must be distinct from date, signal_column, and forward_return_column."
        )
    required_columns = ["date", group_column, signal_column, forward_return_column]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise FactorDiagnosticsError(
            f"factor frame is missing required columns: {missing_text}."
        )

    dataset = frame.loc[:, required_columns].copy()
    if dataset.empty:
        raise FactorDiagnosticsError("factor frame must contain at least one row.")
    dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
    if dataset["date"].isna().any():
        raise FactorDiagnosticsError("factor frame contains invalid date values.")
    dataset[signal_column] = _parse_numeric_column(
        dataset[signal_column],
        column_name=signal_column,
        allow_na=True,
    )
    dataset[forward_return_column] = _parse_numeric_column(
        dataset[forward_return_column],
        column_name=forward_return_column,
        allow_na=True,
    )
    return dataset.sort_values(
        ["date", group_column],
        kind="mergesort",
    ).reset_index(drop=True)


def _prepare_grouped_ic_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate grouped IC series before summarization."""
    required_columns = [
        "date",
        "group_column",
        "group_value",
        "ic",
        "observations",
        "method",
    ]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise FactorDiagnosticsError(
            f"grouped_ic_frame is missing required columns: {missing_text}."
        )

    dataset = frame.loc[:, required_columns].copy()
    if dataset.empty:
        return dataset
    dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
    if dataset["date"].isna().any():
        raise FactorDiagnosticsError("grouped_ic_frame contains invalid date values.")
    dataset["ic"] = _parse_numeric_column(
        dataset["ic"],
        column_name="ic",
        allow_na=True,
    )
    dataset["observations"] = _parse_numeric_column(
        dataset["observations"],
        column_name="observations",
    )
    return dataset.sort_values(
        ["group_column", "group_value", "date"],
        kind="mergesort",
    ).reset_index(drop=True)


def _prepare_ic_frame(
    frame: pd.DataFrame,
    *,
    required_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Validate a dated IC diagnostic frame."""
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise FactorDiagnosticsError(
            f"ic_frame is missing required columns: {missing_text}."
        )

    dataset = frame.loc[:, list(required_columns)].copy()
    if dataset.empty:
        raise FactorDiagnosticsError("ic_frame must contain at least one row.")

    dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
    if dataset["date"].isna().any():
        raise FactorDiagnosticsError("ic_frame contains invalid date values.")

    if "ic" in dataset.columns:
        dataset["ic"] = _parse_numeric_column(
            dataset["ic"],
            column_name="ic",
            allow_na=True,
        )
    return dataset.sort_values("date", kind="mergesort").reset_index(drop=True)


def _normalize_forward_return_columns(
    forward_return_columns: Sequence[str],
) -> tuple[str, ...]:
    """Validate configured label columns for IC decay diagnostics."""
    if isinstance(forward_return_columns, str) or not isinstance(
        forward_return_columns,
        Sequence,
    ):
        raise FactorDiagnosticsError(
            "forward_return_columns must be a non-empty sequence of column names."
        )
    if not forward_return_columns:
        raise FactorDiagnosticsError(
            "forward_return_columns must contain at least one column name."
        )

    normalized = []
    seen = set()
    for column in forward_return_columns:
        if not isinstance(column, str) or not column.strip():
            raise FactorDiagnosticsError(
                "forward_return_columns must contain non-empty string column names."
            )
        column_name = column.strip()
        if column_name in seen:
            raise FactorDiagnosticsError(
                "forward_return_columns must not contain duplicate column names."
            )
        normalized.append(column_name)
        seen.add(column_name)
    return tuple(normalized)


def _infer_forward_return_horizon(forward_return_column: str) -> float:
    """Infer a numeric horizon from the canonical forward_return_<N>d label."""
    prefix = "forward_return_"
    suffix = "d"
    if (
        forward_return_column.startswith(prefix)
        and forward_return_column.endswith(suffix)
    ):
        value = forward_return_column[len(prefix) : -len(suffix)]
        if value.isdigit():
            return float(value)
    return math.nan


def _compute_per_date_quantile_bucket_rows(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    forward_return_column: str,
    n_quantiles: int,
    min_observations: int | None,
) -> pd.DataFrame:
    """Compute one row per date/quantile before cross-date aggregation."""
    n_quantiles = _normalize_quantiles(n_quantiles)
    if min_observations is None:
        min_observations = n_quantiles
    min_observations = _normalize_positive_int(
        min_observations,
        parameter_name="min_observations",
    )
    if min_observations < n_quantiles:
        raise FactorDiagnosticsError(
            "min_observations must be greater than or equal to n_quantiles."
        )

    dataset = _prepare_factor_frame(
        frame,
        signal_column=signal_column,
        forward_return_column=forward_return_column,
    )

    bucket_rows = []
    for date, group in dataset.groupby("date", sort=True):
        usable = group.loc[:, [signal_column, forward_return_column]].dropna()
        if len(usable) < min_observations:
            continue

        ranked_signal = usable[signal_column].rank(method="first")
        try:
            quantiles = pd.qcut(
                ranked_signal,
                q=n_quantiles,
                labels=False,
            ).astype(int) + 1
        except ValueError:
            continue

        dated = usable.assign(quantile=quantiles.to_numpy())
        grouped = (
            dated.groupby("quantile", sort=True)
            .agg(
                mean_forward_return=(forward_return_column, "mean"),
                mean_signal=(signal_column, "mean"),
                count=(signal_column, "size"),
            )
            .reset_index()
        )
        grouped["date"] = date
        bucket_rows.append(grouped)

    if not bucket_rows:
        return pd.DataFrame(
            columns=[
                "quantile",
                "mean_forward_return",
                "mean_signal",
                "count",
                "date",
            ]
        )
    return pd.concat(bucket_rows, ignore_index=True)


def _parse_numeric_column(
    values: pd.Series,
    *,
    column_name: str,
    allow_na: bool = False,
) -> pd.Series:
    """Parse numeric columns without silent coercion."""
    parsed = pd.to_numeric(values, errors="coerce")
    invalid_values = values.notna() & parsed.isna()
    if invalid_values.any():
        raise FactorDiagnosticsError(
            f"factor frame contains invalid numeric values in '{column_name}'."
        )
    if not allow_na and parsed.isna().any():
        raise FactorDiagnosticsError(
            f"factor frame contains missing numeric values in '{column_name}'."
        )
    return parsed


def _compute_ic_from_usable(
    usable: pd.DataFrame,
    *,
    signal_column: str,
    forward_return_column: str,
    method: str,
) -> float:
    """Compute one cross-sectional IC value from already filtered rows."""
    if method == "pearson":
        return usable[signal_column].corr(usable[forward_return_column])
    ranked_signal = usable[signal_column].rank(method="average")
    ranked_forward_return = usable[forward_return_column].rank(method="average")
    return ranked_signal.corr(ranked_forward_return)


def _normalize_ic_method(method: str) -> str:
    """Validate the IC correlation method."""
    normalized = method.lower()
    if normalized not in {"pearson", "spearman"}:
        raise FactorDiagnosticsError(
            "method must be one of {'pearson', 'spearman'}."
        )
    return normalized


def _normalize_quantiles(n_quantiles: int) -> int:
    """Validate quantile count settings."""
    if isinstance(n_quantiles, bool) or not isinstance(n_quantiles, int) or n_quantiles < 2:
        raise FactorDiagnosticsError("n_quantiles must be an integer greater than or equal to 2.")
    return n_quantiles


def _normalize_non_empty_column_name(value: str, *, parameter_name: str) -> str:
    """Validate a user-supplied column name."""
    if not isinstance(value, str) or not value.strip():
        raise FactorDiagnosticsError(f"{parameter_name} must be a non-empty string.")
    return value.strip()


def _normalize_positive_int(value: int, *, parameter_name: str) -> int:
    """Validate positive integer parameters."""
    return _common_positive_int(
        value,
        parameter_name=parameter_name,
        error_factory=FactorDiagnosticsError,
    )

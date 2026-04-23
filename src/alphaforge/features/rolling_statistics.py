"""Benchmark-aware rolling statistics for research datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.data import validate_benchmark_returns, validate_ohlcv


def attach_rolling_benchmark_statistics(
    frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame,
    *,
    window: int,
) -> pd.DataFrame:
    """Attach rolling beta/correlation features using aligned daily benchmark returns.

    Timing convention:
    - each feature row is anchored at the close of ``date``
    - rolling windows use only strategy ``daily_return`` and benchmark return
      observations available through that same close
    - benchmark coverage must match the dataset's global date set exactly so
      research does not silently proceed on a misaligned benchmark series
    """
    window = _normalize_window(window, parameter_name="window")

    dataset = validate_ohlcv(frame, source="rolling benchmark statistics input").copy()
    if "daily_return" not in dataset.columns:
        raise ValueError(
            "rolling benchmark statistics input must contain a 'daily_return' column."
        )

    parsed_returns = pd.to_numeric(dataset["daily_return"], errors="coerce")
    invalid_returns = dataset["daily_return"].notna() & parsed_returns.isna()
    if invalid_returns.any():
        raise ValueError(
            "rolling benchmark statistics input contains invalid numeric values in "
            "'daily_return'."
        )
    dataset["daily_return"] = parsed_returns.astype("float64")

    benchmark = validate_benchmark_returns(
        benchmark_frame,
        source="rolling benchmark statistics benchmark input",
    )
    _validate_exact_date_alignment(dataset, benchmark)

    beta_column = f"rolling_benchmark_beta_{window}d"
    correlation_column = f"rolling_benchmark_correlation_{window}d"
    conflicting_columns = [
        column_name
        for column_name in (beta_column, correlation_column)
        if column_name in dataset.columns
    ]
    if conflicting_columns:
        conflict_text = ", ".join(repr(column) for column in conflicting_columns)
        raise ValueError(
            "rolling benchmark statistics output columns already exist: "
            f"{conflict_text}."
        )

    attached = dataset.merge(
        benchmark.loc[:, ["date", "benchmark_return"]],
        on="date",
        how="left",
        validate="many_to_one",
    )
    attached[beta_column] = np.nan
    attached[correlation_column] = np.nan

    for _, group in attached.groupby("symbol", sort=False):
        strategy_returns = group["daily_return"]
        benchmark_returns = group["benchmark_return"]

        rolling_covariance = strategy_returns.rolling(
            window=window,
            min_periods=window,
        ).cov(benchmark_returns)
        rolling_benchmark_variance = benchmark_returns.rolling(
            window=window,
            min_periods=window,
        ).var(ddof=1)
        rolling_beta = rolling_covariance.div(rolling_benchmark_variance)
        rolling_beta = rolling_beta.mask(~np.isfinite(rolling_beta))

        rolling_correlation = strategy_returns.rolling(
            window=window,
            min_periods=window,
        ).corr(benchmark_returns)

        attached.loc[group.index, beta_column] = rolling_beta.to_numpy()
        attached.loc[group.index, correlation_column] = rolling_correlation.to_numpy()

    return attached.drop(columns=["benchmark_return"])


def _validate_exact_date_alignment(
    dataset: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> None:
    """Require exact benchmark coverage across the dataset's unique market dates."""
    dataset_dates = pd.Index(pd.unique(dataset["date"])).sort_values()
    benchmark_dates = pd.Index(benchmark["date"]).sort_values()

    if dataset_dates.equals(benchmark_dates):
        return

    dataset_set = set(dataset_dates.tolist())
    benchmark_set = set(benchmark_dates.tolist())
    missing_dates = sorted(dataset_set - benchmark_set)
    extra_dates = sorted(benchmark_set - dataset_set)

    problems: list[str] = []
    if missing_dates:
        missing_text = ", ".join(
            timestamp.date().isoformat() for timestamp in missing_dates[:3]
        )
        suffix = "..." if len(missing_dates) > 3 else ""
        problems.append(f"missing dates ({missing_text}{suffix})")
    if extra_dates:
        extra_text = ", ".join(
            timestamp.date().isoformat() for timestamp in extra_dates[:3]
        )
        suffix = "..." if len(extra_dates) > 3 else ""
        problems.append(f"extra dates ({extra_text}{suffix})")

    detail = f": {'; '.join(problems)}" if problems else ""
    raise ValueError(
        "benchmark returns must align exactly to research dataset dates"
        f"{detail}."
    )


def _normalize_window(value: int, *, parameter_name: str) -> int:
    """Validate positive rolling windows."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{parameter_name} must be a positive integer.")
    return value

"""Rolling statistics for research datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.data import validate_benchmark_returns, validate_ohlcv


def attach_realized_volatility_family(
    frame: pd.DataFrame,
    *,
    window: int,
) -> pd.DataFrame:
    """Attach trailing realized-volatility family features from strategy daily returns.

    Current definitions use trailing root-mean-square daily returns over the
    requested window:
    - ``realized_volatility`` uses all daily returns
    - ``downside_realized_volatility`` uses only negative daily returns
    - ``upside_realized_volatility`` uses only positive daily returns

    Timing convention:
    - each feature row is anchored at the close of ``date``
    - rolling windows use only strategy ``daily_return`` observations available
      through that same close
    """
    window = _normalize_window(window, parameter_name="window")
    dataset = _prepare_daily_return_input(
        frame,
        source="realized volatility input",
    )

    realized_volatility_column = f"realized_volatility_{window}d"
    downside_column = f"downside_realized_volatility_{window}d"
    upside_column = f"upside_realized_volatility_{window}d"
    _validate_output_columns_absent(
        dataset,
        output_columns=(
            realized_volatility_column,
            downside_column,
            upside_column,
        ),
        feature_name="realized volatility",
    )

    dataset[realized_volatility_column] = _compute_root_mean_square_by_symbol(
        dataset["daily_return"].pow(2),
        symbols=dataset["symbol"],
        window=window,
    )
    dataset[downside_column] = _compute_root_mean_square_by_symbol(
        dataset["daily_return"].clip(upper=0.0).pow(2),
        symbols=dataset["symbol"],
        window=window,
    )
    dataset[upside_column] = _compute_root_mean_square_by_symbol(
        dataset["daily_return"].clip(lower=0.0).pow(2),
        symbols=dataset["symbol"],
        window=window,
    )

    for column_name in (
        realized_volatility_column,
        downside_column,
        upside_column,
    ):
        dataset[column_name] = dataset[column_name].mask(
            ~np.isfinite(dataset[column_name])
        )

    return dataset


def attach_rolling_higher_moments(
    frame: pd.DataFrame,
    *,
    window: int,
) -> pd.DataFrame:
    """Attach trailing rolling skew/kurtosis features from strategy daily returns.

    Timing convention:
    - each feature row is anchored at the close of ``date``
    - rolling windows use only strategy ``daily_return`` observations available
      through that same close
    - windows smaller than 4 are rejected because kurtosis is not reliable with
      fewer observations
    """
    window = _normalize_higher_moments_window(
        window,
        parameter_name="window",
    )
    dataset = _prepare_daily_return_input(
        frame,
        source="rolling higher moments input",
    )

    skew_column = f"rolling_skew_{window}d"
    kurtosis_column = f"rolling_kurtosis_{window}d"
    _validate_output_columns_absent(
        dataset,
        output_columns=(skew_column, kurtosis_column),
        feature_name="rolling higher moments",
    )

    by_symbol = dataset.groupby("symbol", sort=False)["daily_return"]
    dataset[skew_column] = by_symbol.transform(
        lambda values: values.rolling(
            window=window,
            min_periods=window,
        ).skew()
    )
    dataset[kurtosis_column] = by_symbol.transform(
        lambda values: values.rolling(
            window=window,
            min_periods=window,
        ).kurt()
    )
    dataset[skew_column] = dataset[skew_column].mask(
        ~np.isfinite(dataset[skew_column])
    )
    dataset[kurtosis_column] = dataset[kurtosis_column].mask(
        ~np.isfinite(dataset[kurtosis_column])
    )
    return dataset


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
    dataset = _prepare_daily_return_input(
        frame,
        source="rolling benchmark statistics input",
    )

    benchmark = validate_benchmark_returns(
        benchmark_frame,
        source="rolling benchmark statistics benchmark input",
    )
    _validate_exact_date_alignment(dataset, benchmark)

    beta_column = f"rolling_benchmark_beta_{window}d"
    correlation_column = f"rolling_benchmark_correlation_{window}d"
    _validate_output_columns_absent(
        dataset,
        output_columns=(beta_column, correlation_column),
        feature_name="rolling benchmark statistics",
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


def _prepare_daily_return_input(
    frame: pd.DataFrame,
    *,
    source: str,
) -> pd.DataFrame:
    """Validate a dataset input and normalize the required daily-return column."""
    dataset = validate_ohlcv(frame, source=source).copy()
    if "daily_return" not in dataset.columns:
        raise ValueError(f"{source} must contain a 'daily_return' column.")

    parsed_returns = pd.to_numeric(dataset["daily_return"], errors="coerce")
    invalid_returns = dataset["daily_return"].notna() & parsed_returns.isna()
    if invalid_returns.any():
        raise ValueError(
            f"{source} contains invalid numeric values in 'daily_return'."
        )
    dataset["daily_return"] = parsed_returns.astype("float64")
    return dataset


def _compute_root_mean_square_by_symbol(
    squared_values: pd.Series,
    *,
    symbols: pd.Series,
    window: int,
) -> pd.Series:
    """Compute trailing root-mean-square values within each symbol."""
    return squared_values.groupby(symbols, sort=False).transform(
        lambda values: np.sqrt(
            values.rolling(
                window=window,
                min_periods=window,
            ).mean()
        )
    )


def _validate_output_columns_absent(
    dataset: pd.DataFrame,
    *,
    output_columns: tuple[str, ...],
    feature_name: str,
) -> None:
    """Fail fast if a rolling-feature attach would overwrite existing columns."""
    conflicting_columns = [
        column_name for column_name in output_columns if column_name in dataset.columns
    ]
    if not conflicting_columns:
        return

    conflict_text = ", ".join(repr(column) for column in conflicting_columns)
    raise ValueError(
        f"{feature_name} output columns already exist: {conflict_text}."
    )


def _normalize_higher_moments_window(value: int, *, parameter_name: str) -> int:
    """Validate rolling windows for skew/kurtosis features."""
    normalized = _normalize_window(value, parameter_name=parameter_name)
    if normalized < 4:
        raise ValueError(f"{parameter_name} must be at least 4.")
    return normalized


def _normalize_window(value: int, *, parameter_name: str) -> int:
    """Validate positive rolling windows."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{parameter_name} must be a positive integer.")
    return value

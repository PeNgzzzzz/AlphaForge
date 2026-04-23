"""Rolling statistics for research datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.data import validate_benchmark_returns, validate_ohlcv


def attach_garman_klass_volatility(
    frame: pd.DataFrame,
    *,
    window: int,
) -> pd.DataFrame:
    """Attach trailing Garman-Klass volatility from daily OHLC ranges.

    Current definition uses the daily Garman-Klass variance proxy
    ``0.5 * log(high / low)^2 - (2 * log(2) - 1) * log(close / open)^2``,
    then takes the square root of the trailing window mean when that window
    mean stays strictly positive.

    Timing convention:
    - each feature row is anchored at the close of ``date``
    - rolling windows use only ``open`` / ``high`` / ``low`` / ``close``
      observations available through that same close
    """
    window = _normalize_window(window, parameter_name="window")
    dataset = validate_ohlcv(frame, source="garman-klass volatility input").copy()

    column_name = f"garman_klass_volatility_{window}d"
    _validate_output_columns_absent(
        dataset,
        output_columns=(column_name,),
        feature_name="garman-klass volatility",
    )

    daily_garman_klass_variance = (
        0.5 * np.log(dataset["high"].div(dataset["low"])).pow(2)
        - (2.0 * np.log(2.0) - 1.0)
        * np.log(dataset["close"].div(dataset["open"])).pow(2)
    )
    rolling_variance = daily_garman_klass_variance.groupby(
        dataset["symbol"],
        sort=False,
    ).transform(
        lambda values: values.rolling(
            window=window,
            min_periods=window,
        ).mean()
    )
    dataset[column_name] = np.sqrt(rolling_variance.where(rolling_variance > 0.0))
    dataset[column_name] = dataset[column_name].mask(~np.isfinite(dataset[column_name]))
    return dataset


def attach_parkinson_volatility(
    frame: pd.DataFrame,
    *,
    window: int,
) -> pd.DataFrame:
    """Attach trailing Parkinson volatility from daily high/low ranges.

    Current definition uses the daily Parkinson variance proxy
    ``log(high / low)^2 / (4 * log(2))``, then takes the square root of the
    trailing window mean.

    Timing convention:
    - each feature row is anchored at the close of ``date``
    - rolling windows use only ``high`` / ``low`` observations available
      through that same close
    """
    window = _normalize_window(window, parameter_name="window")
    dataset = validate_ohlcv(frame, source="parkinson volatility input").copy()

    column_name = f"parkinson_volatility_{window}d"
    _validate_output_columns_absent(
        dataset,
        output_columns=(column_name,),
        feature_name="parkinson volatility",
    )

    daily_parkinson_variance = (
        np.log(dataset["high"].div(dataset["low"])).pow(2)
        / (4.0 * np.log(2.0))
    )
    dataset[column_name] = daily_parkinson_variance.groupby(
        dataset["symbol"],
        sort=False,
    ).transform(
        lambda values: np.sqrt(
            values.rolling(
                window=window,
                min_periods=window,
            ).mean()
        )
    )
    dataset[column_name] = dataset[column_name].mask(~np.isfinite(dataset[column_name]))
    return dataset


def attach_average_true_range(
    frame: pd.DataFrame,
    *,
    window: int,
) -> pd.DataFrame:
    """Attach trailing average true range from daily OHLC observations.

    Current definition uses the daily true range
    ``max(high - low, abs(high - close_{t-1}), abs(low - close_{t-1}))``,
    with the first observation per symbol naturally falling back to ``high - low``
    because ``close_{t-1}`` is unavailable. The feature is the trailing window
    mean of that true-range series.

    Timing convention:
    - each feature row is anchored at the close of ``date``
    - rolling windows use only ``high`` / ``low`` observations from ``date``
      plus ``close_{t-1}``, which is already known by that same close
    """
    window = _normalize_window(window, parameter_name="window")
    dataset = validate_ohlcv(frame, source="average true range input").copy()

    column_name = f"average_true_range_{window}d"
    _validate_output_columns_absent(
        dataset,
        output_columns=(column_name,),
        feature_name="average true range",
    )

    dataset[column_name] = _compute_average_true_range_by_symbol(
        dataset,
        window=window,
    )
    dataset[column_name] = dataset[column_name].mask(~np.isfinite(dataset[column_name]))
    return dataset


def attach_normalized_average_true_range(
    frame: pd.DataFrame,
    *,
    window: int,
) -> pd.DataFrame:
    """Attach trailing normalized average true range from daily OHLC observations.

    Current definition computes trailing average true range, then divides that
    ATR level by the same-day close:
    ``normalized_atr_t = average_true_range_t / close_t``.

    Timing convention:
    - each feature row is anchored at the close of ``date``
    - the trailing ATR uses only ``high`` / ``low`` observations from ``date``
      plus ``close_{t-1}``, which is already known by that same close
    - the normalization denominator uses ``close_t``, which is part of the
      same close-anchored observation
    """
    window = _normalize_window(window, parameter_name="window")
    dataset = validate_ohlcv(
        frame,
        source="normalized average true range input",
    ).copy()

    column_name = f"normalized_average_true_range_{window}d"
    _validate_output_columns_absent(
        dataset,
        output_columns=(column_name,),
        feature_name="normalized average true range",
    )

    dataset[column_name] = _compute_average_true_range_by_symbol(
        dataset,
        window=window,
    ).div(dataset["close"])
    dataset[column_name] = dataset[column_name].mask(~np.isfinite(dataset[column_name]))
    return dataset


def _compute_average_true_range_by_symbol(
    dataset: pd.DataFrame,
    *,
    window: int,
) -> pd.Series:
    """Compute a trailing average true range series within each symbol."""
    daily_true_range = _compute_daily_true_range(dataset)
    return daily_true_range.groupby(
        dataset["symbol"],
        sort=False,
    ).transform(
        lambda values: values.rolling(
            window=window,
            min_periods=window,
        ).mean()
    )


def _compute_daily_true_range(dataset: pd.DataFrame) -> pd.Series:
    """Compute daily true range using same-day high/low and previous close."""
    previous_close = dataset.groupby("symbol", sort=False)["close"].shift(1)
    return pd.concat(
        (
            dataset["high"].sub(dataset["low"]),
            dataset["high"].sub(previous_close).abs(),
            dataset["low"].sub(previous_close).abs(),
        ),
        axis=1,
    ).max(axis=1, skipna=True)


def attach_rogers_satchell_volatility(
    frame: pd.DataFrame,
    *,
    window: int,
) -> pd.DataFrame:
    """Attach trailing Rogers-Satchell volatility from daily OHLC ranges.

    Current definition uses the daily Rogers-Satchell variance proxy
    ``log(high / open) * log(high / close) + log(low / open) * log(low / close)``,
    then takes the square root of the trailing window mean.

    Timing convention:
    - each feature row is anchored at the close of ``date``
    - rolling windows use only ``open`` / ``high`` / ``low`` / ``close``
      observations available through that same close
    """
    window = _normalize_window(window, parameter_name="window")
    dataset = validate_ohlcv(frame, source="rogers-satchell volatility input").copy()

    column_name = f"rogers_satchell_volatility_{window}d"
    _validate_output_columns_absent(
        dataset,
        output_columns=(column_name,),
        feature_name="rogers-satchell volatility",
    )

    daily_rogers_satchell_variance = (
        np.log(dataset["high"].div(dataset["open"]))
        * np.log(dataset["high"].div(dataset["close"]))
        + np.log(dataset["low"].div(dataset["open"]))
        * np.log(dataset["low"].div(dataset["close"]))
    )
    dataset[column_name] = daily_rogers_satchell_variance.groupby(
        dataset["symbol"],
        sort=False,
    ).transform(
        lambda values: np.sqrt(
            values.rolling(
                window=window,
                min_periods=window,
            ).mean()
        )
    )
    dataset[column_name] = dataset[column_name].mask(~np.isfinite(dataset[column_name]))
    return dataset


def attach_yang_zhang_volatility(
    frame: pd.DataFrame,
    *,
    window: int,
) -> pd.DataFrame:
    """Attach trailing Yang-Zhang volatility from daily OHLC observations.

    Current definition uses the trailing Yang-Zhang variance estimator over the
    requested window:
    - overnight variance of ``log(open_t / close_{t-1})``
    - open-to-close variance of ``log(close_t / open_t)``
    - Rogers-Satchell variance mean within the same window

    with ``k = 0.34 / (1.34 + (window + 1) / (window - 1))``.

    Timing convention:
    - each feature row is anchored at the close of ``date``
    - rolling windows use only ``open`` / ``high`` / ``low`` / ``close``
      observations available through that same close
    - the overnight component uses ``close_{t-1}``, which is already known by
      the time ``date`` closes
    """
    window = _normalize_yang_zhang_window(window, parameter_name="window")
    dataset = validate_ohlcv(frame, source="yang-zhang volatility input").copy()

    column_name = f"yang_zhang_volatility_{window}d"
    _validate_output_columns_absent(
        dataset,
        output_columns=(column_name,),
        feature_name="yang-zhang volatility",
    )

    previous_close = dataset.groupby("symbol", sort=False)["close"].shift(1)
    overnight_returns = np.log(dataset["open"].div(previous_close))
    open_to_close_returns = np.log(dataset["close"].div(dataset["open"]))
    daily_rogers_satchell_variance = (
        np.log(dataset["high"].div(dataset["open"]))
        * np.log(dataset["high"].div(dataset["close"]))
        + np.log(dataset["low"].div(dataset["open"]))
        * np.log(dataset["low"].div(dataset["close"]))
    )

    rolling_overnight_variance = overnight_returns.groupby(
        dataset["symbol"],
        sort=False,
    ).transform(
        lambda values: values.rolling(
            window=window,
            min_periods=window,
        ).var(ddof=1)
    )
    rolling_open_to_close_variance = open_to_close_returns.groupby(
        dataset["symbol"],
        sort=False,
    ).transform(
        lambda values: values.rolling(
            window=window,
            min_periods=window,
        ).var(ddof=1)
    )
    rolling_rogers_satchell_variance = daily_rogers_satchell_variance.groupby(
        dataset["symbol"],
        sort=False,
    ).transform(
        lambda values: values.rolling(
            window=window,
            min_periods=window,
        ).mean()
    )

    weight = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
    rolling_yang_zhang_variance = (
        rolling_overnight_variance
        + weight * rolling_open_to_close_variance
        + (1.0 - weight) * rolling_rogers_satchell_variance
    )
    dataset[column_name] = np.sqrt(
        rolling_yang_zhang_variance.where(rolling_yang_zhang_variance >= 0.0)
    )
    dataset[column_name] = dataset[column_name].mask(~np.isfinite(dataset[column_name]))
    return dataset


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


def attach_amihud_illiquidity(
    frame: pd.DataFrame,
    *,
    window: int,
) -> pd.DataFrame:
    """Attach trailing Amihud illiquidity from daily returns and dollar volume.

    Current definition uses the daily Amihud-style illiquidity proxy
    ``abs(daily_return) / (close * volume)``, then takes the trailing window
    mean. Days with non-positive dollar volume are treated conservatively as
    unavailable rather than coerced to finite values.

    Timing convention:
    - each feature row is anchored at the close of ``date``
    - rolling windows use only strategy ``daily_return`` and same-day
      ``close`` / ``volume`` observations available through that same close
    """
    window = _normalize_window(window, parameter_name="window")
    dataset = _prepare_daily_return_input(
        frame,
        source="amihud illiquidity input",
    )

    column_name = f"amihud_illiquidity_{window}d"
    _validate_output_columns_absent(
        dataset,
        output_columns=(column_name,),
        feature_name="amihud illiquidity",
    )

    daily_dollar_volume = dataset["close"].mul(dataset["volume"])
    daily_illiquidity = dataset["daily_return"].abs().div(
        daily_dollar_volume.where(daily_dollar_volume > 0.0)
    )
    dataset[column_name] = daily_illiquidity.groupby(
        dataset["symbol"],
        sort=False,
    ).transform(
        lambda values: values.rolling(
            window=window,
            min_periods=window,
        ).mean()
    )
    dataset[column_name] = dataset[column_name].mask(~np.isfinite(dataset[column_name]))
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


def _normalize_yang_zhang_window(value: int, *, parameter_name: str) -> int:
    """Validate rolling windows for Yang-Zhang volatility features."""
    normalized = _normalize_window(value, parameter_name=parameter_name)
    if normalized < 2:
        raise ValueError(f"{parameter_name} must be at least 2.")
    return normalized


def _normalize_window(value: int, *, parameter_name: str) -> int:
    """Validate positive rolling windows."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{parameter_name} must be a positive integer.")
    return value

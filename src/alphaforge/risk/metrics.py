"""Risk analytics for daily backtest outputs and weight panels."""

from __future__ import annotations

import math

import pandas as pd


class RiskError(ValueError):
    """Raised when risk analytics inputs or settings are invalid."""


def summarize_risk(
    frame: pd.DataFrame,
    *,
    return_column: str = "net_return",
    gross_exposure_column: str = "gross_exposure",
    net_exposure_column: str = "net_exposure",
    periods_per_year: int = 252,
    var_confidence: float = 0.95,
) -> pd.Series:
    """Summarize daily backtest output into headline risk metrics."""
    periods_per_year = _normalize_positive_int(
        periods_per_year,
        parameter_name="periods_per_year",
    )
    var_confidence = _normalize_confidence(var_confidence)

    dataset = _prepare_risk_frame(
        frame,
        return_column=return_column,
        gross_exposure_column=gross_exposure_column,
        net_exposure_column=net_exposure_column,
    )
    returns = dataset[return_column]
    downside_returns = returns.clip(upper=0.0)

    realized_volatility = returns.std(ddof=1)
    if pd.isna(realized_volatility):
        annualized_realized_volatility = math.nan
    else:
        annualized_realized_volatility = realized_volatility * math.sqrt(periods_per_year)

    downside_volatility = math.sqrt((downside_returns.pow(2).mean())) * math.sqrt(
        periods_per_year
    )

    tail_quantile = 1.0 - var_confidence
    value_at_risk = returns.quantile(tail_quantile)
    tail_returns = returns.loc[returns <= value_at_risk]
    conditional_value_at_risk = tail_returns.mean()

    return pd.Series(
        {
            "periods": float(len(dataset)),
            "realized_volatility": annualized_realized_volatility,
            "downside_volatility": downside_volatility,
            "value_at_risk": value_at_risk,
            "conditional_value_at_risk": conditional_value_at_risk,
            "var_confidence": var_confidence,
            "average_gross_exposure": dataset[gross_exposure_column].mean(),
            "average_net_exposure": dataset[net_exposure_column].mean(),
        },
        name="risk_summary",
    )


def compute_rolling_benchmark_risk(
    frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame,
    *,
    return_column: str = "net_return",
    benchmark_return_column: str = "benchmark_return",
    window: int = 20,
) -> pd.DataFrame:
    """Compute rolling beta and correlation against a benchmark return series."""
    window = _normalize_positive_int(window, parameter_name="window")

    strategy = _prepare_date_return_frame(
        frame,
        return_column=return_column,
        source="strategy returns",
    )
    benchmark = _prepare_date_return_frame(
        benchmark_frame,
        return_column=benchmark_return_column,
        source="benchmark returns",
    )
    _validate_exact_date_alignment(strategy, benchmark)

    merged = strategy.merge(
        benchmark,
        on="date",
        how="inner",
        validate="one_to_one",
    )

    rolling_covariance = merged[return_column].rolling(
        window=window,
        min_periods=window,
    ).cov(merged[benchmark_return_column])
    rolling_benchmark_variance = merged[benchmark_return_column].rolling(
        window=window,
        min_periods=window,
    ).var(ddof=1)

    result = merged.rename(
        columns={
            return_column: "strategy_return",
            benchmark_return_column: "benchmark_return",
        }
    )
    result["rolling_correlation"] = merged[return_column].rolling(
        window=window,
        min_periods=window,
    ).corr(merged[benchmark_return_column])
    result["rolling_beta"] = rolling_covariance.div(rolling_benchmark_variance)
    return result


def summarize_rolling_benchmark_risk(frame: pd.DataFrame) -> pd.Series:
    """Summarize rolling benchmark beta/correlation outputs."""
    dataset = _prepare_rolling_benchmark_risk_frame(frame)
    valid_mask = dataset["rolling_beta"].notna() & dataset["rolling_correlation"].notna()
    valid_periods = int(valid_mask.sum())

    latest_beta = math.nan
    latest_correlation = math.nan
    if valid_periods > 0:
        latest_row = dataset.loc[valid_mask].iloc[-1]
        latest_beta = float(latest_row["rolling_beta"])
        latest_correlation = float(latest_row["rolling_correlation"])

    return pd.Series(
        {
            "periods": float(len(dataset)),
            "valid_periods": float(valid_periods),
            "average_rolling_beta": dataset["rolling_beta"].mean(),
            "latest_rolling_beta": latest_beta,
            "average_rolling_correlation": dataset["rolling_correlation"].mean(),
            "latest_rolling_correlation": latest_correlation,
        },
        name="benchmark_risk_summary",
    )


def summarize_weight_concentration(
    frame: pd.DataFrame,
    *,
    weight_column: str = "effective_weight",
) -> pd.Series:
    """Summarize daily position concentration from a weight panel."""
    dataset = _prepare_weight_panel(frame, weight_column=weight_column)
    per_day = _compute_daily_weight_profile(dataset, weight_column=weight_column)

    return pd.Series(
        {
            "periods": float(len(per_day)),
            "average_gross_exposure": per_day["gross_exposure"].mean(),
            "max_gross_exposure": per_day["gross_exposure"].max(),
            "average_net_exposure": per_day["net_exposure"].mean(),
            "average_herfindahl_index": per_day["herfindahl_index"].mean(),
            "average_max_abs_weight": per_day["max_abs_weight"].mean(),
        },
        name="weight_concentration_summary",
    )


def summarize_portfolio_diversification(
    frame: pd.DataFrame,
    *,
    weight_column: str = "portfolio_weight",
) -> pd.Series:
    """Summarize target or effective weight diversification from a dated panel."""
    dataset = _prepare_weight_panel(frame, weight_column=weight_column)
    per_day = _compute_daily_weight_profile(dataset, weight_column=weight_column)

    return pd.Series(
        {
            "periods": float(len(per_day)),
            "average_holdings_count": per_day["holdings_count"].mean(),
            "min_holdings_count": per_day["holdings_count"].min(),
            "average_long_count": per_day["long_count"].mean(),
            "average_short_count": per_day["short_count"].mean(),
            "average_effective_number_of_positions": per_day[
                "effective_number_of_positions"
            ].mean(),
            "min_effective_number_of_positions": per_day[
                "effective_number_of_positions"
            ].min(),
            "average_effective_position_ratio": per_day[
                "effective_position_ratio"
            ].mean(),
            "min_effective_position_ratio": per_day[
                "effective_position_ratio"
            ].min(),
            "average_top_position_weight_share": per_day[
                "top_position_weight_share"
            ].mean(),
            "max_top_position_weight_share": per_day[
                "top_position_weight_share"
            ].max(),
            "average_top_five_weight_share": per_day["top_five_weight_share"].mean(),
            "max_top_five_weight_share": per_day["top_five_weight_share"].max(),
        },
        name="portfolio_diversification_summary",
    )


def summarize_group_exposure(
    frame: pd.DataFrame,
    *,
    group_column: str,
    weight_column: str = "portfolio_weight",
) -> pd.DataFrame:
    """Summarize dated gross/net exposure by an explicit group column."""
    group_column = _normalize_non_empty_string(
        group_column,
        parameter_name="group_column",
    )
    dataset = _prepare_group_weight_panel(
        frame,
        group_column=group_column,
        weight_column=weight_column,
    )
    dataset["_group_value"] = _normalize_group_values(dataset[group_column])
    dataset["_is_missing_group"] = dataset["_group_value"].eq("<missing>")
    dataset["_absolute_weight"] = dataset[weight_column].abs()
    dataset["_active_weight"] = dataset["_absolute_weight"] > 1e-12

    per_date_group = (
        dataset.groupby(
            ["date", "_group_value", "_is_missing_group"],
            sort=True,
            dropna=False,
        )
        .agg(
            gross_exposure=("_absolute_weight", "sum"),
            net_exposure=(weight_column, "sum"),
            holdings_count=("_active_weight", "sum"),
        )
        .reset_index()
    )
    per_date_group["abs_net_exposure"] = per_date_group["net_exposure"].abs()

    summary = (
        per_date_group.groupby(
            ["_group_value", "_is_missing_group"],
            sort=True,
            dropna=False,
        )
        .agg(
            periods=("date", "nunique"),
            average_gross_exposure=("gross_exposure", "mean"),
            max_gross_exposure=("gross_exposure", "max"),
            average_net_exposure=("net_exposure", "mean"),
            max_abs_net_exposure=("abs_net_exposure", "max"),
            average_holdings_count=("holdings_count", "mean"),
            max_holdings_count=("holdings_count", "max"),
        )
        .reset_index()
        .rename(
            columns={
                "_group_value": "group_value",
                "_is_missing_group": "is_missing_group",
            }
        )
    )
    summary.insert(0, "group_column", group_column)
    return summary.loc[
        :,
        [
            "group_column",
            "group_value",
            "is_missing_group",
            "periods",
            "average_gross_exposure",
            "max_gross_exposure",
            "average_net_exposure",
            "max_abs_net_exposure",
            "average_holdings_count",
            "max_holdings_count",
        ],
    ]


def format_risk_summary(summary: pd.Series) -> str:
    """Format a headline risk summary as plain text."""
    required_keys = [
        "periods",
        "realized_volatility",
        "downside_volatility",
        "value_at_risk",
        "conditional_value_at_risk",
        "var_confidence",
        "average_gross_exposure",
        "average_net_exposure",
    ]
    missing_keys = [key for key in required_keys if key not in summary.index]
    if missing_keys:
        missing_text = ", ".join(missing_keys)
        raise RiskError(f"summary is missing required risk fields: {missing_text}.")

    confidence_percent = int(round(float(summary["var_confidence"]) * 100))
    lines = [
        "Risk Summary",
        f"Periods: {int(summary['periods'])}",
        f"Realized Volatility: {_format_percent(summary['realized_volatility'])}",
        f"Downside Volatility: {_format_percent(summary['downside_volatility'])}",
        f"VaR ({confidence_percent}%): {_format_percent(summary['value_at_risk'])}",
        f"CVaR ({confidence_percent}%): {_format_percent(summary['conditional_value_at_risk'])}",
        f"Average Gross Exposure: {_format_number(summary['average_gross_exposure'])}",
        f"Average Net Exposure: {_format_number(summary['average_net_exposure'])}",
    ]
    return "\n".join(lines)


def format_benchmark_risk_summary(summary: pd.Series, *, window: int) -> str:
    """Format rolling benchmark beta/correlation diagnostics as plain text."""
    required_keys = [
        "periods",
        "valid_periods",
        "average_rolling_beta",
        "latest_rolling_beta",
        "average_rolling_correlation",
        "latest_rolling_correlation",
    ]
    missing_keys = [key for key in required_keys if key not in summary.index]
    if missing_keys:
        missing_text = ", ".join(missing_keys)
        raise RiskError(
            f"summary is missing required benchmark-risk fields: {missing_text}."
        )

    lines = [
        "Benchmark Risk Summary",
        f"Periods: {int(summary['periods'])}",
        f"Rolling Window: {window}",
        f"Valid Rolling Periods: {int(summary['valid_periods'])}",
        f"Average Rolling Beta: {_format_number(summary['average_rolling_beta'])}",
        f"Latest Rolling Beta: {_format_number(summary['latest_rolling_beta'])}",
        "Average Rolling Correlation: "
        f"{_format_number(summary['average_rolling_correlation'])}",
        "Latest Rolling Correlation: "
        f"{_format_number(summary['latest_rolling_correlation'])}",
    ]
    return "\n".join(lines)


def _prepare_risk_frame(
    frame: pd.DataFrame,
    *,
    return_column: str,
    gross_exposure_column: str,
    net_exposure_column: str,
) -> pd.DataFrame:
    """Validate the daily backtest frame for risk summary calculations."""
    dataset = frame.copy()
    for column in [return_column, gross_exposure_column, net_exposure_column]:
        if column not in dataset.columns:
            raise RiskError(f"backtest results are missing '{column}'.")
        dataset[column] = _parse_numeric_column(dataset[column], column_name=column)

    if dataset.empty:
        raise RiskError("backtest results must contain at least one row.")

    return dataset.reset_index(drop=True)


def _prepare_date_return_frame(
    frame: pd.DataFrame,
    *,
    return_column: str,
    source: str,
) -> pd.DataFrame:
    """Validate a dated return series used for benchmark comparisons."""
    if "date" not in frame.columns:
        raise RiskError(f"{source} are missing 'date'.")
    if return_column not in frame.columns:
        raise RiskError(f"{source} are missing '{return_column}'.")

    dataset = frame.loc[:, ["date", return_column]].copy()
    dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
    if dataset["date"].isna().any():
        raise RiskError(f"{source} contain invalid date values.")
    if dataset["date"].duplicated().any():
        raise RiskError(f"{source} contain duplicate dates.")

    dataset[return_column] = _parse_numeric_column(
        dataset[return_column],
        column_name=return_column,
    )
    if dataset.empty:
        raise RiskError(f"{source} must contain at least one row.")

    return dataset.sort_values("date", kind="mergesort").reset_index(drop=True)


def _prepare_weight_panel(frame: pd.DataFrame, *, weight_column: str) -> pd.DataFrame:
    """Validate a dated weight panel used for concentration summaries."""
    required_columns = ["date", "symbol", weight_column]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise RiskError(f"weight panel is missing required columns: {missing_text}.")

    dataset = frame.loc[:, required_columns].copy()
    dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
    if dataset["date"].isna().any():
        raise RiskError("weight panel contains invalid date values.")
    dataset[weight_column] = _parse_numeric_column(
        dataset[weight_column],
        column_name=weight_column,
    )
    if dataset.empty:
        raise RiskError("weight panel must contain at least one row.")

    return dataset.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)


def _prepare_group_weight_panel(
    frame: pd.DataFrame,
    *,
    group_column: str,
    weight_column: str,
) -> pd.DataFrame:
    """Validate a dated weight panel with an explicit group label column."""
    required_columns = ["date", "symbol", group_column, weight_column]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise RiskError(f"weight panel is missing required columns: {missing_text}.")

    dataset = frame.loc[:, required_columns].copy()
    dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
    if dataset["date"].isna().any():
        raise RiskError("weight panel contains invalid date values.")
    dataset[weight_column] = _parse_numeric_column(
        dataset[weight_column],
        column_name=weight_column,
    )
    if dataset.empty:
        raise RiskError("weight panel must contain at least one row.")

    return dataset.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)


def _compute_daily_weight_profile(
    dataset: pd.DataFrame,
    *,
    weight_column: str,
) -> pd.DataFrame:
    """Compute per-date concentration and diversification primitives."""
    rows: list[dict[str, float | pd.Timestamp]] = []
    tolerance = 1e-12

    for date, group in dataset.groupby("date", sort=True):
        weights = group[weight_column]
        absolute_weights = weights.abs()
        active_absolute_weights = absolute_weights.loc[absolute_weights > tolerance]
        gross_exposure = absolute_weights.sum()
        net_exposure = weights.sum()
        max_abs_weight = absolute_weights.max()
        holdings_count = float(len(active_absolute_weights))
        long_count = float((weights > tolerance).sum())
        short_count = float((weights < -tolerance).sum())

        if active_absolute_weights.empty or gross_exposure <= tolerance:
            herfindahl_index = 0.0
            effective_number_of_positions = 0.0
            effective_position_ratio = 0.0
            top_position_weight_share = 0.0
            top_five_weight_share = 0.0
        else:
            normalized_absolute_weights = active_absolute_weights / gross_exposure
            herfindahl_index = normalized_absolute_weights.pow(2).sum()
            effective_number_of_positions = (
                1.0 / herfindahl_index if herfindahl_index > 0.0 else 0.0
            )
            effective_position_ratio = (
                effective_number_of_positions / holdings_count
                if holdings_count > 0.0
                else 0.0
            )
            sorted_weight_shares = normalized_absolute_weights.sort_values(
                ascending=False,
                kind="mergesort",
            )
            top_position_weight_share = sorted_weight_shares.iloc[0]
            top_five_weight_share = sorted_weight_shares.head(5).sum()

        rows.append(
            {
                "date": date,
                "gross_exposure": float(gross_exposure),
                "net_exposure": float(net_exposure),
                "max_abs_weight": float(max_abs_weight),
                "herfindahl_index": float(herfindahl_index),
                "holdings_count": holdings_count,
                "long_count": long_count,
                "short_count": short_count,
                "effective_number_of_positions": float(
                    effective_number_of_positions
                ),
                "effective_position_ratio": float(effective_position_ratio),
                "top_position_weight_share": float(top_position_weight_share),
                "top_five_weight_share": float(top_five_weight_share),
            }
        )

    return pd.DataFrame(rows)


def _prepare_rolling_benchmark_risk_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate rolling benchmark risk outputs before summarization."""
    required_columns = ["date", "rolling_beta", "rolling_correlation"]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise RiskError(
            "rolling benchmark risk frame is missing required columns: "
            f"{missing_text}."
        )

    dataset = frame.loc[:, required_columns].copy()
    dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
    if dataset["date"].isna().any():
        raise RiskError("rolling benchmark risk frame contains invalid date values.")
    if dataset["date"].duplicated().any():
        raise RiskError("rolling benchmark risk frame contains duplicate dates.")

    for column in ["rolling_beta", "rolling_correlation"]:
        parsed = pd.to_numeric(dataset[column], errors="coerce")
        invalid_values = dataset[column].notna() & parsed.isna()
        if invalid_values.any():
            raise RiskError(
                "rolling benchmark risk frame contains invalid numeric values in "
                f"'{column}'."
            )
        dataset[column] = parsed

    if dataset.empty:
        raise RiskError("rolling benchmark risk frame must contain at least one row.")
    return dataset.sort_values("date", kind="mergesort").reset_index(drop=True)


def _validate_exact_date_alignment(
    strategy: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> None:
    """Require benchmark dates to match strategy dates exactly after sorting."""
    strategy_dates = strategy["date"].reset_index(drop=True)
    benchmark_dates = benchmark["date"].reset_index(drop=True)
    if strategy_dates.equals(benchmark_dates):
        return

    strategy_set = set(strategy_dates.tolist())
    benchmark_set = set(benchmark_dates.tolist())
    missing_dates = sorted(strategy_set - benchmark_set)
    extra_dates = sorted(benchmark_set - strategy_set)
    problems: list[str] = []
    if missing_dates:
        problems.append(f"missing {len(missing_dates)} date(s)")
    if extra_dates:
        problems.append(f"extra {len(extra_dates)} date(s)")
    detail = ""
    if problems:
        detail = " (" + "; ".join(problems) + ")"
    raise RiskError(
        "benchmark returns must align exactly to strategy dates" + detail + "."
    )


def _parse_numeric_column(values: pd.Series, *, column_name: str) -> pd.Series:
    """Parse numeric risk columns without silent coercion."""
    parsed = pd.to_numeric(values, errors="coerce")
    if parsed.isna().any():
        raise RiskError(
            f"risk inputs contain invalid numeric values in '{column_name}'."
        )
    return parsed


def _normalize_non_empty_string(value: str, *, parameter_name: str) -> str:
    """Validate non-empty string parameters."""
    if not isinstance(value, str) or not value.strip():
        raise RiskError(f"{parameter_name} must be a non-empty string.")
    return value.strip()


def _normalize_group_values(values: pd.Series) -> pd.Series:
    """Normalize explicit group labels while surfacing missing labels."""
    normalized = values.astype("string").str.strip()
    missing = values.isna() | normalized.eq("").fillna(False)
    return normalized.mask(missing, "<missing>")


def _normalize_positive_int(value: int, *, parameter_name: str) -> int:
    """Validate positive integer parameters."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RiskError(f"{parameter_name} must be a positive integer.")
    return value


def _normalize_confidence(value: float) -> float:
    """Validate confidence levels used for VaR/CVaR."""
    if isinstance(value, bool):
        raise RiskError("var_confidence must be strictly between 0 and 1.")

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise RiskError(
            "var_confidence must be strictly between 0 and 1."
        ) from exc

    if pd.isna(numeric_value) or numeric_value <= 0.0 or numeric_value >= 1.0:
        raise RiskError("var_confidence must be strictly between 0 and 1.")
    return numeric_value


def _format_percent(value: float) -> str:
    """Format a decimal value as a percentage."""
    if pd.isna(value):
        return "NaN"
    return f"{value:.2%}"


def _format_number(value: float) -> str:
    """Format a scalar metric with two decimals."""
    if pd.isna(value):
        return "NaN"
    return f"{value:.2f}"

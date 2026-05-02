"""Static chart rendering utilities for report-style research artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import pandas as pd

from alphaforge.analytics.performance import compute_drawdown_series
from alphaforge.common.errors import AlphaForgeError

PRIMARY_COLOR = "#0B4F6C"
SECONDARY_COLOR = "#1F9D8B"
TERTIARY_COLOR = "#C68642"
NEGATIVE_COLOR = "#C44536"
GRID_COLOR = "#D9E2EC"


class VisualizationError(AlphaForgeError):
    """Raised when chart inputs or output paths are invalid."""


def save_nav_overview_chart(
    frame: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Render strategy and optional benchmark NAV series into one PNG."""
    dataset = _prepare_frame(
        frame,
        required_columns=["date", "net_nav", "gross_nav"],
        source="nav chart input",
    )

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(dataset["date"], dataset["net_nav"], label="Net NAV", color=PRIMARY_COLOR, linewidth=2.0)
    axis.plot(
        dataset["date"],
        dataset["gross_nav"],
        label="Gross NAV",
        color=SECONDARY_COLOR,
        linewidth=1.5,
        linestyle="--",
    )
    if "benchmark_nav" in dataset.columns:
        dataset["benchmark_nav"] = _parse_numeric_series(
            dataset["benchmark_nav"],
            column_name="benchmark_nav",
            allow_na=True,
        )
        if dataset["benchmark_nav"].notna().any():
            axis.plot(
                dataset["date"],
                dataset["benchmark_nav"],
                label="Benchmark NAV",
                color=TERTIARY_COLOR,
                linewidth=1.5,
            )
    if "relative_nav" in dataset.columns:
        dataset["relative_nav"] = _parse_numeric_series(
            dataset["relative_nav"],
            column_name="relative_nav",
            allow_na=True,
        )
        if dataset["relative_nav"].notna().any():
            axis.plot(
                dataset["date"],
                dataset["relative_nav"],
                label="Relative NAV",
                color=NEGATIVE_COLOR,
                linewidth=1.5,
                linestyle=":",
            )

    axis.set_title("NAV Overview")
    axis.set_xlabel("Date")
    axis.set_ylabel("NAV")
    axis.grid(True, color=GRID_COLOR, linewidth=0.8)
    axis.legend(loc="best")
    figure.autofmt_xdate()
    return _save_figure(figure, path)


def save_drawdown_chart(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    return_column: str = "net_return",
) -> Path:
    """Render drawdown from the selected return column into one PNG."""
    drawdown = compute_drawdown_series(frame, return_column=return_column)
    dataset = _prepare_frame(
        drawdown,
        required_columns=["date", "drawdown"],
        source="drawdown chart input",
    )

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.fill_between(
        dataset["date"],
        dataset["drawdown"],
        0.0,
        color=NEGATIVE_COLOR,
        alpha=0.25,
    )
    axis.plot(dataset["date"], dataset["drawdown"], color=NEGATIVE_COLOR, linewidth=1.5)
    axis.set_title("Drawdown")
    axis.set_xlabel("Date")
    axis.set_ylabel("Drawdown")
    axis.grid(True, color=GRID_COLOR, linewidth=0.8)
    figure.autofmt_xdate()
    return _save_figure(figure, path)


def save_exposure_turnover_chart(
    frame: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Render exposures and turnover in a two-panel PNG."""
    dataset = _prepare_frame(
        frame,
        required_columns=[
            "date",
            "gross_exposure",
            "net_exposure",
            "turnover",
            "target_turnover",
        ],
        source="exposure/turnover chart input",
    )

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(
        dataset["date"],
        dataset["gross_exposure"],
        label="Gross Exposure",
        color=PRIMARY_COLOR,
        linewidth=1.8,
    )
    axes[0].plot(
        dataset["date"],
        dataset["net_exposure"],
        label="Net Exposure",
        color=SECONDARY_COLOR,
        linewidth=1.8,
    )
    axes[0].set_title("Exposure")
    axes[0].set_ylabel("Exposure")
    axes[0].grid(True, color=GRID_COLOR, linewidth=0.8)
    axes[0].legend(loc="best")

    axes[1].bar(
        dataset["date"],
        dataset["turnover"],
        label="Realized Turnover",
        color=TERTIARY_COLOR,
        alpha=0.7,
        width=0.8,
    )
    axes[1].plot(
        dataset["date"],
        dataset["target_turnover"],
        label="Target Turnover",
        color=NEGATIVE_COLOR,
        linewidth=1.6,
    )
    axes[1].set_title("Turnover")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Turnover")
    axes[1].grid(True, color=GRID_COLOR, linewidth=0.8)
    axes[1].legend(loc="best")

    figure.autofmt_xdate()
    return _save_figure(figure, path)


def save_ic_series_chart(
    frame: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Render a dated IC time series and its observation counts into one PNG."""
    dataset = _prepare_frame(
        frame,
        required_columns=["date", "ic", "observations"],
        source="IC chart input",
    )

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].axhline(0.0, color="#52606D", linewidth=1.0, linestyle="--")
    axes[0].plot(dataset["date"], dataset["ic"], color=PRIMARY_COLOR, linewidth=1.8)
    valid_ic = dataset["ic"].dropna()
    if not valid_ic.empty:
        axes[0].axhline(
            float(valid_ic.mean()),
            color=TERTIARY_COLOR,
            linewidth=1.2,
            linestyle=":",
            label="Mean IC",
        )
        axes[0].legend(loc="best")
    axes[0].set_title("IC Series")
    axes[0].set_ylabel("IC")
    axes[0].grid(True, color=GRID_COLOR, linewidth=0.8)

    axes[1].bar(dataset["date"], dataset["observations"], color=SECONDARY_COLOR, alpha=0.8, width=0.8)
    axes[1].set_title("IC Observations")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, color=GRID_COLOR, linewidth=0.8)

    figure.autofmt_xdate()
    return _save_figure(figure, path)


def save_ic_cumulative_chart(
    frame: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Render cumulative IC through time into one PNG."""
    dataset = _prepare_frame(
        frame,
        required_columns=["date", "ic"],
        source="cumulative IC chart input",
    )
    dataset["cumulative_ic"] = dataset["ic"].fillna(0.0).cumsum()

    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.axhline(0.0, color="#52606D", linewidth=1.0, linestyle="--")
    axis.plot(dataset["date"], dataset["cumulative_ic"], color=PRIMARY_COLOR, linewidth=1.8)
    axis.set_title("Cumulative IC")
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative IC")
    axis.grid(True, color=GRID_COLOR, linewidth=0.8)
    figure.autofmt_xdate()
    return _save_figure(figure, path)


def save_ic_decay_chart(
    frame: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Render per-date IC across configured forward-return horizons."""
    dataset = _prepare_frame(
        frame,
        required_columns=["date", "horizon", "ic", "observations"],
        source="IC decay chart input",
    )

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].axhline(0.0, color="#52606D", linewidth=1.0, linestyle="--")

    for horizon, group in dataset.groupby("horizon", sort=True, dropna=False):
        sorted_group = group.sort_values("date", kind="mergesort")
        label = _format_horizon_label(horizon)
        axes[0].plot(
            sorted_group["date"],
            sorted_group["ic"],
            label=label,
            linewidth=1.6,
        )
        axes[1].plot(
            sorted_group["date"],
            sorted_group["observations"],
            label=label,
            linewidth=1.4,
        )

    axes[0].set_title("IC Decay Series")
    axes[0].set_ylabel("IC")
    axes[0].grid(True, color=GRID_COLOR, linewidth=0.8)
    axes[0].legend(loc="best")

    axes[1].set_title("IC Observations by Horizon")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, color=GRID_COLOR, linewidth=0.8)
    axes[1].legend(loc="best")

    figure.autofmt_xdate()
    return _save_figure(figure, path)


def save_coverage_summary_chart(
    summary: pd.Series,
    path: str | Path,
) -> Path:
    """Render signal/label/joint coverage ratios into one PNG."""
    required_keys = [
        "signal_coverage_ratio",
        "forward_return_coverage_ratio",
        "joint_coverage_ratio",
    ]
    missing_keys = [key for key in required_keys if key not in summary.index]
    if missing_keys:
        missing_text = ", ".join(missing_keys)
        raise VisualizationError(
            f"coverage summary is missing required fields: {missing_text}."
        )

    labels = ["Signal", "Forward Return", "Joint"]
    values = [
        _parse_scalar(summary["signal_coverage_ratio"], field_name="signal_coverage_ratio"),
        _parse_scalar(
            summary["forward_return_coverage_ratio"],
            field_name="forward_return_coverage_ratio",
        ),
        _parse_scalar(summary["joint_coverage_ratio"], field_name="joint_coverage_ratio"),
    ]

    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(labels, values, color=[PRIMARY_COLOR, SECONDARY_COLOR, TERTIARY_COLOR])
    axis.set_title("Coverage Summary")
    axis.set_ylabel("Coverage Ratio")
    axis.set_ylim(0.0, 1.05)
    axis.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.02,
            f"{value:.0%}",
            ha="center",
            va="bottom",
        )
    return _save_figure(figure, path)


def save_coverage_timeseries_chart(
    frame: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Render per-date signal/label/joint coverage ratios into one PNG."""
    dataset = _prepare_frame(
        frame,
        required_columns=[
            "date",
            "signal_coverage_ratio",
            "forward_return_coverage_ratio",
            "joint_coverage_ratio",
        ],
        source="coverage timeseries chart input",
    )

    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(
        dataset["date"],
        dataset["signal_coverage_ratio"],
        label="Signal Coverage",
        color=PRIMARY_COLOR,
        linewidth=1.8,
    )
    axis.plot(
        dataset["date"],
        dataset["forward_return_coverage_ratio"],
        label="Forward Return Coverage",
        color=SECONDARY_COLOR,
        linewidth=1.8,
    )
    axis.plot(
        dataset["date"],
        dataset["joint_coverage_ratio"],
        label="Joint Coverage",
        color=TERTIARY_COLOR,
        linewidth=1.8,
    )
    axis.set_title("Coverage Through Time")
    axis.set_xlabel("Date")
    axis.set_ylabel("Coverage Ratio")
    axis.set_ylim(0.0, 1.05)
    axis.grid(True, color=GRID_COLOR, linewidth=0.8)
    axis.legend(loc="best")
    figure.autofmt_xdate()
    return _save_figure(figure, path)


def save_grouped_ic_summary_chart(
    frame: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Render grouped IC summary diagnostics into one PNG."""
    dataset = _prepare_frame(
        frame,
        required_columns=["group_column", "group_value", "mean_ic", "valid_periods"],
        source="grouped IC summary chart input",
    )
    dataset = _with_group_labels(dataset)
    colors = [
        SECONDARY_COLOR if value >= 0.0 else NEGATIVE_COLOR
        for value in dataset["mean_ic"]
    ]

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].bar(dataset["group_label"], dataset["mean_ic"], color=colors, alpha=0.85)
    axes[0].axhline(0.0, color="#52606D", linewidth=1.0, linestyle="--")
    axes[0].set_title("Grouped IC Summary")
    axes[0].set_ylabel("Mean IC")
    axes[0].grid(True, axis="y", color=GRID_COLOR, linewidth=0.8)

    axes[1].bar(
        dataset["group_label"],
        dataset["valid_periods"],
        color=TERTIARY_COLOR,
        alpha=0.8,
    )
    axes[1].set_title("Grouped IC Valid Periods")
    axes[1].set_ylabel("Periods")
    axes[1].grid(True, axis="y", color=GRID_COLOR, linewidth=0.8)
    axes[1].tick_params(axis="x", rotation=25)

    return _save_figure(figure, path)


def save_grouped_ic_timeseries_chart(
    frame: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Render grouped IC through time and observation counts into one PNG."""
    dataset = _prepare_frame(
        frame,
        required_columns=[
            "date",
            "group_column",
            "group_value",
            "ic",
            "observations",
        ],
        source="grouped IC timeseries chart input",
    )
    dataset = _with_group_labels(dataset).sort_values(
        ["group_label", "date"],
        kind="mergesort",
    )

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].axhline(0.0, color="#52606D", linewidth=1.0, linestyle="--")
    for group_label, group in dataset.groupby("group_label", sort=False):
        sorted_group = group.sort_values("date", kind="mergesort")
        axes[0].plot(
            sorted_group["date"],
            sorted_group["ic"],
            label=str(group_label),
            linewidth=1.6,
        )
        axes[1].plot(
            sorted_group["date"],
            sorted_group["observations"],
            label=str(group_label),
            linewidth=1.4,
        )

    axes[0].set_title("Grouped IC Through Time")
    axes[0].set_ylabel("IC")
    axes[0].grid(True, color=GRID_COLOR, linewidth=0.8)
    axes[0].legend(loc="best")

    axes[1].set_title("Grouped IC Observations")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, color=GRID_COLOR, linewidth=0.8)
    axes[1].legend(loc="best")

    figure.autofmt_xdate()
    return _save_figure(figure, path)


def save_grouped_coverage_summary_chart(
    frame: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Render grouped signal/label coverage summary diagnostics into one PNG."""
    dataset = _prepare_frame(
        frame,
        required_columns=[
            "group_column",
            "group_value",
            "signal_coverage_ratio",
            "forward_return_coverage_ratio",
            "joint_coverage_ratio",
        ],
        source="grouped coverage summary chart input",
    )
    dataset = _with_group_labels(dataset)
    bar_width = 0.25
    positions = list(range(len(dataset)))

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(
        [position - bar_width for position in positions],
        dataset["signal_coverage_ratio"],
        width=bar_width,
        label="Signal",
        color=PRIMARY_COLOR,
        alpha=0.85,
    )
    axis.bar(
        positions,
        dataset["forward_return_coverage_ratio"],
        width=bar_width,
        label="Forward Return",
        color=SECONDARY_COLOR,
        alpha=0.85,
    )
    axis.bar(
        [position + bar_width for position in positions],
        dataset["joint_coverage_ratio"],
        width=bar_width,
        label="Joint",
        color=TERTIARY_COLOR,
        alpha=0.85,
    )
    axis.set_title("Grouped Coverage Summary")
    axis.set_ylabel("Coverage Ratio")
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks(positions, dataset["group_label"], rotation=25, ha="right")
    axis.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8)
    axis.legend(loc="best")
    return _save_figure(figure, path)


def save_grouped_coverage_timeseries_chart(
    frame: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Render grouped joint coverage through time and usable rows into one PNG."""
    dataset = _prepare_frame(
        frame,
        required_columns=[
            "date",
            "group_column",
            "group_value",
            "joint_coverage_ratio",
            "usable_rows",
        ],
        source="grouped coverage timeseries chart input",
    )
    dataset = _with_group_labels(dataset).sort_values(
        ["group_label", "date"],
        kind="mergesort",
    )

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for group_label, group in dataset.groupby("group_label", sort=False):
        sorted_group = group.sort_values("date", kind="mergesort")
        axes[0].plot(
            sorted_group["date"],
            sorted_group["joint_coverage_ratio"],
            label=str(group_label),
            linewidth=1.6,
        )
        axes[1].plot(
            sorted_group["date"],
            sorted_group["usable_rows"],
            label=str(group_label),
            linewidth=1.4,
        )

    axes[0].set_title("Grouped Joint Coverage Through Time")
    axes[0].set_ylabel("Coverage Ratio")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, color=GRID_COLOR, linewidth=0.8)
    axes[0].legend(loc="best")

    axes[1].set_title("Grouped Usable Rows")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Rows")
    axes[1].grid(True, color=GRID_COLOR, linewidth=0.8)
    axes[1].legend(loc="best")

    figure.autofmt_xdate()
    return _save_figure(figure, path)


def save_quantile_bucket_chart(
    frame: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Render mean forward returns by signal quantile into one PNG."""
    dataset = _prepare_frame(
        frame,
        required_columns=["quantile", "mean_forward_return"],
        source="quantile chart input",
    )
    if dataset.empty:
        raise VisualizationError("quantile chart input must contain at least one row.")

    labels = dataset["quantile"].astype(int).astype(str).tolist()
    values = dataset["mean_forward_return"].tolist()
    colors = [SECONDARY_COLOR if value >= 0.0 else NEGATIVE_COLOR for value in values]

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, values, color=colors, alpha=0.85)
    axis.axhline(0.0, color="#52606D", linewidth=1.0, linestyle="--")
    axis.set_title("Quantile Bucket Returns")
    axis.set_xlabel("Quantile")
    axis.set_ylabel("Mean Forward Return")
    axis.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8)
    return _save_figure(figure, path)


def save_quantile_cumulative_chart(
    frame: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Render cumulative mean forward-return paths by signal quantile."""
    dataset = _prepare_frame(
        frame,
        required_columns=["date", "quantile", "cumulative_forward_return"],
        source="quantile cumulative chart input",
    )

    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.axhline(0.0, color="#52606D", linewidth=1.0, linestyle="--")
    for quantile, group in dataset.groupby("quantile", sort=True):
        sorted_group = group.sort_values("date", kind="mergesort")
        axis.plot(
            sorted_group["date"],
            sorted_group["cumulative_forward_return"],
            label=f"Q{int(quantile)}",
            linewidth=1.8,
        )
    axis.set_title("Cumulative Quantile Mean Forward Returns")
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative Forward Return")
    axis.grid(True, color=GRID_COLOR, linewidth=0.8)
    axis.legend(loc="best")
    figure.autofmt_xdate()
    return _save_figure(figure, path)


def save_quantile_spread_chart(
    frame: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Render top-minus-bottom quantile spread through time into one PNG."""
    dataset = _prepare_frame(
        frame,
        required_columns=["date", "top_bottom_spread"],
        source="quantile spread chart input",
    )

    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.axhline(0.0, color="#52606D", linewidth=1.0, linestyle="--")
    axis.fill_between(
        dataset["date"],
        dataset["top_bottom_spread"],
        0.0,
        where=dataset["top_bottom_spread"] >= 0.0,
        color=SECONDARY_COLOR,
        alpha=0.25,
        interpolate=True,
    )
    axis.fill_between(
        dataset["date"],
        dataset["top_bottom_spread"],
        0.0,
        where=dataset["top_bottom_spread"] < 0.0,
        color=NEGATIVE_COLOR,
        alpha=0.25,
        interpolate=True,
    )
    axis.plot(dataset["date"], dataset["top_bottom_spread"], color=PRIMARY_COLOR, linewidth=1.8)
    axis.set_title("Top-Bottom Quantile Spread")
    axis.set_xlabel("Date")
    axis.set_ylabel("Spread")
    axis.grid(True, color=GRID_COLOR, linewidth=0.8)
    figure.autofmt_xdate()
    return _save_figure(figure, path)


def save_rolling_benchmark_risk_chart(
    frame: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Render rolling benchmark beta and correlation into one PNG."""
    dataset = _prepare_frame(
        frame,
        required_columns=["date", "rolling_beta", "rolling_correlation"],
        source="benchmark risk chart input",
    )

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].axhline(0.0, color="#52606D", linewidth=1.0, linestyle="--")
    axes[0].plot(dataset["date"], dataset["rolling_beta"], color=PRIMARY_COLOR, linewidth=1.8)
    axes[0].set_title("Rolling Beta")
    axes[0].set_ylabel("Beta")
    axes[0].grid(True, color=GRID_COLOR, linewidth=0.8)

    axes[1].axhline(0.0, color="#52606D", linewidth=1.0, linestyle="--")
    axes[1].plot(
        dataset["date"],
        dataset["rolling_correlation"],
        color=TERTIARY_COLOR,
        linewidth=1.8,
    )
    axes[1].set_title("Rolling Correlation")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Correlation")
    axes[1].grid(True, color=GRID_COLOR, linewidth=0.8)

    figure.autofmt_xdate()
    return _save_figure(figure, path)


def save_compare_summary_chart(
    frame: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Render a multi-run comparison chart across headline summary metrics."""
    dataset = _prepare_frame(
        frame,
        required_columns=[
            "run_id",
            "summary_cumulative_return",
            "summary_sharpe_ratio",
            "summary_mean_ic",
        ],
        source="compare summary chart input",
    )
    run_labels = _build_compare_run_labels(dataset)
    metrics = [
        ("summary_cumulative_return", "Summary Cumulative Return", PRIMARY_COLOR),
        ("summary_sharpe_ratio", "Summary Sharpe Ratio", TERTIARY_COLOR),
        ("summary_mean_ic", "Summary Mean IC", SECONDARY_COLOR),
    ]

    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for axis, (column, title, color) in zip(axes, metrics):
        axis.bar(run_labels, dataset[column], color=color, alpha=0.85)
        axis.axhline(0.0, color="#52606D", linewidth=1.0, linestyle="--")
        axis.set_title(title)
        axis.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8)
        axis.tick_params(axis="x", rotation=20)

    axes[-1].set_xlabel("Run")
    return _save_figure(figure, path)


def _prepare_frame(
    frame: pd.DataFrame,
    *,
    required_columns: list[str],
    source: str,
) -> pd.DataFrame:
    """Validate and parse a dated numeric frame used for charting."""
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise VisualizationError(f"{source} is missing required columns: {missing_text}.")

    dataset = frame.loc[:, required_columns].copy()
    if dataset.empty:
        raise VisualizationError(f"{source} must contain at least one row.")

    if "date" in dataset.columns:
        dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
        if dataset["date"].isna().any():
            raise VisualizationError(f"{source} contains invalid date values.")

    for column in required_columns:
        if column in {"date", "run_id", "command", "group_column", "group_value"}:
            continue
        dataset[column] = _parse_numeric_series(
            dataset[column],
            column_name=column,
            allow_na=True,
        )

    return dataset


def _with_group_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach stable compact labels for grouped diagnostics charts."""
    dataset = frame.copy()
    dataset["group_label"] = (
        dataset["group_column"].astype(str) + "=" + dataset["group_value"].astype(str)
    )
    return dataset.sort_values("group_label", kind="mergesort").reset_index(drop=True)


def _parse_numeric_series(
    series: pd.Series,
    *,
    column_name: str,
    allow_na: bool,
) -> pd.Series:
    """Parse one numeric series for charting."""
    parsed = pd.to_numeric(series, errors="coerce")
    invalid_mask = series.notna() & parsed.isna()
    if invalid_mask.any():
        raise VisualizationError(
            f"chart input contains invalid numeric values in '{column_name}'."
        )
    if not allow_na and parsed.isna().any():
        raise VisualizationError(
            f"chart input contains missing numeric values in '{column_name}'."
        )
    return parsed


def _parse_scalar(value: object, *, field_name: str) -> float:
    """Parse one numeric scalar used for summary charts."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise VisualizationError(f"{field_name} must be numeric.") from exc
    if pd.isna(numeric_value):
        raise VisualizationError(f"{field_name} must be finite.")
    return numeric_value


def _build_compare_run_labels(frame: pd.DataFrame) -> list[str]:
    """Build concise x-axis labels for compare-run charts."""
    labels = []
    commands = frame["command"].astype(str).tolist() if "command" in frame.columns else []
    for index, run_id in enumerate(frame["run_id"].astype(str).tolist()):
        prefix = commands[index] if index < len(commands) else "run"
        labels.append(f"{prefix}\n{run_id[-8:]}")
    return labels


def _format_horizon_label(value: object) -> str:
    """Format a forward-return horizon for chart legends."""
    if pd.isna(value):
        return "Unknown Horizon"
    horizon = float(value)
    if horizon.is_integer():
        return f"{int(horizon)}d"
    return f"{horizon:g}d"


def _save_figure(figure: plt.Figure, path: str | Path) -> Path:
    """Persist one matplotlib figure to disk and close it."""
    output_path = Path(path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    except OSError as exc:
        raise VisualizationError(
            f"Failed to write chart output to {output_path}: {exc}"
        ) from exc
    finally:
        plt.close(figure)
    return output_path

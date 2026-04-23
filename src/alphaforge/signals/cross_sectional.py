"""Cross-sectional signal transforms applied within each market date."""

from __future__ import annotations

import pandas as pd

from alphaforge.data import validate_ohlcv

_NORMALIZATION_CHOICES = {"none", "rank", "zscore"}


def apply_cross_sectional_signal_transform(
    frame: pd.DataFrame,
    *,
    score_column: str,
    winsorize_quantile: float | None = None,
    normalization: str = "none",
) -> tuple[pd.DataFrame, str]:
    """Apply an optional per-date winsorization and normalization pipeline.

    The input signal is assumed to be known at the current row's close. Any
    cross-sectional transform is therefore applied within the same date only,
    after any upstream universe masking has already removed ineligible rows.
    """
    dataset = _prepare_signal_transform_input(
        frame,
        score_column=score_column,
        source="cross-sectional signal transform input",
    )
    normalization = _normalize_normalization_choice(normalization)
    final_column = score_column

    if winsorize_quantile is not None:
        quantile = _normalize_winsorize_quantile(winsorize_quantile)
        final_column = f"{final_column}_winsorized"
        dataset = _append_transformed_signal(
            dataset,
            score_column=score_column,
            output_column=final_column,
            transform=lambda scores: _winsorize_scores(
                scores,
                quantile=quantile,
            ),
        )

    if normalization == "zscore":
        output_column = f"{final_column}_zscore"
        dataset = _append_transformed_signal(
            dataset,
            score_column=final_column,
            output_column=output_column,
            transform=_zscore_scores,
        )
        final_column = output_column
    elif normalization == "rank":
        output_column = f"{final_column}_rank"
        dataset = _append_transformed_signal(
            dataset,
            score_column=final_column,
            output_column=output_column,
            transform=_rank_normalize_scores,
        )
        final_column = output_column

    return dataset, final_column


def winsorize_signal_by_date(
    frame: pd.DataFrame,
    *,
    score_column: str,
    quantile: float = 0.05,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Append a winsorized copy of a signal using within-date quantile caps."""
    dataset = _prepare_signal_transform_input(
        frame,
        score_column=score_column,
        source="winsorized signal input",
    )
    quantile = _normalize_winsorize_quantile(quantile)
    output_column = output_column or f"{score_column}_winsorized"
    return _append_transformed_signal(
        dataset,
        score_column=score_column,
        output_column=output_column,
        transform=lambda scores: _winsorize_scores(scores, quantile=quantile),
    )


def zscore_signal_by_date(
    frame: pd.DataFrame,
    *,
    score_column: str,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Append a within-date z-scored copy of a signal."""
    dataset = _prepare_signal_transform_input(
        frame,
        score_column=score_column,
        source="z-score signal input",
    )
    output_column = output_column or f"{score_column}_zscore"
    return _append_transformed_signal(
        dataset,
        score_column=score_column,
        output_column=output_column,
        transform=_zscore_scores,
    )


def rank_normalize_signal_by_date(
    frame: pd.DataFrame,
    *,
    score_column: str,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Append a within-date rank-normalized copy of a signal on a [0, 1] scale."""
    dataset = _prepare_signal_transform_input(
        frame,
        score_column=score_column,
        source="rank-normalized signal input",
    )
    output_column = output_column or f"{score_column}_rank"
    return _append_transformed_signal(
        dataset,
        score_column=score_column,
        output_column=output_column,
        transform=_rank_normalize_scores,
    )


def _append_transformed_signal(
    dataset: pd.DataFrame,
    *,
    score_column: str,
    output_column: str,
    transform: callable,
) -> pd.DataFrame:
    """Append a per-date transformed score column to an already validated panel."""
    if output_column != score_column and output_column in dataset.columns:
        raise ValueError(
            f"signal transform output column '{output_column}' already exists."
        )

    transformed = dataset.groupby("date", sort=False)[score_column].transform(transform)
    updated = dataset.copy()
    updated[output_column] = transformed.astype("float64")
    return updated


def _prepare_signal_transform_input(
    frame: pd.DataFrame,
    *,
    score_column: str,
    source: str,
) -> pd.DataFrame:
    """Validate a signal panel and parse the selected score column numerically."""
    if score_column not in frame.columns:
        raise ValueError(f"{source} is missing the score column '{score_column}'.")

    dataset = validate_ohlcv(frame, source=source).copy()
    parsed_scores = pd.to_numeric(dataset[score_column], errors="coerce")
    invalid_scores = dataset[score_column].notna() & parsed_scores.isna()
    if invalid_scores.any():
        raise ValueError(
            f"{source} contains invalid numeric values in '{score_column}'."
        )
    dataset[score_column] = parsed_scores.astype("float64")
    return dataset


def _winsorize_scores(scores: pd.Series, *, quantile: float) -> pd.Series:
    """Clip finite scores to symmetric within-date quantile bands."""
    if scores.dropna().empty or quantile == 0.0:
        return scores.astype("float64")

    lower_bound = scores.quantile(quantile)
    upper_bound = scores.quantile(1.0 - quantile)
    return scores.clip(lower=lower_bound, upper=upper_bound).astype("float64")


def _zscore_scores(scores: pd.Series) -> pd.Series:
    """Standardize finite scores within one date using population dispersion."""
    usable = scores.dropna()
    normalized = pd.Series(float("nan"), index=scores.index, dtype="float64")
    if len(usable) < 2:
        return normalized

    standard_deviation = float(usable.std(ddof=0))
    if standard_deviation <= 0.0 or pd.isna(standard_deviation):
        return normalized

    mean_value = float(usable.mean())
    normalized.loc[usable.index] = usable.sub(mean_value).div(standard_deviation)
    return normalized


def _rank_normalize_scores(scores: pd.Series) -> pd.Series:
    """Map finite scores to within-date average ranks on a [0, 1] scale."""
    usable = scores.dropna()
    normalized = pd.Series(float("nan"), index=scores.index, dtype="float64")
    if len(usable) < 2:
        return normalized

    ranks = usable.rank(method="average")
    normalized.loc[usable.index] = ranks.sub(1.0).div(len(usable) - 1)
    return normalized


def _normalize_winsorize_quantile(value: float) -> float:
    """Validate a symmetric winsorization quantile."""
    if isinstance(value, bool):
        raise ValueError("winsorize_quantile must be a float in [0.0, 0.5).")

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("winsorize_quantile must be a float in [0.0, 0.5).") from exc

    if pd.isna(numeric_value) or numeric_value < 0.0 or numeric_value >= 0.5:
        raise ValueError("winsorize_quantile must be a float in [0.0, 0.5).")
    return numeric_value


def _normalize_normalization_choice(value: str) -> str:
    """Validate supported within-date normalization modes."""
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(
            "normalization must be one of {'none', 'rank', 'zscore'}."
        )

    normalized = value.strip().lower()
    if normalized not in _NORMALIZATION_CHOICES:
        raise ValueError(
            "normalization must be one of {'none', 'rank', 'zscore'}."
        )
    return normalized

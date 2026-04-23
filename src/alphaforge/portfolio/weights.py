"""Portfolio weight construction from daily signal panels."""

from __future__ import annotations

import pandas as pd

from alphaforge.data import validate_ohlcv


class PortfolioConstructionError(ValueError):
    """Raised when portfolio inputs or construction settings are invalid."""


def build_long_only_weights(
    frame: pd.DataFrame,
    *,
    score_column: str,
    top_n: int,
    weighting: str = "equal",
    exposure: float = 1.0,
    max_position_weight: float | None = None,
    weight_column: str = "portfolio_weight",
) -> pd.DataFrame:
    """Build a long-only cross-sectional portfolio from daily signal scores."""
    top_n = _normalize_positive_int(top_n, parameter_name="top_n")
    exposure = _normalize_non_negative_float(exposure, parameter_name="exposure")
    max_position_weight = _normalize_optional_positive_float(
        max_position_weight,
        parameter_name="max_position_weight",
    )

    dataset = _prepare_portfolio_input(
        frame,
        score_column=score_column,
        source="long-only portfolio input",
    )
    dataset[weight_column] = 0.0

    for _, group in dataset.groupby("date", sort=False):
        eligible = group.dropna(subset=[score_column])
        if eligible.empty:
            continue

        selected = eligible.sort_values(
            [score_column, "symbol"],
            ascending=[False, True],
            kind="mergesort",
        ).head(top_n)

        side_strength = _compute_side_strength(
            selected[score_column],
            weighting=weighting,
            side="long",
        )
        raw_weights = side_strength.div(side_strength.sum()).mul(exposure)
        dataset.loc[selected.index, weight_column] = _apply_position_limit(
            raw_weights,
            total_exposure=exposure,
            max_position_weight=max_position_weight,
        )

    return dataset


def build_long_short_weights(
    frame: pd.DataFrame,
    *,
    score_column: str,
    top_n: int,
    bottom_n: int | None = None,
    weighting: str = "equal",
    long_exposure: float = 1.0,
    short_exposure: float = 1.0,
    max_position_weight: float | None = None,
    weight_column: str = "portfolio_weight",
) -> pd.DataFrame:
    """Build a long-short cross-sectional portfolio from daily signal scores."""
    top_n = _normalize_positive_int(top_n, parameter_name="top_n")
    bottom_n = _normalize_positive_int(
        bottom_n if bottom_n is not None else top_n,
        parameter_name="bottom_n",
    )
    long_exposure = _normalize_non_negative_float(
        long_exposure,
        parameter_name="long_exposure",
    )
    short_exposure = _normalize_non_negative_float(
        short_exposure,
        parameter_name="short_exposure",
    )
    max_position_weight = _normalize_optional_positive_float(
        max_position_weight,
        parameter_name="max_position_weight",
    )

    dataset = _prepare_portfolio_input(
        frame,
        score_column=score_column,
        source="long-short portfolio input",
    )
    dataset[weight_column] = 0.0

    for _, group in dataset.groupby("date", sort=False):
        eligible = group.dropna(subset=[score_column])
        if len(eligible) < top_n + bottom_n:
            continue

        long_selected = eligible.sort_values(
            [score_column, "symbol"],
            ascending=[False, True],
            kind="mergesort",
        ).head(top_n)
        remaining = eligible.drop(index=long_selected.index)
        short_selected = remaining.sort_values(
            [score_column, "symbol"],
            ascending=[True, True],
            kind="mergesort",
        ).head(bottom_n)

        if len(long_selected) < top_n or len(short_selected) < bottom_n:
            continue

        long_strength = _compute_side_strength(
            long_selected[score_column],
            weighting=weighting,
            side="long",
        )
        short_strength = _compute_side_strength(
            short_selected[score_column],
            weighting=weighting,
            side="short",
        )

        long_raw_weights = long_strength.div(long_strength.sum()).mul(long_exposure)
        dataset.loc[long_selected.index, weight_column] = _apply_position_limit(
            long_raw_weights,
            total_exposure=long_exposure,
            max_position_weight=max_position_weight,
        )
        short_raw_weights = short_strength.div(short_strength.sum()).mul(short_exposure)
        dataset.loc[short_selected.index, weight_column] = -_apply_position_limit(
            short_raw_weights,
            total_exposure=short_exposure,
            max_position_weight=max_position_weight,
        )

    return dataset


def _prepare_portfolio_input(
    frame: pd.DataFrame,
    *,
    score_column: str,
    source: str,
) -> pd.DataFrame:
    """Validate the OHLCV panel and the selected signal column."""
    if score_column not in frame.columns:
        raise PortfolioConstructionError(
            f"{source} is missing the score column '{score_column}'."
        )

    dataset = validate_ohlcv(frame, source=source).copy()
    parsed_scores = pd.to_numeric(dataset[score_column], errors="coerce")
    invalid_scores = dataset[score_column].notna() & parsed_scores.isna()
    if invalid_scores.any():
        raise PortfolioConstructionError(
            f"{source} contains invalid numeric values in '{score_column}'."
        )
    dataset[score_column] = parsed_scores
    return dataset


def _compute_side_strength(
    scores: pd.Series,
    *,
    weighting: str,
    side: str,
) -> pd.Series:
    """Convert selected scores into positive side-specific weight strengths."""
    if weighting == "equal":
        return pd.Series(1.0, index=scores.index)

    if weighting == "score":
        if side == "long":
            return scores.sub(scores.min()).add(1.0)
        if side == "short":
            return scores.max() - scores + 1.0

    raise PortfolioConstructionError(
        "weighting must be one of {'equal', 'score'}."
    )


def _apply_position_limit(
    weights: pd.Series,
    *,
    total_exposure: float,
    max_position_weight: float | None,
) -> pd.Series:
    """Apply a simple per-position cap while preserving as much exposure as possible."""
    if weights.empty:
        return weights.copy()
    if max_position_weight is None or total_exposure == 0.0:
        return weights

    capped = pd.Series(0.0, index=weights.index, dtype="float64")
    remaining_index = pd.Index(weights.index)
    strengths = weights.copy()
    remaining_exposure = total_exposure
    tolerance = 1e-12

    while len(remaining_index) > 0 and remaining_exposure > tolerance:
        remaining_strength = strengths.loc[remaining_index]
        strength_sum = remaining_strength.sum()
        if strength_sum <= 0.0:
            break

        proposed = remaining_strength.div(strength_sum).mul(remaining_exposure)
        capped_mask = proposed > max_position_weight + tolerance
        if not bool(capped_mask.any()):
            capped.loc[remaining_index] = proposed
            remaining_exposure = 0.0
            break

        capped_index = proposed.index[capped_mask]
        capped.loc[capped_index] = max_position_weight
        remaining_exposure -= max_position_weight * float(len(capped_index))
        remaining_index = remaining_index.difference(capped_index)

    return capped


def _normalize_positive_int(value: int, *, parameter_name: str) -> int:
    """Validate positive integer selection parameters."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PortfolioConstructionError(
            f"{parameter_name} must be a positive integer."
        )
    return value


def _normalize_non_negative_float(value: float, *, parameter_name: str) -> float:
    """Validate non-negative exposure targets."""
    if isinstance(value, bool):
        raise PortfolioConstructionError(
            f"{parameter_name} must be a non-negative float."
        )

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise PortfolioConstructionError(
            f"{parameter_name} must be a non-negative float."
        ) from exc

    if pd.isna(numeric_value) or numeric_value < 0.0:
        raise PortfolioConstructionError(
            f"{parameter_name} must be a non-negative float."
        )
    return numeric_value


def _normalize_optional_positive_float(
    value: float | None, *, parameter_name: str
) -> float | None:
    """Validate optional positive float parameters."""
    if value is None:
        return None
    numeric_value = _normalize_non_negative_float(value, parameter_name=parameter_name)
    if numeric_value <= 0.0:
        raise PortfolioConstructionError(
            f"{parameter_name} must be a positive float."
        )
    return numeric_value

"""Portfolio weight construction from daily signal panels."""

from __future__ import annotations

from collections.abc import Sequence
import math

import pandas as pd

from alphaforge.common.errors import AlphaForgeError
from alphaforge.common.validation import (
    normalize_choice_string as _common_choice_string,
    normalize_non_negative_float as _common_non_negative_float,
    normalize_non_empty_string as _common_non_empty_string,
    normalize_optional_finite_float as _common_optional_finite_float,
    normalize_optional_positive_float as _common_optional_positive_float,
    normalize_positive_int as _common_positive_int,
)
from alphaforge.data import validate_ohlcv


class PortfolioConstructionError(AlphaForgeError):
    """Raised when portfolio inputs or construction settings are invalid."""


FactorExposureBound = tuple[str, float | None, float | None]


def build_long_only_weights(
    frame: pd.DataFrame,
    *,
    score_column: str,
    top_n: int,
    weighting: str = "equal",
    exposure: float = 1.0,
    max_position_weight: float | None = None,
    position_cap_column: str | None = None,
    group_column: str | None = None,
    max_group_weight: float | None = None,
    factor_exposure_bounds: Sequence[FactorExposureBound] | None = None,
    weight_column: str = "portfolio_weight",
) -> pd.DataFrame:
    """Build a long-only cross-sectional portfolio from daily signal scores."""
    top_n = _normalize_positive_int(top_n, parameter_name="top_n")
    weighting = _normalize_weighting(weighting)
    exposure = _normalize_non_negative_float(exposure, parameter_name="exposure")
    max_position_weight = _normalize_optional_positive_float(
        max_position_weight,
        parameter_name="max_position_weight",
    )
    position_cap_column = _normalize_optional_column_name(
        position_cap_column,
        parameter_name="position_cap_column",
    )
    group_column = _normalize_group_constraint(
        group_column=group_column,
        max_group_weight=max_group_weight,
    )
    max_group_weight = _normalize_optional_positive_float(
        max_group_weight,
        parameter_name="max_group_weight",
    )
    factor_exposure_bounds = _normalize_factor_exposure_bounds(
        factor_exposure_bounds,
    )

    dataset = _prepare_portfolio_input(
        frame,
        score_column=score_column,
        position_cap_column=position_cap_column,
        factor_exposure_bounds=factor_exposure_bounds,
        source="long-only portfolio input",
    )
    _validate_group_column(dataset, group_column, source="long-only portfolio input")
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
        position_limited = _apply_position_limit(
            raw_weights,
            total_exposure=exposure,
            max_position_weight=max_position_weight,
            position_caps=(
                selected[position_cap_column]
                if position_cap_column is not None
                else None
            ),
        )
        dataset.loc[selected.index, weight_column] = _apply_group_limit(
            position_limited,
            group_labels=selected[group_column] if group_column is not None else None,
            max_group_weight=max_group_weight,
        )
        dataset.loc[selected.index, weight_column] = _apply_factor_exposure_bounds(
            dataset.loc[selected.index, weight_column],
            exposures=selected,
            factor_exposure_bounds=factor_exposure_bounds,
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
    position_cap_column: str | None = None,
    group_column: str | None = None,
    max_group_weight: float | None = None,
    factor_exposure_bounds: Sequence[FactorExposureBound] | None = None,
    weight_column: str = "portfolio_weight",
) -> pd.DataFrame:
    """Build a long-short cross-sectional portfolio from daily signal scores."""
    top_n = _normalize_positive_int(top_n, parameter_name="top_n")
    bottom_n = _normalize_positive_int(
        bottom_n if bottom_n is not None else top_n,
        parameter_name="bottom_n",
    )
    weighting = _normalize_weighting(weighting)
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
    position_cap_column = _normalize_optional_column_name(
        position_cap_column,
        parameter_name="position_cap_column",
    )
    group_column = _normalize_group_constraint(
        group_column=group_column,
        max_group_weight=max_group_weight,
    )
    max_group_weight = _normalize_optional_positive_float(
        max_group_weight,
        parameter_name="max_group_weight",
    )
    factor_exposure_bounds = _normalize_factor_exposure_bounds(
        factor_exposure_bounds,
    )

    dataset = _prepare_portfolio_input(
        frame,
        score_column=score_column,
        position_cap_column=position_cap_column,
        factor_exposure_bounds=factor_exposure_bounds,
        source="long-short portfolio input",
    )
    _validate_group_column(dataset, group_column, source="long-short portfolio input")
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
        long_position_limited = _apply_position_limit(
            long_raw_weights,
            total_exposure=long_exposure,
            max_position_weight=max_position_weight,
            position_caps=(
                long_selected[position_cap_column]
                if position_cap_column is not None
                else None
            ),
        )
        long_limited = _apply_group_limit(
            long_position_limited,
            group_labels=(
                long_selected[group_column] if group_column is not None else None
            ),
            max_group_weight=max_group_weight,
        )
        short_raw_weights = short_strength.div(short_strength.sum()).mul(short_exposure)
        short_position_limited = _apply_position_limit(
            short_raw_weights,
            total_exposure=short_exposure,
            max_position_weight=max_position_weight,
            position_caps=(
                short_selected[position_cap_column]
                if position_cap_column is not None
                else None
            ),
        )
        short_limited = -_apply_group_limit(
            short_position_limited,
            group_labels=(
                short_selected[group_column] if group_column is not None else None
            ),
            max_group_weight=max_group_weight,
        )
        side_limited = pd.concat([long_limited, short_limited])
        selected = pd.concat([long_selected, short_selected], axis=0)
        dataset.loc[side_limited.index, weight_column] = _apply_factor_exposure_bounds(
            side_limited,
            exposures=selected,
            factor_exposure_bounds=factor_exposure_bounds,
        )

    return dataset


def _prepare_portfolio_input(
    frame: pd.DataFrame,
    *,
    score_column: str,
    position_cap_column: str | None,
    factor_exposure_bounds: tuple[FactorExposureBound, ...],
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

    if position_cap_column is not None:
        if position_cap_column not in dataset.columns:
            raise PortfolioConstructionError(
                f"{source} is missing the position cap column "
                f"'{position_cap_column}'."
            )
        parsed_caps = pd.to_numeric(dataset[position_cap_column], errors="coerce")
        invalid_caps = dataset[position_cap_column].notna() & parsed_caps.isna()
        if invalid_caps.any():
            raise PortfolioConstructionError(
                f"{source} contains invalid numeric values in "
                f"'{position_cap_column}'."
            )
        if parsed_caps.lt(0.0).any():
            raise PortfolioConstructionError(
                f"{source} contains negative values in '{position_cap_column}'."
            )
        dataset[position_cap_column] = parsed_caps.fillna(0.0)
    for exposure_column, _, _ in factor_exposure_bounds:
        if exposure_column not in dataset.columns:
            raise PortfolioConstructionError(
                f"{source} is missing the factor exposure column "
                f"'{exposure_column}'."
            )
        parsed_exposures = pd.to_numeric(dataset[exposure_column], errors="coerce")
        invalid_exposures = dataset[exposure_column].notna() & parsed_exposures.isna()
        if invalid_exposures.any():
            raise PortfolioConstructionError(
                f"{source} contains invalid numeric values in "
                f"'{exposure_column}'."
            )
        finite_exposures = parsed_exposures.dropna().map(math.isfinite)
        if not finite_exposures.all():
            raise PortfolioConstructionError(
                f"{source} contains non-finite numeric values in "
                f"'{exposure_column}'."
            )
        dataset[exposure_column] = parsed_exposures
    return dataset


def _compute_side_strength(
    scores: pd.Series,
    *,
    weighting: str,
    side: str,
) -> pd.Series:
    """Convert selected scores into positive side-specific weight strengths."""
    weighting = _normalize_weighting(weighting)
    if weighting == "equal":
        return pd.Series(1.0, index=scores.index)

    if weighting == "score":
        if side == "long":
            return scores.sub(scores.min()).add(1.0)
        if side == "short":
            return scores.max() - scores + 1.0

    raise PortfolioConstructionError("side must be 'long' or 'short'.")


def _normalize_weighting(value: str) -> str:
    """Validate supported portfolio weighting modes."""
    return _common_choice_string(
        value,
        parameter_name="weighting",
        choices={"equal", "score"},
        error_factory=PortfolioConstructionError,
    )


def _apply_position_limit(
    weights: pd.Series,
    *,
    total_exposure: float,
    max_position_weight: float | None,
    position_caps: pd.Series | None = None,
) -> pd.Series:
    """Apply a simple per-position cap while preserving as much exposure as possible."""
    if weights.empty:
        return weights.copy()
    if max_position_weight is None and position_caps is None:
        return weights
    if total_exposure == 0.0:
        return weights

    if max_position_weight is None:
        cap_values = position_caps.reindex(weights.index).fillna(0.0).astype(float)
    else:
        cap_values = pd.Series(max_position_weight, index=weights.index, dtype="float64")
        if position_caps is not None:
            row_caps = position_caps.reindex(weights.index).fillna(0.0).astype(float)
            cap_values = pd.concat([cap_values, row_caps], axis=1).min(axis=1)

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
        remaining_caps = cap_values.loc[remaining_index]
        capped_mask = proposed > remaining_caps + tolerance
        if not bool(capped_mask.any()):
            capped.loc[remaining_index] = proposed
            remaining_exposure = 0.0
            break

        capped_index = proposed.index[capped_mask]
        capped.loc[capped_index] = cap_values.loc[capped_index]
        remaining_exposure -= float(cap_values.loc[capped_index].sum())
        remaining_index = remaining_index.difference(capped_index)

    return capped


def _normalize_optional_column_name(
    value: str | None,
    *,
    parameter_name: str,
) -> str | None:
    """Validate optional input column names."""
    if value is None:
        return None
    return _common_non_empty_string(
        value,
        parameter_name=parameter_name,
        error_factory=PortfolioConstructionError,
    )


def _normalize_factor_exposure_bounds(
    bounds: Sequence[FactorExposureBound] | None,
) -> tuple[FactorExposureBound, ...]:
    """Validate shrink-only factor exposure bounds."""
    if bounds is None:
        return ()
    if isinstance(bounds, str) or not isinstance(bounds, Sequence):
        raise PortfolioConstructionError(
            "factor_exposure_bounds must be a sequence of bounds."
        )

    normalized_bounds: list[FactorExposureBound] = []
    seen_columns: set[str] = set()
    for bound in bounds:
        if not isinstance(bound, tuple) or len(bound) != 3:
            raise PortfolioConstructionError(
                "factor_exposure_bounds entries must be "
                "(column, min_exposure, max_exposure) tuples."
            )
        exposure_column = _normalize_optional_column_name(
            bound[0],
            parameter_name="factor_exposure_bounds.column",
        )
        if exposure_column is None:
            raise PortfolioConstructionError(
                "factor_exposure_bounds.column must be a non-empty string."
            )
        if exposure_column in seen_columns:
            raise PortfolioConstructionError(
                "factor_exposure_bounds must not contain duplicate columns."
            )
        seen_columns.add(exposure_column)

        min_exposure = _normalize_optional_finite_float(
            bound[1],
            parameter_name="factor_exposure_bounds.min_exposure",
        )
        max_exposure = _normalize_optional_finite_float(
            bound[2],
            parameter_name="factor_exposure_bounds.max_exposure",
        )
        if min_exposure is None and max_exposure is None:
            raise PortfolioConstructionError(
                "factor_exposure_bounds entries must include a min or max exposure."
            )
        _validate_shrink_only_factor_bound(
            min_exposure=min_exposure,
            max_exposure=max_exposure,
        )
        normalized_bounds.append((exposure_column, min_exposure, max_exposure))

    return tuple(normalized_bounds)


def _apply_group_limit(
    weights: pd.Series,
    *,
    group_labels: pd.Series | None,
    max_group_weight: float | None,
) -> pd.Series:
    """Cap same-date exposure by an explicit group column without re-optimizing."""
    if weights.empty:
        return weights.copy()
    if group_labels is None or max_group_weight is None:
        return weights

    capped = weights.copy()
    labels = _normalize_group_labels(group_labels.reindex(weights.index))
    missing_group = labels.isna()
    capped.loc[missing_group] = 0.0

    tolerance = 1e-12
    grouped_labels = labels.loc[~missing_group]
    for _, label_values in grouped_labels.groupby(grouped_labels, sort=False):
        group_index = label_values.index
        group_exposure = capped.loc[group_index].sum()
        if group_exposure > max_group_weight + tolerance:
            capped.loc[group_index] = capped.loc[group_index].mul(
                max_group_weight / group_exposure
            )

    return capped


def _apply_factor_exposure_bounds(
    weights: pd.Series,
    *,
    exposures: pd.DataFrame,
    factor_exposure_bounds: tuple[FactorExposureBound, ...],
) -> pd.Series:
    """Apply shrink-only net exposure bounds using explicit numeric columns."""
    if weights.empty or not factor_exposure_bounds:
        return weights.copy()

    bounded = weights.copy()
    for exposure_column, _, _ in factor_exposure_bounds:
        missing_exposure = exposures[exposure_column].reindex(bounded.index).isna()
        bounded.loc[missing_exposure] = 0.0

    max_iterations = max(1, len(factor_exposure_bounds) * 4)
    for _ in range(max_iterations):
        if _factor_exposure_bounds_satisfied(
            bounded,
            exposures=exposures,
            factor_exposure_bounds=factor_exposure_bounds,
        ):
            return bounded

        previous = bounded.copy()
        for exposure_column, min_exposure, max_exposure in factor_exposure_bounds:
            exposure_values = exposures[exposure_column].reindex(bounded.index)
            contributions = bounded.mul(exposure_values).fillna(0.0)
            net_exposure = float(contributions.sum())

            if max_exposure is not None and net_exposure > max_exposure:
                excess = net_exposure - max_exposure
                positive_contributions = contributions.loc[contributions > 0.0]
                bounded = _shrink_contributing_weights(
                    bounded,
                    contributing_index=positive_contributions.index,
                    contribution_total=float(positive_contributions.sum()),
                    required_reduction=excess,
                )
                contributions = bounded.mul(exposure_values).fillna(0.0)
                net_exposure = float(contributions.sum())

            if min_exposure is not None and net_exposure < min_exposure:
                shortfall = min_exposure - net_exposure
                negative_contributions = contributions.loc[contributions < 0.0]
                bounded = _shrink_contributing_weights(
                    bounded,
                    contributing_index=negative_contributions.index,
                    contribution_total=float(-negative_contributions.sum()),
                    required_reduction=shortfall,
                )

        if bounded.equals(previous):
            break

    if _factor_exposure_bounds_satisfied(
        bounded,
        exposures=exposures,
        factor_exposure_bounds=factor_exposure_bounds,
    ):
        return bounded

    return pd.Series(0.0, index=weights.index, dtype="float64")


def _shrink_contributing_weights(
    weights: pd.Series,
    *,
    contributing_index: pd.Index,
    contribution_total: float,
    required_reduction: float,
) -> pd.Series:
    """Scale exposure-contributing weights toward zero by contribution share."""
    if contribution_total <= 0.0 or required_reduction <= 0.0:
        return weights

    reduction_fraction = min(1.0, required_reduction / contribution_total)
    adjusted = weights.copy()
    adjusted.loc[contributing_index] = adjusted.loc[contributing_index].mul(
        1.0 - reduction_fraction
    )
    return adjusted


def _factor_exposure_bounds_satisfied(
    weights: pd.Series,
    *,
    exposures: pd.DataFrame,
    factor_exposure_bounds: tuple[FactorExposureBound, ...],
) -> bool:
    """Check net exposure bounds with a small floating-point tolerance."""
    tolerance = 1e-12
    for exposure_column, min_exposure, max_exposure in factor_exposure_bounds:
        exposure_values = exposures[exposure_column].reindex(weights.index)
        net_exposure = float(weights.mul(exposure_values).fillna(0.0).sum())
        if max_exposure is not None and net_exposure > max_exposure + tolerance:
            return False
        if min_exposure is not None and net_exposure < min_exposure - tolerance:
            return False
    return True


def _normalize_group_constraint(
    *,
    group_column: str | None,
    max_group_weight: float | None,
) -> str | None:
    """Validate that group constraints are configured explicitly as a pair."""
    if group_column is None and max_group_weight is None:
        return None
    if group_column is None or max_group_weight is None:
        raise PortfolioConstructionError(
            "group_column and max_group_weight must be configured together."
        )
    return _common_non_empty_string(
        group_column,
        parameter_name="group_column",
        error_factory=PortfolioConstructionError,
    )


def _validate_group_column(
    dataset: pd.DataFrame,
    group_column: str | None,
    *,
    source: str,
) -> None:
    """Fail fast when a configured group constraint cannot find its label column."""
    if group_column is not None and group_column not in dataset.columns:
        raise PortfolioConstructionError(
            f"{source} is missing the group column '{group_column}'."
        )


def _normalize_group_labels(group_labels: pd.Series) -> pd.Series:
    """Treat missing or blank group labels as unclassified exposure."""
    labels = group_labels.copy()
    blank_labels = labels.astype("string").str.strip().eq("").fillna(False)
    return labels.mask(labels.isna() | blank_labels)


def _normalize_positive_int(value: int, *, parameter_name: str) -> int:
    """Validate positive integer selection parameters."""
    return _common_positive_int(
        value,
        parameter_name=parameter_name,
        error_factory=PortfolioConstructionError,
    )


def _normalize_non_negative_float(value: float, *, parameter_name: str) -> float:
    """Validate non-negative exposure targets."""
    return _common_non_negative_float(
        value,
        parameter_name=parameter_name,
        error_factory=PortfolioConstructionError,
    )


def _normalize_optional_positive_float(
    value: float | None, *, parameter_name: str
) -> float | None:
    """Validate optional positive float parameters."""
    return _common_optional_positive_float(
        value,
        parameter_name=parameter_name,
        error_factory=PortfolioConstructionError,
    )


def _normalize_optional_finite_float(
    value: float | None,
    *,
    parameter_name: str,
) -> float | None:
    """Validate optional finite float parameters."""
    return _common_optional_finite_float(
        value,
        parameter_name=parameter_name,
        error_factory=PortfolioConstructionError,
    )


def _validate_shrink_only_factor_bound(
    *,
    min_exposure: float | None,
    max_exposure: float | None,
) -> None:
    """Require bounds that can be satisfied by shrinking weights toward zero."""
    if (
        min_exposure is not None
        and max_exposure is not None
        and min_exposure > max_exposure
    ):
        raise PortfolioConstructionError(
            "factor_exposure_bounds min_exposure cannot exceed max_exposure."
        )
    if min_exposure is not None and min_exposure > 0.0:
        raise PortfolioConstructionError(
            "factor_exposure_bounds min_exposure must be less than or equal to 0."
        )
    if max_exposure is not None and max_exposure < 0.0:
        raise PortfolioConstructionError(
            "factor_exposure_bounds max_exposure must be greater than or equal to 0."
        )

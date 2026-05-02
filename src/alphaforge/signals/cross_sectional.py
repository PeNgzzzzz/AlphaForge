"""Cross-sectional signal transforms applied within each market date."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from alphaforge.common.validation import (
    normalize_finite_float as _common_finite_float,
    normalize_non_empty_string as _common_non_empty_string,
)
from alphaforge.data import validate_ohlcv

_NORMALIZATION_CHOICES = {"none", "rank", "robust_zscore", "zscore"}
_TRANSFORM_CHOICES = {
    "clip",
    "demean",
    "rank",
    "residualize",
    "robust_zscore",
    "winsorize",
    "zscore",
}
_MAD_TO_NORMAL_STD_SCALE = 1.4826
_SAME_DATE_TRANSFORM_TIMING = (
    "same-date cross-sectional transform; no history or future rows are used"
)


@dataclass(frozen=True)
class SignalTransformDefinition:
    """Metadata and applier for one reusable same-date signal transform."""

    name: str
    family: str
    description: str
    timing: str
    parameter_defaults: Mapping[str, Any]
    output_suffix: str

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return accepted parameter names in stable order."""
        return tuple(self.parameter_defaults.keys())

    def normalize_parameters(
        self,
        parameters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Merge explicit parameters with defaults and fail fast on invalid input."""
        normalized = dict(self.parameter_defaults)
        if parameters is not None:
            unknown_parameters = sorted(
                str(parameter_name)
                for parameter_name in parameters
                if parameter_name not in self.parameter_defaults
            )
            if unknown_parameters:
                unknown_text = ", ".join(unknown_parameters)
                raise ValueError(
                    f"signal transform '{self.name}' does not accept "
                    f"parameter(s): {unknown_text}."
                )
            normalized.update(parameters)

        if self.name == "winsorize":
            normalized["quantile"] = _normalize_winsorize_quantile(
                normalized["quantile"]
            )
        elif self.name == "clip":
            (
                normalized["lower_bound"],
                normalized["upper_bound"],
            ) = _normalize_clip_bounds(
                normalized["lower_bound"],
                normalized["upper_bound"],
            )
        elif self.name == "residualize":
            normalized["exposure_columns"] = _normalize_exposure_columns(
                normalized["exposure_columns"]
            )
        return normalized

    def output_column(self, score_column: str) -> str:
        """Render the default transformed column name."""
        _normalize_score_column_name(score_column)
        return f"{score_column}_{self.output_suffix}"

    def apply(
        self,
        frame: pd.DataFrame,
        *,
        score_column: str,
        parameters: Mapping[str, Any] | None = None,
        group_columns: Sequence[str] = ("date",),
        output_column: str | None = None,
    ) -> tuple[pd.DataFrame, str]:
        """Apply this transform and return the output column name."""
        normalized = self.normalize_parameters(parameters)
        normalized_group_columns = _normalize_group_columns(group_columns)
        dataset = _prepare_signal_transform_input(
            frame,
            score_column=score_column,
            source=f"{self.name} signal transform input",
        )
        output_column = output_column or self.output_column(score_column)

        if self.name == "winsorize":
            updated = _append_transformed_signal(
                dataset,
                score_column=score_column,
                output_column=output_column,
                group_columns=normalized_group_columns,
                transform=lambda scores: _winsorize_scores(
                    scores,
                    quantile=normalized["quantile"],
                ),
            )
        elif self.name == "clip":
            updated = _append_transformed_signal(
                dataset,
                score_column=score_column,
                output_column=output_column,
                group_columns=normalized_group_columns,
                transform=lambda scores: _clip_scores(
                    scores,
                    lower_bound=normalized["lower_bound"],
                    upper_bound=normalized["upper_bound"],
                ),
            )
        elif self.name == "demean":
            updated = _append_transformed_signal(
                dataset,
                score_column=score_column,
                output_column=output_column,
                group_columns=normalized_group_columns,
                transform=_demean_scores,
            )
        elif self.name == "residualize":
            updated = _append_residualized_signal(
                dataset,
                score_column=score_column,
                output_column=output_column,
                group_columns=normalized_group_columns,
                exposure_columns=normalized["exposure_columns"],
            )
        elif self.name == "zscore":
            updated = _append_transformed_signal(
                dataset,
                score_column=score_column,
                output_column=output_column,
                group_columns=normalized_group_columns,
                transform=_zscore_scores,
            )
        elif self.name == "robust_zscore":
            updated = _append_transformed_signal(
                dataset,
                score_column=score_column,
                output_column=output_column,
                group_columns=normalized_group_columns,
                transform=_robust_zscore_scores,
            )
        elif self.name == "rank":
            updated = _append_transformed_signal(
                dataset,
                score_column=score_column,
                output_column=output_column,
                group_columns=normalized_group_columns,
                transform=_rank_normalize_scores,
            )
        else:  # Defensive guard for manually constructed definitions.
            raise ValueError(_signal_transform_name_error_message())
        return updated, output_column

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-friendly transform metadata."""
        return {
            "name": self.name,
            "family": self.family,
            "description": self.description,
            "timing": self.timing,
            "parameter_defaults": dict(self.parameter_defaults),
            "output_suffix": self.output_suffix,
        }


_SIGNAL_TRANSFORM_DEFINITIONS = (
    SignalTransformDefinition(
        name="winsorize",
        family="cross_sectional",
        description="Clip finite same-date scores to symmetric quantile bands.",
        timing=_SAME_DATE_TRANSFORM_TIMING,
        parameter_defaults=MappingProxyType({"quantile": 0.05}),
        output_suffix="winsorized",
    ),
    SignalTransformDefinition(
        name="clip",
        family="cross_sectional",
        description="Clip finite same-date scores to explicit numeric bounds.",
        timing=_SAME_DATE_TRANSFORM_TIMING,
        parameter_defaults=MappingProxyType(
            {"lower_bound": None, "upper_bound": None}
        ),
        output_suffix="clipped",
    ),
    SignalTransformDefinition(
        name="demean",
        family="cross_sectional",
        description="Subtract finite same-date group means from scores.",
        timing=_SAME_DATE_TRANSFORM_TIMING,
        parameter_defaults=MappingProxyType({}),
        output_suffix="demeaned",
    ),
    SignalTransformDefinition(
        name="residualize",
        family="cross_sectional",
        description=(
            "Residualize finite same-date scores against numeric exposure columns."
        ),
        timing=_SAME_DATE_TRANSFORM_TIMING,
        parameter_defaults=MappingProxyType({"exposure_columns": ()}),
        output_suffix="residualized",
    ),
    SignalTransformDefinition(
        name="zscore",
        family="cross_sectional",
        description=(
            "Standardize finite same-date scores using population dispersion."
        ),
        timing=_SAME_DATE_TRANSFORM_TIMING,
        parameter_defaults=MappingProxyType({}),
        output_suffix="zscore",
    ),
    SignalTransformDefinition(
        name="robust_zscore",
        family="cross_sectional",
        description=(
            "Standardize finite same-date scores using median and scaled MAD."
        ),
        timing=_SAME_DATE_TRANSFORM_TIMING,
        parameter_defaults=MappingProxyType({}),
        output_suffix="robust_zscore",
    ),
    SignalTransformDefinition(
        name="rank",
        family="cross_sectional",
        description="Map finite same-date average ranks onto a [0, 1] scale.",
        timing=_SAME_DATE_TRANSFORM_TIMING,
        parameter_defaults=MappingProxyType({}),
        output_suffix="rank",
    ),
)
_SIGNAL_TRANSFORM_DEFINITIONS_BY_NAME = {
    definition.name: definition for definition in _SIGNAL_TRANSFORM_DEFINITIONS
}


def apply_cross_sectional_signal_transform(
    frame: pd.DataFrame,
    *,
    score_column: str,
    winsorize_quantile: float | None = None,
    clip_lower_bound: float | None = None,
    clip_upper_bound: float | None = None,
    residualize_columns: Sequence[str] = (),
    neutralize_group_column: str | None = None,
    normalization: str = "none",
    normalization_group_column: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Apply optional per-date clipping and normalization transforms.

    The input signal is assumed to be known at the current row's close. Any
    cross-sectional transform is therefore applied within the same date only,
    after any upstream universe masking has already removed ineligible rows.
    When ``neutralize_group_column`` is provided, the signal is de-meaned within
    each same-date group before optional normalization.
    When ``residualize_columns`` is provided, the signal is residualized within
    each date against those same-row numeric exposures after clipping and before
    de-meaning / normalization.
    When ``normalization_group_column`` is provided, only the normalization
    step is computed within each same-date group; clipping remains date-wide.
    """
    normalization = _normalize_normalization_choice(normalization)
    residualize_columns = _normalize_exposure_columns(
        residualize_columns,
        allow_empty=True,
    )
    neutralize_group_column = _normalize_optional_group_column_name(
        neutralize_group_column,
        field_name="neutralize_group_column",
    )
    group_column = _normalize_optional_group_column_name(
        normalization_group_column,
        field_name="normalization_group_column",
    )
    if group_column is not None and normalization == "none":
        raise ValueError(
            "normalization_group_column requires a non-'none' normalization."
        )

    pre_normalization_steps: list[tuple[str, Mapping[str, Any]]] = []

    if winsorize_quantile is not None:
        pre_normalization_steps.append(
            ("winsorize", {"quantile": winsorize_quantile})
        )

    if clip_lower_bound is not None or clip_upper_bound is not None:
        pre_normalization_steps.append(
            (
                "clip",
                {
                    "lower_bound": clip_lower_bound,
                    "upper_bound": clip_upper_bound,
                },
            )
        )

    transformed, final_column = apply_signal_transform_pipeline(
        frame,
        score_column=score_column,
        transforms=pre_normalization_steps,
    )
    if residualize_columns:
        definition = get_signal_transform_definition("residualize")
        transformed, final_column = definition.apply(
            transformed,
            score_column=final_column,
            parameters={"exposure_columns": residualize_columns},
        )
    if neutralize_group_column is not None:
        definition = get_signal_transform_definition("demean")
        transformed, final_column = definition.apply(
            transformed,
            score_column=final_column,
            group_columns=("date", neutralize_group_column),
        )
    if normalization == "none":
        return transformed, final_column

    definition = get_signal_transform_definition(normalization)
    return definition.apply(
        transformed,
        score_column=final_column,
        group_columns=_normalization_group_columns(group_column),
    )


def winsorize_signal_by_date(
    frame: pd.DataFrame,
    *,
    score_column: str,
    quantile: float = 0.05,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Append a winsorized copy of a signal using within-date quantile caps."""
    updated, _ = get_signal_transform_definition("winsorize").apply(
        frame,
        score_column=score_column,
        parameters={"quantile": quantile},
        output_column=output_column,
    )
    return updated


def clip_signal_by_date(
    frame: pd.DataFrame,
    *,
    score_column: str,
    lower_bound: float,
    upper_bound: float,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Append a same-date clipped copy of a signal using explicit bounds."""
    updated, _ = get_signal_transform_definition("clip").apply(
        frame,
        score_column=score_column,
        parameters={"lower_bound": lower_bound, "upper_bound": upper_bound},
        output_column=output_column,
    )
    return updated


def demean_signal_by_date(
    frame: pd.DataFrame,
    *,
    score_column: str,
    group_column: str | None = None,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Append a same-date de-meaned signal, optionally within same-date groups."""
    updated, _ = get_signal_transform_definition("demean").apply(
        frame,
        score_column=score_column,
        group_columns=_demean_group_columns(group_column),
        output_column=output_column,
    )
    return updated


def residualize_signal_by_date(
    frame: pd.DataFrame,
    *,
    score_column: str,
    exposure_columns: Sequence[str],
    output_column: str | None = None,
) -> pd.DataFrame:
    """Append same-date OLS residuals versus numeric exposure columns."""
    updated, _ = get_signal_transform_definition("residualize").apply(
        frame,
        score_column=score_column,
        parameters={"exposure_columns": exposure_columns},
        output_column=output_column,
    )
    return updated


def zscore_signal_by_date(
    frame: pd.DataFrame,
    *,
    score_column: str,
    group_column: str | None = None,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Append a within-date z-scored copy of a signal."""
    group_columns = _normalization_group_columns(group_column)
    updated, _ = get_signal_transform_definition("zscore").apply(
        frame,
        score_column=score_column,
        group_columns=group_columns,
        output_column=output_column,
    )
    return updated


def robust_zscore_signal_by_date(
    frame: pd.DataFrame,
    *,
    score_column: str,
    group_column: str | None = None,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Append a same-date robust z-scored copy using median and scaled MAD."""
    group_columns = _normalization_group_columns(group_column)
    updated, _ = get_signal_transform_definition("robust_zscore").apply(
        frame,
        score_column=score_column,
        group_columns=group_columns,
        output_column=output_column,
    )
    return updated


def rank_normalize_signal_by_date(
    frame: pd.DataFrame,
    *,
    score_column: str,
    group_column: str | None = None,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Append a within-date rank-normalized copy of a signal on a [0, 1] scale."""
    group_columns = _normalization_group_columns(group_column)
    updated, _ = get_signal_transform_definition("rank").apply(
        frame,
        score_column=score_column,
        group_columns=group_columns,
        output_column=output_column,
    )
    return updated


def apply_signal_transform_pipeline(
    frame: pd.DataFrame,
    *,
    score_column: str,
    transforms: Sequence[tuple[str, Mapping[str, Any] | None]],
) -> tuple[pd.DataFrame, str]:
    """Apply registered signal transforms sequentially within each date."""
    dataset = _prepare_signal_transform_input(
        frame,
        score_column=score_column,
        source="signal transform pipeline input",
    )
    final_column = score_column
    for transform_name, parameters in transforms:
        definition = get_signal_transform_definition(transform_name)
        dataset, final_column = definition.apply(
            dataset,
            score_column=final_column,
            parameters=parameters,
        )
    return dataset, final_column


def list_signal_transform_definitions() -> tuple[SignalTransformDefinition, ...]:
    """Return built-in signal transform definitions in stable order."""
    return _SIGNAL_TRANSFORM_DEFINITIONS


def get_signal_transform_definition(name: str) -> SignalTransformDefinition:
    """Return one signal transform definition by name."""
    normalized_name = _normalize_signal_transform_name(name)
    return _SIGNAL_TRANSFORM_DEFINITIONS_BY_NAME[normalized_name]


def _append_transformed_signal(
    dataset: pd.DataFrame,
    *,
    score_column: str,
    output_column: str,
    group_columns: Sequence[str],
    transform: Callable[[pd.Series], pd.Series],
) -> pd.DataFrame:
    """Append a grouped transformed score column to an already validated panel."""
    if output_column != score_column and output_column in dataset.columns:
        raise ValueError(
            f"signal transform output column '{output_column}' already exists."
        )
    for group_column in group_columns:
        if group_column not in dataset.columns:
            raise ValueError(
                f"signal transform group column '{group_column}' is missing."
            )

    transformed = dataset.groupby(
        list(group_columns),
        sort=False,
    )[score_column].transform(transform)
    updated = dataset.copy()
    updated[output_column] = transformed.astype("float64")
    return updated


def _append_residualized_signal(
    dataset: pd.DataFrame,
    *,
    score_column: str,
    output_column: str,
    group_columns: Sequence[str],
    exposure_columns: Sequence[str],
) -> pd.DataFrame:
    """Append per-group OLS residuals against same-row numeric exposures."""
    if output_column != score_column and output_column in dataset.columns:
        raise ValueError(
            f"signal transform output column '{output_column}' already exists."
        )
    for group_column in group_columns:
        if group_column not in dataset.columns:
            raise ValueError(
                f"signal transform group column '{group_column}' is missing."
            )

    normalized_exposure_columns = _normalize_exposure_columns(exposure_columns)
    if score_column in normalized_exposure_columns:
        raise ValueError("exposure_columns must not include the score column.")

    parsed = dataset.copy()
    for exposure_column in normalized_exposure_columns:
        if exposure_column not in parsed.columns:
            raise ValueError(
                f"residualize exposure column '{exposure_column}' is missing."
            )
        parsed_exposure = pd.to_numeric(parsed[exposure_column], errors="coerce")
        invalid_exposure = parsed[exposure_column].notna() & ~np.isfinite(
            parsed_exposure.to_numpy(dtype="float64")
        )
        if invalid_exposure.any():
            raise ValueError(
                "residualize exposure column "
                f"'{exposure_column}' contains invalid numeric values."
            )
        parsed[exposure_column] = parsed_exposure.astype("float64")

    residualized = pd.Series(float("nan"), index=parsed.index, dtype="float64")
    for _, group in parsed.groupby(list(group_columns), sort=False):
        residualized.loc[group.index] = _residualize_group_scores(
            group,
            score_column=score_column,
            exposure_columns=normalized_exposure_columns,
        )

    updated = dataset.copy()
    updated[output_column] = residualized.astype("float64")
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


def _normalize_signal_transform_name(name: str) -> str:
    """Normalize and validate a registered signal transform name."""
    if not isinstance(name, str) or name.strip() == "":
        raise ValueError(_signal_transform_name_error_message())
    normalized_name = name.strip().lower()
    if normalized_name not in _TRANSFORM_CHOICES:
        raise ValueError(_signal_transform_name_error_message())
    return normalized_name


def _signal_transform_name_error_message() -> str:
    """Build a deterministic transform-name validation error."""
    allowed = ", ".join(repr(name) for name in sorted(_TRANSFORM_CHOICES))
    return f"signal transform name must be one of {{{allowed}}}."


def _normalize_score_column_name(score_column: str) -> str:
    """Validate a score-column name before rendering derived columns."""
    if not isinstance(score_column, str) or score_column.strip() == "":
        raise ValueError("score_column must be a non-empty string.")
    return score_column


def _normalization_group_columns(group_column: str | None) -> tuple[str, ...]:
    """Return date-only or date-plus-group columns for normalization steps."""
    normalized_group_column = _normalize_optional_group_column_name(
        group_column,
        field_name="normalization_group_column",
    )
    if normalized_group_column is None:
        return ("date",)
    return ("date", normalized_group_column)


def _demean_group_columns(group_column: str | None) -> tuple[str, ...]:
    """Return date-only or date-plus-group columns for de-meaning steps."""
    normalized_group_column = _normalize_optional_group_column_name(
        group_column,
        field_name="neutralize_group_column",
    )
    if normalized_group_column is None:
        return ("date",)
    return ("date", normalized_group_column)


def _normalize_group_columns(group_columns: Sequence[str]) -> tuple[str, ...]:
    """Validate group-by columns used by registered signal transforms."""
    if isinstance(group_columns, str):
        raise ValueError("group_columns must be a non-empty sequence of strings.")
    normalized = tuple(
        _normalize_group_column_name(column, field_name="group_columns")
        for column in group_columns
    )
    if not normalized:
        raise ValueError("group_columns must be a non-empty sequence of strings.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("group_columns must not contain duplicates.")
    return normalized


def _normalize_exposure_columns(
    value: Sequence[str],
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    """Validate exposure columns used for same-date residualization."""
    if value is None or isinstance(value, str):
        raise ValueError("exposure_columns must be a non-empty sequence of strings.")
    try:
        normalized = tuple(
            _normalize_group_column_name(column, field_name="exposure_columns")
            for column in value
        )
    except TypeError as exc:
        raise ValueError(
            "exposure_columns must be a non-empty sequence of strings."
        ) from exc
    if not normalized:
        if allow_empty:
            return ()
        raise ValueError("exposure_columns must contain at least one column.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("exposure_columns must not contain duplicates.")
    return normalized


def _normalize_optional_group_column_name(
    value: str | None,
    *,
    field_name: str,
) -> str | None:
    """Normalize an optional signal-normalization group column name."""
    if value is None:
        return None
    return _normalize_group_column_name(value, field_name=field_name)


def _normalize_group_column_name(value: str, *, field_name: str) -> str:
    """Validate one group column name."""
    return _common_non_empty_string(value, parameter_name=field_name)


def _winsorize_scores(scores: pd.Series, *, quantile: float) -> pd.Series:
    """Clip finite scores to symmetric within-date quantile bands."""
    if scores.dropna().empty or quantile == 0.0:
        return scores.astype("float64")

    lower_bound = scores.quantile(quantile)
    upper_bound = scores.quantile(1.0 - quantile)
    return scores.clip(lower=lower_bound, upper=upper_bound).astype("float64")


def _clip_scores(
    scores: pd.Series,
    *,
    lower_bound: float,
    upper_bound: float,
) -> pd.Series:
    """Clip finite scores within one date to explicit numeric bounds."""
    return scores.clip(lower=lower_bound, upper=upper_bound).astype("float64")


def _demean_scores(scores: pd.Series) -> pd.Series:
    """Subtract the same-date group mean from finite scores."""
    usable = scores.dropna()
    demeaned = pd.Series(float("nan"), index=scores.index, dtype="float64")
    if len(usable) < 2:
        return demeaned

    mean_value = float(usable.mean())
    demeaned.loc[usable.index] = usable.sub(mean_value)
    return demeaned


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


def _robust_zscore_scores(scores: pd.Series) -> pd.Series:
    """Standardize finite scores within one date using median and scaled MAD."""
    usable = scores.dropna()
    normalized = pd.Series(float("nan"), index=scores.index, dtype="float64")
    if len(usable) < 2:
        return normalized

    median_value = float(usable.median())
    absolute_deviation = usable.sub(median_value).abs()
    mad_value = float(absolute_deviation.median())
    scaled_mad = mad_value * _MAD_TO_NORMAL_STD_SCALE
    if scaled_mad <= 0.0 or pd.isna(scaled_mad):
        return normalized

    normalized.loc[usable.index] = usable.sub(median_value).div(scaled_mad)
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


def _residualize_group_scores(
    group: pd.DataFrame,
    *,
    score_column: str,
    exposure_columns: Sequence[str],
) -> pd.Series:
    """Return OLS residuals for one date/group, preserving unusable rows as NaN."""
    residualized = pd.Series(float("nan"), index=group.index, dtype="float64")
    columns = (score_column, *exposure_columns)
    usable = group.loc[:, columns].dropna()
    if len(usable) < len(exposure_columns) + 2:
        return residualized

    scores = usable[score_column].to_numpy(dtype="float64")
    exposures = usable.loc[:, list(exposure_columns)].to_numpy(dtype="float64")
    finite_rows = np.isfinite(scores) & np.isfinite(exposures).all(axis=1)
    if not finite_rows.all():
        usable = usable.loc[finite_rows]
        scores = scores[finite_rows]
        exposures = exposures[finite_rows]
    if len(usable) < len(exposure_columns) + 2:
        return residualized

    design = np.column_stack((np.ones(len(usable)), exposures))
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return residualized

    coefficients, *_ = np.linalg.lstsq(design, scores, rcond=None)
    fitted = design @ coefficients
    residualized.loc[usable.index] = scores - fitted
    return residualized


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


def _normalize_clip_bounds(
    lower_bound: float,
    upper_bound: float,
) -> tuple[float, float]:
    """Validate explicit signal-clipping bounds."""
    lower_value = _normalize_clip_bound_value(
        lower_bound,
        field_name="clip_lower_bound",
    )
    upper_value = _normalize_clip_bound_value(
        upper_bound,
        field_name="clip_upper_bound",
    )
    if lower_value >= upper_value:
        raise ValueError("clip_lower_bound must be smaller than clip_upper_bound.")
    return lower_value, upper_value


def _normalize_clip_bound_value(value: float, *, field_name: str) -> float:
    """Validate one finite signal-clipping bound."""
    return _common_finite_float(value, parameter_name=field_name)


def _normalize_normalization_choice(value: str) -> str:
    """Validate supported within-date normalization modes."""
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(
            "normalization must be one of "
            "{'none', 'rank', 'robust_zscore', 'zscore'}."
        )

    normalized = value.strip().lower()
    if normalized not in _NORMALIZATION_CHOICES:
        raise ValueError(
            "normalization must be one of "
            "{'none', 'rank', 'robust_zscore', 'zscore'}."
        )
    return normalized

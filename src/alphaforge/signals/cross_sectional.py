"""Cross-sectional signal transforms applied within each market date."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import pandas as pd

from alphaforge.data import validate_ohlcv

_NORMALIZATION_CHOICES = {"none", "rank", "zscore"}
_TRANSFORM_CHOICES = {"rank", "winsorize", "zscore"}
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
        output_column: str | None = None,
    ) -> tuple[pd.DataFrame, str]:
        """Apply this transform and return the output column name."""
        normalized = self.normalize_parameters(parameters)
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
                transform=lambda scores: _winsorize_scores(
                    scores,
                    quantile=normalized["quantile"],
                ),
            )
        elif self.name == "zscore":
            updated = _append_transformed_signal(
                dataset,
                score_column=score_column,
                output_column=output_column,
                transform=_zscore_scores,
            )
        elif self.name == "rank":
            updated = _append_transformed_signal(
                dataset,
                score_column=score_column,
                output_column=output_column,
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
    normalization: str = "none",
) -> tuple[pd.DataFrame, str]:
    """Apply an optional per-date winsorization and normalization pipeline.

    The input signal is assumed to be known at the current row's close. Any
    cross-sectional transform is therefore applied within the same date only,
    after any upstream universe masking has already removed ineligible rows.
    """
    normalization = _normalize_normalization_choice(normalization)
    transform_steps: list[tuple[str, Mapping[str, Any]]] = []

    if winsorize_quantile is not None:
        transform_steps.append(
            ("winsorize", {"quantile": winsorize_quantile})
        )

    if normalization != "none":
        transform_steps.append((normalization, {}))

    return apply_signal_transform_pipeline(
        frame,
        score_column=score_column,
        transforms=transform_steps,
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


def zscore_signal_by_date(
    frame: pd.DataFrame,
    *,
    score_column: str,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Append a within-date z-scored copy of a signal."""
    updated, _ = get_signal_transform_definition("zscore").apply(
        frame,
        score_column=score_column,
        output_column=output_column,
    )
    return updated


def rank_normalize_signal_by_date(
    frame: pd.DataFrame,
    *,
    score_column: str,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Append a within-date rank-normalized copy of a signal on a [0, 1] scale."""
    updated, _ = get_signal_transform_definition("rank").apply(
        frame,
        score_column=score_column,
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
    transform: Callable[[pd.Series], pd.Series],
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

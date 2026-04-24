"""Reusable factor definitions for configured research signals."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import pandas as pd

from alphaforge.signals.price_signals import (
    add_mean_reversion_signal,
    add_momentum_signal,
    add_trend_signal,
)

SignalBuilder = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class FactorDefinition:
    """Metadata and builder for one reusable research factor."""

    name: str
    family: str
    description: str
    timing: str
    required_columns: tuple[str, ...]
    parameter_defaults: Mapping[str, int]
    output_column_template: str
    builder: SignalBuilder = field(repr=False, compare=False)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return accepted parameter names in stable order."""
        return tuple(self.parameter_defaults.keys())

    def normalize_parameters(
        self,
        parameters: Mapping[str, Any] | None = None,
    ) -> dict[str, int]:
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
                    f"factor '{self.name}' does not accept parameter(s): "
                    f"{unknown_text}."
                )
            for parameter_name, value in parameters.items():
                normalized[parameter_name] = _normalize_positive_int(
                    value,
                    parameter_name=parameter_name,
                )

        for parameter_name, value in normalized.items():
            normalized[parameter_name] = _normalize_positive_int(
                value,
                parameter_name=parameter_name,
            )
        if (
            self.name == "trend"
            and normalized["short_window"] >= normalized["long_window"]
        ):
            raise ValueError(
                "short_window must be smaller than long_window for factor 'trend'."
            )
        return normalized

    def output_column(self, parameters: Mapping[str, Any] | None = None) -> str:
        """Render the default output column for the supplied parameters."""
        normalized = self.normalize_parameters(parameters)
        return self.output_column_template.format(**normalized)

    def build_signal(
        self,
        frame: pd.DataFrame,
        *,
        parameters: Mapping[str, Any] | None = None,
        signal_column: str | None = None,
    ) -> tuple[pd.DataFrame, str]:
        """Build this factor signal and return the output column name."""
        normalized = self.normalize_parameters(parameters)
        output_column = signal_column or self.output_column_template.format(**normalized)
        return (
            self.builder(frame, signal_column=output_column, **normalized),
            output_column,
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-friendly definition metadata without the callable builder."""
        return {
            "name": self.name,
            "family": self.family,
            "description": self.description,
            "timing": self.timing,
            "required_columns": list(self.required_columns),
            "parameter_defaults": dict(self.parameter_defaults),
            "output_column_template": self.output_column_template,
        }


_CLOSE_ANCHORED_SIGNAL_TIMING = (
    "close-anchored research score; execution delay is applied downstream"
)

_FACTOR_DEFINITIONS = (
    FactorDefinition(
        name="momentum",
        family="price",
        description="Trailing close-to-close return over a configurable lookback.",
        timing=_CLOSE_ANCHORED_SIGNAL_TIMING,
        required_columns=("date", "symbol", "close"),
        parameter_defaults=MappingProxyType({"lookback": 1}),
        output_column_template="momentum_signal_{lookback}d",
        builder=add_momentum_signal,
    ),
    FactorDefinition(
        name="mean_reversion",
        family="price",
        description=(
            "Negative trailing close-to-close return over a configurable lookback."
        ),
        timing=_CLOSE_ANCHORED_SIGNAL_TIMING,
        required_columns=("date", "symbol", "close"),
        parameter_defaults=MappingProxyType({"lookback": 1}),
        output_column_template="mean_reversion_signal_{lookback}d",
        builder=add_mean_reversion_signal,
    ),
    FactorDefinition(
        name="trend",
        family="price",
        description="Short moving average divided by long moving average minus one.",
        timing=_CLOSE_ANCHORED_SIGNAL_TIMING,
        required_columns=("date", "symbol", "close"),
        parameter_defaults=MappingProxyType({"short_window": 20, "long_window": 60}),
        output_column_template="trend_signal_{short_window}_{long_window}d",
        builder=add_trend_signal,
    ),
)
_FACTOR_DEFINITIONS_BY_NAME = {
    definition.name: definition for definition in _FACTOR_DEFINITIONS
}


def list_factor_definitions() -> tuple[FactorDefinition, ...]:
    """Return all built-in factor definitions in stable registry order."""
    return _FACTOR_DEFINITIONS


def get_factor_definition(name: str) -> FactorDefinition:
    """Return one factor definition by name."""
    normalized_name = _normalize_factor_name(name)
    return _FACTOR_DEFINITIONS_BY_NAME[normalized_name]


def build_factor_signal(
    frame: pd.DataFrame,
    *,
    name: str,
    parameters: Mapping[str, Any] | None = None,
    signal_column: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Build a registered factor signal and return its output column name."""
    definition = get_factor_definition(name)
    return definition.build_signal(
        frame,
        parameters=parameters,
        signal_column=signal_column,
    )


def _normalize_factor_name(name: str) -> str:
    """Normalize and validate a factor name."""
    if not isinstance(name, str) or name.strip() == "":
        raise ValueError(_factor_name_error_message())
    normalized_name = name.strip().lower()
    if normalized_name not in _FACTOR_DEFINITIONS_BY_NAME:
        raise ValueError(_factor_name_error_message())
    return normalized_name


def _factor_name_error_message() -> str:
    """Build a deterministic factor-name validation error."""
    allowed = ", ".join(
        repr(name) for name in sorted(_FACTOR_DEFINITIONS_BY_NAME)
    )
    return f"factor name must be one of {{{allowed}}}."


def _normalize_positive_int(value: Any, *, parameter_name: str) -> int:
    """Validate positive integer factor parameters."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{parameter_name} must be a positive integer.")
    return value

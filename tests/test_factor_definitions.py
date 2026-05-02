"""Tests for reusable factor definitions."""

from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.signals import (
    build_factor_signal,
    get_factor_definition,
    list_factor_definitions,
)


def test_factor_definition_registry_exposes_existing_price_factors() -> None:
    """The registry should describe the currently supported built-in factors."""
    definitions = {
        definition.name: definition for definition in list_factor_definitions()
    }

    assert tuple(definitions) == ("momentum", "mean_reversion", "trend")
    momentum_metadata = definitions["momentum"].to_metadata()
    assert momentum_metadata["family"] == "price"
    assert momentum_metadata["required_columns"] == ["date", "symbol", "close"]
    assert momentum_metadata["parameter_defaults"] == {"lookback": 1}
    assert momentum_metadata["output_column_template"] == "momentum_signal_{lookback}d"
    assert "execution delay" in momentum_metadata["timing"]


def test_build_factor_signal_uses_registered_builder_and_default_column() -> None:
    """Registry-based signal construction should preserve existing calculations."""
    signaled, signal_column = build_factor_signal(
        _sample_frame(),
        name=" Momentum ",
        parameters={"lookback": 1},
    )

    assert signal_column == "momentum_signal_1d"
    assert pd.isna(signaled.loc[0, signal_column])
    assert signaled.loc[1, signal_column] == pytest.approx(0.10)
    assert signaled.loc[2, signal_column] == pytest.approx(0.10)


def test_factor_definition_lookup_normalizes_registered_names() -> None:
    """Factor name lookup should trim and lowercase before registry validation."""
    definition = get_factor_definition(" MEAN_REVERSION ")

    assert definition.name == "mean_reversion"


def test_build_factor_signal_supports_custom_output_column() -> None:
    """Definitions should allow callers to override the output column explicitly."""
    signaled, signal_column = build_factor_signal(
        _sample_frame(),
        name="mean_reversion",
        parameters={"lookback": 1},
        signal_column="custom_reversion_score",
    )

    assert signal_column == "custom_reversion_score"
    assert "custom_reversion_score" in signaled.columns
    assert "mean_reversion_signal_1d" not in signaled.columns
    assert signaled.loc[1, signal_column] == pytest.approx(-0.10)


def test_factor_definitions_fail_fast_on_invalid_names_or_parameters() -> None:
    """Definition lookup and parameter handling should reject unsupported input."""
    frame = _sample_frame()

    with pytest.raises(ValueError, match="factor name"):
        get_factor_definition("carry")

    with pytest.raises(ValueError, match="does not accept"):
        build_factor_signal(frame, name="momentum", parameters={"window": 2})

    with pytest.raises(ValueError, match="lookback"):
        build_factor_signal(frame, name="momentum", parameters={"lookback": 0})

    with pytest.raises(ValueError, match="short_window"):
        build_factor_signal(
            frame,
            name="trend",
            parameters={"short_window": 3, "long_window": 3},
        )


def test_factor_definition_renders_parameterized_output_column() -> None:
    """Output column rendering should use normalized factor parameters."""
    trend = get_factor_definition("trend")

    assert trend.parameter_names == ("short_window", "long_window")
    assert trend.output_column({"short_window": 2, "long_window": 5}) == (
        "trend_signal_2_5d"
    )


def _sample_frame() -> pd.DataFrame:
    """Build a minimal valid OHLCV frame for definition tests."""
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "open": [100.0, 110.0, 121.0],
            "high": [101.0, 111.0, 122.0],
            "low": [99.0, 109.0, 120.0],
            "close": [100.0, 110.0, 121.0],
            "volume": [10, 20, 30],
        }
    )

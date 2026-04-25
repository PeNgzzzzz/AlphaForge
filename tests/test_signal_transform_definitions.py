"""Tests for reusable signal transform definitions."""

from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.signals import (
    apply_signal_transform_pipeline,
    get_signal_transform_definition,
    list_signal_transform_definitions,
)


def test_signal_transform_registry_exposes_same_date_transforms() -> None:
    """The registry should describe supported same-date transform steps."""
    definitions = {
        definition.name: definition for definition in list_signal_transform_definitions()
    }

    assert tuple(definitions) == (
        "winsorize",
        "clip",
        "demean",
        "zscore",
        "robust_zscore",
        "rank",
    )
    winsorize_metadata = definitions["winsorize"].to_metadata()
    assert winsorize_metadata["family"] == "cross_sectional"
    assert winsorize_metadata["parameter_defaults"] == {"quantile": 0.05}
    assert winsorize_metadata["output_suffix"] == "winsorized"
    assert "same-date" in winsorize_metadata["timing"]
    clip_metadata = definitions["clip"].to_metadata()
    assert clip_metadata["parameter_defaults"] == {
        "lower_bound": None,
        "upper_bound": None,
    }
    assert clip_metadata["output_suffix"] == "clipped"
    demean_metadata = definitions["demean"].to_metadata()
    assert demean_metadata["parameter_defaults"] == {}
    assert demean_metadata["output_suffix"] == "demeaned"
    robust_metadata = definitions["robust_zscore"].to_metadata()
    assert robust_metadata["parameter_defaults"] == {}
    assert robust_metadata["output_suffix"] == "robust_zscore"


def test_signal_transform_pipeline_composes_registered_steps_in_order() -> None:
    """Registered transforms should compose without changing existing suffixes."""
    frame = _cross_section_frame(
        signal_values=[1.0, 2.0, 3.0, 100.0],
        symbols=["AAPL", "MSFT", "NVDA", "TSLA"],
        dates=["2024-01-02"],
    )

    transformed, signal_column = apply_signal_transform_pipeline(
        frame,
        score_column="raw_signal",
        transforms=(
            ("winsorize", {"quantile": 0.25}),
            ("zscore", {}),
        ),
    )

    assert signal_column == "raw_signal_winsorized_zscore"
    assert "raw_signal" in transformed.columns
    assert "raw_signal_winsorized" in transformed.columns
    assert transformed[signal_column].mean() == pytest.approx(0.0)
    assert transformed[signal_column].std(ddof=0) == pytest.approx(1.0)


def test_signal_transform_pipeline_supports_explicit_clipping() -> None:
    """Explicit clipping should compose between winsorization and normalization."""
    frame = _cross_section_frame(
        signal_values=[-5.0, -1.0, 1.0, 5.0],
        symbols=["AAPL", "MSFT", "NVDA", "TSLA"],
        dates=["2024-01-02"],
    )

    transformed, signal_column = apply_signal_transform_pipeline(
        frame,
        score_column="raw_signal",
        transforms=(
            ("clip", {"lower_bound": -2.0, "upper_bound": 2.0}),
            ("rank", {}),
        ),
    )

    assert signal_column == "raw_signal_clipped_rank"
    assert transformed["raw_signal_clipped"].tolist() == pytest.approx(
        [-2.0, -1.0, 1.0, 2.0]
    )
    assert transformed[signal_column].tolist() == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])


def test_signal_transform_pipeline_supports_robust_zscore() -> None:
    """Robust z-score should compose as a registered same-date transform."""
    frame = _cross_section_frame(
        signal_values=[0.0, 1.0, 2.0, 3.0],
        symbols=["AAPL", "MSFT", "NVDA", "TSLA"],
        dates=["2024-01-02"],
    )

    transformed, signal_column = apply_signal_transform_pipeline(
        frame,
        score_column="raw_signal",
        transforms=(("robust_zscore", {}),),
    )

    assert signal_column == "raw_signal_robust_zscore"
    assert transformed[signal_column].tolist() == pytest.approx(
        [-1.5 / 1.4826, -0.5 / 1.4826, 0.5 / 1.4826, 1.5 / 1.4826]
    )


def test_signal_transform_pipeline_supports_same_date_demeaning() -> None:
    """De-meaning should compose as a registered same-date transform."""
    frame = _cross_section_frame(
        signal_values=[1.0, 3.0, 10.0, 20.0],
        symbols=["AAPL", "MSFT", "NVDA", "TSLA"],
        dates=["2024-01-02"],
    )

    transformed, signal_column = apply_signal_transform_pipeline(
        frame,
        score_column="raw_signal",
        transforms=(("demean", {}),),
    )

    assert signal_column == "raw_signal_demeaned"
    assert transformed[signal_column].tolist() == pytest.approx(
        [-7.5, -5.5, 1.5, 11.5]
    )


def test_signal_transform_definition_supports_custom_output_column() -> None:
    """Individual transform definitions should allow explicit output naming."""
    frame = _cross_section_frame(
        signal_values=[10.0, 20.0, 20.0, 40.0],
        symbols=["AAPL", "MSFT", "NVDA", "TSLA"],
        dates=["2024-01-02"],
    )

    transformed, output_column = get_signal_transform_definition("rank").apply(
        frame,
        score_column="raw_signal",
        output_column="custom_rank_score",
    )

    assert output_column == "custom_rank_score"
    assert "raw_signal_rank" not in transformed.columns
    assert transformed[output_column].tolist() == pytest.approx([0.0, 0.5, 0.5, 1.0])


def test_signal_transform_definitions_fail_fast_on_invalid_input() -> None:
    """Transform definitions should reject unsupported names and parameters."""
    frame = _cross_section_frame(
        signal_values=[1.0, 2.0],
        symbols=["AAPL", "MSFT"],
        dates=["2024-01-02"],
    )

    with pytest.raises(ValueError, match="signal transform name"):
        get_signal_transform_definition("neutralize")

    with pytest.raises(ValueError, match="does not accept"):
        get_signal_transform_definition("rank").apply(
            frame,
            score_column="raw_signal",
            parameters={"quantile": 0.1},
        )

    with pytest.raises(ValueError, match="winsorize_quantile"):
        get_signal_transform_definition("winsorize").apply(
            frame,
            score_column="raw_signal",
            parameters={"quantile": 0.5},
        )

    with pytest.raises(ValueError, match="clip_lower_bound"):
        get_signal_transform_definition("clip").apply(
            frame,
            score_column="raw_signal",
            parameters={"lower_bound": 2.0, "upper_bound": 1.0},
        )

    with pytest.raises(ValueError, match="clip_upper_bound"):
        get_signal_transform_definition("clip").apply(
            frame,
            score_column="raw_signal",
            parameters={"lower_bound": -1.0},
        )


def _cross_section_frame(
    *,
    signal_values: list[float],
    symbols: list[str],
    dates: list[str],
) -> pd.DataFrame:
    """Build a valid OHLCV panel with an extra raw signal column."""
    rows: list[dict[str, object]] = []
    value_index = 0
    for date in dates:
        for symbol in symbols:
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000,
                    "raw_signal": signal_values[value_index],
                }
            )
            value_index += 1
    return pd.DataFrame(rows)

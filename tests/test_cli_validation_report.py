"""Tests for validate-data report assembly."""

from __future__ import annotations

from pathlib import Path

import pytest

from alphaforge.cli.validation_report import build_validate_data_text
from alphaforge.common import load_pipeline_config
from alphaforge.data import DataValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_build_validate_data_text_prints_market_sections() -> None:
    """Basic validation reports should include market data and quality summaries."""
    config = load_pipeline_config(PROJECT_ROOT / "configs" / "sample_pipeline.toml")

    report_text = build_validate_data_text(config)

    assert "Data Summary" in report_text
    assert "Data Quality Summary" in report_text
    assert "Symbols:" in report_text


def test_build_validate_data_text_prints_benchmark_and_universe_sections() -> None:
    """Configured benchmark and universe previews should remain part of validate-data."""
    config = load_pipeline_config(
        PROJECT_ROOT / "configs" / "stage4_flagship_example.toml"
    )

    report_text = build_validate_data_text(config)

    assert "Benchmark Configuration" in report_text
    assert "Benchmark Summary" in report_text
    assert "Universe Rules" in report_text
    assert "Universe Summary" in report_text


def test_build_validate_data_text_rejects_off_calendar_market_dates(
    tmp_path: Path,
) -> None:
    """Calendar validation should still fail loudly for market dates off calendar."""
    data_path = tmp_path / "market.csv"
    data_path.write_text(
        "\n".join(
            [
                "date,symbol,open,high,low,close,volume",
                "2024-01-02,AAPL,100,101,99,100,1000",
                "2024-01-03,AAPL,101,102,100,101,1100",
                "2024-01-04,AAPL,102,103,101,102,1200",
            ]
        ),
        encoding="utf-8",
    )
    calendar_path = tmp_path / "calendar.csv"
    calendar_path.write_text("date\n2024-01-02\n2024-01-03\n", encoding="utf-8")
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(
        "\n\n".join(
            [
                "[data]",
                'path = "market.csv"',
                "[calendar]",
                'path = "calendar.csv"',
            ]
        ),
        encoding="utf-8",
    )
    config = load_pipeline_config(config_path)

    with pytest.raises(DataValidationError, match="configured trading calendar"):
        build_validate_data_text(config)

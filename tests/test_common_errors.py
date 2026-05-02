"""Tests for the shared AlphaForge exception hierarchy."""

from __future__ import annotations

from alphaforge.analytics import (
    AnalyticsError,
    FactorDiagnosticsError,
    ParameterSweepError,
    VisualizationError,
    WalkForwardError,
)
from alphaforge.backtest import BacktestError
from alphaforge.cli.errors import WorkflowError
from alphaforge.common import AlphaForgeError, ConfigError
from alphaforge.data import DataValidationError
from alphaforge.features import FeatureCacheError
from alphaforge.portfolio import PortfolioConstructionError
from alphaforge.risk import RiskError


def test_alpha_forge_error_preserves_value_error_compatibility() -> None:
    """The shared base should not break existing ValueError-based callers."""
    assert issubclass(AlphaForgeError, ValueError)


def test_public_errors_share_alpha_forge_base() -> None:
    """Public package errors should support one AlphaForge-specific catch boundary."""
    error_types = (
        AnalyticsError,
        BacktestError,
        ConfigError,
        DataValidationError,
        FactorDiagnosticsError,
        FeatureCacheError,
        ParameterSweepError,
        PortfolioConstructionError,
        RiskError,
        VisualizationError,
        WalkForwardError,
        WorkflowError,
    )

    for error_type in error_types:
        assert issubclass(error_type, AlphaForgeError)
        assert issubclass(error_type, ValueError)

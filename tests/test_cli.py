"""Smoke tests for the AlphaForge package scaffold."""

from __future__ import annotations

from pathlib import Path

import pytest

import alphaforge
from alphaforge.cli.main import main


def test_package_version_is_exposed() -> None:
    """The package should expose a non-empty version string."""
    assert isinstance(alphaforge.__version__, str)
    assert alphaforge.__version__


def test_cli_help_exits_successfully(capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI help output should exit cleanly."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "AlphaForge" in captured.out


def test_cli_entrypoint_uses_focused_modules_instead_of_legacy_workflows() -> None:
    """The production CLI entrypoint should not depend on the legacy workflow surface."""
    source = Path("src/alphaforge/cli/main.py").read_text(encoding="utf-8")

    assert "alphaforge.cli.workflows" not in source

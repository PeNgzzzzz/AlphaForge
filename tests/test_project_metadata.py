"""Tests for project runtime and CI metadata."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_requires_python_310_plus_and_modern_dependencies() -> None:
    """Package metadata should reflect the supported runtime floor."""
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]

    assert project["requires-python"] == ">=3.10"
    assert "matplotlib>=3.10.9" in project["dependencies"]
    assert "numpy>=2.2.6" in project["dependencies"]
    assert "pandas>=2.3.3" in project["dependencies"]
    assert "pyarrow>=24.0.0" in project["dependencies"]
    assert "tomli>=2.4.1; python_version < '3.11'" in project["dependencies"]
    assert project["optional-dependencies"]["dev"] == ["pytest>=9,<10"]


def test_ci_matrix_covers_supported_python_versions() -> None:
    """CI should test every supported minor version from 3.10 through 3.14."""
    workflow_text = (
        PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
    ).read_text()

    assert '- "3.9"' not in workflow_text
    for version in ["3.10", "3.11", "3.12", "3.13", "3.14"]:
        assert f'- "{version}"' in workflow_text


def test_local_python_version_targets_314() -> None:
    """Local version-manager metadata should target the newest CI runtime."""
    assert (PROJECT_ROOT / ".python-version").read_text().strip() == "3.14"

"""Materialized research feature cache helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import pandas as pd


FEATURE_CACHE_SCHEMA_VERSION = 1
FEATURE_CACHE_DATA_FILE = "features.parquet"
FEATURE_CACHE_MANIFEST_FILE = "manifest.json"
FEATURE_CACHE_KEY_COLUMNS = ("date", "symbol")


class FeatureCacheError(ValueError):
    """Raised when feature-cache metadata or files are invalid."""


def write_research_feature_cache(
    frame: pd.DataFrame,
    cache_dir: str | Path,
    *,
    cache_metadata: Mapping[str, Any],
) -> dict[str, Path]:
    """Write a PIT-safe materialized feature cache and manifest.

    Only key columns plus reusable feature/signal columns are written. Future
    return labels are tracked in the manifest but are never materialized.
    """
    manifest_inputs = _normalize_cache_metadata(cache_metadata)
    cached_columns = _cached_columns_from_metadata(manifest_inputs)
    output_columns = [*FEATURE_CACHE_KEY_COLUMNS, *cached_columns]
    _require_columns(frame, output_columns, source="feature cache input frame")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    data_path = cache_path / FEATURE_CACHE_DATA_FILE
    manifest_path = cache_path / FEATURE_CACHE_MANIFEST_FILE

    frame.loc[:, output_columns].to_parquet(data_path, index=False)
    manifest = {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "cache_key": manifest_inputs["cache_key"],
        "cache_key_algorithm": manifest_inputs["cache_key_algorithm"],
        "materialization": "parquet_feature_frame",
        "key_columns": list(FEATURE_CACHE_KEY_COLUMNS),
        "feature_columns": list(manifest_inputs["feature_columns"]),
        "signal_columns": dict(manifest_inputs["signal_columns"]),
        "cached_columns": list(cached_columns),
        "excluded_label_columns": list(manifest_inputs["label_columns"]),
        "row_count": int(len(frame)),
        "data_file": FEATURE_CACHE_DATA_FILE,
    }
    _write_json(manifest, manifest_path)
    return {"data_path": data_path, "manifest_path": manifest_path}


def load_research_feature_cache(
    cache_dir: str | Path,
    *,
    expected_cache_key: str | None = None,
) -> pd.DataFrame:
    """Load a materialized feature cache after validating its manifest."""
    cache_path = Path(cache_dir)
    manifest = load_research_feature_cache_manifest(cache_path)
    if expected_cache_key is not None and manifest["cache_key"] != expected_cache_key:
        raise FeatureCacheError(
            "feature cache key mismatch: "
            f"expected {expected_cache_key!r}, found {manifest['cache_key']!r}."
        )

    data_path = cache_path / manifest["data_file"]
    if not data_path.exists():
        raise FeatureCacheError(f"feature cache data file does not exist: {data_path}.")

    frame = pd.read_parquet(data_path)
    expected_columns = [*manifest["key_columns"], *manifest["cached_columns"]]
    _require_columns(frame, expected_columns, source="feature cache data file")
    if len(frame) != manifest["row_count"]:
        raise FeatureCacheError(
            "feature cache row count mismatch: "
            f"manifest has {manifest['row_count']}, data has {len(frame)}."
        )
    return frame.loc[:, expected_columns].copy()


def load_research_feature_cache_manifest(cache_dir: str | Path) -> dict[str, Any]:
    """Load and validate a materialized feature-cache manifest."""
    manifest_path = Path(cache_dir) / FEATURE_CACHE_MANIFEST_FILE
    if not manifest_path.exists():
        raise FeatureCacheError(f"feature cache manifest does not exist: {manifest_path}.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FeatureCacheError(f"feature cache manifest is invalid JSON: {exc}.") from exc
    if not isinstance(manifest, Mapping):
        raise FeatureCacheError("feature cache manifest must be a JSON object.")
    return _normalize_manifest(manifest)


def _normalize_cache_metadata(cache_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate metadata produced by build_research_feature_cache_metadata."""
    if not isinstance(cache_metadata, Mapping):
        raise FeatureCacheError("cache_metadata must be a mapping.")
    cache_key = _normalize_cache_key(cache_metadata.get("cache_key"))
    feature_columns = _normalize_string_sequence(
        cache_metadata.get("feature_columns"),
        field_name="feature_columns",
    )
    label_columns = _normalize_string_sequence(
        cache_metadata.get("label_columns"),
        field_name="label_columns",
    )
    overlap = sorted(set(feature_columns).intersection(label_columns))
    if overlap:
        raise FeatureCacheError(
            "feature cache metadata must not list label columns as reusable "
            f"features: {', '.join(overlap)}."
        )
    return {
        "cache_key": cache_key,
        "cache_key_algorithm": _normalize_non_empty_string(
            cache_metadata.get("cache_key_algorithm"),
            field_name="cache_key_algorithm",
        ),
        "feature_columns": feature_columns,
        "label_columns": label_columns,
        "signal_columns": _normalize_signal_columns(
            cache_metadata.get("signal_columns", {})
        ),
    }


def _normalize_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the on-disk materialized feature-cache manifest."""
    schema_version = manifest.get("schema_version")
    if schema_version != FEATURE_CACHE_SCHEMA_VERSION:
        raise FeatureCacheError(
            "feature cache schema_version must be "
            f"{FEATURE_CACHE_SCHEMA_VERSION}, got {schema_version!r}."
        )
    data_file = _normalize_non_empty_string(
        manifest.get("data_file"),
        field_name="data_file",
    )
    if Path(data_file).name != data_file:
        raise FeatureCacheError("feature cache data_file must be a local file name.")
    key_columns = _normalize_string_sequence(
        manifest.get("key_columns"),
        field_name="key_columns",
    )
    if tuple(key_columns) != FEATURE_CACHE_KEY_COLUMNS:
        raise FeatureCacheError(
            "feature cache key_columns must be "
            f"{list(FEATURE_CACHE_KEY_COLUMNS)!r}."
        )
    feature_columns = _normalize_string_sequence(
        manifest.get("feature_columns"),
        field_name="feature_columns",
    )
    label_columns = _normalize_string_sequence(
        manifest.get("excluded_label_columns"),
        field_name="excluded_label_columns",
    )
    cached_columns = _normalize_string_sequence(
        manifest.get("cached_columns"),
        field_name="cached_columns",
    )
    signal_columns = _normalize_signal_columns(manifest.get("signal_columns", {}))
    expected_cached_columns = _deduplicate(
        [*feature_columns, *signal_columns.values()]
    )
    if cached_columns != expected_cached_columns:
        raise FeatureCacheError(
            "feature cache cached_columns must match feature plus signal columns."
        )
    overlap = sorted(set(cached_columns).intersection(label_columns))
    if overlap:
        raise FeatureCacheError(
            "feature cache manifest includes excluded label columns in cached "
            f"columns: {', '.join(overlap)}."
        )
    row_count = manifest.get("row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
        raise FeatureCacheError("feature cache row_count must be a non-negative integer.")
    materialization = _normalize_non_empty_string(
        manifest.get("materialization"),
        field_name="materialization",
    )
    if materialization != "parquet_feature_frame":
        raise FeatureCacheError(
            "feature cache materialization must be 'parquet_feature_frame'."
        )
    return {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "cache_key": _normalize_cache_key(manifest.get("cache_key")),
        "cache_key_algorithm": _normalize_non_empty_string(
            manifest.get("cache_key_algorithm"),
            field_name="cache_key_algorithm",
        ),
        "materialization": materialization,
        "key_columns": list(key_columns),
        "feature_columns": list(feature_columns),
        "signal_columns": signal_columns,
        "cached_columns": list(cached_columns),
        "excluded_label_columns": list(label_columns),
        "row_count": row_count,
        "data_file": data_file,
    }


def _cached_columns_from_metadata(cache_metadata: Mapping[str, Any]) -> tuple[str, ...]:
    """Return reusable feature and signal columns, excluding labels."""
    return _deduplicate(
        [
            *cache_metadata["feature_columns"],
            *cache_metadata["signal_columns"].values(),
        ]
    )


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, source: str) -> None:
    """Fail fast when required cache columns are missing."""
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise FeatureCacheError(
            f"{source} is missing required columns: {', '.join(missing_columns)}."
        )


def _normalize_cache_key(value: object) -> str:
    """Validate the SHA-256 cache key format."""
    if not isinstance(value, str) or len(value) != 64:
        raise FeatureCacheError("feature cache cache_key must be a 64-character string.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise FeatureCacheError("feature cache cache_key must be hexadecimal.") from exc
    return value


def _normalize_signal_columns(value: object) -> dict[str, str]:
    """Normalize optional raw/final signal-column metadata."""
    if not isinstance(value, Mapping):
        raise FeatureCacheError("signal_columns must be a mapping.")
    normalized: dict[str, str] = {}
    for key, column in value.items():
        normalized[str(key)] = _normalize_non_empty_string(
            column,
            field_name=f"signal_columns.{key}",
        )
    return normalized


def _normalize_string_sequence(value: object, *, field_name: str) -> tuple[str, ...]:
    """Normalize a unique sequence of non-empty strings."""
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise FeatureCacheError(f"{field_name} must be a sequence of strings.")
    normalized = tuple(
        _normalize_non_empty_string(item, field_name=field_name)
        for item in value
    )
    if len(set(normalized)) != len(normalized):
        raise FeatureCacheError(f"{field_name} must not contain duplicates.")
    return normalized


def _normalize_non_empty_string(value: object, *, field_name: str) -> str:
    """Normalize one required non-empty string."""
    if not isinstance(value, str) or value.strip() == "":
        raise FeatureCacheError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _deduplicate(values: Sequence[str]) -> tuple[str, ...]:
    """Deduplicate values while preserving order."""
    deduplicated: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        deduplicated.append(value)
        seen.add(value)
    return tuple(deduplicated)


def _write_json(data: Mapping[str, Any], path: Path) -> None:
    """Write a deterministic JSON document."""
    path.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

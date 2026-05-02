"""Tests for reference-data field and index selection validation."""

from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.data import DataValidationError
from alphaforge.features import build_research_dataset
from alphaforge.features.borrow_join import attach_borrow_availability_asof
from alphaforge.features.classifications_join import attach_classifications_asof
from alphaforge.features.membership_join import attach_memberships_asof
from alphaforge.features.trading_status_join import attach_trading_status_asof


def test_reference_field_selectors_strip_and_lowercase_requested_fields() -> None:
    """Reference field selectors should normalize whitespace and case."""
    dataset = _dataset_frame()

    classified = attach_classifications_asof(
        dataset,
        _classification_frame(),
        fields=(" Sector ",),
    )
    assert classified["classification_sector"].tolist() == ["Technology", "Technology"]
    assert "classification_industry" not in classified.columns

    borrow = attach_borrow_availability_asof(
        dataset,
        _borrow_frame(),
        fields=(" BORROW_FEE_BPS ",),
    )
    assert borrow["borrow_fee_bps"].tolist() == [125.0, 125.0]
    assert "borrow_is_borrowable" not in borrow.columns

    trading_status = attach_trading_status_asof(
        dataset,
        _trading_status_frame(),
        fields=(" STATUS_REASON ",),
    )
    assert trading_status["trading_status_reason"].tolist() == ["regular", "regular"]
    assert "trading_is_tradable" not in trading_status.columns


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        (("sector", " Sector "), "classification_fields.*duplicate fields"),
        (("is_borrowable", " IS_BORROWABLE "), "borrow_fields.*duplicate fields"),
        (("status_reason", " STATUS_REASON "), "trading_status_fields.*duplicate fields"),
    ],
)
def test_reference_field_selectors_reject_case_insensitive_duplicates(
    fields: tuple[str, str],
    match: str,
) -> None:
    """Duplicate checks should run after field-name case normalization."""
    dataset = _dataset_frame()

    with pytest.raises(ValueError, match=match):
        if fields[0].lower() == "sector":
            attach_classifications_asof(dataset, _classification_frame(), fields=fields)
        elif fields[0].lower() == "is_borrowable":
            attach_borrow_availability_asof(dataset, _borrow_frame(), fields=fields)
        else:
            attach_trading_status_asof(dataset, _trading_status_frame(), fields=fields)


@pytest.mark.parametrize(
    ("field_value", "match"),
    [
        ("sector", "classification_fields must be a sequence of strings"),
        ("is_borrowable", "borrow_fields must be a sequence of strings"),
        ("is_tradable", "trading_status_fields must be a sequence of strings"),
    ],
)
def test_reference_field_selectors_reject_scalar_strings(
    field_value: str,
    match: str,
) -> None:
    """Scalar strings should not be treated as iterable field selections."""
    dataset = _dataset_frame()

    with pytest.raises(ValueError, match=match):
        if field_value == "sector":
            attach_classifications_asof(
                dataset,
                _classification_frame(),
                fields=field_value,
            )
        elif field_value == "is_borrowable":
            attach_borrow_availability_asof(
                dataset,
                _borrow_frame(),
                fields=field_value,
            )
        else:
            attach_trading_status_asof(
                dataset,
                _trading_status_frame(),
                fields=field_value,
            )


def test_membership_index_selector_reuses_common_string_validation() -> None:
    """Membership index selection should share string sequence validation."""
    dataset = _dataset_frame()
    memberships = _membership_frame()

    attached = attach_memberships_asof(
        dataset,
        memberships,
        indexes=(" S&P 500 ",),
    )
    assert attached["membership_s_p_500"].tolist() == [True, True]

    with pytest.raises(ValueError, match="membership_indexes.*duplicate index names"):
        attach_memberships_asof(
            dataset,
            memberships,
            indexes=("S&P 500", " S&P 500 "),
        )

    with pytest.raises(ValueError, match="membership_indexes must be a sequence"):
        attach_memberships_asof(dataset, memberships, indexes="S&P 500")

    with pytest.raises(
        DataValidationError,
        match="missing configured index_name values: NASDAQ 100",
    ):
        attach_memberships_asof(dataset, memberships, indexes=(" NASDAQ 100 ",))


def test_research_dataset_membership_options_reject_scalar_strings() -> None:
    """Dataset-level membership options should reject scalar string selections."""
    with pytest.raises(ValueError, match="membership_indexes must be a sequence"):
        build_research_dataset(
            _ohlcv_frame(),
            memberships=_membership_frame(),
            membership_indexes="S&P 500",
        )


def _dataset_frame() -> pd.DataFrame:
    """Return a minimal sorted research panel for reference joins."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": ["AAPL", "AAPL"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [1000, 1100],
        }
    )


def _ohlcv_frame() -> pd.DataFrame:
    """Return raw OHLCV input for dataset-builder validation coverage."""
    frame = _dataset_frame()
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame


def _classification_frame() -> pd.DataFrame:
    """Return one effective classification row."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "effective_date": ["2024-01-02"],
            "sector": ["Technology"],
            "industry": ["Software"],
        }
    )


def _borrow_frame() -> pd.DataFrame:
    """Return one effective borrow availability row."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "effective_date": ["2024-01-02"],
            "is_borrowable": [True],
            "borrow_fee_bps": [125.0],
        }
    )


def _membership_frame() -> pd.DataFrame:
    """Return one effective membership row."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "effective_date": ["2024-01-02"],
            "index_name": ["S&P 500"],
            "is_member": [True],
        }
    )


def _trading_status_frame() -> pd.DataFrame:
    """Return one effective trading-status row."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "effective_date": ["2024-01-02"],
            "is_tradable": [True],
            "status_reason": ["regular"],
        }
    )

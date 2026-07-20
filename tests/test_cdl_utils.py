"""
test_cdl_utils.py
-----------------
Offline tests for the CDL classification threshold logic
(src.cdl_utils.classify_cdl_histogram) — one mock histogram per label branch,
evaluated in spec order:

    1. Corn >= 70%                      -> "Corn"
    2. Soybeans >= 70%                  -> "Soybeans"
    3. Dominant Corn/Soybeans 50-69%    -> "Predominantly {crop} ({pct}%)"
    4. Non-ag share > 10%               -> "Mixed / boundary review flagged"
    5. Else                             -> "Mixed rotation (p1% c1 / p2% c2)"

Plus the missing-year guard on get_cdl_classification (year > CDL_LATEST_YEAR
short-circuits before any GEE call, so it is testable without credentials).

Run:
    python -m pytest tests/test_cdl_utils.py -v
"""

import pytest

from src.cdl_utils import (
    CDL_CLASS_MAP,
    CDL_LATEST_YEAR,
    NON_AG_CODES,
    classify_cdl_histogram,
    get_cdl_classification,
)


# GEE returns string keys and float counts — fixtures mirror that shape.

def test_corn_branch():
    # 80% corn, 20% soybeans
    hist = {"1": 800.0, "5": 200.0}
    r = classify_cdl_histogram(hist, 2025)
    assert r["label"] == "Corn"
    assert r["dominant_class"] == "Corn"
    assert r["dominant_pct"] == 80.0
    assert r["top_two"] == [("Corn", 80.0), ("Soybeans", 20.0)]
    assert r["non_ag_pct"] == 0.0
    assert r["boundary_warning"] is False
    assert r["year"] == 2025


def test_soybeans_branch():
    # 75% soybeans, 25% corn
    hist = {"5": 750.0, "1": 250.0}
    r = classify_cdl_histogram(hist, 2024)
    assert r["label"] == "Soybeans"
    assert r["dominant_class"] == "Soybeans"
    assert r["dominant_pct"] == 75.0


def test_predominantly_branch():
    # 58% corn, 42% soybeans — dominant corn in the 50-69% band
    hist = {"1": 580.0, "5": 420.0}
    r = classify_cdl_histogram(hist, 2023)
    assert r["label"] == "Predominantly Corn (58%)"
    assert r["dominant_class"] == "Corn"
    assert r["dominant_pct"] == 58.0


def test_boundary_flag_branch():
    # 45% corn, 40% soybeans, 15% grassland (non-ag > 10%), no class >= 50%
    hist = {"1": 450.0, "5": 400.0, "176": 150.0}
    r = classify_cdl_histogram(hist, 2025)
    assert r["label"] == "Mixed / boundary review flagged"
    assert r["non_ag_pct"] == 15.0
    assert r["boundary_warning"] is True


def test_mixed_rotation_branch():
    # 45% corn / 30% soybeans / 25% oats — no branch above fires
    hist = {"1": 450.0, "5": 300.0, "28": 250.0}
    r = classify_cdl_histogram(hist, 2025)
    assert r["label"] == "Mixed rotation (45% Corn / 30% Soybeans)"
    assert r["top_two"] == [("Corn", 45.0), ("Soybeans", 30.0)]
    assert r["boundary_warning"] is False


def test_rule_order_predominant_beats_boundary_flag():
    # Spec order: rule 3 (dominant corn 50-69%) fires before rule 4 even
    # though non-ag exceeds 10%; the warning flag is still set.
    hist = {"1": 600.0, "176": 150.0, "5": 250.0}
    r = classify_cdl_histogram(hist, 2025)
    assert r["label"] == "Predominantly Corn (60%)"
    assert r["boundary_warning"] is True


def test_unknown_code_named_generically():
    # 100% of a code outside CDL_CLASS_MAP falls to the single-class
    # mixed-rotation form with a generic "CDL {code}" name.
    hist = {"37": 100.0}
    r = classify_cdl_histogram(hist, 2025)
    assert r["dominant_class"] == "CDL 37"
    assert r["label"] == "Mixed rotation (100% CDL 37)"


def test_empty_histogram():
    r = classify_cdl_histogram({}, 2025)
    assert r["label"] == "No CDL data"
    assert r["dominant_class"] is None
    assert r["dominant_pct"] is None
    assert r["boundary_warning"] is False


def test_missing_year_not_yet_published():
    # Year beyond the latest published CDL short-circuits before any GEE
    # call — geometry argument is never touched.
    r = get_cdl_classification(None, CDL_LATEST_YEAR + 1)
    assert r["label"] == "Not yet published"
    assert r["year"] == CDL_LATEST_YEAR + 1
    assert r["dominant_class"] is None
    assert r["top_two"] == []


def test_class_map_covers_spec_codes():
    for code in (1, 5, 24, 28, 36, 61):
        assert code in CDL_CLASS_MAP
    assert NON_AG_CODES == {82, 111, 121, 122, 123, 124, 141, 142, 143, 152, 176}
    assert NON_AG_CODES <= set(CDL_CLASS_MAP)


# --- confident_previous_crop (Previous-crop autofill guard) -----------------

from src.cdl_utils import confident_previous_crop, _placeholder_result


def test_prev_crop_clean_label_fills():
    rows = [
        classify_cdl_histogram({"1": 950, "5": 50}, 2025),   # Corn 95%
        classify_cdl_histogram({"5": 800, "1": 200}, 2024),
    ]
    r = confident_previous_crop(rows)
    assert r == {"crop": "Corn", "year": 2025, "pct": 95.0}


def test_prev_crop_predominant_does_not_fill():
    rows = [classify_cdl_histogram({"1": 580, "5": 420}, 2025)]  # 58% — below 70
    assert confident_previous_crop(rows) is None


def test_prev_crop_unpublished_year_never_falls_back():
    # Most recent season unknown -> blank, even though 2024 is a clean Corn
    # year. An older year must never be substituted as "previous crop".
    rows = [
        _placeholder_result(2026, "Not yet published"),
        classify_cdl_histogram({"1": 900, "5": 100}, 2024),
    ]
    assert confident_previous_crop(rows) is None


def test_prev_crop_empty_rows():
    assert confident_previous_crop([]) is None

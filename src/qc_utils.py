"""
qc_utils.py
-----------
Single source of truth for the Tech Guide QC signals surfaced across the three
UI surfaces: the Streamlit app top box, the 45Z Verification Report, and the
CCA/producer PDF reports. Display/QC only — no scoring, no pipeline plumbing.

Signals (Tech Guide QC rules table):
  1. Valid pixel fraction, three-tier language (always shown).
  2. Single-scene composite notice (scene_count == 1; conditional).
  3. Saturation warning (mean NDVI > 0.75; conditional).
  4. Negative mean NDVI (< -0.05) — hard-blocked upstream via st.stop() in
     app.py, so the user never reaches a report. Not surfaced here; the
     constant is kept for reference only.

All phrasing lives here so the three surfaces cannot drift (per the QC ticket:
same computation logic and same threshold language on every surface).
"""

from typing import Any, Dict, Optional

import numpy as np

# Thresholds — from the Tech Guide QC rules table.
VALID_NDVI_FLOOR = 0.05    # a pixel counts as "valid" above this NDVI
QC_PASS_PCT      = 75.0    # >= this -> QC pass
QC_YELLOW_PCT    = 50.0    # [50, 75) -> yellow; < 50 -> red
SATURATION_NDVI  = 0.75    # mean NDVI above this -> saturation flag
NEGATIVE_MEAN_NDVI = -0.05  # mean NDVI below this -> hard error (gated upstream)

# Three-tier phrasing for Signal 1 (valid pixel fraction).
TIER_PASS_TEXT   = "QC pass"
TIER_YELLOW_TEXT = "Yellow — results may be cloud-contaminated"
TIER_RED_TEXT    = "Red — results flagged unreliable, widen date range"

# Conditional signal phrasing.
SINGLE_SCENE_TEXT = (
    "Single-scene composite — cloud contamination risk elevated; median "
    "compositing cannot filter cloud shadows from a single scene"
)
SATURATION_TEXT = (
    "Saturation warning — likely mature cash crop not cover crop; verify "
    "image date"
)


def valid_pixel_fraction(ndvi_array: np.ndarray) -> float:
    """Percent of boundary (non-NaN) pixels with NDVI above the valid floor.

    Canonical denominator = non-NaN field pixels, matching app.py's
    ``_valid_px_count = np.sum(~np.isnan(ndvi_array))``. Returns a float in
    [0, 100]; 0.0 when the field has no valid pixels.
    """
    nonnan = ndvi_array[~np.isnan(ndvi_array)]
    denom = nonnan.size
    if denom == 0:
        return 0.0
    return float(np.sum(nonnan > VALID_NDVI_FLOOR)) / denom * 100.0


def valid_tier(valid_pct: float) -> tuple:
    """Return ``(tier, phrase)`` for the three-tier valid-pixel rule.

    tier is one of ``'pass' | 'yellow' | 'red'``.
    """
    if valid_pct >= QC_PASS_PCT:
        return "pass", TIER_PASS_TEXT
    if valid_pct >= QC_YELLOW_PCT:
        return "yellow", TIER_YELLOW_TEXT
    return "red", TIER_RED_TEXT


def qc_signals(
    ndvi_array: np.ndarray,
    scene_count: Optional[int] = None,
    mean_ndvi: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute the surfaced QC signals for one field's NDVI array.

    Returns a dict:
      valid_pct     float 0-100
      tier          'pass' | 'yellow' | 'red'
      valid_phrase  the three-tier phrase (e.g. "QC pass")
      valid_text    "<pct>% valid pixels (NDVI > 0.05) — <phrase>"  (Signal 1)
      single_scene  Signal 2 text, or None when not triggered
      saturation    Signal 3 text, or None when not triggered

    ``mean_ndvi`` defaults to the non-NaN mean of ``ndvi_array``.
    """
    vpf = valid_pixel_fraction(ndvi_array)
    tier, phrase = valid_tier(vpf)

    if mean_ndvi is None:
        nonnan = ndvi_array[~np.isnan(ndvi_array)]
        mean_ndvi = float(np.mean(nonnan)) if nonnan.size else 0.0

    return {
        "valid_pct":    vpf,
        "tier":         tier,
        "valid_phrase": phrase,
        "valid_text": (
            f"{vpf:.0f}% valid pixels (NDVI > {VALID_NDVI_FLOOR:.2f}) — {phrase}"
        ),
        "single_scene": SINGLE_SCENE_TEXT if scene_count == 1 else None,
        "saturation":   SATURATION_TEXT if mean_ndvi > SATURATION_NDVI else None,
    }

"""
scoring.py
----------
Erosion concern scoring using RUSLE C-factor lookup table
calibrated to Iowa cover crop species and growth stages.

Replaces the original binary threshold approach with a
science-based score that NRCS advisors can defend in EQIP documentation.

RUSLE C-factor reference:
    Laflen & Roose (1998), Iowa NRCS Technical Note Agronomy-4,
    ISU Extension PM-1209 (Cover Crop Management in Iowa)

C-factor scale:
    0.0 = perfect cover (no erosion)
    1.0 = bare soil (maximum erosion)

NDVI-to-C-factor calibration:
    Breakpoints are calibrated to cereal rye biomass thresholds per the national
    cereal rye cover crop database (mean 3,428 kg/ha) and NRCS Practice Code 340
    minimum stand requirement of 1,500 kg/ha at approximately NDVI 0.25.
    NDVI 0.20 is the minimum detectable green cover threshold for Sentinel-2 10m
    resolution under Iowa early-spring cloud conditions.
    Source: Iowa RUSLE C-factor calibration to cereal rye biomass thresholds per
    national cereal rye database (mean 3,428 kg/ha) and NRCS Practice Code 340
    minimum of 1,500 kg/ha at approximately NDVI 0.25.
"""

from typing import Dict, Any
import numpy as np

# ---------------------------------------------------------------------------
# Default thresholds (kept for backward compatibility with sidebar sliders)
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {
    "ndvi_low":     0.20,
    "slope_steep":  9.0,
}

# ---------------------------------------------------------------------------
# Iowa cover crop RUSLE C-factor lookup table
# Keyed by NDVI range midpoint → C-factor value
# Source: NRCS Iowa Technical Note + ISU Extension PM-1209
# ---------------------------------------------------------------------------
IOWA_C_FACTOR_TABLE = {
    # (ndvi_min, ndvi_max): c_factor
    # Calibrated to cereal rye biomass per national database (mean 3,428 kg/ha)
    # and NRCS Practice Code 340 minimum (~1,500 kg/ha at NDVI ~0.25)
    (0.00, 0.15): 0.90,   # Failed stand — essentially bare soil
    (0.15, 0.20): 0.75,   # Inadequate — <1,000 kg/ha biomass
    (0.20, 0.35): 0.45,   # Marginal — 1,000–2,500 kg/ha, NRCS 340 borderline
    (0.35, 0.50): 0.20,   # Adequate — >2,500 kg/ha, meets NRCS minimum
    (0.50, 0.65): 0.08,   # Good stand
    (0.65, 1.00): 0.03,   # Excellent — near canopy saturation
}

# Slope-based LS-factor adjustment (simplified for field advisory use)
# Steeper slopes amplify erosion risk multiplicatively
LS_FACTOR_TABLE = {
    # (slope_pct_min, slope_pct_max): ls_factor
    (0,   2):  0.2,
    (2,   4):  0.5,
    (4,   6):  1.0,   # Baseline — NRCS typical concern threshold
    (6,   9):  1.8,
    (9,  12):  2.8,
    (12, 20):  4.5,
    (20, 100): 7.0,
}

# Concern level thresholds based on combined RUSLE C x LS score
CONCERN_THRESHOLDS = {
    "Low":      0.3,
    "Moderate": 0.7,
    "High":     1.5,
    "Critical": float("inf"),
}

# NRCS cover crop species C-factor targets for Iowa (for report context)
SPECIES_C_TARGETS = {
    "Cereal Rye":           0.10,
    "Winter Wheat":         0.12,
    "Radish/Turnip":        0.15,
    "Oats":                 0.18,
    "Crimson Clover":       0.20,
    "Legume Blend":         0.22,
    "Bare Soil (no cover)": 0.95,
}


def _lookup_c_factor(ndvi_mean: float) -> float:
    """Map mean NDVI to RUSLE C-factor using Iowa lookup table."""
    for (ndvi_min, ndvi_max), c_factor in IOWA_C_FACTOR_TABLE.items():
        if ndvi_min <= ndvi_mean < ndvi_max:
            return c_factor
    return 0.95  # fallback — treat as bare soil if out of range


def _lookup_ls_factor(slope_mean: float) -> float:
    """Map mean slope % to simplified LS-factor."""
    for (slope_min, slope_max), ls in LS_FACTOR_TABLE.items():
        if slope_min <= slope_mean < slope_max:
            return ls
    return 7.0  # fallback — steepest category


def _concern_level(rusle_score: float) -> str:
    """Map combined RUSLE score to concern level label."""
    for level, threshold in CONCERN_THRESHOLDS.items():
        if rusle_score < threshold:
            return level
    return "Critical"


def score_erosion_concern(
    ndvi_mean: float,
    slope_mean: float,
    ndvi_threshold: float = DEFAULT_THRESHOLDS["ndvi_low"],
    slope_threshold: float = DEFAULT_THRESHOLDS["slope_steep"],
) -> Dict[str, Any]:
    """
    Score field-level erosion concern using RUSLE C x LS proxy.

    Parameters
    ----------
    ndvi_mean       : Mean NDVI across the clipped field
    slope_mean      : Mean slope (%) across the clipped field
    ndvi_threshold  : Legacy threshold (kept for sidebar slider compatibility)
    slope_threshold : Legacy threshold (kept for sidebar slider compatibility)

    Returns
    -------
    Dict with keys:
        concern_level   : "Low" | "Moderate" | "High" | "Critical"
        score           : int 1-4 (for color coding)
        c_factor        : RUSLE C-factor from Iowa lookup table
        ls_factor       : Simplified LS-factor from slope
        rusle_score     : C × LS combined score
        low_cover       : bool (legacy — NDVI below threshold)
        steep_slope     : bool (legacy — slope above threshold)
        ndvi_threshold  : float (echoed back for report)
        slope_threshold : float (echoed back for report)
        recommendation  : str — plain-English NRCS advisory text
    """
    c_factor   = _lookup_c_factor(ndvi_mean)
    ls_factor  = _lookup_ls_factor(slope_mean)
    rusle_score = c_factor * ls_factor

    concern    = _concern_level(rusle_score)
    score_int  = {"Low": 1, "Moderate": 2, "High": 3, "Critical": 4}.get(concern, 2)

    # Plain-English recommendations by concern level
    recommendations = {
        "Critical": (
            "Satellite imagery indicates low cover crop establishment "
            "on high-risk slope units. Stand density appears below effective "
            "erosion protection thresholds for this terrain based on NDVI analysis."
        ),
        "High": (
            "Cover crop establishment is marginal on identified slope areas. "
            "Erosion protection is likely insufficient on steeper terrain units "
            "based on NDVI analysis."
        ),
        "Moderate": (
            "Cover crop stand is variable across slope positions based "
            "on NDVI analysis. Erosion risk increases on steeper units during "
            "spring rainfall events."
        ),
        "Low": (
            "Cover crop establishment is adequate for current slope conditions. "
            "Canopy development appears sufficient to reduce erosion risk based on "
            "NDVI analysis."
        ),
    }

    return {
        "concern_level":   concern,
        "score":           score_int,
        "c_factor":        round(c_factor, 3),
        "ls_factor":       round(ls_factor, 2),
        "rusle_score":     round(rusle_score, 3),
        "low_cover":       ndvi_mean < ndvi_threshold,
        "steep_slope":     slope_mean > slope_threshold,
        "ndvi_threshold":  ndvi_threshold,
        "slope_threshold": slope_threshold,
        "recommendation":  recommendations.get(concern, ""),
    }


def pixel_level_concern(
    ndvi_array: np.ndarray,
    slope_array: np.ndarray,
) -> np.ndarray:
    """
    Apply RUSLE C×LS scoring at every pixel for map visualization.
    Returns a float array of rusle_score values (same shape as inputs).
    """
    rusle = np.full(ndvi_array.shape, np.nan)
    for (ndvi_min, ndvi_max), c in IOWA_C_FACTOR_TABLE.items():
        mask = (ndvi_array >= ndvi_min) & (ndvi_array < ndvi_max)
        for (s_min, s_max), ls in LS_FACTOR_TABLE.items():
            slope_mask = (slope_array >= s_min) & (slope_array < s_max)
            combined = mask & slope_mask
            rusle[combined] = c * ls
    return rusle

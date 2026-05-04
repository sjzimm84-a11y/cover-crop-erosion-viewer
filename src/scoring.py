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

Limitation — LS-factor:
    LS-factor now uses continuous RUSLE S-factor formula (McCool et al. 1987)
    with fixed 100m assumed slope length. Slope length from flow accumulation
    remains a planned Phase 2 improvement.
"""

from typing import Dict, Any, Optional
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

# ---------------------------------------------------------------------------
# Residue adjustment multipliers — applied to NDVI-derived C-factor
# to account for crop residue protection not captured by satellite imagery.
# Source: ISU Extension PM-1901, NRCS RUSLE2 Iowa State File guidance
# ---------------------------------------------------------------------------
RESIDUE_ADJUSTMENTS = {
    "No-till corn (high residue ~80% cover)":         0.30,
    "No-till soybeans (moderate residue — fragile)":  0.55,
    "Tillage — > 30% residue (conservation tillage)": 0.75,
    "Tillage — < 30% residue (conventional tillage)": 1.00,
    "Unknown — not recorded (conservative default)":  1.00,
}

RESIDUE_OPTIONS = list(RESIDUE_ADJUSTMENTS.keys())

# ---------------------------------------------------------------------------
# Iowa R-factor zones — annual erosivity index (MJ·mm/ha·hr·yr)
# Northwest Iowa counties use R=150; all remaining Iowa counties use R=175.
# Source: Iowa NRCS FOTG Section I USLE Erosion Prediction,
#         Figure 2 — Rainfall Factors (Updated September 2002)
# R=150: northwest Iowa (~34 counties)
# R=175: all remaining Iowa counties (default)
# Shelby County = R=175 (confirmed from FOTG map)
# ---------------------------------------------------------------------------
IOWA_R_FACTOR_150_COUNTIES = {
    "lyon", "osceola", "dickinson", "emmet", "kossuth",
    "winnebago", "worth", "mitchell", "howard", "winneshiek",
    "sioux", "obrien", "clay", "palo alto", "hancock",
    "cerro gordo", "floyd", "chickasaw", "plymouth", "cherokee",
    "buena vista", "pocahontas", "humboldt", "wright", "franklin",
    "butler", "woodbury", "ida", "sac", "calhoun", "webster",
    "hamilton", "monona", "crawford",
}

# ---------------------------------------------------------------------------
# Iowa soil loss tolerance (T-value) by dominant series
# T = tolerable annual soil loss in tons/acre/year (NRCS SSURGO default = 5)
# Source: USDA NRCS SSURGO; ISU Extension Iowa Soil Properties
# ---------------------------------------------------------------------------
IOWA_T_VALUES: Dict[str, int] = {
    "Monona":   5,
    "Ida":      4,
    "Judson":   5,
    "Burchard": 5,
    "Tama":     5,
    "Clarion":  5,
    "Nicollet": 5,
    "Webster":  5,
    "Canisteo": 5,
    "default":  5,
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


def get_iowa_r_factor(boundary_gdf) -> tuple:
    """
    Look up Iowa R-factor zone from field centroid using FCC Census Block API.
    Returns (r_factor: float, source_note: str).
    Northwest Iowa counties use R=150; all other Iowa counties default to
    R=175 per Iowa NRCS FOTG Section I USLE Figure 2 (September 2002).
    Falls back to R=175 if the API call fails.
    """
    import urllib.request
    import json as _json

    try:
        centroid = (boundary_gdf.to_crs("EPSG:4326")
                    .geometry.centroid.iloc[0])
        lat, lon = centroid.y, centroid.x
        url = (
            f"https://geo.fcc.gov/api/census/block/find"
            f"?latitude={lat:.6f}&longitude={lon:.6f}"
            f"&format=json"
        )
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = _json.loads(resp.read())
        county_raw  = data.get("County", {}).get("name", "")
        county_name = (county_raw.lower()
                       .replace(" county", "").strip())
        if county_name in IOWA_R_FACTOR_150_COUNTIES:
            return (
                150.0,
                f"R=150 (northwest Iowa — "
                f"{county_name.title()} County, NRCS FOTG)"
            )
        note = (
            f"R=175 (standard Iowa — "
            f"{county_name.title()} County, NRCS FOTG)"
            if county_name
            else "R=175 (standard Iowa zone, NRCS FOTG)"
        )
        return (175.0, note)
    except Exception:
        return (
            175.0,
            "R=175 (default — county lookup unavailable, NRCS FOTG)"
        )


def estimate_soil_loss(
    c_factor: float,
    ls_factor: float,
    k_factor: Any,
    t_value: int = 5,
    r_factor: float = 175.0,
) -> Dict[str, Any]:
    """
    Estimate annual soil loss using simplified RUSLE: A = R × K × LS × C × P.
    P-factor = 1.0 (no conservation practice factor applied).
    Returns dict with soil_loss_tons_ac_yr, t_value, ratio_to_t,
    conservation_status, and status_code.
    """
    try:
        k = float(k_factor)
    except (TypeError, ValueError):
        return {
            "soil_loss_tons_ac_yr": None,
            "t_value":              t_value,
            "ratio_to_t":          None,
            "conservation_status": "K-factor unavailable — soil loss not estimated",
            "status_code":         "unavailable",
        }

    soil_loss = r_factor * k * ls_factor * c_factor  # P = 1.0
    ratio     = soil_loss / t_value if t_value > 0 else None

    if ratio is None:
        status_code = "unavailable"
        status = "T-value unavailable"
    elif ratio <= 1.0:
        status_code = "within_t"
        status = (f"Within tolerable soil loss limit — "
                  f"{soil_loss:.1f} t/ac/yr \u2264 T={t_value}")
    elif ratio <= 2.0:
        status_code = "near_t"
        status = (f"Near tolerable limit — "
                  f"{soil_loss:.1f} t/ac/yr ({ratio:.1f}\u00d7 T={t_value})")
    elif ratio <= 5.0:
        status_code = "over_t"
        status = (f"Exceeds tolerable limit — "
                  f"{soil_loss:.1f} t/ac/yr ({ratio:.1f}\u00d7 T={t_value})")
    else:
        status_code = "critical_t"
        status = (f"Significantly exceeds tolerable limit — "
                  f"{soil_loss:.1f} t/ac/yr ({ratio:.1f}\u00d7 T={t_value})")

    return {
        "soil_loss_tons_ac_yr": round(soil_loss, 2),
        "t_value":              t_value,
        "ratio_to_t":          round(ratio, 2) if ratio is not None else None,
        "conservation_status": status,
        "status_code":         status_code,
    }


def _lookup_c_factor(ndvi_mean: float) -> float:
    """Map mean NDVI to RUSLE C-factor using Iowa lookup table."""
    for (ndvi_min, ndvi_max), c_factor in IOWA_C_FACTOR_TABLE.items():
        if ndvi_min <= ndvi_mean < ndvi_max:
            return c_factor
    return 0.95  # fallback — treat as bare soil if out of range


def _lookup_ls_factor(slope_mean: float) -> float:
    """Map mean slope % to LS-factor via continuous analytical formula."""
    return float(_analytical_ls_factor(slope_mean))


def _analytical_ls_factor(slope_pct):
    """Continuous RUSLE S-factor formula (McCool et al. 1987)
    with fixed 100m assumed slope length. Replaces 7-bin stepped lookup.
    Handles scalar and numpy array input."""
    slope_pct = np.asarray(slope_pct, dtype=float)
    theta = np.arctan(slope_pct / 100.0)
    S = np.where(
        slope_pct < 9,
        10.8 * np.sin(theta) + 0.03,
        16.8 * np.sin(theta) - 0.50,
    )
    S = np.maximum(S, 0.03)
    m = np.where(slope_pct < 1, 0.2,
        np.where(slope_pct < 3, 0.3,
        np.where(slope_pct < 5, 0.4, 0.5)))
    L = (100.0 / 22.13) ** m
    return L * S


def _concern_level(rusle_score: float) -> str:
    """Map combined RUSLE score to concern level label."""
    for level, threshold in CONCERN_THRESHOLDS.items():
        if rusle_score < threshold:
            return level
    return "Critical"


def pixel_risk_index(
    ndvi_array: np.ndarray,
    slope_array: np.ndarray,
    residue_multiplier: float = 1.0,
) -> np.ndarray:
    """
    Compute per-pixel RUSLE Risk Index (C x LS) for every pixel in the field.
    Returns array of same shape as inputs.
    C-factor derived from NDVI via Iowa lookup table, then scaled by
    residue_multiplier to match field-level C-factor adjustment.
    LS-factor derived from slope percent via simplified lookup.
    """
    c_array = np.full(ndvi_array.shape, np.nan, dtype=float)
    c_array = np.where(ndvi_array < 0.15,                                    0.90, c_array)
    c_array = np.where((ndvi_array >= 0.15) & (ndvi_array < 0.20),           0.75, c_array)
    c_array = np.where((ndvi_array >= 0.20) & (ndvi_array < 0.35),           0.45, c_array)
    c_array = np.where((ndvi_array >= 0.35) & (ndvi_array < 0.50),           0.20, c_array)
    c_array = np.where((ndvi_array >= 0.50) & (ndvi_array < 0.65),           0.08, c_array)
    c_array = np.where(ndvi_array >= 0.65,                                    0.03, c_array)
    c_array = c_array * residue_multiplier

    ls_array = _analytical_ls_factor(slope_array)

    risk_array = c_array * ls_array
    risk_array = np.where(
        np.isnan(ndvi_array) | np.isnan(slope_array), np.nan, risk_array
    )
    return risk_array


def classify_risk_zones(risk_array: np.ndarray) -> np.ndarray:
    """
    Classify pixel-level Risk Index into 4 concern zones.
    Returns float array: 1=Low, 2=Moderate, 3=High, 4=Critical (NaN where no data).
    Matches Concern Level labels used in field summary.
    """
    zones = np.full(risk_array.shape, np.nan, dtype=float)
    zones = np.where(risk_array < 0.3,                            1, zones)
    zones = np.where((risk_array >= 0.3) & (risk_array < 0.7),   2, zones)
    zones = np.where((risk_array >= 0.7) & (risk_array < 1.5),   3, zones)
    zones = np.where(risk_array >= 1.5,                           4, zones)
    zones = np.where(np.isnan(risk_array),                        np.nan, zones)
    return zones


def compute_ndvi_zone_summary(
    ndvi_array: np.ndarray,
    ndvi_threshold: float = 0.20,
) -> "pd.DataFrame":
    """Three-zone NDVI classification: Low cover / Marginal / Good cover."""
    import pandas as pd
    valid = ~np.isnan(ndvi_array)
    total = float(np.sum(valid))
    mid_upper = ndvi_threshold + 0.15
    zones_def = [
        ("Low cover",  (ndvi_array < ndvi_threshold) & valid),
        ("Marginal",   (ndvi_array >= ndvi_threshold) & (ndvi_array < mid_upper) & valid),
        ("Good cover", (ndvi_array >= mid_upper) & valid),
    ]
    rows = []
    for label, mask in zones_def:
        count = float(np.sum(mask))
        rows.append({
            "zone":      label,
            "percent":   count / total * 100 if total > 0 else 0.0,
            "ndvi_mean": float(np.nanmean(ndvi_array[mask])) if np.any(mask) else 0.0,
        })
    return pd.DataFrame(rows)


def score_erosion_concern(
    ndvi_stats: dict,
    slope_stats: dict,
    ndvi_threshold: float = DEFAULT_THRESHOLDS["ndvi_low"],
    slope_threshold: float = DEFAULT_THRESHOLDS["slope_steep"],
    residue_system: str = "Unknown — not recorded (conservative default)",
    ndvi_array: np.ndarray = None,
    slope_array: np.ndarray = None,
    k_factor: Any = None,
    soil_series: str = "default",
    r_factor: float = 175.0,
) -> Dict[str, Any]:
    """
    Score field-level erosion concern using RUSLE C x LS proxy.

    Accepts ndvi_stats and slope_stats dicts (as returned by raster_stats()).
    When ndvi_array and slope_array are provided, concern_level is derived
    from the distribution of pixel-level Risk Index scores.
    residue_system applies a research-based multiplier to the NDVI-derived
    C-factor to account for crop residue not captured by satellite imagery.
    When k_factor is provided, estimate_soil_loss() is called and the result
    is included in the return dict under the "soil_loss" key.

    Returns
    -------
    Dict with keys:
        concern_level        : "Low" | "Moderate" | "High" | "Critical"
        score                : int 1–4
        c_factor             : residue-adjusted C-factor
        c_factor_unadjusted  : NDVI-only C-factor before residue adjustment
        residue_multiplier   : float multiplier applied
        residue_system       : str label selected
        ls_factor            : mean-based LS-factor
        rusle_score          : adjusted C × LS
        risk_array           : per-pixel Risk Index (None if no arrays given)
        zone_array           : per-pixel zone 1–4 (None if no arrays given)
        soil_loss            : dict from estimate_soil_loss() (None if k_factor missing)
        low_cover / steep_slope / ndvi_threshold / slope_threshold : legacy
        recommendation       : plain-English advisory text
    """
    ndvi_mean  = ndvi_stats["mean"]
    slope_mean = slope_stats["mean"]

    c_factor_unadjusted = _lookup_c_factor(ndvi_mean)
    residue_multiplier  = RESIDUE_ADJUSTMENTS.get(residue_system, 1.00)
    c_factor_adjusted   = c_factor_unadjusted * residue_multiplier

    ls_factor   = _lookup_ls_factor(slope_mean)
    rusle_score = c_factor_adjusted * ls_factor

    # Soil loss estimation (A = R × K × LS × C)
    _series_key      = (soil_series or "default").split()[0]
    t_value          = IOWA_T_VALUES.get(_series_key, IOWA_T_VALUES["default"])
    soil_loss_result = estimate_soil_loss(
        c_factor=c_factor_adjusted,
        ls_factor=ls_factor,
        k_factor=k_factor,
        t_value=t_value,
        r_factor=r_factor,
    )

    risk_array_out = None
    zone_array_out = None

    if ndvi_array is not None and slope_array is not None:
        raw_risk       = pixel_risk_index(ndvi_array, slope_array)
        risk_array_out = raw_risk * residue_multiplier
        zone_array_out = classify_risk_zones(risk_array_out)
        valid_mask  = ~np.isnan(zone_array_out)
        valid_count = valid_mask.sum()
        if valid_count > 0:
            pct_critical = float((zone_array_out[valid_mask] == 4).sum() / valid_count * 100)
            pct_high     = float((zone_array_out[valid_mask] == 3).sum() / valid_count * 100)
            # Concern level thresholds — CoverMap v1.3
            # Critical: >=20% Critical pixels
            # High:     >=5% Critical OR >40% High+Critical
            # Moderate: 20-40% High+Critical AND Critical <5%
            # Low:      <20% High+Critical AND Critical <5%
            # Priority: evaluated top-down, first match wins
            # Source: CoverMap Technical Guide v1.3 Section 6.3
            pct_high_critical = pct_critical + pct_high
            if pct_critical >= 20:
                concern = "Critical"
            elif pct_critical >= 5 or pct_high_critical > 40:
                concern = "High"
            elif pct_high_critical >= 20 and pct_critical < 5:
                concern = "Moderate"
            else:
                concern = "Low"
        else:
            concern = _concern_level(rusle_score)
    else:
        concern = _concern_level(rusle_score)

    score_int = {"Low": 1, "Moderate": 2, "High": 3, "Critical": 4}.get(concern, 2)

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

    zone_counts_out: Dict[int, int] = {}
    if zone_array_out is not None:
        valid_mask = ~np.isnan(ndvi_array)
        zone_counts_out = {
            1: int(np.sum((zone_array_out == 1) & valid_mask)),
            2: int(np.sum((zone_array_out == 2) & valid_mask)),
            3: int(np.sum((zone_array_out == 3) & valid_mask)),
            4: int(np.sum((zone_array_out == 4) & valid_mask)),
        }

    return {
        "concern_level":       concern,
        "score":               score_int,
        "c_factor":            round(c_factor_adjusted, 3),
        "c_factor_unadjusted": round(c_factor_unadjusted, 3),
        "residue_multiplier":  residue_multiplier,
        "residue_system":      residue_system,
        "ls_factor":           round(ls_factor, 2),
        "rusle_score":         round(rusle_score, 3),
        "risk_array":          risk_array_out,
        "zone_array":          zone_array_out,
        "zone_counts":         zone_counts_out,
        "soil_loss":           soil_loss_result,
        "low_cover":           ndvi_mean < ndvi_threshold,
        "steep_slope":         slope_mean > slope_threshold,
        "ndvi_threshold":      ndvi_threshold,
        "slope_threshold":     slope_threshold,
        "recommendation":      recommendations.get(concern, ""),
    }


def pixel_level_concern(
    ndvi_array: np.ndarray,
    slope_array: np.ndarray,
) -> np.ndarray:
    """
    Apply RUSLE C×LS scoring at every pixel for map visualization.
    Returns a float array of rusle_score values (same shape as inputs).
    """
    c_array = np.full(ndvi_array.shape, np.nan, dtype=float)
    c_array = np.where(ndvi_array < 0.15,                                    0.90, c_array)
    c_array = np.where((ndvi_array >= 0.15) & (ndvi_array < 0.20),           0.75, c_array)
    c_array = np.where((ndvi_array >= 0.20) & (ndvi_array < 0.35),           0.45, c_array)
    c_array = np.where((ndvi_array >= 0.35) & (ndvi_array < 0.50),           0.20, c_array)
    c_array = np.where((ndvi_array >= 0.50) & (ndvi_array < 0.65),           0.08, c_array)
    c_array = np.where(ndvi_array >= 0.65,                                    0.03, c_array)
    ls_array = _analytical_ls_factor(slope_array)
    rusle = c_array * ls_array
    rusle = np.where(np.isnan(ndvi_array) | np.isnan(slope_array), np.nan, rusle)
    return rusle

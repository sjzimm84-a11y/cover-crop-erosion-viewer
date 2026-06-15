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
# Continuous C-factor parameters — exponential decay model
# C(NDVI) = floor + (intercept - floor) * exp(-k * NDVI)
#   intercept : C-factor at NDVI=0 (failed cover crop, residue alone drives erosion)
#   floor     : C-factor asymptote at high NDVI (excellent stand + residue system)
#   k         : exponential decay constant (higher = faster benefit from early biomass)
# Initial parameters based on published RUSLE2 value ranges for Iowa cropland.
# Calibration against RUSLE2 Iowa State File runs (map units 100D2, 100E2, 100F2,
# Monona silt loam) in progress — Shelby County NRCS (W. Dittmer, 2026).
# Parameters subject to revision once RUSLE2 comparison data are available.
# Keys match RESIDUE_OPTIONS exactly.
# ---------------------------------------------------------------------------
CONTINUOUS_C_PARAMS = {
    "No-till corn (high residue ~80% cover)":         {"intercept": 0.05, "floor": 0.005, "k": 8},
    "No-till soybeans (moderate residue — fragile)":  {"intercept": 0.10, "floor": 0.015, "k": 7},
    "Tillage — > 30% residue (conservation tillage)": {"intercept": 0.25, "floor": 0.050, "k": 6},
    "Tillage — < 30% residue (conventional tillage)": {"intercept": 0.45, "floor": 0.100, "k": 5},
    "Unknown — not recorded (conservative default)":  {"intercept": 0.45, "floor": 0.100, "k": 5},
}

_CONTINUOUS_C_FALLBACK = "Tillage — < 30% residue (conventional tillage)"

# NDVI threshold below which no living-cover credit is given — C = intercept (residue only).
# Data: 15 Iowa field observations, five counties, three tillage systems, corn and soybean,
# March–April 2026 Sentinel-2 L2A imagery. Mean bare/residue NDVI = 0.179, SD = 0.013.
# Threshold set at mean + 0.5 SD. No significant difference by residue type or tillage found.
UNIVERSAL_NDVI_BASELINE = 0.185

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
    Returns (r_factor: float, source_note: str, county_display: str | None).
    county_display is formatted as "Name County, IA" for use in reports,
    or None if the lookup fails.
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
        county_raw     = data.get("County", {}).get("name", "")
        county_name    = (county_raw.lower()
                          .replace(" county", "").strip())
        county_display = f"{county_name.title()} County, IA" if county_name else None
        if county_name in IOWA_R_FACTOR_150_COUNTIES:
            return (
                150.0,
                f"R=150 (northwest Iowa — "
                f"{county_name.title()} County, NRCS FOTG)",
                county_display,
            )
        note = (
            f"R=175 (standard Iowa — "
            f"{county_name.title()} County, NRCS FOTG)"
            if county_name
            else "R=175 (standard Iowa zone, NRCS FOTG)"
        )
        return (175.0, note, county_display)
    except Exception:
        return (
            175.0,
            "R=175 (default — county lookup unavailable, NRCS FOTG)",
            None,
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


def _continuous_c_factor(ndvi: float, residue_system: str) -> float:
    """Piecewise exponential C-factor for a scalar NDVI value.

    NDVI <= UNIVERSAL_NDVI_BASELINE:
        C = intercept  (residue only — no living cover credit)
    NDVI > UNIVERSAL_NDVI_BASELINE:
        C = floor + (intercept - floor) * exp(-k * (NDVI - UNIVERSAL_NDVI_BASELINE))

    Negative NDVI clamped to 0 (returns intercept).
    Unknown residue_system falls back to conventional tillage (most conservative).
    """
    p    = CONTINUOUS_C_PARAMS.get(residue_system, CONTINUOUS_C_PARAMS[_CONTINUOUS_C_FALLBACK])
    ndvi = max(0.0, float(ndvi))
    if ndvi <= UNIVERSAL_NDVI_BASELINE:
        return p["intercept"]
    return p["floor"] + (p["intercept"] - p["floor"]) * np.exp(-p["k"] * (ndvi - UNIVERSAL_NDVI_BASELINE))


def _continuous_c_array(ndvi_array: np.ndarray, residue_system: str) -> np.ndarray:
    """Vectorized piecewise exponential C-factor for a pixel array.

    NaN pixels propagate as NaN (np.where evaluates both branches but NaN in
    the exponential branch propagates through; NaN <= threshold is False so
    np.where selects the exponential branch, which is also NaN). Shape preserved.
    """
    p    = CONTINUOUS_C_PARAMS.get(residue_system, CONTINUOUS_C_PARAMS[_CONTINUOUS_C_FALLBACK])
    ndvi = np.maximum(ndvi_array, 0.0)  # clamp negatives; np.maximum preserves NaN
    return np.where(
        ndvi <= UNIVERSAL_NDVI_BASELINE,
        p["intercept"],
        p["floor"] + (p["intercept"] - p["floor"]) * np.exp(-p["k"] * (ndvi - UNIVERSAL_NDVI_BASELINE)),
    )


# DEPRECATED — superseded by _continuous_c_factor(). Retained for compare_methods.py.
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
    residue_system: str = "Unknown — not recorded (conservative default)",
) -> np.ndarray:
    """
    Compute per-pixel RUSLE Risk Index (C × LS) for every pixel in the field.
    Returns array of same shape as inputs. NaN propagates from either input.
    C-factor from continuous exponential model (_continuous_c_array).
    LS-factor from McCool et al. 1987 analytical formula (_analytical_ls_factor).
    Old bin-based implementation preserved in pixel_level_concern() for compare_methods.py.
    """
    c_array  = _continuous_c_array(ndvi_array, residue_system)
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
    zones = np.digitize(risk_array, bins=[0.3, 0.7, 1.5]).astype(float) + 1
    zones[np.isnan(risk_array)] = np.nan
    return zones


def _compute_zone_erosion_summary(
    ndvi_array: np.ndarray,
    slope_array: np.ndarray,
    zone_array: np.ndarray,
    residue_system: str,
    k_factor: Any,
    r_factor: float,
    field_t_value: Any = None,
    zone_geometries: Optional[dict] = None,
    mupolygons: Optional[list] = None,
) -> list:
    """Per-risk-zone RUSLE erosion and reduction summary.
    Returns list of dicts keyed: zone_label, mean_ndvi, c_adj, c_baseline,
    mean_ls, a_current_zone, pct_reduction, a_saved_zone, area_fraction,
    plus zone-level soil-tolerance fields t_zone_weighted, t_zone_min,
    t_zone_max, within_t_zone.
    Only zones with at least one valid pixel are included.

    Zone-level T (zone_geometries + mupolygons): Risk zones and SSURGO map
    units are different spatial partitions, so each zone's T is area-weighted
    from the map units it overlaps via soil_data.zone_weighted_t. When
    zone_geometries (keyed by zone value 1-4, WGS84 shapely) or mupolygons
    (``(mukey, geom, t)`` tuples) are unavailable, the t_zone_* fields are set
    to None and within_t_zone falls back to comparing A against field_t_value
    (the field-level T). No network call is made here; zone_weighted_t is a
    pure shapely helper imported lazily only when geometry is supplied."""
    try:
        k = float(k_factor)
    except (TypeError, ValueError):
        k = None

    zone_labels = {1: "Low", 2: "Moderate", 3: "High", 4: "Critical"}
    valid_mask  = ~np.isnan(ndvi_array) & ~np.isnan(zone_array)
    total_valid = int(np.sum(valid_mask))
    if total_valid == 0:
        return []

    # Lazily bind the pure shapely zone-T helper only when geometry is given.
    _zone_weighted_t = None
    if zone_geometries is not None and mupolygons:
        try:
            from src.soil_data import zone_weighted_t as _zone_weighted_t
        except Exception:
            _zone_weighted_t = None

    c_baseline = _continuous_c_factor(0.0, residue_system)  # C at NDVI=0 = intercept

    results = []
    for zone_val, zone_label in zone_labels.items():
        zone_mask   = (zone_array == zone_val) & valid_mask
        pixel_count = int(np.sum(zone_mask))
        if pixel_count == 0:
            continue

        mean_ndvi      = float(np.nanmean(ndvi_array[zone_mask]))
        mean_ls        = float(np.nanmean(_analytical_ls_factor(slope_array[zone_mask])))
        mean_slope_pct = float(np.nanmean(slope_array[zone_mask]))
        area_fraction  = pixel_count / total_valid

        c_adj = _continuous_c_factor(mean_ndvi, residue_system)

        a_current_zone  = r_factor * k * mean_ls * c_adj      if k is not None else None
        a_baseline_zone = r_factor * k * mean_ls * c_baseline if k is not None else None

        if k is not None and c_baseline > 0:
            # >= handles the flat region (NDVI <= UNIVERSAL_NDVI_BASELINE)
            # where c_adj == c_baseline and saving is legitimately zero
            pct_reduction = (c_baseline - c_adj) / c_baseline * 100
            a_saved_zone  = r_factor * k * mean_ls * (c_baseline - c_adj)
        else:
            pct_reduction = None
            a_saved_zone  = None

        # ----------------------------------------------------------------
        # Zone-level area-weighted T from the SSURGO map units this zone
        # overlaps. Degrades to field-level T (zone fields None) when zone
        # geometry / mupolygons are unavailable — never invents per-pixel T.
        # ----------------------------------------------------------------
        t_zone_weighted = t_zone_min = t_zone_max = None
        if _zone_weighted_t is not None:
            _zgeom = zone_geometries.get(zone_val)
            if _zgeom is not None:
                try:
                    _zt = _zone_weighted_t(_zgeom, mupolygons)
                    t_zone_weighted = _zt["t_zone_weighted"]
                    t_zone_min      = _zt["t_min"]
                    t_zone_max      = _zt["t_max"]
                except Exception:
                    t_zone_weighted = t_zone_min = t_zone_max = None

        # within_t_zone uses the zone's own T when available, otherwise reuses
        # the field-level T (so the per-zone flag still carries a signal).
        _t_for_flag = t_zone_weighted if t_zone_weighted is not None else field_t_value
        if a_current_zone is not None and _t_for_flag is not None:
            within_t_zone = bool(a_current_zone <= _t_for_flag)
        else:
            within_t_zone = None

        results.append({
            "zone_label":      zone_label,
            "mean_ndvi":       round(mean_ndvi, 3),
            "mean_slope_pct":  round(mean_slope_pct, 1),
            "c_adj":           round(c_adj, 3),
            "c_baseline":      round(c_baseline, 3),
            "mean_ls":         round(mean_ls, 2),
            "a_current_zone":  round(a_current_zone, 2)  if a_current_zone  is not None else None,
            "a_baseline_zone": round(a_baseline_zone, 2) if a_baseline_zone is not None else None,
            "pct_reduction":   round(pct_reduction, 1)   if pct_reduction   is not None else None,
            "a_saved_zone":    round(a_saved_zone, 2)    if a_saved_zone    is not None else None,
            "area_fraction":   round(area_fraction, 4),
            "t_zone_weighted": round(t_zone_weighted, 2) if t_zone_weighted is not None else None,
            "t_zone_min":      t_zone_min,
            "t_zone_max":      t_zone_max,
            "within_t_zone":   within_t_zone,
        })

    return results


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
    soil_summary: Optional[dict] = None,
    zone_geometries: Optional[dict] = None,
    mupolygons: Optional[list] = None,
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
    When soil_summary (the dict from soil_data.soil_summary_for_boundary) is
    provided with non-null area-weighted "k_factor" and "t_value", those values
    replace the passed-in k_factor and the soil_series-derived T at the
    soil-loss and zone-erosion sites, and soil_source_fallback is False;
    otherwise the passed-in k_factor / soil_series values are used and
    soil_source_fallback is True. ksat_r, hydgrp and the drainage-class
    breakdown from soil_summary are stored verbatim under "soil_attrs"
    (Phase 3 handoff — no computation done here). This module performs no
    network call; soil_summary is computed upstream and passed in.
    When zone_geometries (keyed by zone value 1-4, WGS84 shapely) and
    mupolygons (``(mukey, geom, t)`` tuples) are supplied, each zone in the
    zone_erosion_summary gains an area-weighted t_zone_weighted / t_zone_min /
    t_zone_max and a within_t_zone flag; otherwise those fields are None and
    within_t_zone reuses the field-level T. Field-level soil_loss A/T outputs
    are unchanged either way.

    Returns
    -------
    Dict with keys:
        concern_level        : "Low" | "Moderate" | "High" | "Critical"
        score                : int 1–4
        c_factor             : C-factor from continuous exponential model
        c_factor_baseline    : C at NDVI=0 for this residue system (the intercept)
        c_factor_method      : "exponential_v2"
        residue_multiplier   : legacy multiplier (retained for compare_methods.py reference)
        residue_system       : str label selected
        ls_factor            : mean-based LS-factor
        rusle_score          : adjusted C × LS
        risk_array           : per-pixel Risk Index (None if no arrays given)
        zone_array           : per-pixel zone 1–4 (None if no arrays given)
        soil_loss            : dict from estimate_soil_loss() (None if k_factor missing)
        soil_source_fallback : True when soil_summary was absent/invalid and the
                               passed-in k_factor / soil_series T were used
        soil_attrs           : dict of ksat_r / hydgrp / drainage_fraction from
                               soil_summary (None on fallback)
        low_cover / steep_slope / ndvi_threshold / slope_threshold : legacy
        recommendation       : plain-English advisory text
    """
    ndvi_mean  = ndvi_stats["mean"]
    slope_mean = slope_stats["mean"]

    residue_multiplier = RESIDUE_ADJUSTMENTS.get(residue_system, 1.00)  # retained for compare_methods.py reference
    c_factor_adjusted  = _continuous_c_factor(ndvi_mean, residue_system)
    c_factor_baseline  = _continuous_c_factor(0.0,       residue_system)  # C at NDVI=0 = intercept

    ls_factor   = _lookup_ls_factor(slope_mean)
    rusle_score = c_factor_adjusted * ls_factor

    # ------------------------------------------------------------------
    # Resolve K and T source. Prefer the area-weighted soil_summary (SDA)
    # when it is provided and carries non-null K and T; otherwise fall back
    # to the passed-in k_factor and the soil_series-derived T (current WSS
    # path). This changes only the *source* of K/T — the RUSLE formula,
    # LS/R/C models, and Risk Index zone thresholds are untouched.
    # ------------------------------------------------------------------
    _use_soil_summary = (
        isinstance(soil_summary, dict)
        and soil_summary.get("k_factor") is not None
        and soil_summary.get("t_value") is not None
    )
    if _use_soil_summary:
        effective_k          = soil_summary["k_factor"]
        t_value              = soil_summary["t_value"]
        soil_source_fallback = False
        soil_attrs           = {
            "ksat_r":            soil_summary.get("ksat_r"),
            "hydgrp":            soil_summary.get("hydgrp"),
            "drainage_fraction": soil_summary.get("drainage_fraction"),
        }
    else:
        effective_k          = k_factor
        _series_key          = (soil_series or "default").split()[0]
        t_value              = IOWA_T_VALUES.get(_series_key, IOWA_T_VALUES["default"])
        soil_source_fallback = True
        soil_attrs           = None

    # Soil loss estimation (A = R × K × LS × C)
    soil_loss_result = estimate_soil_loss(
        c_factor=c_factor_adjusted,
        ls_factor=ls_factor,
        k_factor=effective_k,
        t_value=t_value,
        r_factor=r_factor,
    )

    risk_array_out = None
    zone_array_out = None

    if ndvi_array is not None and slope_array is not None:
        risk_array_out = pixel_risk_index(ndvi_array, slope_array, residue_system)
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

    zone_erosion_out = []
    if zone_array_out is not None and ndvi_array is not None and slope_array is not None:
        zone_erosion_out = _compute_zone_erosion_summary(
            ndvi_array=ndvi_array,
            slope_array=slope_array,
            zone_array=zone_array_out,
            residue_system=residue_system,
            k_factor=effective_k,
            r_factor=r_factor,
            field_t_value=t_value,
            zone_geometries=zone_geometries,
            mupolygons=mupolygons,
        )

    return {
        "concern_level":        concern,
        "score":                score_int,
        "c_factor":             round(c_factor_adjusted, 3),
        "c_factor_baseline":    round(c_factor_baseline, 3),
        "c_factor_method":      "exponential_v2",
        "residue_multiplier":   residue_multiplier,   # retained for compare_methods.py reference
        "residue_system":       residue_system,
        "ls_factor":            round(ls_factor, 2),
        "rusle_score":          round(rusle_score, 3),
        "risk_array":           risk_array_out,
        "zone_array":           zone_array_out,
        "zone_counts":          zone_counts_out,
        "zone_erosion_summary": zone_erosion_out,
        "soil_loss":            soil_loss_result,
        "soil_source_fallback": soil_source_fallback,
        "soil_attrs":           soil_attrs,
        "low_cover":            ndvi_mean < ndvi_threshold,
        "steep_slope":          slope_mean > slope_threshold,
        "ndvi_threshold":       ndvi_threshold,
        "slope_threshold":      slope_threshold,
        "recommendation":       recommendations.get(concern, ""),
    }


# DEPRECATED — dead code (not called in current app). Superseded by pixel_risk_index().
# Retained for compare_methods.py reference.
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

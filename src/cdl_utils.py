"""
cdl_utils.py
------------
USDA NASS Cropland Data Layer (CDL) crop classification per field.

Pulls ee.Image("USDA/NASS/CDL/{year}") band "cropland" and reduces a
pixel-frequency histogram over the actual field polygon (not the bounding
rectangle the Sentinel-2 NDVI pipeline uses — a rectangle would count
neighboring fields and roads in the crop shares).

Classification (confidence-based labeling, evaluated in order):
  1. Corn (class 1) share >= 70%              -> "Corn"
  2. Soybeans (class 5) share >= 70%          -> "Soybeans"
  3. Dominant Corn/Soybeans share 50-69%      -> "Predominantly {crop} ({pct}%)"
  4. Non-ag share > 10%                       -> "Mixed / boundary review flagged"
  5. Else                                     -> "Mixed rotation ({p1}% {c1} / {p2}% {c2})"

`classify_cdl_histogram` is pure (no GEE, no network) so the threshold logic
is unit-testable offline; `get_cdl_classification` does the GEE pull and
delegates to it. GEE auth must already be initialized by the caller (see
gee_ndvi_utils.init_gee_from_streamlit_secrets).
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import geopandas as gpd
from shapely.geometry import mapping as _shp_mapping

# GEE imports with graceful fallback (mirrors gee_ndvi_utils)
_GEE_IMPORT_ERROR = None
GEE_AVAILABLE = False
try:
    import ee
    GEE_AVAILABLE = True
except Exception as _e:
    _GEE_IMPORT_ERROR = str(_e)

# Latest published CDL year. CDL for year N publishes Jan-Feb of N+1
# (CDL 2026 publishes Jan-Feb 2027). Years beyond this are reported as
# "Not yet published" without touching GEE.
CDL_LATEST_YEAR = 2025

CDL_SCALE_M = 30

# Iowa-relevant CDL class codes (USDA NASS CDL legend).
CDL_CLASS_MAP: Dict[int, str] = {
    1:   "Corn",
    5:   "Soybeans",
    24:  "Winter Wheat",
    28:  "Oats",
    36:  "Alfalfa",
    61:  "Fallow/Idle Cropland",
    # Non-agricultural codes (boundary-contamination signals)
    82:  "Developed",
    111: "Open Water",
    121: "Developed/Open Space",
    122: "Developed/Low Intensity",
    123: "Developed/Med Intensity",
    124: "Developed/High Intensity",
    141: "Deciduous Forest",
    142: "Evergreen Forest",
    143: "Mixed Forest",
    152: "Shrubland",
    176: "Grassland/Pasture",
}

NON_AG_CODES = {82, 111, 121, 122, 123, 124, 141, 142, 143, 152, 176}

CORN_CODE     = 1
SOYBEANS_CODE = 5


def _class_name(code: int) -> str:
    return CDL_CLASS_MAP.get(code, f"CDL {code}")


def _placeholder_result(year: int, label: str) -> Dict[str, Any]:
    """Result shape for years with no usable CDL data."""
    return {
        "year": year,
        "label": label,
        "dominant_class": None,
        "dominant_pct": None,
        "top_two": [],
        "non_ag_pct": 0.0,
        "boundary_warning": False,
    }


def classify_cdl_histogram(histogram: Dict[Any, float], year: int) -> Dict[str, Any]:
    """
    Pure classification of a CDL pixel-frequency histogram.

    histogram: {code: pixel_count} — codes may be int or str (GEE returns
    string keys); counts may be float (edge pixels are fractionally weighted).
    """
    counts = {int(k): float(v) for k, v in (histogram or {}).items() if float(v) > 0}
    total = sum(counts.values())
    if total <= 0:
        return _placeholder_result(year, "No CDL data")

    shares = {code: cnt / total * 100.0 for code, cnt in counts.items()}
    ranked = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)

    dominant_code, dominant_pct = ranked[0]
    top_two = [(_class_name(c), round(p, 1)) for c, p in ranked[:2]]
    non_ag_pct = sum(p for c, p in shares.items() if c in NON_AG_CODES)
    boundary_warning = non_ag_pct > 10.0

    corn_pct = shares.get(CORN_CODE, 0.0)
    soy_pct  = shares.get(SOYBEANS_CODE, 0.0)

    if corn_pct >= 70.0:
        label = "Corn"
    elif soy_pct >= 70.0:
        label = "Soybeans"
    elif dominant_code in (CORN_CODE, SOYBEANS_CODE) and dominant_pct >= 50.0:
        label = f"Predominantly {_class_name(dominant_code)} ({dominant_pct:.0f}%)"
    elif non_ag_pct > 10.0:
        label = "Mixed / boundary review flagged"
    elif len(top_two) >= 2:
        (c1, p1), (c2, p2) = top_two
        label = f"Mixed rotation ({p1:.0f}% {c1} / {p2:.0f}% {c2})"
    else:
        c1, p1 = top_two[0]
        label = f"Mixed rotation ({p1:.0f}% {c1})"

    return {
        "year": year,
        "label": label,
        "dominant_class": _class_name(dominant_code),
        "dominant_pct": round(dominant_pct, 1),
        "top_two": top_two,
        "non_ag_pct": round(non_ag_pct, 1),
        "boundary_warning": boundary_warning,
    }


def get_cdl_classification(field_geom_ee: "ee.Geometry", year: int) -> Dict[str, Any]:
    """
    CDL crop classification for one field-year via GEE.

    Returns the classify_cdl_histogram dict; years beyond CDL_LATEST_YEAR
    (or any GEE failure) degrade to a placeholder row rather than raising,
    so one bad year never drops the whole rotation table.
    """
    if year > CDL_LATEST_YEAR:
        return _placeholder_result(year, "Not yet published")
    if not GEE_AVAILABLE:
        raise RuntimeError(f"GEE packages not installed: {_GEE_IMPORT_ERROR}")

    try:
        image = ee.Image(f"USDA/NASS/CDL/{year}").select("cropland")
        hist = image.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=field_geom_ee,
            scale=CDL_SCALE_M,
            maxPixels=1e9,
        ).getInfo().get("cropland") or {}
    except Exception:
        return _placeholder_result(year, "CDL unavailable")

    return classify_cdl_histogram(hist, year)


def boundary_to_ee_geometry(boundary_gdf: gpd.GeoDataFrame) -> "ee.Geometry":
    """Actual field polygon (union, EPSG:4326) as an ee.Geometry."""
    if not GEE_AVAILABLE:
        raise RuntimeError(f"GEE packages not installed: {_GEE_IMPORT_ERROR}")
    boundary_ll = boundary_gdf.to_crs("EPSG:4326")
    try:
        geom = boundary_ll.geometry.union_all()
    except AttributeError:                      # geopandas < 1.0
        geom = boundary_ll.geometry.unary_union
    return ee.Geometry(_shp_mapping(geom))


def confident_previous_crop(rotation_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    The previous crop for the current cover crop season, but only when CDL is
    confident about it.

    Uses the most recent rotation row only (rows are most-recent-first; that
    year is the cash crop the cover crop was seeded into). Returns
    {"crop": str, "year": int, "pct": float} when that row carries a clean
    >=70% single-crop label (label == dominant class, i.e. the Corn/Soybeans
    threshold branches) — None otherwise, including "Predominantly"/"Mixed"
    labels and unpublished years. An older year is never substituted: if the
    immediately preceding season is unknown, the right answer is blank.
    """
    if not rotation_rows:
        return None
    row = rotation_rows[0]
    if (
        row.get("dominant_pct") is not None
        and row["dominant_pct"] >= 70.0
        and row.get("label") == row.get("dominant_class")
    ):
        return {"crop": row["label"], "year": row["year"], "pct": row["dominant_pct"]}
    return None


def get_cdl_rotation_rows(
    boundary_gdf: gpd.GeoDataFrame,
    n_years: int = 3,
    current_season_year: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Rotation history rows for the producer report: years N-1 .. N-n_years
    relative to the current cover crop season (most recent first).
    """
    if current_season_year is None:
        current_season_year = datetime.now().year
    geom = boundary_to_ee_geometry(boundary_gdf)
    return [
        get_cdl_classification(geom, current_season_year - offset)
        for offset in range(1, n_years + 1)
    ]

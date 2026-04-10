"""
ndvi_scheduler.py
-----------------
Weekly rolling NDVI retrieval with automatic cloud-cover fallback.

Logic:
  1. Try the last 7 days first (ideal — most current data)
  2. If no valid pixels returned, widen to 14 days automatically
  3. If still no valid pixels, widen to 30 days and warn the user
  4. If nothing found in 30 days, raise a clear error with the reason

Designed for Iowa early-season cover crop monitoring (March–April).
Can be called on demand from app.py or scheduled via Streamlit's
session_state refresh pattern.

Usage:
    from src.ndvi_scheduler import fetch_best_available_ndvi

    ndvi_array, transform, profile, meta = fetch_best_available_ndvi(
        boundary_gdf=field_boundary,
        token=cdse_token,
    )
    st.info(meta["message"])
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rasterio
import geopandas as gpd

from src.sentinel_utils import fetch_ndvi_for_field


# ---------------------------------------------------------------------------
# Window configurations (days lookback, label shown to user)
# ---------------------------------------------------------------------------
WINDOWS = [
    {"days": 7,  "label": "last 7 days"},
    {"days": 14, "label": "last 14 days"},
    {"days": 30, "label": "last 30 days"},
]

# Minimum fraction of non-NaN pixels to consider a scene "valid"
MIN_VALID_FRACTION = 0.10


def _count_valid(ndvi_array: np.ndarray) -> float:
    """Return fraction of pixels that are not NaN."""
    total = ndvi_array.size
    if total == 0:
        return 0.0
    valid = int(np.sum(~np.isnan(ndvi_array)))
    return valid / total


def fetch_best_available_ndvi(
    boundary_gdf: gpd.GeoDataFrame,
    token: str,
    reference_date: Optional[datetime] = None,
    min_valid_fraction: float = MIN_VALID_FRACTION,
) -> Tuple[np.ndarray, rasterio.Affine, Dict[str, Any], Dict[str, Any]]:
    """
    Fetch the most current NDVI available for a field boundary,
    automatically widening the time window if cloud cover blocks data.

    Parameters
    ----------
    boundary_gdf       : Field boundary GeoDataFrame (any CRS)
    token              : CDSE bearer token from sentinel_utils.get_cdse_token()
    reference_date     : End date for the search window (default: today)
    min_valid_fraction : Minimum non-NaN pixel fraction to accept a result

    Returns
    -------
    (ndvi_array, affine_transform, rasterio_profile, meta)

    meta keys:
        window_days   : int   — how many days back we ended up searching
        window_label  : str   — human-readable label e.g. "last 14 days"
        date_from     : str   — ISO date string for window start
        date_to       : str   — ISO date string for window end
        valid_fraction: float — fraction of non-NaN pixels in result
        message       : str   — ready-to-display Streamlit info message
        year          : int   — year used for the pull
    """
    if reference_date is None:
        reference_date = datetime.utcnow()

    last_error = None

    for window in WINDOWS:
        days = window["days"]
        label = window["label"]

        date_to   = reference_date
        date_from = reference_date - timedelta(days=days)

        # Format as MM-DD for sentinel_utils API
        spring_start = date_from.strftime("%m-%d")
        spring_end   = date_to.strftime("%m-%d")
        year         = date_from.year

        try:
            ndvi_array, transform, profile = fetch_ndvi_for_field(
                boundary_gdf=boundary_gdf,
                token=token,
                year=year,
                spring_start=spring_start,
                spring_end=spring_end,
            )

            valid_frac = _count_valid(ndvi_array)

            if valid_frac < min_valid_fraction:
                # Not enough clear pixels — try wider window
                last_error = (
                    f"Window '{label}' returned only "
                    f"{valid_frac*100:.1f}% valid pixels (need {min_valid_fraction*100:.0f}%). "
                    f"Likely cloud cover. Widening search window..."
                )
                continue

            # Success — build metadata and return
            meta = {
                "window_days":    days,
                "window_label":   label,
                "date_from":      date_from.strftime("%Y-%m-%d"),
                "date_to":        date_to.strftime("%Y-%m-%d"),
                "valid_fraction": valid_frac,
                "year":           year,
                "message": (
                    f"✅ NDVI pulled from Sentinel-2 | "
                    f"{date_from.strftime('%b %d')} – {date_to.strftime('%b %d, %Y')} | "
                    f"{valid_frac*100:.0f}% clear pixels"
                ),
            }
            return ndvi_array, transform, profile, meta

        except Exception as exc:
            last_error = str(exc)
            continue

    # All windows exhausted
    raise RuntimeError(
        f"No valid Sentinel-2 NDVI found in the last {WINDOWS[-1]['days']} days. "
        f"Iowa cloud cover may be blocking all scenes. Last error: {last_error}\n\n"
        f"Tip: Try uploading a manual NDVI GeoTIFF from Earthdata or ESA Browser "
        f"as a fallback."
    )


# ---------------------------------------------------------------------------
# Year-over-year comparison helper
# ---------------------------------------------------------------------------

def fetch_ndvi_comparison(
    boundary_gdf: gpd.GeoDataFrame,
    token: str,
    years: list,
    spring_start: str = "03-01",
    spring_end: str = "04-20",
) -> Dict[int, Dict[str, Any]]:
    """
    Pull NDVI for multiple years over the same spring window.
    Returns a dict keyed by year: {ndvi, transform, profile, valid_fraction}

    Example:
        results = fetch_ndvi_comparison(gdf, token, years=[2023, 2024, 2025])
        for year, data in results.items():
            st.write(f"{year} mean NDVI: {np.nanmean(data['ndvi']):.3f}")
    """
    results = {}
    for year in years:
        try:
            ndvi_array, transform, profile = fetch_ndvi_for_field(
                boundary_gdf=boundary_gdf,
                token=token,
                year=year,
                spring_start=spring_start,
                spring_end=spring_end,
            )
            results[year] = {
                "ndvi":           ndvi_array,
                "transform":      transform,
                "profile":        profile,
                "valid_fraction": _count_valid(ndvi_array),
                "mean_ndvi":      float(np.nanmean(ndvi_array)),
            }
        except Exception as exc:
            results[year] = {"error": str(exc)}

    return results

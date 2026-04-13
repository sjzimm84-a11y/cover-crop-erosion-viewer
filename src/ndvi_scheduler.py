"""
ndvi_scheduler.py
-----------------
Weekly rolling NDVI retrieval with automatic cloud-cover fallback.
Uses sentinelhub-py library for reliable CDSE Process API access.

Window logic:
  1. Try last 7 days first (most current)
  2. Widen to 14 days if < 10% valid pixels
  3. Widen to 30 days if still insufficient
  4. Raise clear error if nothing found in 30 days
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import numpy as np
import geopandas as gpd

from src.sentinel_utils import fetch_ndvi_for_field, get_config_from_streamlit_secrets

MIN_VALID_FRACTION = 0.10

WINDOWS = [
    {"days": 7,  "label": "last 7 days"},
    {"days": 14, "label": "last 14 days"},
    {"days": 30, "label": "last 30 days"},
]


def fetch_best_available_ndvi(
    boundary_gdf: gpd.GeoDataFrame,
    reference_date: Optional[datetime] = None,
    min_valid_fraction: float = MIN_VALID_FRACTION,
) -> Tuple[np.ndarray, Any, Dict[str, Any], Dict[str, Any]]:
    """
    Fetch most current NDVI, widening window if clouds block data.

    Returns (ndvi_array, transform, profile, meta)
    meta includes: window_label, date_from, date_to, valid_fraction, message
    """
    if reference_date is None:
        reference_date = datetime.now()

    config = get_config_from_streamlit_secrets()
    last_error = None

    for window in WINDOWS:
        days  = window["days"]
        label = window["label"]
        date_from = reference_date - timedelta(days=days)
        date_to   = reference_date

        try:
            ndvi_array, transform, profile = fetch_ndvi_for_field(
                boundary_gdf=boundary_gdf,
                config=config,
                date_from=date_from,
                date_to=date_to,
            )

            valid = ndvi_array[~np.isnan(ndvi_array)]
            valid_frac = valid.size / ndvi_array.size if ndvi_array.size > 0 else 0

            if valid_frac < min_valid_fraction:
                last_error = (
                    f"Window '{label}': only {valid_frac*100:.1f}% valid pixels. "
                    f"Widening search window..."
                )
                continue

            meta = {
                "window_days":    days,
                "window_label":   label,
                "date_from":      date_from.strftime("%Y-%m-%d"),
                "date_to":        date_to.strftime("%Y-%m-%d"),
                "valid_fraction": valid_frac,
                "message": (
                    f"✅ Sentinel-2 NDVI | "
                    f"{date_from.strftime('%b %d')}–"
                    f"{date_to.strftime('%b %d, %Y')} | "
                    f"{valid_frac*100:.0f}% clear pixels"
                ),
            }
            return ndvi_array, transform, profile, meta

        except Exception as exc:
            last_error = str(exc)
            continue

    raise RuntimeError(
        f"No valid NDVI found in last {WINDOWS[-1]['days']} days. "
        f"Iowa cloud cover may be blocking all scenes. "
        f"Last error: {last_error}"
    )


def fetch_ndvi_comparison(
    boundary_gdf: gpd.GeoDataFrame,
    years: list,
    spring_start_month: int = 3,
    spring_end_month: int = 4,
) -> Dict[int, Dict[str, Any]]:
    """
    Pull NDVI for multiple years over the same spring window.
    Returns dict keyed by year.
    """
    config = get_config_from_streamlit_secrets()
    results = {}

    for year in years:
        try:
            date_from = datetime(year, spring_start_month, 1)
            date_to   = datetime(year, spring_end_month, 30)
            ndvi_array, transform, profile = fetch_ndvi_for_field(
                boundary_gdf=boundary_gdf,
                config=config,
                date_from=date_from,
                date_to=date_to,
            )
            valid = ndvi_array[~np.isnan(ndvi_array)]
            results[year] = {
                "ndvi":           ndvi_array,
                "transform":      transform,
                "profile":        profile,
                "valid_fraction": valid.size / ndvi_array.size,
                "mean_ndvi":      float(np.nanmean(ndvi_array)),
            }
        except Exception as exc:
            results[year] = {"error": str(exc)}

    return results

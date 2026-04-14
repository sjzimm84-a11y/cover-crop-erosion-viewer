"""
gee_ndvi_utils.py
-----------------
Sentinel-2 NDVI via Google Earth Engine — production version.
Auth method matches confirmed-working test script exactly.

Streamlit secrets.toml format:
    [gee]
    project = "cover-crop-erosion"

    [gee.service_account]
    type = "service_account"
    project_id = "cover-crop-erosion"
    private_key_id = "..."
    private_key = "-----BEGIN RSA PRIVATE KEY-----\\n...\\n-----END RSA PRIVATE KEY-----\\n"
    client_email = "...@cover-crop-erosion.iam.gserviceaccount.com"
    client_id = "..."
    auth_uri = "https://accounts.google.com/o/oauth2/auth"
    token_uri = "https://oauth2.googleapis.com/token"
"""

import json
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import numpy as np
import geopandas as gpd

# GEE imports with graceful fallback
_GEE_IMPORT_ERROR = None
GEE_AVAILABLE = False
try:
    import ee
    from google.oauth2 import service_account
    GEE_AVAILABLE = True
except Exception as _e:
    _GEE_IMPORT_ERROR = str(_e)

S2_COLLECTION  = "COPERNICUS/S2_SR_HARMONIZED"
GEE_SCOPE      = "https://www.googleapis.com/auth/earthengine"
MAX_CLOUD_PCT  = 80
TARGET_SCALE_M = 10


# ---------------------------------------------------------------------------
# Authentication — matches working test script exactly
# ---------------------------------------------------------------------------

def init_gee_from_streamlit_secrets() -> None:
    """Initialize GEE using service account from Streamlit secrets."""
    if not GEE_AVAILABLE:
        raise RuntimeError(
            f"GEE packages not installed: {_GEE_IMPORT_ERROR}. "
            "Add earthengine-api and google-auth to requirements.txt"
        )

    import streamlit as st

    try:
        project      = st.secrets["gee"]["project"]
        sa_info_raw  = dict(st.secrets["gee"]["service_account"])
    except KeyError as exc:
        raise RuntimeError(
            f"GEE credentials missing from Streamlit secrets: {exc}. "
            "Add [gee] project and [gee.service_account] to secrets.toml"
        ) from exc

    # Convert AttrDict to plain dict with all required fields
    sa_info = {
        "type":                        sa_info_raw.get("type", "service_account"),
        "project_id":                  sa_info_raw.get("project_id", project),
        "private_key_id":              sa_info_raw.get("private_key_id", ""),
        "private_key":                 sa_info_raw.get("private_key", "").replace("\\n", "\n"),
        "client_email":                sa_info_raw.get("client_email", ""),
        "client_id":                   sa_info_raw.get("client_id", ""),
        "auth_uri":                    sa_info_raw.get("auth_uri", "https://accounts.google.com/o/oauth2/auth"),
        "token_uri":                   sa_info_raw.get("token_uri", "https://oauth2.googleapis.com/token"),
        "auth_provider_x509_cert_url": sa_info_raw.get("auth_provider_x509_cert_url", ""),
        "client_x509_cert_url":        sa_info_raw.get("client_x509_cert_url", ""),
    }

    # Use same auth method as confirmed-working test script
    credentials = service_account.Credentials.from_service_account_info(
        sa_info,
        scopes=[GEE_SCOPE],
    )
    ee.Initialize(credentials=credentials, project=project)


def init_gee_local(key_file: str, project: str) -> None:
    """Initialize GEE from local JSON key file (for testing)."""
    if not GEE_AVAILABLE:
        raise RuntimeError("GEE packages not installed.")
    with open(key_file, "r") as f:
        credentials_dict = json.load(f)
    credentials = service_account.Credentials.from_service_account_info(
        credentials_dict,
        scopes=[GEE_SCOPE],
    )
    ee.Initialize(credentials=credentials, project=project)


# ---------------------------------------------------------------------------
# NDVI computation helpers — matches working test script
# ---------------------------------------------------------------------------

def _add_ndvi(image: "ee.Image") -> "ee.Image":
    """Add NDVI band using normalizedDifference (same as test script)."""
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return image.addBands(ndvi)


def _get_collection(
    aoi: "ee.Geometry",
    date_from: datetime,
    date_to: datetime,
    max_cloud_pct: int,
) -> "ee.ImageCollection":
    """Build filtered Sentinel-2 collection with NDVI band."""
    collection = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(aoi)
        .filterDate(
            date_from.strftime("%Y-%m-%d"),
            date_to.strftime("%Y-%m-%d"),
        )
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .map(_add_ndvi)
        .select("NDVI")
    )
    return collection


# ---------------------------------------------------------------------------
# Main NDVI fetch
# ---------------------------------------------------------------------------

def fetch_ndvi_for_field(
    boundary_gdf: gpd.GeoDataFrame,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    scale_m: int = TARGET_SCALE_M,
    max_cloud_pct: int = MAX_CLOUD_PCT,
) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
    """
    Pull Sentinel-2 NDVI for a field boundary via GEE.

    Returns
    -------
    (ndvi_array, affine_transform, profile_dict)
    ndvi_array: float32, NaN where no data, values in [-1, 1]
    """
    if not GEE_AVAILABLE:
        raise RuntimeError(f"GEE not available: {_GEE_IMPORT_ERROR}")

    # Default to current year Jan 1 → today for Iowa spring coverage
    now = datetime.now()
    if date_from is None:
        date_from = datetime(now.year, 1, 1)
    if date_to is None:
        date_to = now

    # Convert boundary to WGS84
    boundary_ll = boundary_gdf.to_crs("EPSG:4326")
    bounds = boundary_ll.total_bounds  # [minx, miny, maxx, maxy]

    # Use Rectangle geometry (simpler, more reliable than Polygon extraction)
    aoi = ee.Geometry.Rectangle([
        bounds[0], bounds[1], bounds[2], bounds[3]
    ])

    # Try with requested cloud threshold, widen if no scenes found
    for cloud_pct in [max_cloud_pct, 80, 100]:
        collection = _get_collection(aoi, date_from, date_to, cloud_pct)
        count = collection.size().getInfo()
        if count > 0:
            break

    if count == 0:
        raise RuntimeError(
            f"No Sentinel-2 scenes found for this field "
            f"between {date_from.strftime('%Y-%m-%d')} "
            f"and {date_to.strftime('%Y-%m-%d')}. "
            f"Try a wider date range."
        )

    # Median composite — robust for Iowa partly-cloudy spring conditions
    ndvi_image = collection.median().clip(aoi)

    # Download as GeoTIFF via getDownloadURL
    url = ndvi_image.getDownloadURL({
        "bands":  ["NDVI"],
        "region": aoi,
        "scale":  scale_m,
        "format": "GEO_TIFF",
        "crs":    "EPSG:4326",
    })

    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            tiff_bytes = response.read()
    except Exception as exc:
        raise RuntimeError(f"GEE download failed: {exc}") from exc

    # Parse GeoTIFF
    import rasterio
    from rasterio.io import MemoryFile

    with MemoryFile(tiff_bytes) as memfile:
        with memfile.open() as dataset:
            ndvi_raw  = dataset.read(1).astype(np.float32)
            transform = dataset.transform
            profile   = dataset.profile.copy()

    # Mask invalid values
    ndvi_raw[ndvi_raw < -1.0] = np.nan
    ndvi_raw[ndvi_raw >  1.0] = np.nan

    profile.update({
        "dtype":  "float32",
        "nodata": float("nan"),
        "crs":    "EPSG:4326",
    })

    valid = ndvi_raw[~np.isnan(ndvi_raw)]
    if valid.size == 0:
        raise RuntimeError(
            "GEE returned all-NaN NDVI. "
            "Field boundary may be outside Sentinel-2 coverage "
            "or all scenes are fully clouded."
        )

    return ndvi_raw, transform, profile


# ---------------------------------------------------------------------------
# Streamlit wrapper
# ---------------------------------------------------------------------------

def fetch_ndvi_streamlit(
    boundary_gdf: gpd.GeoDataFrame,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> Tuple[np.ndarray, Any, Dict[str, Any], str]:
    """
    GEE NDVI fetch for Streamlit. Initializes auth from secrets.
    Returns (array, transform, profile, status_message).
    """
    init_gee_from_streamlit_secrets()

    ndvi, transform, profile = fetch_ndvi_for_field(
        boundary_gdf=boundary_gdf,
        date_from=date_from,
        date_to=date_to,
    )

    valid     = ndvi[~np.isnan(ndvi)]
    valid_pct = valid.size / ndvi.size * 100 if ndvi.size > 0 else 0
    d_from    = date_from.strftime("%b %d") if date_from else "Jan 1"
    d_to      = date_to.strftime("%b %d, %Y") if date_to else "Today"

    message = (
        f"✅ Sentinel-2 NDVI via Google Earth Engine | "
        f"{d_from} – {d_to} | "
        f"{valid_pct:.0f}% valid pixels | "
        f"Mean NDVI: {float(np.nanmean(ndvi)):.3f}"
    )
    return ndvi, transform, profile, message

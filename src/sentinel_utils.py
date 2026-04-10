"""
sentinel_utils.py
-----------------
Automated Sentinel-2 NDVI retrieval via the Copernicus Data Space Ecosystem (CDSE).
Free account required at: https://dataspace.copernicus.eu

Replaces manual NDVI GeoTIFF uploads with automatic early-season pulls
scoped to the farmer's uploaded field boundary.

Usage:
    token = get_cdse_token(client_id, client_secret)
    ndvi_array, transform, profile = fetch_ndvi_for_field(boundary_gdf, token)
"""

import io
import time
from datetime import date, datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rasterio
import requests
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import box, mapping

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# Default early-spring window for Iowa cover crop assessment
IOWA_SPRING_START = "03-01"   # March 1
IOWA_SPRING_END   = "04-20"   # April 20

# Sentinel-2 L2A band resolution (10 m native for B04/B08)
TARGET_RESOLUTION_M = 10

# Max acceptable cloud cover for a usable scene
MAX_CLOUD_PCT = 30


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def get_cdse_token(client_id: str, client_secret: str) -> str:
    """
    Retrieve a short-lived OAuth2 bearer token from CDSE.

    Parameters
    ----------
    client_id     : From CDSE dashboard → User Settings → OAuth Clients
    client_secret : From same location

    Returns
    -------
    Bearer token string (valid ~10 min; call again if you get 401)
    """
    resp = requests.post(
        CDSE_TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"CDSE authentication failed ({resp.status_code}): {resp.text[:300]}"
        )
    return resp.json()["access_token"]


# ---------------------------------------------------------------------------
# Scene search (OData catalog)
# ---------------------------------------------------------------------------

def search_best_scene(
    boundary_gdf: gpd.GeoDataFrame,
    year: int,
    token: str,
    spring_start: str = IOWA_SPRING_START,
    spring_end: str = IOWA_SPRING_END,
    max_cloud_pct: int = MAX_CLOUD_PCT,
) -> Optional[Dict[str, Any]]:
    """
    Search the CDSE catalog for the least-cloudy Sentinel-2 L2A scene
    that intersects the field boundary in the early-spring window.

    Returns the best scene metadata dict or None if nothing found.
    """
    boundary_ll = boundary_gdf.to_crs("EPSG:4326")
    bbox = boundary_ll.total_bounds  # [minx, miny, maxx, maxy]
    aoi_wkt = (
        f"POLYGON(({bbox[0]} {bbox[1]},{bbox[2]} {bbox[1]},"
        f"{bbox[2]} {bbox[3]},{bbox[0]} {bbox[3]},{bbox[0]} {bbox[1]}))"
    )

    date_from = f"{year}-{spring_start}T00:00:00Z"
    date_to   = f"{year}-{spring_end}T23:59:59Z"

    odata_url = (
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
        f"$filter=Collection/Name eq 'SENTINEL-2' "
        f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        f"and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') "
        f"and ContentDate/Start gt {date_from} "
        f"and ContentDate/Start lt {date_to} "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}') "
        f"and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
        f"and att/OData.CSC.DoubleAttribute/Value lt {max_cloud_pct})"
        f"&$orderby=Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
        f"and att/OData.CSC.DoubleAttribute/Value) asc"
        f"&$top=5"
    )

    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(odata_url, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Catalog search failed ({resp.status_code}): {resp.text[:300]}")

    results = resp.json().get("value", [])
    if not results:
        return None

    # Return the scene with lowest cloud cover (already sorted above)
    return results[0]


# ---------------------------------------------------------------------------
# NDVI fetch via Process API (evalscript)
# ---------------------------------------------------------------------------

NDVI_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B08", "SCL"], units: "REFLECTANCE" }],
    output: { bands: 1, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  // Cloud / shadow mask using Scene Classification Layer
  if ([3,8,9,10,11].includes(s.SCL[0])) return [-9999];
  let ndvi = (s.B08[0] - s.B04[0]) / (s.B08[0] + s.B04[0] + 1e-10);
  return [ndvi];
}
"""


def fetch_ndvi_for_field(
    boundary_gdf: gpd.GeoDataFrame,
    token: str,
    year: Optional[int] = None,
    spring_start: str = IOWA_SPRING_START,
    spring_end: str = IOWA_SPRING_END,
    resolution_m: int = TARGET_RESOLUTION_M,
) -> Tuple[np.ndarray, rasterio.Affine, Dict[str, Any]]:
    """
    Pull cloud-masked NDVI from Sentinel-2 L2A for a field boundary.

    Parameters
    ----------
    boundary_gdf  : Field boundary as GeoDataFrame (any CRS)
    token         : Bearer token from get_cdse_token()
    year          : Defaults to current year; falls back to previous year
                    if current year has no valid scenes yet.
    spring_start  : MM-DD string for window start (default "03-01")
    spring_end    : MM-DD string for window end   (default "04-20")
    resolution_m  : Output pixel size in metres (default 10)

    Returns
    -------
    (ndvi_array, affine_transform, rasterio_profile)
    ndvi_array values in [-1, 1]; nodata = -9999
    """
    if year is None:
        year = datetime.now().year

    boundary_ll = boundary_gdf.to_crs("EPSG:4326")
    bounds = boundary_ll.total_bounds  # [minx, miny, maxx, maxy]

    # Estimate output image size (cap at 2500 px per side for free tier)
    width_deg  = bounds[2] - bounds[0]
    height_deg = bounds[3] - bounds[1]
    # ~111 320 m per degree latitude
    width_m  = width_deg  * 111_320 * np.cos(np.radians((bounds[1] + bounds[3]) / 2))
    height_m = height_deg * 111_320
    px_w = min(int(width_m  / resolution_m) + 1, 2500)
    px_h = min(int(height_m / resolution_m) + 1, 2500)

    payload = {
        "input": {
            "bounds": {
                "bbox": [bounds[0], bounds[1], bounds[2], bounds[3]],
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [
                {
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{year}-{spring_start}T00:00:00Z",
                            "to":   f"{year}-{spring_end}T23:59:59Z",
                        },
                        "mosaickingOrder": "leastCC",   # least cloud cover composite
                        "maxCloudCoverage": MAX_CLOUD_PCT,
                    },
                    "type": "sentinel-2-l2a",
                }
            ],
        },
        "output": {
            "width":  px_w,
            "height": px_h,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": NDVI_EVALSCRIPT,
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "image/tiff",
    }

    resp = requests.post(CDSE_PROCESS_URL, json=payload, headers=headers, timeout=120)

    # If current year has no data yet, fall back to previous year automatically
    if resp.status_code == 200 and len(resp.content) < 1000:
        year -= 1
        payload["input"]["data"][0]["dataFilter"]["timeRange"]["from"] = f"{year}-{spring_start}T00:00:00Z"
        payload["input"]["data"][0]["dataFilter"]["timeRange"]["to"]   = f"{year}-{spring_end}T23:59:59Z"
        resp = requests.post(CDSE_PROCESS_URL, json=payload, headers=headers, timeout=120)

    if resp.status_code != 200:
        raise RuntimeError(
            f"CDSE Process API error ({resp.status_code}): {resp.text[:500]}"
        )

    # Parse the returned GeoTIFF from memory
    with MemoryFile(resp.content) as memfile:
        with memfile.open() as dataset:
            ndvi_array = dataset.read(1).astype(np.float32)
            transform  = dataset.transform
            profile    = dataset.profile.copy()

    profile.update({"nodata": -9999.0, "dtype": "float32"})

    # Mask fill / nodata
    ndvi_array[ndvi_array == -9999.0] = np.nan

    return ndvi_array, transform, profile


# ---------------------------------------------------------------------------
# Streamlit helper — handles auth via st.secrets
# ---------------------------------------------------------------------------

def get_token_from_streamlit_secrets() -> str:
    """
    Pull CDSE credentials from Streamlit secrets.toml and return a token.

    Add this to your .streamlit/secrets.toml:
        [cdse]
        client_id     = "your-client-id"
        client_secret = "your-client-secret"
    """
    try:
        import streamlit as st
        cid = st.secrets["cdse"]["client_id"]
        csecret = st.secrets["cdse"]["client_secret"]
    except (KeyError, FileNotFoundError) as exc:
        raise RuntimeError(
            "CDSE credentials not found in st.secrets. "
            "Add [cdse] client_id and client_secret to .streamlit/secrets.toml"
        ) from exc
    return get_cdse_token(cid, csecret)


# ---------------------------------------------------------------------------
# Quick test (run directly: python sentinel_utils.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import json

    client_id     = os.environ.get("CDSE_CLIENT_ID", "")
    client_secret = os.environ.get("CDSE_CLIENT_SECRET", "")

    if not client_id:
        print("Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET env vars to test.")
    else:
        # Shelby County IA test boundary (approximate)
        shelby_geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-95.42, 41.52], [-95.40, 41.52],
                        [-95.40, 41.54], [-95.42, 41.54],
                        [-95.42, 41.52]
                    ]]
                },
                "properties": {}
            }]
        }
        gdf = gpd.GeoDataFrame.from_features(shelby_geojson["features"], crs="EPSG:4326")
        token = get_cdse_token(client_id, client_secret)
        print("Token obtained.")
        ndvi, tfm, prof = fetch_ndvi_for_field(gdf, token, year=2025)
        valid = ndvi[~np.isnan(ndvi)]
        print(f"NDVI shape: {ndvi.shape}  mean: {valid.mean():.3f}  min: {valid.min():.3f}  max: {valid.max():.3f}")

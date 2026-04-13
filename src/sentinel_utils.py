"""
sentinel_utils.py
-----------------
Automated Sentinel-2 NDVI retrieval using the official sentinelhub-py library.
This replaces the manual API request approach which had formatting issues.

Library handles:
- OAuth2 authentication and token refresh automatically
- Correct request formatting for CDSE Process API
- Response validation and GeoTIFF parsing
- Cloud masking via SCL band

Setup:
    pip install sentinelhub

Credentials in .streamlit/secrets.toml:
    [cdse]
    client_id     = "your-shapps-client-id"
    client_secret = "your-shapps-client-secret"

Account requirements:
    - Register at dataspace.copernicus.eu
    - Create OAuth client at shapps.dataspace.copernicus.eu
    - Free tier: 30,000 processing units/month
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import numpy as np
import geopandas as gpd

_SENTINELHUB_IMPORT_ERROR = None
try:
    from sentinelhub import (
        SHConfig,
        SentinelHubRequest,
        DataCollection,
        MimeType,
        BBox,
        CRS,
        bbox_to_dimensions,
        MosaickingOrder,
    )
    SENTINELHUB_AVAILABLE = True
except Exception as _e:
    SENTINELHUB_AVAILABLE = False
    _SENTINELHUB_IMPORT_ERROR = str(_e)

# ---------------------------------------------------------------------------
# Iowa spring window defaults
# ---------------------------------------------------------------------------
IOWA_SPRING_START_MONTH = 3   # March
IOWA_SPRING_END_MONTH   = 4   # April
MAX_CLOUD_PCT = 80
TARGET_RESOLUTION_M = 10

# ---------------------------------------------------------------------------
# Cloud-masked NDVI evalscript — SCL uses DN units (correct format)
# ---------------------------------------------------------------------------
# Raw NDVI evalscript — no cloud masking
# Cloud masking via SCL was rejecting all Iowa spring pixels
# Use mosaickingOrder=LEAST_CC instead to get clearest available scene
NDVI_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B08"] }],
    output: { bands: 1, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  var ndvi = (s.B08[0] - s.B04[0]) / (s.B08[0] + s.B04[0] + 1e-10);
  return [ndvi];
}
"""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_sh_config(client_id: str, client_secret: str) -> "SHConfig":
    """Build SentinelHub config pointed at CDSE endpoint."""
    config = SHConfig()
    config.sh_client_id     = client_id
    config.sh_client_secret = client_secret
    config.sh_base_url      = "https://sh.dataspace.copernicus.eu"
    config.sh_token_url     = (
        "https://identity.dataspace.copernicus.eu"
        "/auth/realms/CDSE/protocol/openid-connect/token"
    )
    return config


def get_config_from_streamlit_secrets() -> "SHConfig":
    """Load CDSE credentials from Streamlit secrets.toml."""
    try:
        import streamlit as st
        cid     = st.secrets["cdse"]["client_id"]
        csecret = st.secrets["cdse"]["client_secret"]
    except (KeyError, Exception) as exc:
        raise RuntimeError(
            "CDSE credentials not found in Streamlit secrets. "
            "Add [cdse] client_id and client_secret to .streamlit/secrets.toml"
        ) from exc
    return get_sh_config(cid, csecret)


# ---------------------------------------------------------------------------
# Main NDVI fetch
# ---------------------------------------------------------------------------

def fetch_ndvi_for_field(
    boundary_gdf: gpd.GeoDataFrame,
    config: "SHConfig",
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    resolution_m: int = TARGET_RESOLUTION_M,
) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
    """
    Pull cloud-masked Sentinel-2 NDVI for a field boundary.

    Parameters
    ----------
    boundary_gdf  : Field boundary GeoDataFrame (any CRS)
    config        : SHConfig with CDSE credentials
    date_from     : Start date (default: March 1 of current year)
    date_to       : End date (default: today)
    resolution_m  : Output resolution in meters (default 10)

    Returns
    -------
    (ndvi_array, affine_transform, profile_dict)
    ndvi_array: float32 array, NaN where cloud-masked, values in [-1, 1]
    """
    if not SENTINELHUB_AVAILABLE:
        raise RuntimeError(
            f"sentinelhub import failed: {_SENTINELHUB_IMPORT_ERROR}"
        )

    # Default to current spring window
    now = datetime.now()
    if date_from is None:
        date_from = datetime(now.year, IOWA_SPRING_START_MONTH, 1)
    if date_to is None:
        date_to = now

    # Convert boundary to WGS84 for SentinelHub BBox
    boundary_ll = boundary_gdf.to_crs("EPSG:4326")
    bounds = boundary_ll.total_bounds  # [minx, miny, maxx, maxy]

    bbox = BBox(
        bbox=[bounds[0], bounds[1], bounds[2], bounds[3]],
        crs=CRS.WGS84,
    )

    # Calculate pixel dimensions from bbox and resolution
    size = bbox_to_dimensions(bbox, resolution=resolution_m)
    # Cap at 2500px per side (free tier limit)
    max_px = 2500
    if size[0] > max_px or size[1] > max_px:
        scale = max(size[0], size[1]) / max_px
        size = (int(size[0] / scale), int(size[1] / scale))

    # Build request using sentinelhub-py
    request = SentinelHubRequest(
        evalscript=NDVI_EVALSCRIPT,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A.define_from(
                    "s2l2a",
                    service_url=config.sh_base_url,
                ),
                time_interval=(date_from, date_to),
                mosaicking_order=MosaickingOrder.LEAST_CC,
                other_args={
                    "dataFilter": {
                        "maxCloudCoverage": 100,
                    }
                },
            )
        ],
        responses=[
            SentinelHubRequest.output_response("default", MimeType.TIFF),
        ],
        bbox=bbox,
        size=size,
        config=config,
    )

    # Execute request — library handles auth, formatting, retries
    data = request.get_data()

    if not data or len(data) == 0:
        raise RuntimeError(
            "SentinelHub returned empty response. "
            "No valid scenes found in time window."
        )

    ndvi_raw = data[0].squeeze().astype(np.float32)

    # Convert nodata sentinel to NaN
    ndvi_raw[ndvi_raw == -9999.0] = np.nan

    # Build affine transform from bbox
    from rasterio.transform import from_bounds as rasterio_from_bounds
    import rasterio
    height, width = ndvi_raw.shape
    transform = rasterio_from_bounds(
        bounds[0], bounds[1], bounds[2], bounds[3],
        width, height
    )

    profile = {
        "driver":    "GTiff",
        "dtype":     "float32",
        "count":     1,
        "height":    height,
        "width":     width,
        "crs":       "EPSG:4326",
        "transform": transform,
        "nodata":    float("nan"),
    }

    return ndvi_raw, transform, profile


# ---------------------------------------------------------------------------
# Streamlit-ready wrapper
# ---------------------------------------------------------------------------

def fetch_ndvi_streamlit(
    boundary_gdf: gpd.GeoDataFrame,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> Tuple[np.ndarray, Any, Dict[str, Any], str]:
    """
    Fetch NDVI with Streamlit secrets auth. Returns (array, transform, profile, message).
    """
    config = get_config_from_streamlit_secrets()
    ndvi, transform, profile = fetch_ndvi_for_field(
        boundary_gdf=boundary_gdf,
        config=config,
        date_from=date_from,
        date_to=date_to,
    )
    valid = ndvi[~np.isnan(ndvi)]
    valid_pct = valid.size / ndvi.size * 100 if ndvi.size > 0 else 0
    message = (
        f"✅ Sentinel-2 NDVI pulled via sentinelhub-py | "
        f"{date_from.strftime('%b %d') if date_from else 'Spring'} – "
        f"{date_to.strftime('%b %d, %Y') if date_to else 'Today'} | "
        f"{valid_pct:.0f}% clear pixels"
    )
    return ndvi, transform, profile, message


# ---------------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import geopandas as gpd
    from shapely.geometry import box

    cid     = os.environ.get("CDSE_CLIENT_ID", "")
    csecret = os.environ.get("CDSE_CLIENT_SECRET", "")

    if not cid:
        print("Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET env vars to test.")
    else:
        config = get_sh_config(cid, csecret)
        # Iowa City — confirmed clear scenes June 2024
        field = gpd.GeoDataFrame(
            geometry=[box(-91.60, 41.63, -91.45, 41.73)],
            crs="EPSG:4326"
        )
        print("Fetching NDVI via sentinelhub-py...")
        ndvi, tfm, prof = fetch_ndvi_for_field(
            field, config,
            date_from=datetime(2024, 6, 24),
            date_to=datetime(2024, 6, 26),
        )
        valid = ndvi[~np.isnan(ndvi)]
        print(f"Shape:     {ndvi.shape}")
        print(f"Valid px:  {valid.size}/{ndvi.size} ({valid.size/ndvi.size*100:.0f}%)")
        if valid.size > 0:
            print(f"NDVI mean: {valid.mean():.3f}")
            print(f"NDVI min:  {valid.min():.3f}")
            print(f"NDVI max:  {valid.max():.3f}")
            print("SUCCESS")

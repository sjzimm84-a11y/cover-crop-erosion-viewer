"""
iowa_dem_utils.py
-----------------
Automatic Iowa 3m DEM retrieval via the Iowa Geospatial Data Clearinghouse
Web Coverage Service (WCS) endpoint.

Source: Iowa 3-Meter Digital Elevation Model
URL: https://geodata.iowa.gov
WCS Endpoint: https://geodata.iowa.gov/arcgis/services/DEM_3M/ImageServer/WCSServer

Fallback: user-uploaded DEM GeoTIFF if API is unavailable.

Usage:
    from src.iowa_dem_utils import fetch_iowa_dem

    dem_array, transform, profile = fetch_iowa_dem(boundary_gdf)
"""

import io
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds
import geopandas as gpd


# ---------------------------------------------------------------------------
# Iowa 3m DEM WCS endpoint
# ---------------------------------------------------------------------------
IOWA_WCS_URL = (
    "https://geodata.iowa.gov/arcgis/services/DEM_3M/ImageServer/WCSServer"
)

# Target output resolution in meters
TARGET_RESOLUTION_M = 3

# Buffer around field boundary (meters) to ensure full edge coverage
BOUNDARY_BUFFER_M = 50

# Request timeout seconds
REQUEST_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Core fetch function
# ---------------------------------------------------------------------------

def fetch_iowa_dem(
    boundary_gdf: gpd.GeoDataFrame,
    resolution_m: int = TARGET_RESOLUTION_M,
    buffer_m: int = BOUNDARY_BUFFER_M,
    timeout: int = REQUEST_TIMEOUT,
) -> Tuple[np.ndarray, rasterio.Affine, Dict[str, Any]]:
    """
    Fetch Iowa 3m DEM for a field boundary via WCS.

    Parameters
    ----------
    boundary_gdf  : Field boundary GeoDataFrame (any CRS)
    resolution_m  : Output resolution in meters (default 3)
    buffer_m      : Buffer around boundary in meters (default 50)
    timeout       : Request timeout in seconds (default 60)

    Returns
    -------
    (dem_array, affine_transform, rasterio_profile)
    dem_array values in meters (correctly scaled from source)

    Raises
    ------
    RuntimeError if WCS request fails
    """
    # Reproject to Iowa state plane (meters) for accurate buffering
    boundary_utm = boundary_gdf.to_crs("EPSG:26915")
    boundary_buffered = boundary_utm.buffer(buffer_m)
    bounds_utm = boundary_buffered.total_bounds  # [minx, miny, maxx, maxy]

    # Calculate output dimensions
    width_m  = bounds_utm[2] - bounds_utm[0]
    height_m = bounds_utm[3] - bounds_utm[1]
    px_w = max(int(width_m  / resolution_m), 32)
    px_h = max(int(height_m / resolution_m), 32)

    # Cap at 4000px to avoid timeout on very large fields
    if px_w > 4000 or px_h > 4000:
        scale = max(px_w, px_h) / 4000
        px_w = int(px_w / scale)
        px_h = int(px_h / scale)

    # Build WCS GetCoverage request
    # Iowa clearinghouse WCS uses EPSG:26915 (NAD83 UTM Zone 15N)
    params = {
        "SERVICE":    "WCS",
        "VERSION":    "1.0.0",
        "REQUEST":    "GetCoverage",
        "COVERAGE":   "DEM_3M",
        "CRS":        "EPSG:26915",
        "BBOX":       f"{bounds_utm[0]},{bounds_utm[1]},{bounds_utm[2]},{bounds_utm[3]}",
        "WIDTH":      str(px_w),
        "HEIGHT":     str(px_h),
        "FORMAT":     "GeoTIFF",
        "RESX":       str(resolution_m),
        "RESY":       str(resolution_m),
    }

    try:
        resp = requests.get(
            IOWA_WCS_URL,
            params=params,
            timeout=timeout,
            headers={"User-Agent": "CoverCropErosionViewer/1.0"},
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Iowa WCS request timed out after {timeout}s. "
            "Try uploading a DEM manually."
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot reach Iowa Geospatial Data Clearinghouse. "
            "Check internet connection or upload DEM manually."
        )

    if resp.status_code != 200:
        raise RuntimeError(
            f"Iowa WCS error ({resp.status_code}): {resp.text[:300]}"
        )

    # Verify response is a valid GeoTIFF not an XML error
    if resp.content[:4] != b'II*\x00' and resp.content[:4] != b'MM\x00*':
        # Not a TIFF — likely an XML error response
        raise RuntimeError(
            f"Iowa WCS returned non-TIFF response: {resp.text[:300]}"
        )

    # Parse GeoTIFF from memory
    with MemoryFile(resp.content) as memfile:
        with memfile.open() as dataset:
            dem_array = dataset.read(1).astype(np.float32)
            transform = dataset.transform
            profile   = dataset.profile.copy()

    # Iowa 3m DEM from clearinghouse is in meters — no unit conversion needed
    # Values typically 150-600m for Iowa elevations
    # Mask nodata
    nodata_val = profile.get("nodata", -9999)
    if nodata_val is not None:
        dem_array[dem_array == nodata_val] = np.nan

    profile.update({
        "dtype":  "float32",
        "nodata": np.nan,
        "crs":    "EPSG:26915",
    })

    return dem_array, transform, profile


# ---------------------------------------------------------------------------
# Streamlit-aware wrapper with fallback
# ---------------------------------------------------------------------------

def get_dem_with_fallback(
    boundary_gdf: gpd.GeoDataFrame,
    uploaded_dem_path: Optional[str] = None,
    sample_dem_path: Optional[str] = None,
) -> Tuple[np.ndarray, rasterio.Affine, Dict[str, Any], str]:
    """
    Try Iowa WCS first, fall back to uploaded file, then sample data.

    Returns
    -------
    (dem_array, transform, profile, source_label)
    source_label: one of "Iowa 3m WCS" | "Uploaded DEM" | "Sample DEM"
    """
    # Try Iowa WCS first
    try:
        dem_array, transform, profile = fetch_iowa_dem(boundary_gdf)
        return dem_array, transform, profile, "Iowa 3m WCS (auto)"
    except RuntimeError as wcs_err:
        wcs_error_msg = str(wcs_err)

    # Fall back to uploaded DEM
    if uploaded_dem_path is not None:
        try:
            from src.raster_utils import clip_raster_to_geometry
            dem_array, transform, profile = clip_raster_to_geometry(
                uploaded_dem_path, boundary_gdf
            )
            return dem_array, transform, profile, "Uploaded DEM"
        except Exception:
            pass

    # Fall back to sample DEM
    if sample_dem_path is not None:
        try:
            from src.raster_utils import clip_raster_to_geometry
            dem_array, transform, profile = clip_raster_to_geometry(
                sample_dem_path, boundary_gdf
            )
            return dem_array, transform, profile, "Sample DEM"
        except Exception:
            pass

    raise RuntimeError(
        f"All DEM sources failed.\n"
        f"Iowa WCS: {wcs_error_msg}\n"
        f"Please upload a DEM GeoTIFF manually."
    )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    # Shelby County IA test boundary
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-95.42, 41.52], [-95.40, 41.52],
                    [-95.40, 41.54], [-95.42, 41.54],
                    [-95.42, 41.52],
                ]]
            },
            "properties": {}
        }]
    }
    gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")

    print("Testing Iowa 3m WCS DEM fetch for Shelby County...")
    try:
        dem, tfm, prof = fetch_iowa_dem(gdf)
        valid = dem[~np.isnan(dem)]
        print(f"Shape:      {dem.shape}")
        print(f"Valid px:   {valid.size}/{dem.size}")
        print(f"Elev mean:  {valid.mean():.1f} m")
        print(f"Elev min:   {valid.min():.1f} m")
        print(f"Elev max:   {valid.max():.1f} m")
        print(f"CRS:        {prof['crs']}")
        print("SUCCESS — Iowa WCS DEM pipeline working!")
    except RuntimeError as e:
        print(f"FAILED: {e}")

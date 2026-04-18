"""
iowa_dem_utils.py
-----------------
Automatic Iowa 3m DEM retrieval via Iowa DNR ArcGIS ImageServer.

Confirmed endpoint (verified April 2026):
  https://programs.iowadnr.gov/geospatial/rest/services/Elevation/DEM_3M_I/ImageServer

Service details:
  - Pixel type: UInt16 (values in centimeters)
  - CRS: EPSG:26915 (NAD83 UTM Zone 15N)
  - Resolution: 3m native
  - Coverage: Full Iowa statewide
  - No authentication required

Unit conversion:
  Raw values ~10,000-50,000 = centimeters
  Divide by 100 to get meters (150-600m, correct for Iowa)
"""

import json
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
import geopandas as gpd


# ---------------------------------------------------------------------------
# Confirmed Iowa DNR ImageServer endpoint
# ---------------------------------------------------------------------------
IOWA_DEM_URL = (
    "https://programs.iowadnr.gov/geospatial/rest/services"
    "/Elevation/DEM_3M_I/ImageServer/exportImage"
)

TARGET_RESOLUTION_M = 3
BOUNDARY_BUFFER_M   = 50
REQUEST_TIMEOUT     = 60


def fetch_iowa_dem(
    boundary_gdf: gpd.GeoDataFrame,
    resolution_m: int = TARGET_RESOLUTION_M,
    buffer_m: int = BOUNDARY_BUFFER_M,
    timeout: int = REQUEST_TIMEOUT,
) -> Tuple[np.ndarray, rasterio.Affine, Dict[str, Any]]:
    """
    Fetch Iowa 3m DEM for a field boundary via Iowa DNR ImageServer.

    Returns
    -------
    (dem_array_meters, affine_transform, rasterio_profile)
    dem_array values are in METERS (converted from source centimeters)
    """
    # Reproject to EPSG:26915 for accurate buffering and bbox
    boundary_utm = boundary_gdf.to_crs("EPSG:26915")
    bounds = boundary_utm.buffer(buffer_m).total_bounds
    # [minx, miny, maxx, maxy]

    # Calculate output pixel dimensions
    width_m  = bounds[2] - bounds[0]
    height_m = bounds[3] - bounds[1]
    px_w = max(int(width_m  / resolution_m), 32)
    px_h = max(int(height_m / resolution_m), 32)

    # Cap at service max (15000 x 4100)
    if px_w > 15000:
        px_w = 15000
    if px_h > 4100:
        px_h = 4100

    # ArcGIS ImageServer exportImage parameters
    params = {
        "bbox":          f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}",
        "bboxSR":        "26915",
        "size":          f"{px_w},{px_h}",
        "imageSR":       "26915",
        "format":        "tiff",
        "pixelType":     "U16",
        "noData":        "0",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_BilinearInterpolation",
        "f":             "image",
    }

    try:
        resp = requests.get(
            IOWA_DEM_URL,
            params=params,
            timeout=timeout,
            headers={"User-Agent": "CoverMap/1.0"},
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Iowa DEM request timed out after {timeout}s. "
            "Upload a DEM manually."
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot reach Iowa DNR server. "
            "Check internet connection or upload DEM manually."
        )

    if resp.status_code != 200:
        raise RuntimeError(
            f"Iowa DEM server error ({resp.status_code}): {resp.text[:300]}"
        )

    # Verify GeoTIFF magic bytes
    if len(resp.content) < 100:
        raise RuntimeError(
            f"Iowa DEM returned too-small response ({len(resp.content)} bytes). "
            f"Content: {resp.text[:200]}"
        )

    is_tiff = (
        resp.content[:4] == b'II*\x00' or   # little-endian TIFF
        resp.content[:4] == b'MM\x00*'       # big-endian TIFF
    )
    if not is_tiff:
        raise RuntimeError(
            f"Iowa DEM returned non-TIFF: {resp.content[:100]}"
        )

    # Parse GeoTIFF
    with MemoryFile(resp.content) as memfile:
        with memfile.open() as dataset:
            dem_raw   = dataset.read(1).astype(np.float32)
            transform = dataset.transform
            profile   = dataset.profile.copy()

    # Mask nodata (0 values)
    dem_raw[dem_raw == 0] = np.nan

    # Convert centimeters → meters
    # Iowa 3m DEM stores UInt16 values in centimeters
    # Raw range ~10,000-50,000 cm = 100-500m elevation
    dem_meters = dem_raw / 100.0

    profile.update({
        "dtype":  "float32",
        "nodata": float("nan"),
        "crs":    "EPSG:26915",
        "height": dem_meters.shape[0],
        "width":  dem_meters.shape[1],
    })

    return dem_meters, transform, profile


def get_dem_with_fallback(
    boundary_gdf: gpd.GeoDataFrame,
    uploaded_dem_path: Optional[str] = None,
    sample_dem_path: Optional[str] = None,
) -> Tuple[np.ndarray, rasterio.Affine, Dict[str, Any], str]:
    """
    Try Iowa DNR WCS first, fall back to uploaded file, then sample data.

    Returns
    -------
    (dem_array_meters, transform, profile, source_label)
    """
    # Try Iowa DNR ImageServer
    try:
        dem_array, transform, profile = fetch_iowa_dem(boundary_gdf)
        valid = dem_array[~np.isnan(dem_array)]
        if valid.size == 0:
            raise RuntimeError("Iowa DEM returned all nodata pixels.")
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
        f"Iowa DNR: {wcs_error_msg}\n"
        f"Please upload a DEM GeoTIFF manually."
    )


if __name__ == "__main__":
    # Quick test — Shelby County IA
    import geopandas as gpd
    from shapely.geometry import box

    # ~1.5km field in Shelby County
    field = gpd.GeoDataFrame(
        geometry=[box(-95.42, 41.52, -95.405, 41.535)],
        crs="EPSG:4326"
    )

    print("Testing Iowa DNR 3m DEM fetch for Shelby County...")
    try:
        dem, tfm, prof = fetch_iowa_dem(field)
        valid = dem[~np.isnan(dem)]
        print(f"Shape:     {dem.shape}")
        print(f"Valid px:  {valid.size}/{dem.size} ({valid.size/dem.size*100:.0f}%)")
        print(f"Elev mean: {valid.mean():.1f} m")
        print(f"Elev min:  {valid.min():.1f} m")
        print(f"Elev max:  {valid.max():.1f} m")
        print("SUCCESS")
    except RuntimeError as e:
        print(f"FAILED: {e}")
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import geopandas as gpd


def clip_raster_to_geometry(
    raster_path: str,
    boundary: gpd.GeoDataFrame,
) -> Tuple[np.ndarray, rasterio.Affine, Dict[str, Any]]:
    with rasterio.open(raster_path) as src:
        if boundary.crs != src.crs:
            boundary = boundary.to_crs(src.crs)

        geometry = [mapping(boundary.unary_union)]
        out_image, out_transform = mask(src, geometry, crop=True, nodata=src.nodata)
	out_image = np.where(out_image == src.nodata, np.nan, out_image)
        profile = src.profile.copy()
        profile.update(
            {
                "height": out_image.shape[1],
                "width":  out_image.shape[2],
                "transform": out_transform,
            }
        )

    data = out_image[0]
    return data.astype(float), out_transform, profile


def _detect_elevation_units(dem_array: np.ndarray) -> str:
    """
    Detect whether DEM elevation values are in meters or millimeters.

    Iowa 3m DEM from Iowa Geospatial Data Clearinghouse uses UInt16
    with values typically in the range 30,000-50,000 which are
    millimeters (30m - 50m elevation = 300-500 feet, typical Iowa range).

    Real meter values for Iowa would be 200-600 (not 30,000+).
    """
    valid = dem_array[~np.isnan(dem_array)]
    if valid.size == 0:
        return "meters"
    mean_val = float(np.nanmean(valid))
    # Iowa elevations in meters: 150-600m
    # Iowa elevations in millimeters: 150,000-600,000 (too high for UInt16)
    # Iowa elevations in centimeters: 15,000-60,000 (matches observed 38,972-43,002)
    if mean_val > 10000:
        return "centimeters"   # divide by 100 to get meters
    elif mean_val > 1000:
        return "decimeters"    # divide by 10 to get meters
    else:
        return "meters"


def compute_slope_from_dem(
    dem_array: np.ndarray,
    transform: rasterio.Affine,
    elevation_units: str = "auto",
) -> np.ndarray:
    """
    Compute percent slope from a DEM array.

    Parameters
    ----------
    dem_array       : 2D elevation array
    transform       : Rasterio affine transform (pixel size in CRS units)
    elevation_units : "auto" | "meters" | "centimeters" | "millimeters"
                      "auto" detects from value range — important for
                      Iowa 3m DEM which stores values in centimeters (UInt16)

    Returns
    -------
    slope_pct : 2D array of slope in percent
    """
    x_res = abs(transform.a)  # pixel width in CRS units (meters for UTM)
    y_res = abs(transform.e)  # pixel height in CRS units
    if x_res == 0 or y_res == 0:
        raise ValueError("DEM transform has invalid pixel size.")

    # Convert elevation to meters if needed
    elev = dem_array.copy().astype(float)

    if elevation_units == "auto":
        elevation_units = _detect_elevation_units(elev)

    if elevation_units == "centimeters":
        elev = elev / 100.0
    elif elevation_units == "millimeters":
        elev = elev / 1000.0
    elif elevation_units == "decimeters":
        elev = elev / 10.0
    # "meters" needs no conversion

    # Nodata (0 for Iowa DEM) → NaN before gradient
    elev[elev == 0] = np.nan

    # Fill NaN edges with nearest valid value for gradient stability
    from scipy.ndimage import generic_filter
    nan_mask = np.isnan(elev)
    if nan_mask.any() and not nan_mask.all():
        # Simple fill: replace NaN with local mean of valid neighbors
        elev_filled = elev.copy()
        elev_filled[nan_mask] = np.nanmean(elev)
    else:
        elev_filled = elev

    dz_dx, dz_dy = np.gradient(elev_filled, x_res, y_res)
    slope_pct = np.hypot(dz_dx, dz_dy) * 100.0

    # Restore NaN where original nodata was
    slope_pct[nan_mask] = np.nan

    return slope_pct


def raster_stats(data: np.ndarray, nodata: Any = None) -> Dict[str, float]:
    values = np.array(data, dtype=float)
    if nodata is not None:
        values = values[values != nodata]
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}

    return {
        "mean":  float(np.nanmean(values)),
        "min":   float(np.nanmin(values)),
        "max":   float(np.nanmax(values)),
        "count": int(values.size),
    }


def zone_risk_summary(
    ndvi_array: np.ndarray,
    slope_array: np.ndarray,
    ndvi_threshold: float = 0.35,
    slope_threshold: float = 6.0,
) -> Any:
    category = np.full(ndvi_array.shape, "Normal", dtype=object)
    low_cover  = ndvi_array < ndvi_threshold
    steep_slope = slope_array > slope_threshold

    # Ignore NaN slope pixels
    valid_slope = ~np.isnan(slope_array)
    steep_slope = steep_slope & valid_slope

    category[low_cover  &  steep_slope] = "High concern"
    category[low_cover  & ~steep_slope] = "Low cover"
    category[~low_cover &  steep_slope] = "Steep slope"

    labels = ["High concern", "Low cover", "Steep slope", "Normal"]
    rows = []
    total_pixels = float(category.size)
    for label in labels:
        mask_data = category == label
        if not np.any(mask_data):
            continue
        ndvi_values  = ndvi_array[mask_data]
        slope_values = slope_array[mask_data]
        rows.append(
            {
                "zone":      label,
                "percent":   float(mask_data.sum() / total_pixels * 100.0),
                "ndvi_mean": float(np.nanmean(ndvi_values)),
                "slope_mean": float(np.nanmean(slope_values)),
            }
        )

    import pandas as pd
    return pd.DataFrame(rows)

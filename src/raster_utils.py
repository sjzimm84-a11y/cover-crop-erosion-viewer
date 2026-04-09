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
        profile = src.profile.copy()
        profile.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )

    data = out_image[0]
    return data.astype(float), out_transform, profile


def compute_slope_from_dem(dem_array: np.ndarray, transform: rasterio.Affine) -> np.ndarray:
    x_res = abs(transform.a)
    y_res = abs(transform.e)
    if x_res == 0 or y_res == 0:
        raise ValueError("DEM transform has invalid pixel size.")

    dz_dx, dz_dy = np.gradient(dem_array, x_res, y_res)
    slope_pct = np.hypot(dz_dx, dz_dy) * 100.0
    return slope_pct


def raster_stats(data: np.ndarray, nodata: Any = None) -> Dict[str, float]:
    values = np.array(data, dtype=float)
    if nodata is not None:
        values = values[values != nodata]
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}

    return {
        "mean": float(np.nanmean(values)),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "count": int(values.size),
    }


def zone_risk_summary(
    ndvi_array: np.ndarray,
    slope_array: np.ndarray,
    ndvi_threshold: float = 0.35,
    slope_threshold: float = 6.0,
) -> Any:
    category = np.full(ndvi_array.shape, "Normal", dtype=object)
    low_cover = ndvi_array < ndvi_threshold
    steep_slope = slope_array > slope_threshold
    category[np.logical_and(low_cover, steep_slope)] = "High concern"
    category[np.logical_and(low_cover, ~steep_slope)] = "Low cover"
    category[np.logical_and(~low_cover, steep_slope)] = "Steep slope"

    labels = ["High concern", "Low cover", "Steep slope", "Normal"]
    rows = []
    total_pixels = float(category.size)
    for label in labels:
        mask_data = category == label
        if not np.any(mask_data):
            continue
        ndvi_values = ndvi_array[mask_data]
        slope_values = slope_array[mask_data]
        rows.append(
            {
                "zone": label,
                "percent": float(mask_data.sum() / total_pixels * 100.0),
                "ndvi_mean": float(np.nanmean(ndvi_values)),
                "slope_mean": float(np.nanmean(slope_values)),
            }
        )

    import pandas as pd

    return pd.DataFrame(rows)

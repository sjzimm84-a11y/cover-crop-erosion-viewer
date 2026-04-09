from pathlib import Path
from typing import Dict

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon


def ensure_sample_data(data_dir: Path) -> Dict[str, str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_field = data_dir / "sample_field.geojson"
    sample_ndvi = data_dir / "sample_ndvi.tif"
    sample_dem = data_dir / "sample_dem.tif"

    if not sample_field.exists() or not sample_ndvi.exists() or not sample_dem.exists():
        _write_sample_field(sample_field)
        _write_sample_ndvi(sample_ndvi)
        _write_sample_dem(sample_dem)

    return {
        "field": str(sample_field),
        "ndvi": str(sample_ndvi),
        "dem": str(sample_dem),
    }


def _write_sample_field(sample_path: Path) -> None:
    field_polygon = Polygon(
        [
            (500000.0, 4500000.0),
            (500800.0, 4500000.0),
            (500800.0, 4500800.0),
            (500000.0, 4500800.0),
            (500000.0, 4500000.0),
        ]
    )
    field = gpd.GeoDataFrame(
        {"field_id": ["sample_field"]}, geometry=[field_polygon], crs="EPSG:32615"
    )
    field.to_file(sample_path, driver="GeoJSON")


def _write_sample_ndvi(sample_path: Path) -> None:
    width = 120
    height = 100
    values = np.linspace(0.2, 0.75, num=width * height, dtype=np.float32).reshape(height, width)
    transform = from_origin(500000.0, 4500800.0, 10.0, 10.0)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": height,
        "width": width,
        "crs": "EPSG:32615",
        "transform": transform,
        "nodata": -9999.0,
    }

    with rasterio.open(sample_path, "w", **profile) as dst:
        dst.write(values, 1)


def _write_sample_dem(sample_path: Path) -> None:
    width = 120
    height = 100
    base = np.linspace(220.0, 260.0, num=width, dtype=np.float32)
    elevation = np.broadcast_to(base, (height, width))
    elevation = elevation + np.linspace(0.0, 10.0, num=height, dtype=np.float32).reshape(height, 1)
    transform = from_origin(500000.0, 4500800.0, 10.0, 10.0)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": height,
        "width": width,
        "crs": "EPSG:32615",
        "transform": transform,
        "nodata": -9999.0,
    }

    with rasterio.open(sample_path, "w", **profile) as dst:
        dst.write(elevation, 1)

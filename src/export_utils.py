import io
import os
import tempfile
import zipfile
from typing import Optional

import numpy as np
import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes

_ZONE_META = {
    1: ("Low",      "< 0.3",   "#22C55E"),
    2: ("Moderate", "0.3–0.7", "#FACC15"),
    3: ("High",     "0.7–1.5", "#F97316"),
    4: ("Critical", "≥ 1.5",   "#EF4444"),
}


def export_risk_zones_shp(
    risk_zone_array: np.ndarray,
    transform,
    crs,
    field_name: str = "field",
) -> Optional[bytes]:
    """
    Vectorize the risk zone raster and return a zipped shapefile bundle.

    Returns zip bytes (shp + shx + dbf + prj), or None if the array contains
    no valid zone pixels.
    """
    if transform is None or crs is None:
        return None

    # rasterio.features.shapes requires an integer array and a uint8 mask
    zone_int = np.where(np.isnan(risk_zone_array), 0, risk_zone_array).astype(np.int32)
    valid_mask = (zone_int > 0).astype(np.uint8)

    if valid_mask.sum() == 0:
        return None

    records = []
    for geom_dict, val in shapes(zone_int, mask=valid_mask, transform=transform):
        val = int(val)
        if val not in _ZONE_META:
            continue
        label, cx_ls_range, color_hex = _ZONE_META[val]
        records.append({
            "geometry":    shape(geom_dict),
            "zone_val":    val,
            "zone_label":  label,
            "cx_ls_range": cx_ls_range,
            "color_hex":   color_hex,
        })

    if not records:
        return None

    gdf = gpd.GeoDataFrame(records, crs=crs)

    # Dissolve adjacent same-zone pixels into clean polygons
    gdf = (
        gdf.dissolve(by="zone_val", as_index=False)
        [["zone_val", "zone_label", "cx_ls_range", "color_hex", "geometry"]]
    )

    gdf = gdf.to_crs("EPSG:4326")

    stem = field_name.replace(" ", "_") or "field"
    with tempfile.TemporaryDirectory() as tmpdir:
        shp_path = os.path.join(tmpdir, f"{stem}_erosion_risk_zones.shp")
        gdf.to_file(shp_path, driver="ESRI Shapefile")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir(tmpdir):
                zf.write(os.path.join(tmpdir, fname), arcname=fname)
        buf.seek(0)
        return buf.read()

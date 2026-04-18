"""
wss_utils.py
------------
USDA Web Soil Survey (SSURGO) Soil Data Access API helpers.
Free public API — no authentication required.
"""

import requests
import geopandas as gpd
from typing import Dict, Any


def get_dominant_soil_series(boundary_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    """
    Query USDA Soil Data Access API for dominant soil series and K-factor
    within a field boundary.
    Returns dict with keys: series_name, k_factor, map_unit_name, pct_of_aoi
    """
    boundary_ll = boundary_gdf.to_crs("EPSG:4326")
    geom = boundary_ll.geometry.iloc[0]
    coords = list(geom.exterior.coords)
    coord_str = ",".join([f"{x} {y}" for x, y in coords])
    wkt_polygon = f"POLYGON(({coord_str}))"

    simple_query = f"""SELECT TOP 1
        mu.muname, c.compname, c.comppct_r,
        (SELECT TOP 1 kwfact FROM chorizon ch
         JOIN component c2 ON ch.cokey = c2.cokey
         WHERE c2.cokey = c.cokey
         AND ch.hzdept_r = 0) AS k_factor
    FROM mapunit mu
    INNER JOIN component c ON mu.mukey = c.mukey
    WHERE mu.mukey IN (
        SELECT * FROM SDA_Get_Mukey_from_intersection_with_WktWgs84(
            '{wkt_polygon}')
    )
    AND c.majcompflag = 'Yes'
    ORDER BY c.comppct_r DESC"""

    try:
        resp = requests.post(
            "https://SDMDataAccess.nrcs.usda.gov/Tabular/post.rest",
            data={
                "REQUEST": "query",
                "QUERY":   simple_query,
                "FORMAT":  "JSON+COLUMNNAME",
            },
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            rows = data.get("Table", [])
            if rows and len(rows) > 1:
                row = rows[1]  # row 0 is column headers
                return {
                    "map_unit_name": row[0] or "Unknown",
                    "series_name":   row[1] or "Unknown",
                    "pct_of_aoi":    row[2] or 0,
                    "k_factor":      row[3] or "N/A",
                }
    except Exception:
        pass

    return {
        "series_name":   "Not available",
        "k_factor":      None,
        "map_unit_name": "Not available",
        "pct_of_aoi":    0,
    }
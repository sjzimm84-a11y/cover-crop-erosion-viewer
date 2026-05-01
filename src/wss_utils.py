"""
wss_utils.py
------------
USDA Web Soil Survey (SSURGO) Soil Data Access API helpers.
Free public API — no authentication required.
"""

import requests
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from typing import Dict, Any


def get_dominant_soil_series(boundary_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    """
    Query USDA Soil Data Access API for dominant soil series and K-factor
    within a field boundary.
    Returns dict with keys: series_name, k_factor, map_unit_name, pct_of_aoi
    """
    boundary_ll = boundary_gdf.to_crs("EPSG:4326")
    # Safe MultiPolygon extraction — GeoJSON fix
    from shapely.ops import unary_union
    unified = unary_union(boundary_ll.geometry)
    if unified.geom_type == "MultiPolygon":
        geom = max(unified.geoms, key=lambda g: g.area)
    elif unified.geom_type == "Polygon":
        geom = unified
    else:
        geom = unified.convex_hull
    # Simplify geometry to reduce SSURGO query complexity
    # 0.0001 degrees ≈ 10m tolerance at Iowa latitude - appropriate for soil mapping
    simplified_geom = geom.simplify(0.0001, preserve_topology=True)

    # Fallback if simplification fails
    if simplified_geom.is_empty or not simplified_geom.is_valid:
        print("DEBUG: Simplification failed, using original geometry")
        simplified_geom = geom
    else:
        orig_vertices = len(list(geom.exterior.coords))
        new_vertices = len(list(simplified_geom.exterior.coords))
        print(f"DEBUG: Geometry simplified: {orig_vertices} → {new_vertices} vertices ({100*(orig_vertices-new_vertices)/orig_vertices:.1f}% reduction)")

    # Extract coordinates and ensure polygon closure
    coords = list(simplified_geom.exterior.coords)
    # SSURGO requires closed polygon (first = last coordinate)
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    # Create WKT with explicit lon,lat ordering
    coord_str = ",".join([f"{coord[0]} {coord[1]}" for coord in coords])
    wkt_polygon = f"POLYGON(({coord_str}))"

    # Debug output
    print(f"DEBUG: Polygon coords: {len(coords)} points")
    print(f"DEBUG: First coord: {coords[0]}, Last coord: {coords[-1]}")
    print(f"DEBUG: WKT sample: {wkt_polygon[:100]}...")

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

    print(f"DEBUG: About to call SSURGO API...")
    print(f"DEBUG: Query length: {len(simple_query)} chars")

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
        print(f"DEBUG: SSURGO API status: {resp.status_code}")
        print(f"DEBUG: Response length: {len(resp.text)} chars")

        if resp.status_code == 200:
            data = resp.json()
            print(f"DEBUG: JSON keys: {list(data.keys())}")
            rows = data.get("Table", [])
            print(f"DEBUG: Row count: {len(rows)}")
            if rows:
                print(f"DEBUG: First row (headers): {rows[0] if len(rows) > 0 else 'None'}")
                print(f"DEBUG: Second row (data): {rows[1] if len(rows) > 1 else 'None'}")

            if rows and len(rows) > 1:
                row = rows[1]  # row 0 is column headers
                return {
                    "map_unit_name": row[0] or "Unknown",
                    "series_name":   row[1] or "Unknown",
                    "pct_of_aoi":    row[2] or 0,
                    "k_factor":      row[3] or "N/A",
                }
    except Exception as e:
        print(f"DEBUG: SSURGO API Exception: {type(e).__name__}: {e}")
        print(f"DEBUG: Query that failed: {simple_query[:200]}...")

    return {
        "series_name":   "Not available",
        "k_factor":      None,
        "map_unit_name": "Not available",
        "pct_of_aoi":    0,
    }
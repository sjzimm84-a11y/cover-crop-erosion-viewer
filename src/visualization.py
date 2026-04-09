from pathlib import Path
import tempfile
from typing import Any, Tuple
import base64
from io import BytesIO

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import rasterio
from folium import GeoJson
from PIL import Image
from rasterio.warp import transform_bounds


def build_map_with_rasters(
    boundary: gpd.GeoDataFrame,
    ndvi_array: np.ndarray,
    slope_array: np.ndarray,
    transform: rasterio.Affine,
    raster_crs: Any,
    ndvi_opacity: float = 0.6,
    slope_opacity: float = 0.4,
) -> folium.Map:
    # Convert boundary to lat/lon
    boundary_ll = boundary.to_crs("EPSG:4326")
    bounds = boundary_ll.total_bounds  # [minx, miny, maxx, maxy]
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    # Create folium map
    m = folium.Map(location=center, zoom_start=14)

    # Add boundary
    geojson = boundary_ll.__geo_interface__
    GeoJson(geojson, style_function=lambda x: {'color': 'black', 'weight': 2, 'fillOpacity': 0.1}).add_to(m)

    # Get bounds for raster overlay in the raster CRS
    height, width = ndvi_array.shape
    raster_bounds = rasterio.transform.array_bounds(height, width, transform)
    raster_bounds_ll = transform_bounds(raster_crs, 'EPSG:4326', *raster_bounds)
    sw = [raster_bounds_ll[1], raster_bounds_ll[0]]
    ne = [raster_bounds_ll[3], raster_bounds_ll[2]]

    # Normalize NDVI for visualization using robust min/max.
    ndvi_valid = np.nan_to_num(ndvi_array, nan=0.0)
    ndvi_min = np.nanpercentile(ndvi_valid, 2)
    ndvi_max = np.nanpercentile(ndvi_valid, 98)
    ndvi_norm = (ndvi_valid - ndvi_min) / max(ndvi_max - ndvi_min, 1e-6)
    ndvi_norm = np.clip(ndvi_norm, 0, 1)
    ndvi_img = plt.cm.viridis(ndvi_norm)
    ndvi_img = (ndvi_img * 255).astype(np.uint8)
    ndvi_pil = Image.fromarray(ndvi_img)
    ndvi_buffer = BytesIO()
    ndvi_pil.save(ndvi_buffer, format='PNG')
    ndvi_data = base64.b64encode(ndvi_buffer.getvalue()).decode('utf-8')
    ndvi_url = f'data:image/png;base64,{ndvi_data}'

    # Normalize slope for visualization using the actual slope distribution.
    slope_valid = np.nan_to_num(slope_array, nan=0.0)
    slope_min = float(np.nanpercentile(slope_valid, 2))
    slope_max = float(np.nanpercentile(slope_valid, 98))
    if slope_max - slope_min < 1e-3:
        slope_max = slope_min + 1.0

    slope_norm = (slope_valid - slope_min) / (slope_max - slope_min)
    slope_norm = np.clip(slope_norm, 0, 1)
    slope_norm = np.sqrt(slope_norm)  # enhance contrast for low-to-moderate slope values

    slope_rgba = plt.cm.terrain(slope_norm)
    slope_rgba[..., :3] = (slope_rgba[..., :3] * 255).astype(np.uint8)
    slope_rgba[..., 3] = np.round(np.clip(slope_norm * 255.0, 0, 255)).astype(np.uint8)
    slope_pil = Image.fromarray(slope_rgba.astype(np.uint8), mode='RGBA')
    slope_buffer = BytesIO()
    slope_pil.save(slope_buffer, format='PNG')
    slope_data = base64.b64encode(slope_buffer.getvalue()).decode('utf-8')
    slope_url = f'data:image/png;base64,{slope_data}'

    # Add NDVI overlay
    folium.raster_layers.ImageOverlay(
        image=ndvi_url,
        bounds=[sw, ne],
        opacity=ndvi_opacity,
        name='NDVI'
    ).add_to(m)

    # Add slope overlay
    folium.raster_layers.ImageOverlay(
        image=slope_url,
        bounds=[sw, ne],
        opacity=slope_opacity,
        name='Slope'
    ).add_to(m)

    # Add layer control and fit the map to the overlay bounds
    folium.LayerControl().add_to(m)
    m.fit_bounds([sw, ne])

    return m


def build_zone_risk_chart(zone_summary: Any) -> Any:
    if zone_summary.empty:
        return px.bar(title="No zone risk categories were found.")

    return px.bar(
        zone_summary,
        x="zone",
        y="percent",
        color="zone",
        text="percent",
        title="Risk zone share by percentage",
        labels={"zone": "Zone", "percent": "Percent of field (%)"},
    ).update_layout(showlegend=False).update_traces(texttemplate='%{text:.1f}%', textposition='outside', marker_line_color='white', marker_line_width=1)

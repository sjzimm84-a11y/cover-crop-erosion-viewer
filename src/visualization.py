from typing import Any
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
    zoom_start: int = 15,
) -> folium.Map:
    boundary_ll = boundary.to_crs("EPSG:4326")
    bounds = boundary_ll.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )

    GeoJson(
        boundary_ll.__geo_interface__,
        style_function=lambda x: {
            "color": "#f0c040",
            "weight": 2.5,
            "fillOpacity": 0.0,
        },
    ).add_to(m)

    height, width = ndvi_array.shape
    raster_bounds = rasterio.transform.array_bounds(height, width, transform)
    raster_bounds_ll = transform_bounds(raster_crs, "EPSG:4326", *raster_bounds)
    sw = [raster_bounds_ll[1], raster_bounds_ll[0]]
    ne = [raster_bounds_ll[3], raster_bounds_ll[2]]

    # --- NDVI: RdYlGn (red=low cover, green=good) ---
    ndvi_clean = ndvi_array.copy().astype(float)
    ndvi_clean[ndvi_clean <= -9999] = np.nan
    valid_pixels = ndvi_clean[~np.isnan(ndvi_clean)]

    if valid_pixels.size > 0:
        p2 = float(np.percentile(valid_pixels, 2))
        p98 = float(np.percentile(valid_pixels, 98))
        stretch = max(p98 - p2, 0.01)
        p2 = p2 - (stretch * 0.1)
        ndvi_norm = np.where(
            np.isnan(ndvi_clean),
            np.nan,
            np.clip((ndvi_clean - p2) / stretch, 0.0, 1.0),
        )
        cmap_ndvi = plt.cm.viridis
        ndvi_rgba = cmap_ndvi(np.where(np.isnan(ndvi_norm), 0.0, ndvi_norm))
        ndvi_rgba[np.isnan(ndvi_norm), 3] = 0.0
        ndvi_rgba[~np.isnan(ndvi_norm), 3] = ndvi_opacity
        ndvi_img = (ndvi_rgba * 255).astype(np.uint8)
        ndvi_pil = Image.fromarray(ndvi_img, mode="RGBA")
    else:
        ndvi_pil = Image.fromarray(
            np.zeros((height, width, 4), dtype=np.uint8), mode="RGBA"
        )

    ndvi_buffer = BytesIO()
    ndvi_pil.save(ndvi_buffer, format="PNG")
    ndvi_url = "data:image/png;base64," + base64.b64encode(ndvi_buffer.getvalue()).decode()

    # --- Slope: RdYlBu reversed (dark red=steep, blue=flat) ---
    # Higher contrast than YlOrRd — agronomically steep slopes show as red
    slope_clean = slope_array.copy().astype(float)
    slope_clean[slope_clean <= -9999] = np.nan
    slope_valid = slope_clean[~np.isnan(slope_clean)]

    if slope_valid.size > 0:
        # Absolute NRCS slope thresholds for Iowa (percent):
        # 0-2% = flat (blue), 2-6% = moderate (yellow), 6-12% = steep (orange), 12%+ = critical (red)
        SLOPE_MIN = 0.0    # flat
        SLOPE_MAX = 15.0   # cap at 15% — anything steeper still shows max red
        slope_norm = np.where(
            np.isnan(slope_clean),
            np.nan,
            np.clip((slope_clean - SLOPE_MIN) / (SLOPE_MAX - SLOPE_MIN), 0.0, 1.0),
        )
        slope_norm_safe = np.where(np.isnan(slope_norm), 0.0, slope_norm)
        # RdYlBu_r: steep=dark red, moderate=yellow, flat=blue
        slope_rgba = plt.cm.RdYlBu_r(slope_norm_safe)
        slope_rgba[np.isnan(slope_clean), 3] = 0.0
        slope_rgba[~np.isnan(slope_clean), 3] = slope_opacity
        slope_img = (slope_rgba * 255).astype(np.uint8)
        slope_pil = Image.fromarray(slope_img, mode="RGBA")
    else:
        slope_pil = Image.fromarray(
            np.zeros((height, width, 4), dtype=np.uint8), mode="RGBA"
        )

    slope_buffer = BytesIO()
    slope_pil.save(slope_buffer, format="PNG")
    slope_url = "data:image/png;base64," + base64.b64encode(slope_buffer.getvalue()).decode()

    folium.raster_layers.ImageOverlay(
        image=ndvi_url, bounds=[sw, ne], opacity=1.0,
        name="NDVI (red=low cover, green=good)", show=True,
    ).add_to(m)

    folium.raster_layers.ImageOverlay(
        image=slope_url, bounds=[sw, ne], opacity=1.0,
        name="Slope (red=steep, blue=flat)", show=True,
    ).add_to(m)

    colorbar_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
        background:rgba(14,17,23,0.88);padding:12px 16px;
        border-radius:8px;border:1px solid #30363d;
        font-family:monospace;font-size:12px;color:#c9d1d9;">
        <b style="color:#79c0ff;">NDVI Cover Quality</b><br>
        <span style="color:#440154;">&#9632;</span> Low cover — reseed target<br>
        <span style="color:#31688e;">&#9632;</span> Marginal stand<br>
        <span style="color:#fde725;">&#9632;</span> Good cover<br>
        <hr style="border-color:#30363d;margin:6px 0;">
        <b style="color:#79c0ff;">Slope</b><br>
        <span style="color:#d73027;">&#9632;</span> Steep &nbsp;
        <span style="color:#ffffbf;">&#9632;</span> Moderate &nbsp;
        <span style="color:#4575b4;">&#9632;</span> Flat
    </div>
    """
    m.get_root().html.add_child(folium.Element(colorbar_html))
    folium.LayerControl().add_to(m)
    m.fit_bounds([sw, ne])
    return m


def build_zone_risk_chart(zone_summary: Any) -> Any:
    if zone_summary.empty:
        return px.bar(title="No zone risk categories found.")

    color_map = {
        "High concern": "#cf222e",
        "Low cover":    "#9a6700",
        "Steep slope":  "#0550ae",
        "Normal":       "#1a7f37",
    }
    fig = px.bar(
        zone_summary,
        x="zone",
        y="percent",
        color="zone",
        color_discrete_map=color_map,
        text="percent",
        title="Risk Zone Distribution",
        labels={"zone": "Zone", "percent": "% of Field"},
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        marker_line_color="#30363d",
        marker_line_width=1,
    )
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#c9d1d9",
    )
    return fig
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
    ndvi_threshold: float = 0.20,
    risk_zone_array: np.ndarray = None,
) -> folium.Map:
    # Expose threshold to colormap logic below
    ndvi_opacity_threshold = ndvi_threshold
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

    # --- NDVI: 3-class discrete colormap tied to threshold ---
    # Orange=low cover, Blue=marginal, Yellow=good
    # Colors chosen for red-green colorblind accessibility
    COLOR_LOW      = np.array([249, 115,  22, 255], dtype=np.uint8)  # #F97316 orange
    COLOR_MARGINAL = np.array([ 56, 189, 248, 255], dtype=np.uint8)  # #38BDF8 steel blue
    COLOR_GOOD     = np.array([250, 204,  21, 255], dtype=np.uint8)  # #FACC15 bright yellow
    COLOR_NODATA   = np.array([  0,   0,   0,   0], dtype=np.uint8)  # transparent

    ndvi_clean = ndvi_array.copy().astype(float)
    ndvi_clean[ndvi_clean <= -9999] = np.nan
    valid_pixels = ndvi_clean[~np.isnan(ndvi_clean)]

    ndvi_img = np.zeros((height, width, 4), dtype=np.uint8)

    if valid_pixels.size > 0:
        # Marginal band = threshold to threshold+0.15
        marginal_upper = ndvi_opacity_threshold + 0.15

        low_mask      = (~np.isnan(ndvi_clean)) & (ndvi_clean < ndvi_opacity_threshold)
        marginal_mask = (~np.isnan(ndvi_clean)) & (ndvi_clean >= ndvi_opacity_threshold) & (ndvi_clean < marginal_upper)
        good_mask     = (~np.isnan(ndvi_clean)) & (ndvi_clean >= marginal_upper)
        nodata_mask   = np.isnan(ndvi_clean)

        ndvi_img[low_mask]      = COLOR_LOW
        ndvi_img[marginal_mask] = COLOR_MARGINAL
        ndvi_img[good_mask]     = COLOR_GOOD
        ndvi_img[nodata_mask]   = COLOR_NODATA

        # Apply opacity to alpha channel only
        ndvi_img[low_mask,      3] = int(ndvi_opacity * 255)
        ndvi_img[marginal_mask, 3] = int(ndvi_opacity * 255)
        ndvi_img[good_mask,     3] = int(ndvi_opacity * 255)

    ndvi_pil = Image.fromarray(ndvi_img, mode="RGBA")

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

    # --- Risk Index Zones layer (optional) ---
    if risk_zone_array is not None:
        ZONE_COLORS = {
            1: np.array([ 34, 197,  94, 255], dtype=np.uint8),  # #22C55E green  — Low
            2: np.array([250, 204,  21, 255], dtype=np.uint8),  # #FACC15 yellow — Moderate
            3: np.array([249, 115,  22, 255], dtype=np.uint8),  # #F97316 orange — High
            4: np.array([239,  68,  68, 255], dtype=np.uint8),  # #EF4444 red    — Critical
        }
        rz_h, rz_w = risk_zone_array.shape
        zone_img = np.zeros((rz_h, rz_w, 4), dtype=np.uint8)
        for val, color in ZONE_COLORS.items():
            mask = risk_zone_array == val
            zone_img[mask] = color
            zone_img[mask, 3] = int(ndvi_opacity * 255)
        zone_pil = Image.fromarray(zone_img, mode="RGBA")
        zone_buf = BytesIO()
        zone_pil.save(zone_buf, format="PNG")
        zone_url = "data:image/png;base64," + base64.b64encode(zone_buf.getvalue()).decode()
        folium.raster_layers.ImageOverlay(
            image=zone_url, bounds=[sw, ne], opacity=1.0,
            name="Risk Index Zones (C\u00d7LS)", show=False,
        ).add_to(m)

    colorbar_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
        background:rgba(14,17,23,0.88);padding:12px 16px;
        border-radius:8px;border:1px solid #30363d;
        font-family:monospace;font-size:12px;color:#c9d1d9;">
        <b style="color:#79c0ff;">NDVI Cover Quality</b><br>
        <span style="color:#F97316;">&#9632;</span> Low cover — reseed target<br>
        <span style="color:#38BDF8;">&#9632;</span> Marginal stand<br>
        <span style="color:#FACC15;">&#9632;</span> Good cover<br>
        <hr style="border-color:#30363d;margin:6px 0;">
        <b style="color:#79c0ff;">Slope</b><br>
        <span style="color:#d73027;">&#9632;</span> Steep &nbsp;
        <span style="color:#ffffbf;">&#9632;</span> Moderate &nbsp;
        <span style="color:#4575b4;">&#9632;</span> Flat<br>
        <hr style="border-color:#30363d;margin:6px 0;">
        <b style="color:#79c0ff;">Risk Index Zones (C&times;LS)</b><br>
        <span style="color:#22C55E;">&#9632;</span> Low &nbsp;
        <span style="color:#FACC15;">&#9632;</span> Moderate &nbsp;
        <span style="color:#F97316;">&#9632;</span> High &nbsp;
        <span style="color:#EF4444;">&#9632;</span> Critical
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
        "Critical risk": "#EF4444",
        "High risk":     "#F97316",
        "Moderate risk": "#FACC15",
        "Low risk":      "#22C55E",
        # legacy labels (fallback)
        "High concern":  "#cf222e",
        "Low cover":     "#9a6700",
        "Steep slope":   "#0550ae",
        "Normal":        "#1a7f37",
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

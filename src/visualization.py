from typing import Any, Optional
import base64
import json
from io import BytesIO

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import rasterio
from branca.element import MacroElement, Template
from folium import GeoJson
from PIL import Image
from rasterio.warp import transform_bounds
from shapely.geometry import mapping

# Minimum polygon size (acres) for an at-a-glance text label. Polygons below
# this floor still render — we just suppress their label so slivers don't clutter
# the map. Consistent with the 0.5-ac overlap floor used by the flagged table.
_LABEL_MIN_ACRES = 0.5

# Neutral SSURGO style — the soil map-unit layer no longer implies erosion risk
# by color (T-based ramp removed). Uniform grey outline + faint grey fill so the
# polygons stay hoverable for the tooltip without reading as a risk signal.
_SOIL_NEUTRAL_FILL    = "#9aa0a6"
_SOIL_NEUTRAL_OUTLINE = "#5a6068"

# Shared legend rows (color swatches + labels). Single source of truth used both
# for the in-map Leaflet control (standalone/exported HTML) and the native
# Streamlit legend rendered beneath the map in the app (streamlit-folium strips
# the in-map control, so the native copy is what users actually see).
LEGEND_ROWS_HTML = """
        <b style="color:#79c0ff;">NDVI Cover Quality</b><br>
        <span style="color:#F97316;">&#9632;</span> Low Cover<br>
        <span style="color:#38BDF8;">&#9632;</span> Marginal<br>
        <span style="color:#FACC15;">&#9632;</span> Good Cover<br>
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
        <hr style="border-color:#30363d;margin:6px 0;">
        <b style="color:#79c0ff;">Flagged Soil (A vs T)</b><br>
        <span style="color:#EF9F27;">&#9632;</span> Exceeds (2&ndash;5&times;T) &nbsp;
        <span style="color:#D85A30;">&#9632;</span> Significantly (&gt;5&times;T)
    """


def _text_label_marker(lat: float, lon: float, text: str) -> folium.Marker:
    """A small DivIcon text marker centered on (lat, lon).

    Used as an at-a-glance identifier over a polygon's representative point. The
    text is high-contrast white with a dark halo so it reads on any basemap;
    ``icon_size=(0, 0)`` plus the centering transform anchors the text on the
    point rather than offsetting it.
    """
    html = (
        '<div style="font-family:monospace;font-size:11px;font-weight:600;'
        'color:#f5f7fa;white-space:nowrap;'
        'text-shadow:0 0 2px #000,0 0 2px #000,0 0 2px #000;'
        f'transform:translate(-50%,-50%);">{text}</div>'
    )
    return folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(html=html, icon_size=(0, 0), icon_anchor=(0, 0)),
    )


def _add_legend_control(m: folium.Map, inner_html: str) -> None:
    """Render the legend as a Leaflet control (bottom-left), not a fixed div.

    A ``position:fixed`` body-level div is fragile under streamlit-folium: the
    component iframe (or an ancestor that establishes a containing block) can
    clip or hide it regardless of z-index — which is why bumping z-index did not
    bring it back. A Leaflet control lives inside the map's own
    ``.leaflet-control-container``, so Leaflet positions and shows it reliably.
    ``disableClickPropagation`` stops clicks on the legend from zooming the map.
    """
    style = (
        "background:rgba(14,17,23,0.88);padding:12px 16px;border-radius:8px;"
        "border:1px solid #30363d;font-family:monospace;font-size:12px;"
        "color:#c9d1d9;line-height:1.45;"
    )
    macro = MacroElement()
    macro._name = "LegendControl"
    macro.inner_json = json.dumps(inner_html)   # JS string literal, safely escaped
    macro.style_json = json.dumps(style)
    macro._template = Template(
        """
        {% macro script(this, kwargs) %}
        var {{ this.get_name() }} = L.control({position: 'bottomleft'});
        {{ this.get_name() }}.onAdd = function (map) {
            var div = L.DomUtil.create('div');
            div.innerHTML = {{ this.inner_json }};
            div.style.cssText = {{ this.style_json }};
            L.DomEvent.disableClickPropagation(div);
            L.DomEvent.disableScrollPropagation(div);
            return div;
        };
        {{ this.get_name() }}.addTo({{ this._parent.get_name() }});
        {% endmacro %}
        """
    )
    macro.add_to(m)


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
    soil_polygons: Optional[list] = None,
    flagged_soil_polygons: Optional[list] = None,
) -> folium.Map:
    # Expose threshold to colormap logic below
    ndvi_opacity_threshold = ndvi_threshold
    boundary_ll = boundary.to_crs("EPSG:4326")
    bounds = boundary_ll.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        # Esri legacy Dark Gray Canvas: free, keyless XYZ service. Swapped in
        # after CARTO's "dark_matter" basemap began requiring an API key.
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
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

    ndvi_clean = ndvi_array.astype(float)
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
    slope_clean = slope_array.astype(float)
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

    # --- SSURGO soil map-unit layer (optional, OFF by default) ---
    # Added BEFORE the Risk Index overlay so it sits beneath it. Each polygon is
    # colored by ITS OWN per-mukey T on the blue→orange ramp (low T = orange =
    # attention). Geometries are WGS84 shapely; tooltip shows musym / T / K.
    if soil_polygons:
        _features = []
        _soil_labels = []   # (lat, lon, text) at each unit's representative point
        for _sp in soil_polygons:
            _geom = _sp.get("geometry")
            if _geom is None or _geom.is_empty:
                continue
            _t = _sp.get("t")
            _k = _sp.get("k")
            _musym = _sp.get("musym") or f"mukey {_sp.get('mukey')}"
            _features.append({
                "type": "Feature",
                "geometry": mapping(_geom),
                "properties": {
                    "musym": _musym,
                    "T": "n/a" if _t is None else round(float(_t), 1),
                    "K": "n/a" if _k is None else round(float(_k), 3),
                },
            })
            # musym-only label (one polygon per unit on this layer — no ambiguity);
            # suppress on slivers below the area floor.
            _ac = _sp.get("acres")
            if _ac is None or float(_ac) >= _LABEL_MIN_ACRES:
                _rp = _geom.representative_point()
                _soil_labels.append((_rp.y, _rp.x, _musym))
        if _features:
            _soil_group = folium.FeatureGroup(name="SSURGO Soil Map Units", show=False)
            folium.GeoJson(
                {"type": "FeatureCollection", "features": _features},
                style_function=lambda feat: {
                    "fillColor": _SOIL_NEUTRAL_FILL,
                    "color": _SOIL_NEUTRAL_OUTLINE,
                    "weight": 0.7,
                    "fillOpacity": 0.12,
                },
                highlight_function=lambda feat: {"weight": 2, "color": "#f0c040"},
                tooltip=folium.GeoJsonTooltip(
                    fields=["musym", "T", "K"],
                    aliases=["Map unit:", "T (t/ac/yr):", "K factor:"],
                ),
            ).add_to(_soil_group)
            for _lat, _lon, _txt in _soil_labels:
                _text_label_marker(_lat, _lon, _txt).add_to(_soil_group)
            _soil_group.add_to(m)

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

    # --- Flagged Soil (Exceeds Tolerance) outline layer (optional, OFF) ---
    # Added AFTER the Risk Index overlay so outlines sit ABOVE the colored zones.
    # Only the two highest severity tiers arrive here ("Near tolerable limit" is
    # excluded upstream). Outline-only (no fill). Two-tier styling on the existing
    # blue\u2192orange palette: "Exceeds tolerable limit" = orange #EF9F27, lighter +
    # dashed; "Significantly exceeds limit" = #D85A30, darker + thicker + solid so
    # it reads as more severe. Geometry is the WGS84 zone\u2229mukey intersection \u2014 the
    # actual overlap, not the soil's or zone's full extent.
    if flagged_soil_polygons:
        # Two severity tiers, now FILLED (not outline-only). The fill color +
        # intensity carries severity: "Significantly exceeds limit" = deeper
        # orange at higher opacity; "Exceeds tolerable limit" = lighter orange at
        # lower opacity. A thin dark border keeps each polygon defined.
        _SEV_STYLE = {
            "Significantly exceeds limit": {"fill": "#D85A30", "fillop": 0.60},
            "Exceeds tolerable limit":     {"fill": "#EF9F27", "fillop": 0.38},
        }
        _flag_feats = []
        _flag_labels = []   # (lat, lon, text) at each polygon's representative point
        for _fp in flagged_soil_polygons:
            _geom = _fp.get("geometry")
            if _geom is None or getattr(_geom, "is_empty", True):
                continue
            _sev   = _fp.get("severity")
            _style = _SEV_STYLE.get(_sev, _SEV_STYLE["Exceeds tolerable limit"])
            _aot   = _fp.get("a_over_t")
            _ac    = _fp.get("overlap_acres")
            _musym = _fp.get("musym") or "n/a"
            _aot_txt = None if _aot is None else f"{float(_aot):.1f}\u00d7"
            _flag_feats.append({
                "type": "Feature",
                "geometry": mapping(_geom),
                "properties": {
                    "musym":    _musym,
                    "severity": _sev or "n/a",
                    # Property KEY must be a valid JS identifier: folium may pick
                    # it as the style-switch feature identifier and emits
                    # `switch(feature.properties.<key>)`. A "/" parses as division
                    # ("A/T" -> A ÷ T -> "T is not defined"), aborting the whole
                    # map render. Display label stays "A/T:" via the alias below.
                    "A_T":      "n/a" if _aot_txt is None else _aot_txt,
                    "acres":    "n/a" if _ac is None else round(float(_ac), 2),
                    "_fill":    _style["fill"],
                    "_fillop":  _style["fillop"],
                },
            })
            # Label "{musym} {A/T}\u00d7" \u2014 the same musym can appear as multiple
            # polygons (one per risk-zone \u00d7 soil intersection), so appending A/T
            # disambiguates and matches the table's A/T column. Suppress on slivers.
            if _ac is None or float(_ac) >= _LABEL_MIN_ACRES:
                _label = _musym if _aot_txt is None else f"{_musym} {_aot_txt}"
                _rp = _geom.representative_point()
                _flag_labels.append((_rp.y, _rp.x, _label))
        if _flag_feats:
            _flag_group = folium.FeatureGroup(
                name="Flagged Soil (Exceeds Tolerance)", show=False,
            )
            folium.GeoJson(
                {"type": "FeatureCollection", "features": _flag_feats},
                style_function=lambda feat: {
                    "color":       "#1b1f24",
                    "weight":      0.8,
                    "fill":        True,
                    "fillColor":   feat["properties"]["_fill"],
                    "fillOpacity": feat["properties"]["_fillop"],
                },
                highlight_function=lambda feat: {"weight": 2.5, "color": "#f0c040"},
                tooltip=folium.GeoJsonTooltip(
                    fields=["musym", "severity", "A_T", "acres"],
                    aliases=["Map unit:", "Severity:", "A/T:", "Overlap (ac):"],
                ),
            ).add_to(_flag_group)
            for _lat, _lon, _txt in _flag_labels:
                _text_label_marker(_lat, _lon, _txt).add_to(_flag_group)
            _flag_group.add_to(m)

    # In-map legend (works for standalone/exported HTML). NOTE: streamlit-folium
    # strips custom Leaflet controls, so the app ALSO renders LEGEND_ROWS_HTML as
    # a native Streamlit block beneath the map — that is the copy users see in the
    # app. Both share LEGEND_ROWS_HTML so the content cannot drift.
    folium.LayerControl().add_to(m)
    _add_legend_control(m, LEGEND_ROWS_HTML)
    m.fit_bounds([sw, ne])
    return m


def build_zone_risk_chart(
    zone_summary: Any,
    ndvi_low_label:  str = "Low cover",
    ndvi_mid_label:  str = "Marginal",
    ndvi_good_label: str = "Good cover",
) -> Any:
    if zone_summary.empty:
        return px.bar(title="No zone risk categories found.")

    color_map = {
        # Dynamic NDVI zone labels (param-based, match _chart_label_map output in app.py)
        ndvi_low_label:   "#F97316",
        ndvi_mid_label:   "#38BDF8",
        ndvi_good_label:  "#FACC15",
        # Risk Index zone labels
        "Critical risk":  "#EF4444",
        "High risk":      "#F97316",
        "Moderate risk":  "#FACC15",
        "Low risk":       "#22C55E",
        # Raw NDVI zone keys from compute_ndvi_zone_summary (fallback)
        "Low cover":      "#F97316",
        "Marginal":       "#38BDF8",
        "Good cover":     "#FACC15",
        # Capitalized variants
        "Low Cover":      "#F97316",
        "Good Cover":     "#FACC15",
        # Legacy fallback labels
        "High concern":   "#cf222e",
        "Steep slope":    "#0550ae",
        "Normal":         "#1a7f37",
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


def build_yoy_ndvi_figure(yoy_rows: list):
    """
    Year-over-year early-season NDVI bar chart (Plotly figure).

    Pure figure builder — no Streamlit calls, no I/O. Figure construction
    matches the original inline app code (title updated for the Mar 15 –
    Apr 20 window); the producer report reuses the same rows for its
    matplotlib PNG replica (kaleido is not a project dependency, so the
    Plotly figure itself cannot be exported to PNG).

    yoy_rows: [{"Year": 2023, "Mean NDVI": 0.300}, ...]
    """
    yoy_df = pd.DataFrame(yoy_rows)
    fig_yoy = px.bar(
        yoy_df, x="Year", y="Mean NDVI",
        title="Early-Season NDVI Trend (Mar 15–Apr 20)",
        color="Mean NDVI",
        color_continuous_scale="RdYlGn",
        text="Mean NDVI",
    )
    fig_yoy.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#c9d1d9",
    )
    return fig_yoy

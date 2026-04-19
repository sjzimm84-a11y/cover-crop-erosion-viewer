from pathlib import Path
import tempfile
from typing import Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import folium

import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import array_bounds

from src.io_utils import load_boundary_file, save_uploaded_file
from src.raster_utils import (
    clip_raster_to_geometry,
    compute_slope_from_dem,
    raster_stats,
    zone_risk_summary,
)
from src.sample_data import ensure_sample_data
from src.scoring import (
    DEFAULT_THRESHOLDS,
    score_erosion_concern,
    pixel_level_concern,
    pixel_risk_index,
    classify_risk_zones,
    RESIDUE_OPTIONS,
)
from src.visualization import build_map_with_rasters, build_zone_risk_chart
from src.report_generator import generate_field_report
from src.iowa_dem_utils import get_dem_with_fallback

# GEE NDVI imports — graceful fallback if not configured
SENTINEL_AVAILABLE = False
SENTINEL_IMPORT_ERROR = None
try:
    from src.gee_ndvi_utils import (
        init_gee_from_streamlit_secrets,
        fetch_ndvi_for_field as gee_fetch_ndvi,
        fetch_ndvi_streamlit,
    )
    SENTINEL_AVAILABLE = True
except Exception as _sentinel_exc:
    SENTINEL_IMPORT_ERROR = str(_sentinel_exc)

APP_TITLE       = "CoverMap"
APP_DESCRIPTION = (
    "Field-scale cover crop stand assessment using Sentinel-2 NDVI and terrain slope. "
    "Iowa RUSLE C-factor scoring · Automated satellite pull · CCA-ready documentation."
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — professional ag-tech aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0e1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
    }

    /* Concern level badges */
    .badge-low      { background:#1a7f37; color:white; padding:4px 12px; border-radius:12px; font-weight:600; }
    .badge-moderate { background:#9a6700; color:white; padding:4px 12px; border-radius:12px; font-weight:600; }
    .badge-high     { background:#cf222e; color:white; padding:4px 12px; border-radius:12px; font-weight:600; }
    .badge-critical { background:#6e1c1c; color:white; padding:4px 12px; border-radius:12px; font-weight:600; }

    /* Section headers */
    h2 { color: #58a6ff !important; border-bottom: 1px solid #30363d; padding-bottom: 6px; }
    h3 { color: #79c0ff !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("## 🌾")
with col_title:
    st.title(APP_TITLE)
    st.caption(APP_DESCRIPTION)

st.divider()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "ndvi_source" not in st.session_state:
    st.session_state.ndvi_source = "auto"   # "auto" | "upload" | "sample"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📍 Field Setup")

    residue_system = st.selectbox(
        "Previous crop and tillage system",
        options=RESIDUE_OPTIONS,
        index=4,
        help=(
            "Select the previous crop and tillage system. "
            "This adjusts the C-factor to account for residue "
            "protection not captured by satellite NDVI. "
            "Defaults to conservative (no adjustment) if unknown."
        ),
    )

    boundary_file = st.file_uploader(
        "Upload field boundary",
        type=["geojson", "json", "zip", "kml"],
        help="GeoJSON, zipped shapefile, or KML accepted.",
    )

    st.divider()
    st.markdown("### 🛰️ NDVI Source")

    ndvi_mode = st.radio(
        "How to get NDVI?",
        options=["Auto (Sentinel-2 API)", "Upload GeoTIFF", "Use sample data"],
        index=0,
    )

    # Date window selector — only shown in Auto mode
    if ndvi_mode == "Auto (Sentinel-2 API)":
        st.session_state.ndvi_source = "auto"
        ndvi_window = st.selectbox(
            "NDVI time window",
            options=["Last 7 days", "Last 14 days", "Last 30 days", "Custom spring window"],
            index=0,
        )
        if ndvi_window == "Custom spring window":
            ndvi_year = st.slider("Year", 2021, datetime.now().year, datetime.now().year)
            ndvi_start = st.text_input("Start (MM-DD)", value="03-01")
            ndvi_end   = st.text_input("End (MM-DD)",   value="04-20")
        else:
            ndvi_year  = None
            ndvi_start = None
            ndvi_end   = None

        # Year-over-year comparison
        yoy_compare = st.checkbox("📈 Year-over-year comparison (2023–present)")

    elif ndvi_mode == "Upload GeoTIFF":
        st.session_state.ndvi_source = "upload"
        ndvi_file = st.file_uploader("Upload NDVI GeoTIFF", type=["tif", "tiff"])
    else:
        st.session_state.ndvi_source = "sample"
        ndvi_file = None

    st.divider()
    st.markdown("### 🏔️ DEM")
    dem_file = st.file_uploader(
        "Upload DEM GeoTIFF (optional)",
        type=["tif", "tiff", "img"],
        help="Leave blank to use the 3m Iowa sample DEM.",
    )

    st.divider()
    st.markdown("### ⚙️ Thresholds")
    ndvi_threshold  = st.slider("Low cover NDVI", 0.0, 1.0, 0.20, 0.01)
    slope_threshold = st.slider(
        "Steep slope — Iowa HEL threshold (%)",
        0.0, 30.0, 9.0, 0.5,
        help="Iowa NRCS HEL threshold for Shelby County Monona-Nira-Ida soils. "
             "Adjust based on local soil survey data.",
    )

    st.divider()
    st.markdown("### 🗺️ Map Overlays")
    ndvi_opacity  = st.slider("NDVI opacity",  0.0, 1.0, 0.8, 0.1)
    slope_opacity = st.slider("Slope opacity", 0.0, 1.0, 0.1, 0.1)

# ---------------------------------------------------------------------------
# Temp dir and sample data
# ---------------------------------------------------------------------------
temp_dir     = Path(tempfile.mkdtemp())
data_dir     = Path(__file__).parent / "data"
sample_paths = ensure_sample_data(data_dir)

boundary_path = None
ndvi_path     = None
dem_path      = None
ndvi_array    = None
ndvi_transform = None
ndvi_profile   = None

# ---------------------------------------------------------------------------
# Boundary resolution
# ---------------------------------------------------------------------------
if boundary_file is not None:
    boundary_path = save_uploaded_file(boundary_file, temp_dir)
else:
    boundary_path = sample_paths["field"]

# DEM resolution — uploaded file path (used as fallback if WCS fails)
dem_path = save_uploaded_file(dem_file, temp_dir) if dem_file else None

# ---------------------------------------------------------------------------
# Load boundary
# ---------------------------------------------------------------------------
progress = st.progress(0)
status   = st.empty()

try:
    status.text("Loading field boundary...")
    progress.progress(10)
    field_boundary = load_boundary_file(boundary_path)
except Exception as exc:
    st.error(f"Could not load field boundary: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# NDVI acquisition — Auto (Sentinel-2) or Upload or Sample
# ---------------------------------------------------------------------------
status.text("Acquiring NDVI data...")
progress.progress(25)

if ndvi_mode == "Auto (Sentinel-2 API)":
    if not SENTINEL_AVAILABLE:
        st.warning(
            f"⚠️ GEE module not loaded. Error: {SENTINEL_IMPORT_ERROR}"
        )
        ndvi_path = sample_paths["ndvi"]
    else:
        try:
            from datetime import datetime as _dt, timedelta as _td

            # Initialize GEE authentication
            init_gee_from_streamlit_secrets()

            if ndvi_window == "Custom spring window":
                date_from = _dt(ndvi_year, int(ndvi_start.split('-')[0]), int(ndvi_start.split('-')[1]))
                date_to   = _dt(ndvi_year, int(ndvi_end.split('-')[0]),   int(ndvi_end.split('-')[1]))
            else:
                days_map  = {"Last 7 days": 7, "Last 14 days": 14, "Last 30 days": 30}
                days_back = days_map.get(ndvi_window, 7)
                date_to   = _dt.now()
                date_from = date_to - _td(days=days_back)

            ndvi_array, ndvi_transform, ndvi_profile, ndvi_msg, scene_meta, ndvi_warning = fetch_ndvi_streamlit(
                boundary_gdf=field_boundary,
                date_from=date_from,
                date_to=date_to,
            )
            st.success(ndvi_msg)
            if ndvi_warning:
                st.warning(ndvi_warning)

            # Store actual scene acquisition dates (not just the query window)
            latest_scene  = scene_meta.get("latest_date")
            earliest_scene = scene_meta.get("earliest_date")
            scene_count   = scene_meta.get("count", 0)
            if latest_scene:
                st.session_state.ndvi_scene_latest  = latest_scene.strftime("%b %d, %Y")
                st.session_state.ndvi_scene_earliest = (
                    earliest_scene.strftime("%b %d, %Y") if earliest_scene else None
                )
                st.session_state.ndvi_scene_count = scene_count
            else:
                st.session_state.ndvi_scene_latest   = None
                st.session_state.ndvi_scene_earliest = None
                st.session_state.ndvi_scene_count    = scene_count
            # Keep query-window dates for the PDF report
            st.session_state.ndvi_date_from = date_from.strftime("%b %d, %Y")
            st.session_state.ndvi_date_to   = date_to.strftime("%b %d, %Y")

            # Year-over-year comparison chart
            if yoy_compare:
                with st.spinner("Pulling year-over-year NDVI via GEE (30-60s)..."):
                    current_year = _dt.now().year
                    yoy_rows = []
                    for yr in range(2023, current_year + 1):
                        try:
                            arr, _, _, _ = gee_fetch_ndvi(
                                field_boundary,
                                date_from=_dt(yr, 3, 1),
                                date_to=_dt(yr, 4, 30),
                            )
                            valid = arr[~np.isnan(arr)]
                            if valid.size > 0:
                                yoy_rows.append({"Year": yr, "Mean NDVI": round(float(valid.mean()), 3)})
                        except Exception:
                            pass
                    if yoy_rows:
                        yoy_df = pd.DataFrame(yoy_rows)
                        fig_yoy = px.bar(
                            yoy_df, x="Year", y="Mean NDVI",
                            title="Early-Season NDVI Trend (March–April)",
                            color="Mean NDVI",
                            color_continuous_scale="RdYlGn",
                            text="Mean NDVI",
                        )
                        fig_yoy.update_layout(
                            plot_bgcolor="#0e1117",
                            paper_bgcolor="#0e1117",
                            font_color="#c9d1d9",
                        )
                        st.subheader("📈 Year-over-Year NDVI Trend")
                        st.plotly_chart(fig_yoy, width='stretch')

        except Exception as exc:
            st.warning(f"GEE NDVI unavailable: {exc}. Falling back to sample NDVI.")
            ndvi_path = sample_paths["ndvi"]

else:
    # Upload or sample mode
    if ndvi_mode == "Upload GeoTIFF" and ndvi_file is not None:
        ndvi_path = save_uploaded_file(ndvi_file, temp_dir)
    else:
        ndvi_path = sample_paths["ndvi"]

# ---------------------------------------------------------------------------
# Clip rasters
# ---------------------------------------------------------------------------
try:
    status.text("Clipping NDVI raster...")
    progress.progress(40)

    # If ndvi_array already populated from API, skip file clip
    if ndvi_array is None:
        ndvi_array, ndvi_transform, ndvi_profile = clip_raster_to_geometry(
            ndvi_path, field_boundary
        )
except Exception as exc:
    st.error(f"Could not clip NDVI: {exc}")
    st.stop()

try:
    status.text("Fetching DEM (Iowa 3m WCS)...")
    progress.progress(55)
    dem_array, dem_transform, dem_profile, dem_source = get_dem_with_fallback(
        boundary_gdf=field_boundary,
        uploaded_dem_path=dem_path,
        sample_dem_path=sample_paths["dem"],
    )
    if dem_source == "Iowa 3m WCS (auto)":
        st.success(f"🛰️ DEM auto-fetched from Iowa 3-meter Digital Elevation Model (Iowa DNR)")
        st.session_state.dem_source_label = "Iowa 3-meter Digital Elevation Model (Iowa DNR)"
    else:
        st.info(f"📁 DEM source: {dem_source}")
except Exception as exc:
    st.error(f"Could not load DEM: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Slope computation — BEFORE reprojection while DEM is still in UTM meters
# Critical: slope must be computed in a projected CRS (meters)
# Computing slope after reprojection to EPSG:4326 gives wrong values
# because pixel size would be in degrees not meters
# ---------------------------------------------------------------------------
status.text("Computing slope...")
progress.progress(70)

# Compute slope while DEM is in its native CRS (UTM meters from Iowa WCS)
slope_percent = compute_slope_from_dem(dem_array, dem_transform)
slope_crs     = dem_profile.get("crs")
slope_transform = dem_transform

# ---------------------------------------------------------------------------
# CRS alignment — reproject both slope AND DEM to match NDVI CRS
# ---------------------------------------------------------------------------
if ndvi_profile.get("crs") != dem_profile.get("crs"):
    left, bottom, right, top = array_bounds(
        dem_profile["height"], dem_profile["width"], dem_transform
    )
    transform_new, width_new, height_new = calculate_default_transform(
        dem_profile["crs"], ndvi_profile["crs"],
        dem_profile["width"], dem_profile["height"],
        left, bottom, right, top,
    )

    # Reproject DEM elevation (for display reference only)
    dem_reproj = np.empty((height_new, width_new), dtype=dem_array.dtype)
    reproject(
        source=dem_array, destination=dem_reproj,
        src_transform=dem_transform, src_crs=dem_profile["crs"],
        dst_transform=transform_new, dst_crs=ndvi_profile["crs"],
        resampling=Resampling.bilinear,
    )
    dem_array     = dem_reproj
    dem_transform = transform_new

    # Reproject slope — values stay correct because computed before reprojection
    slope_reproj = np.empty((height_new, width_new), dtype=slope_percent.dtype)
    reproject(
        source=slope_percent, destination=slope_reproj,
        src_transform=slope_transform, src_crs=slope_crs,
        dst_transform=transform_new, dst_crs=ndvi_profile["crs"],
        resampling=Resampling.bilinear,
    )
    slope_percent   = slope_reproj
    slope_transform = transform_new

# Final shape alignment between slope and NDVI
if ndvi_array.shape != slope_percent.shape:
    slope_resampled = np.empty(ndvi_array.shape, dtype=slope_percent.dtype)
    reproject(
        source=slope_percent, destination=slope_resampled,
        src_transform=slope_transform, dst_transform=ndvi_transform,
        src_crs=ndvi_profile.get("crs"), dst_crs=ndvi_profile.get("crs"),
        resampling=Resampling.bilinear,
    )
    slope_percent = slope_resampled

# ---------------------------------------------------------------------------
# WSS dominant soil series lookup
# ---------------------------------------------------------------------------
try:
    from src.wss_utils import get_dominant_soil_series
    soil_info = get_dominant_soil_series(field_boundary)
    st.session_state.soil_series   = soil_info.get("series_name", "Not available")
    st.session_state.soil_k_factor = soil_info.get("k_factor", None)
except Exception:
    st.session_state.soil_series   = "Not available"
    st.session_state.soil_k_factor = None

# ---------------------------------------------------------------------------
# Iowa R-factor zone lookup (FCC Census Block API)
# ---------------------------------------------------------------------------
try:
    from src.scoring import get_iowa_r_factor
    _r_factor, _r_factor_note = get_iowa_r_factor(field_boundary)
    st.session_state.r_factor      = _r_factor
    st.session_state.r_factor_note = _r_factor_note
except Exception:
    st.session_state.r_factor      = 150.0
    st.session_state.r_factor_note = "R=150 (default — county lookup failed)"

# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
status.text("Building map...")
progress.progress(85)
# Preserve zoom level across reruns so sliders don't reset the map
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 15
if "map_center" not in st.session_state:
    st.session_state.map_center = None
if "ndvi_date_from" not in st.session_state:
    st.session_state.ndvi_date_from = None
if "ndvi_date_to" not in st.session_state:
    st.session_state.ndvi_date_to = None
if "ndvi_scene_latest" not in st.session_state:
    st.session_state.ndvi_scene_latest = None
if "ndvi_scene_earliest" not in st.session_state:
    st.session_state.ndvi_scene_earliest = None
if "ndvi_scene_count" not in st.session_state:
    st.session_state.ndvi_scene_count = None
if "dem_source_label" not in st.session_state:
    st.session_state.dem_source_label = "Sample DEM"
if "soil_series" not in st.session_state:
    st.session_state.soil_series = "Not available"
if "soil_k_factor" not in st.session_state:
    st.session_state.soil_k_factor = None
if "r_factor" not in st.session_state:
    st.session_state.r_factor = 150.0
if "r_factor_note" not in st.session_state:
    st.session_state.r_factor_note = "R=150 (standard Iowa zone)"


_risk_zone_preview = classify_risk_zones(pixel_risk_index(ndvi_array, slope_percent))

folium_map = build_map_with_rasters(
    field_boundary, ndvi_array, slope_percent,
    ndvi_transform, ndvi_profile.get("crs"),
    ndvi_opacity, slope_opacity,
    zoom_start=st.session_state.map_zoom,
    ndvi_threshold=ndvi_threshold,
    risk_zone_array=_risk_zone_preview,
)

progress.progress(100)
status.empty()
progress.empty()

st.subheader("🗺️ Field Risk Map")
try:
    from streamlit_folium import st_folium
    map_data = st_folium(
        folium_map,
        height=520,
        width='stretch',
        returned_objects=["last_zoom", "last_center"],
        key="field_map",
    )
    # Save zoom and center so next rerun starts at same position
    if map_data and map_data.get("last_zoom"):
        st.session_state.map_zoom = map_data["last_zoom"]
    if map_data and map_data.get("last_center"):
        st.session_state.map_center = map_data["last_center"]
except ImportError:
    # Fallback if streamlit-folium not installed
    st.components.v1.html(folium_map._repr_html_(), height=520)

with st.expander("How are risk zones calculated?"):
    st.markdown("""
**Risk Index = C-factor × LS-factor (computed per pixel)**

| Zone | Risk Index | Meaning |
|------|-----------|---------|
| 🟢 Low | < 0.3 | Adequate cover for slope conditions |
| 🟡 Moderate | 0.3–0.7 | Variable cover — monitor steep units |
| 🟠 High | 0.7–1.5 | Marginal cover on identified slopes |
| 🔴 Critical | > 1.5 | Low cover on high-risk slope units |

C-factor is derived from satellite NDVI using Iowa cereal rye calibration. LS-factor is derived from Iowa 3m DEM slope. Field-level Concern Level reflects the distribution of pixel-level Risk Index scores across the field.
""")

# ---------------------------------------------------------------------------
# Stats and scoring
# ---------------------------------------------------------------------------
ndvi_stats  = raster_stats(ndvi_array, ndvi_profile.get("nodata"))
slope_stats = raster_stats(slope_percent)
risk_result = score_erosion_concern(
    ndvi_stats=ndvi_stats,
    slope_stats=slope_stats,
    ndvi_threshold=ndvi_threshold,
    slope_threshold=slope_threshold,
    residue_system=residue_system,
    ndvi_array=ndvi_array,
    slope_array=slope_percent,
    k_factor=st.session_state.get("soil_k_factor"),
    soil_series=st.session_state.get("soil_series", "default"),
    r_factor=st.session_state.get("r_factor", 150.0),
)

# Hoist image date string — used in Section 4 and PDF call
_s_latest   = st.session_state.ndvi_scene_latest
_s_earliest = st.session_state.ndvi_scene_earliest
_s_count    = st.session_state.ndvi_scene_count
if _s_latest:
    if _s_count and _s_count > 1 and _s_earliest and _s_earliest != _s_latest:
        _image_date_str = f"{_s_count} scenes: {_s_earliest} – {_s_latest}"
    else:
        _image_date_str = _s_latest
else:
    _image_date_str = "Upload date unknown"

# === SECTION 2: EROSION CONCERN + ADVISORY ===
badge_class = f"badge-{risk_result['concern_level'].lower()}"
st.markdown(
    f"### Erosion Concern: "
    f"<span class='{badge_class}'>{risk_result['concern_level']}</span>",
    unsafe_allow_html=True,
)

_latest = st.session_state.ndvi_scene_latest
_count  = st.session_state.ndvi_scene_count
if _latest:
    if _count and _count > 1 and _s_earliest and _s_earliest != _latest:
        _scene_label = (
            f"Latest flight: {_latest} "
            f"({_count} scenes composited: {_s_earliest} – {_latest})"
        )
    else:
        _scene_label = f"Flight date: {_latest}"
    st.caption(f"🛰️ Sentinel-2 image · {_scene_label} · via Google Earth Engine")
elif st.session_state.ndvi_date_from and st.session_state.ndvi_date_to:
    st.caption(
        f"🛰️ NDVI collected: {st.session_state.ndvi_date_from} – "
        f"{st.session_state.ndvi_date_to} via Sentinel-2 / Google Earth Engine"
    )

concern_colors = {"Low": "✅", "Moderate": "⚠️", "High": "🔴", "Critical": "🚨"}
icon = concern_colors.get(risk_result["concern_level"], "ℹ️")
st.info(f"{icon} **CoverMap Advisory:** {risk_result['recommendation']}")

# === SECTION 3: FIELD RISK ZONE SUMMARY ===
zone_summary = zone_risk_summary(
    ndvi_array, slope_percent,
    ndvi_threshold=ndvi_threshold,
    slope_threshold=slope_threshold,
    zone_array=risk_result.get("zone_array"),
)

st.subheader("📋 Cover Crop Stand — NDVI Zone Summary")
zone_summary_display = zone_summary.copy()
if "slope_mean" in zone_summary_display.columns:
    zone_summary_display["slope_mean"] = zone_summary_display["slope_mean"].apply(
        lambda x: f"{x:.1f}%"
    )
zone_summary_display = zone_summary_display.rename(columns={
    "zone":       "Zone",
    "percent":    "% of Field",
    "ndvi_mean":  "NDVI Mean",
    "slope_mean": "Slope Mean (3m DEM, UTM)",
})

# Dynamic NDVI zone labels based on current threshold slider
ndvi_low_label  = f"Low Cover (NDVI < {ndvi_threshold:.2f})"
ndvi_mid_upper  = ndvi_threshold + 0.15
ndvi_mid_label  = f"Marginal (NDVI {ndvi_threshold:.2f}–{ndvi_mid_upper:.2f})"
ndvi_good_label = f"Good Cover (NDVI > {ndvi_mid_upper:.2f})"
zone_label_map = {
    "Low cover":                 ndvi_low_label,
    "Low cover — reseed target": ndvi_low_label,
    "Marginal stand":            ndvi_mid_label,
    "Marginal":                  ndvi_mid_label,
    "Good cover":                ndvi_good_label,
}
zone_summary_display["Zone"] = (
    zone_summary_display["Zone"]
    .map(zone_label_map)
    .fillna(zone_summary_display["Zone"])
)

st.dataframe(zone_summary_display, hide_index=True, use_container_width=True)

zone_chart = build_zone_risk_chart(zone_summary)
zone_chart.update_layout(
    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#c9d1d9"
)
st.plotly_chart(zone_chart, width='stretch')

# Risk Index zone table (C×LS pixel distribution)
zone_counts  = risk_result.get("zone_counts", {})
total_pixels = sum(zone_counts.values())

if zone_counts and total_pixels > 0:
    pixel_size_m    = 10.0
    acres_per_pixel = (pixel_size_m ** 2) / 4046.86

    risk_zone_rows = []
    risk_zone_config = [
        (4, "Critical Risk",  "#EF4444", "> 1.5"),
        (3, "High Risk",      "#F97316", "0.7–1.5"),
        (2, "Moderate Risk",  "#FACC15", "0.3–0.7"),
        (1, "Low Risk",       "#22C55E", "< 0.3"),
    ]
    for zone_val, label, _color, threshold in risk_zone_config:
        count = zone_counts.get(zone_val, 0)
        acres = count * acres_per_pixel
        pct   = count / total_pixels * 100
        risk_zone_rows.append({
            "Zone":       label,
            "C×LS Range": threshold,
            "Acres":      f"{acres:.1f}",
            "% of Field": f"{pct:.0f}%",
        })

    risk_zone_df = pd.DataFrame(risk_zone_rows)

    st.subheader("📊 Erosion Risk Zone Summary (C×LS)")
    st.dataframe(
        risk_zone_df,
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Risk Index = C-factor (from NDVI) × LS-factor "
        "(from slope). Unitless composite erosion "
        "vulnerability score. Critical > 1.5 · "
        "High 0.7–1.5 · Moderate 0.3–0.7 · Low < 0.3"
    )
else:
    st.info(
        "ℹ️ Risk Index zone distribution requires "
        "pixel-level C×LS computation. Ensure "
        "risk_zone_array is being passed to the "
        "scoring pipeline."
    )

# === SECTION 4: COVER CROP STAND ASSESSMENT ===
st.subheader("📋 Cover Crop Stand Assessment — Satellite Documentation")

_ndvi_mean     = ndvi_stats["mean"]
_biomass_kgha  = max(0.0, (_ndvi_mean - 0.10) / 0.40 * 3500)
_biomass_lbac  = _biomass_kgha * 0.891
_biomass_low   = max(0, round(_biomass_lbac * 0.6 / 50) * 50)
_biomass_high  = round(_biomass_lbac * 1.4 / 50) * 50
_valid_px      = ndvi_array[~np.isnan(ndvi_array)]
_pct_above_020 = (np.sum(_valid_px > 0.20) / _valid_px.size * 100) if _valid_px.size > 0 else 0.0
_valid_pct     = (_valid_px.size / ndvi_array.size * 100) if ndvi_array.size > 0 else 0.0

_valid_warning = None
if _valid_pct < 50:
    _valid_warning = (
        f"⚠️ Only {_valid_pct:.0f}% of field pixels returned valid NDVI values. "
        f"Results are unreliable — widen date range before using for documentation."
    )
elif _valid_pct < 75:
    _valid_warning = (
        f"⚠️ {_valid_pct:.0f}% valid pixels. Results may be affected by cloud cover. "
        f"Consider widening date range."
    )
if _valid_warning:
    st.warning(_valid_warning)

_cover_status = (
    f"✅ NDVI {_ndvi_mean:.3f} — cover crop confirmed"
    if _ndvi_mean > 0.20 else
    f"⚠️ NDVI {_ndvi_mean:.3f} — inadequate cover"
)
_ground_cover_status = (
    "✅ Estimated adequate cover zones based on NDVI threshold — field verification recommended"
    if _pct_above_020 > 50 else
    "⚠️ Estimated adequate cover zones below 50% of field — field verification recommended"
)

_eqip_rows = {
    "Cover crop present":    ("Sentinel-2 NDVI > 0.20",  _cover_status),
    "Field boundary":        ("Operator provided",        "📋 Verify against FSA CLU records"),
    "Image date":            ("GEE metadata",             _image_date_str),
    "Estimated biomass":     ("NDVI proxy",               f"~{_biomass_low}–{_biomass_high} lb/acre (±40% NDVI proxy)"),
    "30% ground cover":      ("NDVI threshold",           _ground_cover_status),
    "Valid pixels":          ("NDVI > 0.05 threshold",    f"{'✅' if _valid_pct >= 75 else '⚠️'} {_valid_pct:.0f}% valid pixels ({'reliable' if _valid_pct >= 75 else 'below 75% — verify date range'})"),
    "Seeding rate":          ("Field records required",   "📋 CCA to verify on-site"),
    "Species confirmation":  ("Field records required",   "📋 CCA to verify on-site"),
    "Termination date":      ("Not yet applicable",       "⏳ Pending — document at termination"),
    "Cooperator signature":  ("Physical form required",   "📋 Required for EQIP submission"),
}

_eqip_df = pd.DataFrame(
    [(req, src, stat) for req, (src, stat) in _eqip_rows.items()],
    columns=["Requirement", "Data Source", "Status"],
)
st.dataframe(_eqip_df, hide_index=True, use_container_width=True)
st.caption(
    "Remote sensing confirms spatial cover crop presence. "
    "Seeding rate, species, and termination compliance require CCA field verification "
    "per NRCS Practice Code 340."
)

# === SECTION 5: COVER CROP METRICS ===
st.subheader("📊 Cover Crop Metrics")
_soil_label = st.session_state.get("soil_series", "—") or "—"
_soil_kf    = st.session_state.get("soil_k_factor")
if _soil_kf:
    _soil_label = f"{_soil_label} (K={_soil_kf})"
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("NDVI Mean",      f"{ndvi_stats['mean']:.3f}")
c2.metric("NDVI Min",       f"{ndvi_stats['min']:.3f}")
c3.metric("NDVI Max",       f"{ndvi_stats['max']:.3f}")
c4.metric("Slope Mean (%)", f"{slope_stats['mean']:.1f}%")
c5.metric("C-Factor",       f"{risk_result['c_factor']:.3f}",
          help="RUSLE C-factor (residue-adjusted). Lower = better cover.")
c6.metric("Risk Index",     f"{risk_result['rusle_score']:.3f}",
          help="Unitless erosion risk index (C-factor × LS-factor). "
               "Scale: <0.3 Minimal · 0.3-0.7 Moderate · 0.7-1.5 High · >1.5 Critical")
c7.metric("Dominant Soil",  _soil_label,
          help="Dominant soil series from USDA Web Soil Survey SSURGO")

if risk_result["residue_multiplier"] < 1.0:
    st.caption(
        f"C-Factor adjusted from {risk_result['c_factor_unadjusted']:.3f} to "
        f"{risk_result['c_factor']:.3f} "
        f"({int((1 - risk_result['residue_multiplier']) * 100)}% reduction for residue — "
        f"{residue_system})"
    )
else:
    st.caption(
        "C-Factor: no residue adjustment applied (unknown or conventional tillage)"
    )

if ndvi_stats["mean"] > 0.75:
    st.warning(
        "⚠️ High NDVI may indicate mature cash crops rather than cover crops. "
        "Verify image date — early spring pull recommended for accurate assessment."
    )

# === SECTION 6: ESTIMATED SOIL LOSS ===
st.subheader("📊 Estimated Soil Loss vs. Soil Loss Tolerance")
_sl_result = risk_result.get("soil_loss")
if _sl_result and _sl_result.get("status_code") != "unavailable":
    _sl = _sl_result["soil_loss_tons_ac_yr"]
    _tv = _sl_result["t_value"]
    _rt = _sl_result["ratio_to_t"]
    _sc = _sl_result["status_code"]
    sl1, sl2, sl3 = st.columns(3)
    sl1.metric(
        "Est. Soil Loss",
        f"{_sl:.1f} t/ac/yr",
        help="A = R × K × LS × C (simplified RUSLE; P=1.0 assumed)",
    )
    sl2.metric(
        "Soil Loss Tolerance (T)",
        f"{_tv} t/ac/yr",
        help="NRCS tolerable soil loss limit for the dominant soil series",
    )
    sl3.metric(
        "Ratio to T",
        f"{_rt:.2f}×",
        delta="Within T" if _sc == "within_t" else "Over T",
        delta_color="normal" if _sc == "within_t" else "inverse",
    )
    _status_fn = {
        "within_t":   st.success,
        "near_t":     st.warning,
        "over_t":     st.error,
        "critical_t": st.error,
    }.get(_sc, st.info)
    _status_fn(f"**{_sl_result['conservation_status']}**")
    st.caption(f"Iowa R-factor: {st.session_state.get('r_factor_note', 'R=150 (standard Iowa)')}")
    st.caption(
        "⚠️ Simplified RUSLE estimate for advisory use only. "
        "Not a substitute for a site-specific RUSLE2 run or official NRCS determination."
    )
else:
    st.info(
        "Soil loss estimate unavailable — K-factor not returned from USDA "
        "Web Soil Survey for this field location."
    )

# === SECTION 7: GENERATE REPORT ===
report_df = pd.DataFrame([
    {"Metric": "NDVI Mean",        "Value": ndvi_stats["mean"]},
    {"Metric": "NDVI Min",         "Value": ndvi_stats["min"]},
    {"Metric": "NDVI Max",         "Value": ndvi_stats["max"]},
    {"Metric": "NDVI Date From",   "Value": st.session_state.ndvi_date_from or "N/A"},
    {"Metric": "NDVI Date To",     "Value": st.session_state.ndvi_date_to   or "N/A"},
    {"Metric": "Slope Mean (%)",   "Value": slope_stats["mean"]},
    {"Metric": "C-Factor (RUSLE)", "Value": risk_result["c_factor"]},
    {"Metric": "LS-Factor",        "Value": risk_result["ls_factor"]},
    {"Metric": "RUSLE CxLS Score", "Value": risk_result["rusle_score"]},
    {"Metric": "Erosion Concern",  "Value": risk_result["concern_level"]},
    {"Metric": "Recommendation",   "Value": risk_result["recommendation"]},
])

st.subheader("📄 Generate Field Report")
col_a, col_b, col_c = st.columns(3)
with col_a:
    _default_field_name = Path(boundary_file.name).stem if boundary_file else "North Field"
    pdf_field_name = st.text_input("Field name", value=_default_field_name)
with col_b:
    pdf_farm_name  = st.text_input("Farm name",  value="")
with col_c:
    pdf_county     = st.text_input("County",     value="Shelby County, IA")

col_d, col_e, col_f = st.columns(3)
with col_d:
    pdf_termination_date = st.text_input(
        "Termination date (optional)", value="", placeholder="e.g. May 10, 2026")
with col_e:
    pdf_cca_name = st.text_input("CCA name", value="Stephen Zimmerman, CCA MS")
with col_f:
    pdf_previous_crop = st.text_input(
        "Previous crop (optional)", value="", placeholder="e.g. Corn, Soybeans")

col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    if st.button("📋 Generate PDF Report", type="primary", use_container_width=True):
        with st.spinner("Building PDF report..."):
            try:
                pdf_bytes = generate_field_report(
                    field_name=pdf_field_name or "Field",
                    farm_name=pdf_farm_name   or "",
                    county=pdf_county         or "Iowa",
                    ndvi_array=ndvi_array,
                    slope_array=slope_percent,
                    ndvi_stats=ndvi_stats,
                    slope_stats=slope_stats,
                    risk_result=risk_result,
                    zone_summary=zone_summary,
                    ndvi_threshold=ndvi_threshold,
                    slope_threshold=slope_threshold,
                    ndvi_date_from=st.session_state.ndvi_date_from,
                    ndvi_date_to=st.session_state.ndvi_date_to,
                    ndvi_scene_date=_image_date_str,
                    dem_source=st.session_state.get("dem_source_label", "Iowa 3-meter Digital Elevation Model (Iowa DNR)"),
                    termination_date=pdf_termination_date or None,
                    cca_name=pdf_cca_name or "Stephen Zimmerman, CCA MS",
                    previous_crop=pdf_previous_crop or None,
                    soil_series=st.session_state.get("soil_series"),
                    soil_k_factor=st.session_state.get("soil_k_factor"),
                    residue_system=residue_system,
                    soil_loss_result=risk_result.get("soil_loss"),
                    r_factor=st.session_state.get("r_factor", 150.0),
                    r_factor_note=st.session_state.get("r_factor_note"),
                )
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"covermap_report_{(pdf_field_name or 'field').replace(' ','_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
                st.success("PDF ready — click Download above.")
            except Exception as pdf_exc:
                st.error(f"PDF generation failed: {pdf_exc}")

with col_dl2:
    st.download_button(
        label="⬇️ Download CSV Data",
        data=report_df.to_csv(index=False).encode("utf-8"),
        file_name="erosion_report_nrcs.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()
st.caption(
    "CoverMap · Stephen Zimmerman CCA MS · Ankeny IA · "
    "Sentinel-2 via Google Earth Engine · Iowa RUSLE C-factor calibration"
)
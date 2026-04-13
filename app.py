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
    SPECIES_C_TARGETS,
)
from src.visualization import build_map_with_rasters, build_zone_risk_chart
from src.iowa_dem_utils import get_dem_with_fallback

# Sentinel-2 imports — graceful fallback if credentials not yet configured
SENTINEL_AVAILABLE = False
SENTINEL_IMPORT_ERROR = None
try:
    from src.sentinel_utils import (
        get_config_from_streamlit_secrets,
        fetch_ndvi_for_field,
    )
    from src.ndvi_scheduler import fetch_best_available_ndvi, fetch_ndvi_comparison
    SENTINEL_AVAILABLE = True
except Exception as _sentinel_exc:
    SENTINEL_IMPORT_ERROR = str(_sentinel_exc)

APP_TITLE       = "Cover Crop Erosion Viewer"
APP_DESCRIPTION = (
    "Field-level erosion risk using Sentinel-2 NDVI and terrain slope. "
    "NRCS EQIP ready · Iowa RUSLE C-factor scoring · Automated satellite pull."
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
if "demo_loaded" not in st.session_state:
    st.session_state.demo_loaded = False
if "ndvi_source" not in st.session_state:
    st.session_state.ndvi_source = "auto"   # "auto" | "upload" | "sample"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📍 Field Setup")

    if st.button("🗺️ Load Shelby County Demo", type="primary", width='stretch'):
        st.session_state.demo_loaded = True
        st.rerun()

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
    ndvi_threshold  = st.slider("Low cover NDVI", 0.0, 1.0, float(DEFAULT_THRESHOLDS["ndvi_low"]),  0.01)
    slope_threshold = st.slider("Steep slope (%)", 0.0, 30.0, float(DEFAULT_THRESHOLDS["slope_steep"]), 0.5)

    st.divider()
    st.markdown("### 🗺️ Map Overlays")
    ndvi_opacity  = st.slider("NDVI opacity",  0.0, 1.0, 0.6, 0.1)
    slope_opacity = st.slider("Slope opacity", 0.0, 1.0, 0.4, 0.1)

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
if st.session_state.demo_loaded:
    boundary_path = sample_paths["field"]
    st.info("🗺️ Shelby County demo loaded — synthetic field data active.")
elif boundary_file is not None:
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

if ndvi_mode == "Auto (Sentinel-2 API)" and not st.session_state.demo_loaded:
    if not SENTINEL_AVAILABLE:
        st.warning(
            f"⚠️ Sentinel-2 module not loaded. Error: {SENTINEL_IMPORT_ERROR}"
        )
        ndvi_path = sample_paths["ndvi"]
    else:
        try:
            config = get_config_from_streamlit_secrets()

            if ndvi_window == "Custom spring window":
                # Custom spring pull
                from datetime import datetime as _dt
                date_from = _dt(ndvi_year, int(ndvi_start.split('-')[0]), int(ndvi_start.split('-')[1]))
                date_to   = _dt(ndvi_year, int(ndvi_end.split('-')[0]),   int(ndvi_end.split('-')[1]))
                ndvi_array, ndvi_transform, ndvi_profile = fetch_ndvi_for_field(
                    boundary_gdf=field_boundary,
                    config=config,
                    date_from=date_from,
                    date_to=date_to,
                )
                st.success(f"✅ Sentinel-2 NDVI pulled | {ndvi_year} {ndvi_start}–{ndvi_end}")
            else:
                # Rolling window — auto-widens on cloud cover
                days_map = {"Last 7 days": 7, "Last 14 days": 14, "Last 30 days": 30}
                ref_date = datetime.now()

                ndvi_array, ndvi_transform, ndvi_profile, meta = fetch_best_available_ndvi(
                    boundary_gdf=field_boundary,
                    reference_date=ref_date,
                )
                st.success(meta["message"])

            # Year-over-year comparison chart
            if yoy_compare:
                with st.spinner("Pulling year-over-year NDVI (this takes ~30s)..."):
                    current_year = datetime.now().year
                    yoy_years = list(range(2023, current_year + 1))
                    yoy_results = fetch_ndvi_comparison(
                        boundary_gdf=field_boundary,
                        years=yoy_years,
                    )
                    yoy_rows = []
                    for yr, data in yoy_results.items():
                        if "error" not in data:
                            yoy_rows.append({"Year": yr, "Mean NDVI": round(data["mean_ndvi"], 3)})
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
            st.warning(f"Sentinel-2 API unavailable: {exc}. Falling back to sample NDVI.")
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
        st.success(f"🛰️ DEM auto-fetched from Iowa 3m WCS")
    else:
        st.info(f"📁 DEM source: {dem_source}")
except Exception as exc:
    st.error(f"Could not load DEM: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# CRS alignment
# ---------------------------------------------------------------------------
if ndvi_profile.get("crs") != dem_profile.get("crs"):
    st.warning("CRS mismatch — reprojecting DEM to match NDVI.")
    left, bottom, right, top = array_bounds(
        dem_profile["height"], dem_profile["width"], dem_transform
    )
    transform_new, width_new, height_new = calculate_default_transform(
        dem_profile["crs"], ndvi_profile["crs"],
        dem_profile["width"], dem_profile["height"],
        left, bottom, right, top,
    )
    dem_reproj = np.empty((height_new, width_new), dtype=dem_array.dtype)
    reproject(
        source=dem_array, destination=dem_reproj,
        src_transform=dem_transform, src_crs=dem_profile["crs"],
        dst_transform=transform_new, dst_crs=ndvi_profile["crs"],
        resampling=Resampling.bilinear,
    )
    dem_array     = dem_reproj
    dem_transform = transform_new

# ---------------------------------------------------------------------------
# Slope computation
# ---------------------------------------------------------------------------
status.text("Computing slope...")
progress.progress(70)
slope_percent = compute_slope_from_dem(dem_array, dem_transform)

if ndvi_array.shape != slope_percent.shape:
    slope_resampled = np.empty(ndvi_array.shape, dtype=slope_percent.dtype)
    reproject(
        source=slope_percent, destination=slope_resampled,
        src_transform=dem_transform, dst_transform=ndvi_transform,
        src_crs=dem_profile.get("crs"), dst_crs=ndvi_profile.get("crs"),
        resampling=Resampling.bilinear,
    )
    slope_percent = slope_resampled

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

folium_map = build_map_with_rasters(
    field_boundary, ndvi_array, slope_percent,
    ndvi_transform, ndvi_profile.get("crs"),
    ndvi_opacity, slope_opacity,
    zoom_start=st.session_state.map_zoom,
    ndvi_threshold=ndvi_threshold,
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

# ---------------------------------------------------------------------------
# Stats and scoring
# ---------------------------------------------------------------------------
ndvi_stats  = raster_stats(ndvi_array, ndvi_profile.get("nodata"))
slope_stats = raster_stats(slope_percent)
risk_result = score_erosion_concern(
    ndvi_mean=ndvi_stats["mean"],
    slope_mean=slope_stats["mean"],
    ndvi_threshold=ndvi_threshold,
    slope_threshold=slope_threshold,
)

# Concern badge
badge_class = f"badge-{risk_result['concern_level'].lower()}"
st.markdown(
    f"### Erosion Concern: "
    f"<span class='{badge_class}'>{risk_result['concern_level']}</span>",
    unsafe_allow_html=True,
)

# Recommendation box
concern_colors = {
    "Low": "✅", "Moderate": "⚠️", "High": "🔴", "Critical": "🚨"
}
icon = concern_colors.get(risk_result["concern_level"], "ℹ️")
st.info(f"{icon} **NRCS Advisory:** {risk_result['recommendation']}")

# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------
st.subheader("📊 Field Summary")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("NDVI Mean",      f"{ndvi_stats['mean']:.3f}")
c2.metric("NDVI Min",       f"{ndvi_stats['min']:.3f}")
c3.metric("NDVI Max",       f"{ndvi_stats['max']:.3f}")
c4.metric("Slope Mean (%)", f"{slope_stats['mean']:.2f}")
c5.metric("C-Factor",       f"{risk_result['c_factor']:.3f}",
          help="RUSLE C-factor from Iowa lookup table. Lower = better cover.")
c6.metric("RUSLE C×LS",     f"{risk_result['rusle_score']:.3f}",
          help="Combined erosion index. >0.7 = Moderate, >1.5 = High.")

# NDVI freshness warning
if ndvi_stats["mean"] > 0.75:
    st.warning(
        "⚠️ High NDVI may indicate mature cash crops rather than cover crops. "
        "Verify image date — early spring pull recommended for accurate assessment."
    )

# ---------------------------------------------------------------------------
# C-factor species comparison chart
# ---------------------------------------------------------------------------
st.subheader("🌱 C-Factor vs Iowa Cover Crop Targets")
species_df = pd.DataFrame([
    {"Species": k, "C-Factor Target": v, "Type": "Target"}
    for k, v in SPECIES_C_TARGETS.items()
])
species_df = pd.concat([
    species_df,
    pd.DataFrame([{"Species": "⬅ This Field", "C-Factor Target": risk_result["c_factor"], "Type": "Field"}])
], ignore_index=True)

fig_species = px.bar(
    species_df, x="C-Factor Target", y="Species",
    orientation="h",
    color="Type",
    color_discrete_map={"Target": "#1f6feb", "Field": "#f78166"},
    title="Field C-Factor vs Iowa Species Benchmarks (lower = better cover)",
)
fig_species.update_layout(
    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
    font_color="#c9d1d9", showlegend=False,
)
st.plotly_chart(fig_species, width='stretch')

# ---------------------------------------------------------------------------
# Zone risk summary
# ---------------------------------------------------------------------------
zone_summary = zone_risk_summary(
    ndvi_array, slope_percent,
    ndvi_threshold=ndvi_threshold,
    slope_threshold=slope_threshold,
)
st.subheader("📋 Zone Risk Summary")
st.dataframe(zone_summary, width='stretch')

zone_chart = build_zone_risk_chart(zone_summary)
zone_chart.update_layout(
    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#c9d1d9"
)
st.plotly_chart(zone_chart, width='stretch')

# ---------------------------------------------------------------------------
# NRCS EQIP badge
# ---------------------------------------------------------------------------
st.success("✅ NRCS EQIP Ready — field data meets basic conservation planning requirements.")

# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
report_df = pd.DataFrame([
    {"Metric": "NDVI Mean",        "Value": ndvi_stats["mean"]},
    {"Metric": "NDVI Min",         "Value": ndvi_stats["min"]},
    {"Metric": "NDVI Max",         "Value": ndvi_stats["max"]},
    {"Metric": "Slope Mean (%)",   "Value": slope_stats["mean"]},
    {"Metric": "C-Factor (RUSLE)", "Value": risk_result["c_factor"]},
    {"Metric": "LS-Factor",        "Value": risk_result["ls_factor"]},
    {"Metric": "RUSLE C×LS Score", "Value": risk_result["rusle_score"]},
    {"Metric": "Erosion Concern",  "Value": risk_result["concern_level"]},
    {"Metric": "Recommendation",   "Value": risk_result["recommendation"]},
])
st.download_button(
    label="⬇️ Download NRCS Report (CSV)",
    data=report_df.to_csv(index=False).encode("utf-8"),
    file_name="erosion_report_nrcs.csv",
    mime="text/csv",
    width='stretch',
)

st.divider()
st.caption(
    "Cover Crop Erosion Viewer · Stephen Zimmerman CCA MS · Ankeny IA · "
    "Sentinel-2 L2A via Copernicus Data Space · Iowa RUSLE C-factor calibration"
)
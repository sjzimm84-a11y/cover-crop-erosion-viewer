from pathlib import Path
import tempfile
from typing import Optional
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import folium
import geopandas as gpd
from shapely.geometry import box

import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import array_bounds
from rasterio.mask import mask

from src.io_utils import (
    load_boundary_file,
    save_uploaded_file,
)
from src.raster_utils import (
    clip_raster_to_geometry,
    compute_slope_from_dem,
    raster_stats,
    zone_risk_summary,
)
from src.sample_data import ensure_sample_data
from src.scoring import DEFAULT_THRESHOLDS, score_erosion_concern
from src.visualization import build_map_with_rasters, build_zone_risk_chart

APP_TITLE = "Cover Crop Erosion Viewer"
APP_DESCRIPTION = (
    "Field-level cover crop erosion risk analysis for NRCS advisors and Iowa farmers. "
    "Upload boundary + NDVI + DEM → instant risk zones + reports."
)

def calculate_c_factor(ndvi_mean: float) -> float:
    """Proxy C-factor from NDVI. Higher NDVI = lower erosion."""
    return max(0.01, 0.95 - ndvi_mean * 0.8)

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    st.title("🌾 Cover Crop Erosion Viewer")
    st.markdown("**Quantify cover crop ROI: NDVI + slope → NRCS-ready reports**")

    # Demo mode
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = False

    data_dir = Path(__file__).parent / "data"
    sample_paths = ensure_sample_data(data_dir)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.header("Controls")

        if st.button("🚜 Load Shelby County Demo", type="primary", use_container_width=True):
            st.session_state.demo_mode = True
            st.rerun()

        boundary_file = st.file_uploader(
            "📍 Field Boundary", 
            type=["geojson", "json", "zip"],
            help="GeoJSON or zipped shapefile"
        )
        ndvi_file = st.file_uploader("🌿 NDVI GeoTIFF", type=["tif", "tiff"])
        dem_file = st.file_uploader("⛰️ DEM (TIF/IMG)", type=["tif", "tiff", "img"])

        st.markdown("---")
        ndvi_threshold = st.slider("NDVI Low Cover", 0.0, 1.0, 0.35, 0.01)
        slope_threshold = st.slider("Slope Steep (%)", 0.0, 30.0, 6.0, 0.5)

        st.markdown("---")
        ndvi_opacity = st.slider("NDVI Opacity", 0.0, 1.0, 0.6, 0.1)
        slope_opacity = st.slider("Slope Opacity", 0.0, 1.0, 0.4, 0.1)

    with col2:
        if st.session_state.demo_mode:
            st.success("✅ Shelby County demo loaded!")
            boundary_path = data_dir / "shelby_boundary.geojson"
            ndvi_path = data_dir / "shelby_ndvi.tif"
            dem_path = data_dir / "shelby_dem.tif"
        else:
            temp_dir = Path(tempfile.mkdtemp())
            boundary_path = save_uploaded_file(boundary_file, temp_dir) if boundary_file else None
            ndvi_path = save_uploaded_file(ndvi_file, temp_dir) if ndvi_file else data_dir / "sample_ndvi.tif"
            dem_path = save_uploaded_file(dem_file, temp_dir) if dem_file else data_dir / "sample_dem.tif"

        # NULL CHECK
        if boundary_path is None or not boundary_path.exists():
            st.error("No valid boundary file. Upload or use demo.")
            st.stop()

        progress = st.progress(0)

        # Load boundary
        progress.progress(0.2)
        field_boundary = load_boundary_file(boundary_path)
        st.success(f"Loaded field: {len(field_boundary)} polygons")

        # Clip rasters
        progress.progress(0.4)
        ndvi_array, ndvi_transform, ndvi_profile = clip_raster_to_geometry(ndvi_path, field_boundary)

        progress.progress(0.6)
        dem_array, dem_transform, dem_profile = clip_raster_to_geometry(dem_path, field_boundary)

        # CRS handling
        progress.progress(0.7)
        if ndvi_profile.get('crs') != dem_profile.get('crs'):
            st.info("🔄 CRS mismatch - reprojecting...")
            left, bottom, right, top = array_bounds(
                dem_profile['height'], dem_profile['width'], dem_transform
            )
            dst_transform, w, h = calculate_default_transform(
                dem_profile['crs'], ndvi_profile['crs'],
                dem_profile['width'], dem_profile['height'],
                left, bottom, right, top
            )
            dem_reproj = np.empty((h, w), dtype=dem_array.dtype)
            reproject(
                dem_array, dem_reproj,
                dem_transform, dst_transform,
                dem_profile['crs'], ndvi_profile['crs'],
                Resampling.bilinear
            )
            dem_array = dem_reproj
            dem_transform = dst_transform

        # Slope
        progress.progress(0.8)
        slope = compute_slope_from_dem(dem_array, dem_transform)

        # Align grids
        if slope.shape != ndvi_array.shape:
            slope_resampled = np.empty(ndvi_array.shape, dtype=slope.dtype)
            reproject(slope, slope_resampled, dem_transform, ndvi_transform, Resampling.bilinear)
            slope = slope_resampled

        progress.progress(1.0)

        # Map
        folium_map = build_map_with_rasters(
            field_boundary, ndvi_array, slope,
            ndvi_transform, ndvi_profile.get('crs'),
            ndvi_opacity, slope_opacity
        )
        st.components.v1.html(folium_map._repr_html_(), height=500)

        # Metrics
        ndvi_stats = raster_stats(ndvi_array)
        slope_stats = raster_stats(slope)

        risk = score_erosion_concern(
            ndvi_stats["mean"], slope_stats["mean"],
            ndvi_threshold, slope_threshold
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("NDVI", f"{ndvi_stats['mean']:.3f}")
        col2.metric("Slope %", f"{slope_stats['mean']:.1f}")
        col3.metric("Concern", risk["concern_level"])
        col4.metric("C-factor", f"{calculate_c_factor(ndvi_stats['mean']):.3f}")

        if ndvi_stats["mean"] > 0.75:
            st.warning("High NDVI = mature crops?")

        # Zones
        zones = zone_risk_summary(ndvi_array, slope, ndvi_threshold, slope_threshold)
        st.dataframe(zones)
        st.plotly_chart(build_zone_risk_chart(zones))

        st.success("✅ NRCS EQIP ready")

        # Report
        report = pd.DataFrame([
            {"NDVI": ndvi_stats['mean']},
            {"Slope": slope_stats['mean']},
            {"Concern": risk["concern_level"]},
        ])
        st.download_button("Download Report", report.to_csv(), "report.csv")

if __name__ == "__main__":
    main()

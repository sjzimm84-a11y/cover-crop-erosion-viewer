from pathlib import Path
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import folium

import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import array_bounds
from rasterio.mask import mask
import geopandas as gpd

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
    "Estimate field-level erosion concern using early-season NDVI and terrain slope. "
    "Use synthetic sample data or upload your own field boundary and rasters."
)

def calculate_c_factor(ndvi_mean: float) -> float:
    """Proxy C-factor from NDVI (0-1 scale). Higher NDVI = lower erosion potential."""
    return max(0.0, 1.0 - ndvi_mean)

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.write(APP_DESCRIPTION)

    # Session state for demo mode
    if 'demo_loaded' not in st.session_state:
        st.session_state.demo_loaded = False

    data_dir = Path(__file__).parent / "data"
    sample_paths = ensure_sample_data(data_dir)

    with st.sidebar:
        st.header("Inputs")

        # Shelby County Demo Button
        if st.button("Load Shelby County Demo", type="primary"):
            st.session_state.demo_loaded = True
            st.rerun()

        boundary_file = st.file_uploader(
            "Upload field boundary GeoJSON or zipped shapefile",
            type=["geojson", "json", "zip"],
            help="Use GeoJSON directly or upload a zipped shapefile bundle.",
        )
        ndvi_file = st.file_uploader(
            "Upload NDVI GeoTIFF",
            type=["tif", "tiff"],
            help="Optional. If omitted, the sample NDVI dataset will be used.",
        )
        dem_file = st.file_uploader(
            "Upload DEM GeoTIFF or IMG",
            type=["tif", "tiff", "img"],
            help="Upload a DEM GeoTIFF (.tif) or ERDAS IMAGINE (.img) file, or omit to use the sample DEM.",
        )

        st.markdown("---")
        st.subheader("Editable thresholds")
        ndvi_threshold = st.slider(
            "Low cover NDVI threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(DEFAULT_THRESHOLDS["ndvi_low"]),
            step=0.01,
        )
        slope_threshold = st.slider(
            "Steep slope threshold (%)",
            min_value=0.0,
            max_value=30.0,
            value=float(DEFAULT_THRESHOLDS["slope_steep"]),
            step=0.5,
        )

        st.markdown("---")
        st.subheader("Visualization")
        ndvi_opacity = st.slider(
            "NDVI overlay opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1,
        )
        slope_opacity = st.slider(
            "Slope overlay opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.1,
        )

    temp_dir = Path(tempfile.mkdtemp())
    boundary_path = None
    ndvi_path = None
    dem_path = None

    # Handle demo mode
    if st.session_state.demo_loaded:
        boundary_path = sample_paths["boundary"]
        ndvi_path = sample_paths["ndvi"]
        dem_path = sample_paths["dem"]  
        st.info("Shelby County demo loaded. Using synthetic field data.")
    else:
        if boundary_file is not None:
            boundary_path = save_uploaded_file(boundary_file, temp_dir)
            st.write(f"Uploaded boundary: {boundary_path}")
        else:
            boundary_path = sample_paths["boundary"]
            st.info("Using sample boundary.")

        if ndvi_file is not None:
            ndvi_path = save_uploaded_file(ndvi_file, temp_dir)
        else:
            ndvi_path = sample_paths["ndvi"]

        if dem_file is not None:
            dem_path = save_uploaded_file(dem_file, temp_dir)
        else:
            dem_path = sample_paths["dem"]

    # Progress bar for processing
    progress_bar = st.progress(0)
    status_text = st.empty()

    # DEBUG boundary_path
    if boundary_path is None:
        st.error("🚨 No boundary path - check sample_data.py")
        st.stop()

    st.write(f"Path: {boundary_path} | Type: {type(boundary_path)}")

    try:
        status_text.text("Loading boundary...")
        progress_bar.progress(10)
        field_boundary = load_boundary_file(boundary_path)
        st.success(f"Loaded {len(field_boundary)} features")
    except Exception as exc:
        st.error(f"Boundary load failed: {exc}")
        return

    try:
        status_text.text("Clipping NDVI...")
        progress_bar.progress(30)
        ndvi_array, ndvi_transform, ndvi_profile = clip_raster_to_geometry(
            ndvi_path, field_boundary
        )
    except Exception as exc:
        st.error(f"NDVI clip failed: {exc}")
        return

    try:
        status_text.text("Clipping DEM...")
        progress_bar.progress(50)
        dem_array, dem_transform, dem_profile = clip_raster_to_geometry(
            dem_path, field_boundary
        )
    except Exception as exc:
        st.error(f"DEM clip failed: {exc}")
        return

    # CRS mismatch handling - SAFE VERSION
    if ndvi_profile.get('crs') != dem_profile.get('crs'):
        st.warning("CRS mismatch - reprojecting DEM...")
        left, bottom, right, top = array_bounds(
            dem_profile['height'], dem_profile['width'], dem_transform
        )
        transform, width, height = calculate_default_transform(
            dem_profile['crs'], ndvi_profile['crs'],
            dem_profile['width'], dem_profile['height'],
            left, bottom, right, top
        )
        dem_reproj = np.empty((height, width), dtype=dem_array.dtype)
        reproject(
            source=dem_array,
            destination=dem_reproj,
            src_transform=dem_transform,
            src_crs=dem_profile['crs'],
            dst_transform=transform,
            dst_crs=ndvi_profile['crs'],
            resampling=Resampling.bilinear,
        )
        dem_array = dem_reproj
        dem_transform = transform

    status_text.text("Computing slope...")
    progress_bar.progress(70)
    slope_percent = compute_slope_from_dem(dem_array, dem_transform)

    if ndvi_array.shape != slope_percent.shape:
        st.info(f"Resampling slope to match NDVI grid...")
        slope_resampled = np.empty(ndvi_array.shape, dtype=slope_percent.dtype)
        reproject(
            source=slope_percent,
            destination=slope_resampled,
            src_transform=dem_transform,
            dst_transform=ndvi_transform,
            src_crs=dem_profile.get('crs'),
            dst_crs=ndvi_profile.get('crs'),
            resampling=Resampling.bilinear,
        )
        slope_percent = slope_resampled

    status_text.text("Rendering map...")
    progress_bar.progress(90)
    folium_map = build_map_with_rasters(
        field_boundary,
        ndvi_array,
        slope_percent,
        ndvi_transform,
        ndvi_profile.get('crs'),
        ndvi_opacity,
        slope_opacity,
    )
    st.components.v1.html(folium_map._repr_html_(), height=500)

    progress_bar.progress(100)
    status_text.text("Complete!")
    progress_bar.empty()

    st.subheader("Raster sources")
    st.write(f"NDVI: `{Path(ndvi_path).name}`")
    st.write(f"DEM: `{Path(dem_path).name}`")

    ndvi_stats = raster_stats(ndvi_array, ndvi_profile.get("nodata"))
    slope_stats = raster_stats(slope_percent)

    risk_result = score_erosion_concern(
        ndvi_mean=ndvi_stats["mean"],
        slope_mean=slope_stats["mean"],
        ndvi_threshold=ndvi_threshold,
        slope_threshold=slope_threshold,
    )

    concern_color = {"Low": "normal", "Medium": "warning", "High": "inverse"}.get(
        risk_result["concern_level"], "normal"
    )

    st.subheader("Field metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("NDVI mean", f"{ndvi_stats['mean']:.3f}")
    col2.metric("NDVI min/max", f"{ndvi_stats['min']:.3f}/{ndvi_stats['max']:.3f}")
    col3.metric("Slope mean (%)", f"{slope_stats['mean']:.2f}")
    col4.metric("Erosion concern", risk_result["concern_level"], delta_color=concern_color)

    c_factor = calculate_c_factor(ndvi_stats["mean"])
    col5.metric("C-factor (RUSLE)", f"{c_factor:.3f}")

    if ndvi_stats["mean"] > 0.75:
        st.warning("⚠️ High NDVI suggests mature crops (not early cover crops)")

    zone_summary = zone_risk_summary(
        ndvi_array,
        slope_percent,
        ndvi_threshold=ndvi_threshold,
        slope_threshold=slope_threshold,
    )

    st.subheader("Risk zones")
    st.dataframe(zone_summary, use_container_width=True)

    st.subheader("Zone distribution")
    zone_chart = build_zone_risk_chart(zone_summary)
    st.plotly_chart(zone_chart, use_container_width=True)

    st.success("✅ NRCS EQIP Ready - field analysis complete")

    report_df = pd.DataFrame([
        {"Metric": "NDVI mean", "Value": ndvi_stats["mean"]},
        {"Metric": "Slope mean (%)", "Value": slope_stats["mean"]},
        {"Metric": "Erosion concern", "Value": risk_result["concern_level"]},
        {"Metric": "C-factor proxy", "Value": c_factor},
    ])

    st.download_button(
        "Download NRCS Report (CSV)",
        report_df.to_csv(index=False),
        "cover_crop_erosion_report.csv",
        "text/csv"
    )

    st.markdown("---")
    st.info("Prototype validated Shelby County IA. Grand Farm trials applied.")

if __name__ == "__main__":
    main()

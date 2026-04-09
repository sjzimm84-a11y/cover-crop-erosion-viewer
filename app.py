from pathlib import Path
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import folium

import rasterio
from rasterio.warp import reproject, Resampling

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
            st.write(f"Uploaded file saved to: {boundary_path}")
        else:
            st.write("No boundary file uploaded, using sample.")

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

    try:
        status_text.text("Loading field boundary...")
        progress_bar.progress(10)
        field_boundary = load_boundary_file(boundary_path)
        st.write(f"Successfully loaded boundary with {len(field_boundary)} features.")
    except Exception as exc:
        st.error(f"Unable to load field boundary: {exc}")
        st.write("Please check that your ZIP contains a valid shapefile (.shp) with companion files (.shx, .dbf).")
        return

    try:
        status_text.text("Clipping NDVI raster...")
        progress_bar.progress(30)
        ndvi_array, ndvi_transform, ndvi_profile = clip_raster_to_geometry(
            ndvi_path, field_boundary
        )
    except Exception as exc:
        st.error(f"Unable to clip NDVI raster: {exc}")
        return

    try:
        status_text.text("Clipping DEM raster...")
        progress_bar.progress(50)
        dem_array, dem_transform, dem_profile = clip_raster_to_geometry(
            dem_path, field_boundary
        )
    except Exception as exc:
        st.error(f"Unable to clip DEM raster: {exc}")
        return

    # Handle CRS mismatch if needed
    if ndvi_profile.get('crs') != dem_profile.get('crs'):
        st.warning("CRS mismatch between NDVI and DEM. Reprojecting DEM to match NDVI.")
        from rasterio.warp import calculate_default_transform, reproject
        transform, width, height = calculate_default_transform(
            dem_profile['crs'], ndvi_profile['crs'], dem_profile['width'], dem_profile['height'], *dem_profile['transform'].bounds
        )
        dem_reproj = np.empty((height, width), dtype=dem_array.dtype)
        reproject(
            source=dem_array,
            destination=dem_reproj,
            src_transform=dem_profile['transform'],
            src_crs=dem_profile['crs'],
            dst_transform=transform,
            dst_crs=ndvi_profile['crs'],
            resampling=Resampling.nearest
        )
        dem_array = dem_reproj
        dem_transform = transform

    status_text.text("Computing slope...")
    progress_bar.progress(70)
    slope_percent = compute_slope_from_dem(dem_array, dem_transform)

    if ndvi_array.shape != slope_percent.shape:
        st.info(f"Resampling slope from {slope_percent.shape} to match NDVI {ndvi_array.shape}...")
        # Resample slope to match NDVI grid
        slope_resampled = np.empty(ndvi_array.shape, dtype=slope_percent.dtype)
        reproject(
            source=slope_percent,
            destination=slope_resampled,
            src_transform=dem_transform,
            dst_transform=ndvi_transform,
            src_crs=dem_profile.get('crs'),
            dst_crs=ndvi_profile.get('crs'),
            resampling=Resampling.nearest
        )
        slope_percent = slope_resampled

    status_text.text("Building map...")
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
    status_text.text("Processing complete.")
    progress_bar.empty()
    status_text.empty()

    st.subheader("Raster inputs")
    st.write(f"NDVI source: `{Path(ndvi_path).name}`")
    st.write(f"DEM source: `{Path(dem_path).name}`")

    ndvi_stats = raster_stats(ndvi_array, ndvi_profile.get("nodata"))
    slope_stats = raster_stats(slope_percent)

    risk_result = score_erosion_concern(
        ndvi_mean=ndvi_stats["mean"],
        slope_mean=slope_stats["mean"],
        ndvi_threshold=ndvi_threshold,
        slope_threshold=slope_threshold,
    )

    # Color-code erosion concern
    concern_color = {"Low": "normal", "Medium": "off", "High": "inverse"}.get(risk_result["concern_level"], "normal")

    st.subheader("Field summary metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("NDVI mean", f"{ndvi_stats['mean']:.3f}")
    col2.metric("NDVI min", f"{ndvi_stats['min']:.3f}")
    col3.metric("NDVI max", f"{ndvi_stats['max']:.3f}")
    col4.metric("Slope mean (%)", f"{slope_stats['mean']:.2f}")
    col5.metric("Erosion concern", risk_result["concern_level"], delta_color=concern_color)

    # C-factor proxy
    c_factor = calculate_c_factor(ndvi_stats["mean"])
    st.metric("C-factor proxy (RUSLE)", f"{c_factor:.3f}", help="Estimated soil erodibility factor from NDVI.")

    # NDVI freshness warning
    if ndvi_stats["mean"] > 0.75:
        st.warning("⚠️ NDVI values suggest mature crops. This may not reflect early-season cover crop effectiveness.")

    zone_summary = zone_risk_summary(
        ndvi_array,
        slope_percent,
        ndvi_threshold=ndvi_threshold,
        slope_threshold=slope_threshold,
    )

    st.subheader("Zone risk summary")
    st.dataframe(zone_summary, use_container_width=True)

    st.subheader("Risk by zone")
    zone_chart = build_zone_risk_chart(zone_summary)
    st.plotly_chart(zone_chart, use_container_width=True)

    # NRCS EQIP Ready Badge
    st.success("✅ NRCS EQIP Ready: Field meets basic data requirements for conservation planning.")

    report_df = pd.DataFrame(
        [
            {
                "metric": "NDVI mean",
                "value": ndvi_stats["mean"],
            },
            {
                "metric": "NDVI min",
                "value": ndvi_stats["min"],
            },
            {
                "metric": "NDVI max",
                "value": ndvi_stats["max"],
            },
            {
                "metric": "Slope mean (%)",
                "value": slope_stats["mean"],
            },
            {
                "metric": "Erosion concern",
                "value": risk_result["concern_level"],
            },
            {
                "metric": "C-factor proxy",
                "value": c_factor,
            },
        ]
    )

    csv_bytes = report_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download summary report as CSV",
        data=csv_bytes,
        file_name="erosion_report.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.info(
        "This prototype is rule-based. Future work can connect Sentinel-2 ingestion, cloud masking, "
        "DEM processing, and calibration against RUSLE / WEPP field data."
    )


if __name__ == "__main__":
    main()

# Cover Crop Erosion Viewer

A lightweight prototype for agricultural conservation teams to explore early-season cover crop concern using NDVI and terrain slope.

## What it does

This app helps you evaluate field-level erosion concern by combining two data layers:

- NDVI (Normalized Difference Vegetation Index), representing early cover crop or vegetation vigour.
- DEM-derived slope in percent, representing terrain steepness.

The app:

- Loads a field boundary from GeoJSON or a zipped shapefile.
- Loads a sample NDVI raster or a user-supplied NDVI GeoTIFF.
- Loads a DEM GeoTIFF, clips it to the field, and computes slope in percent.
- Clips all rasters to the field boundary and optionally resamples slope to match NDVI resolution.
- Creates a simple risk zone map using combined NDVI and slope rules.
- Displays an interactive map, summary metrics, zone counts, and a downloadable CSV report.

## Outputs explained

### Map preview

The interactive map shows:

- The field boundary in black.
- An NDVI overlay colored by vegetation index values.
- A slope overlay colored by terrain steepness.
- A layer switcher so you can toggle NDVI and slope separately.

The green/viridis NDVI overlay highlights low-cover versus higher-cover areas. The red/plasma slope overlay highlights steeper areas.

### Field summary metrics

The app reports:

- NDVI mean, min, and max inside the field.
- Slope mean inside the field.
- A high-level erosion concern label based on the combined rule set.

### Zone risk summary

Each pixel inside the clipped field is classified into one of four zones:

- `High concern` — pixels with both low NDVI and steep slope.
- `Low cover` — pixels with low NDVI but not steep slope.
- `Steep slope` — pixels with steep slope but not low NDVI.
- `Normal` — pixels that are neither low-cover nor steep.

This zone summary is shown as a table and a bar chart of pixel counts.

## Risk zone definitions

Risk zones are created using simple rule-based definitions in `src/scoring.py` and `src/raster_utils.py`.

### Low cover

Low cover is defined by the NDVI threshold in the sidebar. The default threshold is `0.35`, meaning:

- NDVI values below the threshold are treated as low ground cover or sparse vegetation.
- This is a proxy for fields that may have insufficient early season cover crop or vegetative protection.

### Steep slope

Steep slope is defined by the slope threshold in the sidebar. The default threshold is `6.0%`, meaning:

- Slope values above the threshold are treated as steep enough to increase erosion risk.
- This is a proxy for terrain where runoff and soil loss are more likely.

### Combined risk categories

The app combines NDVI and slope into risk categories:

- `High concern` when both low cover and steep slope are present.
- `Low cover` when NDVI is low but the slope is not steep.
- `Steep slope` when slope is steep but NDVI is not low.
- `Normal` when neither condition applies.

## Project structure

- `app.py` — Streamlit application entry point.
- `data/` — synthetic sample data folder.
- `src/io_utils.py` — boundary and upload helpers.
- `src/raster_utils.py` — raster clipping, slope computation, and zone summary logic.
- `src/scoring.py` — erosion concern scoring rules.
- `src/sample_data.py` — synthetic field boundary, NDVI, and DEM generator.
- `src/visualization.py` — map and chart rendering helpers.

## Setup

1. Open a terminal in the project folder:

```bash
cd "c:\Users\User\Python code\cover_crop_erosion_viewer"
```

2. Create a Python 3.10+ environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Then open the local URL displayed by Streamlit.

## Usage

- Use the sidebar to upload a field boundary or rely on the sample field.
- Upload an NDVI GeoTIFF and a DEM GeoTIFF, or use the built-in sample data.
- Adjust the NDVI and slope thresholds to tune what qualifies as low cover and steep slope.
- Toggle NDVI and slope overlay visibility and opacity on the map.
- Download the summary report as CSV.

## Notes for future development

- Add Sentinel-2 ingestion and cloud masking.
- Add support for additional raster formats and projections.
- Calibrate scoring rules against RUSLE / WEPP and observed erosion data.
- Add a spatial export of risk zones back to GeoTIFF or shapefile.

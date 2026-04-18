# How to Set Up DEM Data for CoverMap

## Overview
This guide covers two paths for DEM data:
1. **Automatic** — Iowa 3m WCS API (recommended, zero setup for Iowa fields)
2. **Manual** — Creating a correctly formatted GeoTIFF for upload

---

## Path 1: Automatic Iowa 3m WCS (No Setup Required)

The app now auto-fetches the Iowa 3m DEM from the Iowa Geospatial Data
Clearinghouse whenever a field boundary is uploaded. No DEM file needed.

### What happens automatically:
1. User uploads field boundary (GeoJSON or shapefile)
2. App buffers boundary by 50m to ensure full edge coverage
3. App requests DEM tiles from `geodata.iowa.gov` WCS endpoint
4. DEM arrives in meters, correctly scaled — no unit conversion issues
5. If WCS is unavailable, app falls back to uploaded DEM or sample data

### When automatic fetch will fail:
- Field is outside Iowa state boundary
- No internet connection
- Iowa Geospatial server is down (rare)
- Field boundary CRS is unreadable

In any of these cases the app shows a warning and prompts for manual upload.

---

## Path 2: Manual DEM GeoTIFF Creation

Use this path when:
- Working outside Iowa
- Need higher resolution than 3m
- WCS is unavailable
- Using your own lidar data

### Step 1: Download the Iowa 3m DEM

1. Go to: https://geodata.iowa.gov
2. Search for **"3 Meter Digital Elevation Model"**
3. Navigate to the county tile covering your field
4. Download the GeoTIFF tile (.tif)

### Step 2: Verify correct units in QGIS

**Critical:** The Iowa 3m DEM from the clearinghouse is stored in
**meters** natively. Do NOT use scaled UInt16 versions — these store
values in centimeters (38,000-43,000 range) and will produce
incorrect slope calculations unless the app detects and converts them.

To check units in QGIS:
1. Right-click layer → Properties → Information
2. Check Band 1 Min/Max values
3. If values are 150-600 → meters ✅ correct
4. If values are 15,000-60,000 → centimeters ⚠️ needs conversion
5. If values are 150,000-600,000 → millimeters ⚠️ needs conversion

### Step 3: Reproject to correct CRS in QGIS

The app expects the DEM in a projected CRS (meters), not geographic (degrees).
EPSG:26915 (NAD83 UTM Zone 15N) is the standard for Iowa.

1. Raster → Projections → Warp (Reproject)
2. Input: your DEM file
3. Target CRS: EPSG:26915
4. Resampling: Bilinear
5. Output: save as new .tif file

### Step 4: Compute percent slope in QGIS (for verification)

**Important:** Set Scale correctly based on your DEM units:

| DEM units | Scale value |
|-----------|-------------|
| Meters    | 1           |
| Centimeters | 100       |
| Millimeters | 1000      |

Steps:
1. Raster → Analysis → Slope
2. Input: your reprojected DEM
3. ✅ Check "Slope expressed as percent" (AS_PERCENT = True)
4. Scale: set based on table above (1 for meters)
5. Run → check output Min/Max
6. Expected range for Iowa: 0-60% (steep Shelby County loess hills)
7. If you see 0-6000% → Scale was wrong, rerun with correct value

### Step 5: Export correctly for app upload

In QGIS:
1. Right-click the DEM layer → Export → Save As
2. Format: GeoTIFF
3. CRS: EPSG:26915
4. Data type: Float32 (important — do not use UInt16)
5. NoData value: -9999
6. File name: something descriptive like `shelby_north_dem_meters.tif`

### Step 6: Upload to the app

1. Open the app sidebar → DEM section
2. Upload your .tif file
3. App will detect units automatically and convert if needed
4. Check the slope map — steep fields should show orange/red zones

---

## Troubleshooting Common DEM Issues

### Problem: Slope map shows all blue (flat) on a steep field
**Cause:** DEM units are centimeters but app computed slope in wrong units
**Fix:** The app now auto-detects centimeter DEMs. If still showing flat,
check that you uploaded the latest `src/raster_utils.py`

### Problem: Slope values in table show 40%+ on a moderate field
**Cause:** Same unit mismatch as above but in opposite direction
**Fix:** Export DEM as Float32 meters from QGIS before uploading

### Problem: "Input shapes do not overlap raster" error
**Cause:** Field boundary and DEM are in different geographic areas
or the DEM does not cover the field extent
**Fix:** Make sure your DEM tile covers the full field boundary.
Download adjacent tiles if field is near a tile edge and merge them
in QGIS: Raster → Miscellaneous → Merge

### Problem: WCS auto-fetch fails with connection error
**Cause:** Iowa Geospatial server temporarily unavailable
**Fix:** Upload a manual DEM file. The app will use it automatically
as a fallback without any code changes needed.

---

## Iowa WCS Technical Details (for developers)

```
Endpoint: https://geodata.iowa.gov/arcgis/services/DEM_3M/ImageServer/WCSServer
Version:  WCS 1.0.0
Coverage: DEM_3M
CRS:      EPSG:26915 (NAD83 UTM Zone 15N)
Format:   GeoTIFF
Units:    Meters (native, no conversion needed)
Resolution: 3m native
Max request: 4000 x 4000 pixels (enforced by iowa_dem_utils.py)
```

---

## NDVI GeoTIFF Creation Guide

*(See companion document: HOW_TO_NDVI_SETUP.md)*

For now, recommended sources for Iowa early-season NDVI:
- **Google Earth Engine** — free, cloud-based, exports GeoTIFF directly
- **Copernicus Browser** (browser.dataspace.copernicus.eu) — manual download
- **EarthExplorer** (earthexplorer.usgs.gov) — Landsat/Sentinel archive

Key requirements for NDVI upload:
- Single band Float32 GeoTIFF
- Values in range -1.0 to 1.0
- Projected CRS (not geographic degrees)
- Captured March-April for early spring cover crop assessment
- NoData value set (typically -9999 or 0)

---

*CoverMap — Stephen Zimmerman CCA MS — Ankeny IA*
*Last updated: April 2026*

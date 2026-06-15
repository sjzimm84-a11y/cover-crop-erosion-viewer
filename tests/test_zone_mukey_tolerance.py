"""
test_zone_mukey_tolerance.py
----------------------------
Tests for the Phase-4 zone x map-unit soil-tolerance wiring
(src.soil_data.zone_mukey_tolerance_rows and its _repair / _utm_epsg_for_lonlat
helpers).

The four offline scenarios mirror the Phase-4 validation harness using purely
synthetic geometry (no network):

    (a) single map unit, T=5  -> per-zone within_t_zone + G2 acre reconciliation
    (b) multi map unit T=4/T=6 -> flag/roll filtering behaviour
    (c) None/None inputs       -> graceful degrade, no crash
    (d) bowtie / zero-area     -> G3 make_valid repair vs skip

One live-SDA test reconciles real Monona SSURGO polygons (K=0.37, T=5) and is
skipped automatically when the SDA endpoint is unreachable, so offline / CI runs
never block on it.

CRS contract guarded explicitly: intersection in EPSG:4326 (G1), acreage in the
field UTM EPSG (G2) — never degree-area.

Run:
    python -m pytest tests/test_zone_mukey_tolerance.py -v
"""

import functools

import numpy as np
import geopandas as gpd
import pytest
import requests
from rasterio.transform import from_origin
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape as shp_shape, box, Polygon
from shapely.ops import unary_union

from src import soil_data
from src.soil_data import (
    zone_mukey_tolerance_rows,
    _repair,
    _utm_epsg_for_lonlat,
)

# ---------------------------------------------------------------------------
# Synthetic raster grid over Shelby County, IA (WGS84 degrees) — deterministic.
# 50x50 pixels at ~0.0001 deg (~8.3 m lon x 11.06 m lat) -> ~57 acres total.
# ---------------------------------------------------------------------------
WEST, NORTH = -95.3520, 41.7520
PX = 0.0001
N = 50
_TRANSFORM = from_origin(WEST, NORTH, PX, PX)
_NDVI_CRS = "EPSG:4326"
_ACRE_DIVISOR = 4046.8564224
EXPECTED_UTM = "EPSG:32615"  # field UTM zone 15N for Iowa


def _polygonize_zones(zone_array, transform, raster_crs):
    """Replicates the app.py polygonize + G1 reproject path.

    Returns ({zone_val: WGS84 geometry}, auto-selected UTM acre CRS string).
    """
    zint = np.where(np.isnan(zone_array), 0, zone_array).astype(np.int32)
    zvalid = (zint > 0).astype(np.uint8)
    parts = {}
    for gdict, val in rio_shapes(zint, mask=zvalid, transform=transform):
        v = int(val)
        if v in (1, 2, 3, 4):
            parts.setdefault(v, []).append(shp_shape(gdict))
    native = {v: unary_union(p) for v, p in parts.items()}
    zgs = gpd.GeoSeries(list(native.values()), crs=raster_crs)
    acre_crs = str(zgs.estimate_utm_crs())
    zgs_ll = zgs.to_crs("EPSG:4326")
    return dict(zip(native.keys(), zgs_ll.geometry)), acre_crs


def _acres_in(geom, acre_crs):
    return float(
        gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(acre_crs).area.iloc[0]
        / _ACRE_DIVISOR
    )


@pytest.fixture(scope="module")
def zones():
    """Top half Critical(4) / bottom half Low(1), polygonized + reprojected."""
    zone_arr = np.empty((N, N), dtype=float)
    zone_arr[:25, :] = 4.0
    zone_arr[25:, :] = 1.0
    geoms, acre_crs = _polygonize_zones(zone_arr, _TRANSFORM, _NDVI_CRS)
    return geoms, acre_crs


@pytest.fixture(scope="module")
def field_box():
    return box(WEST, NORTH - N * PX, WEST + N * PX, NORTH)


# ---------------------------------------------------------------------------
# (a) single map unit, T=5 — no-op equivalence + G2 acreage reconciliation
# ---------------------------------------------------------------------------
def test_single_unit_soil_t_and_within(zones, field_box):
    zone_geometries, acre_crs = zones
    mupolys = [("MU_MONONA", field_box, 5.0)]
    a_by_zone = {"Critical": 10.0, "Low": 2.0}  # 10>5 -> False ; 2<=5 -> True

    res = zone_mukey_tolerance_rows(zone_geometries, mupolys, a_by_zone, acre_crs=acre_crs)
    rows = {r["risk_zone"]: r for r in res["rows_full"]}

    assert set(rows) == {"Critical", "Low"}
    assert all(r["soil_T"] == 5.0 for r in res["rows_full"])
    assert rows["Critical"]["within_t_zone"] is False
    assert rows["Low"]["within_t_zone"] is True


def test_single_unit_acreage_reconciles_field(zones, field_box):
    """G2: zone x mukey overlap acres must sum to the field acreage."""
    zone_geometries, acre_crs = zones
    mupolys = [("MU_MONONA", field_box, 5.0)]
    res = zone_mukey_tolerance_rows(
        zone_geometries, mupolys, {"Critical": 10.0, "Low": 2.0}, acre_crs=acre_crs
    )
    sum_acres = sum(r["overlap_acres"] for r in res["rows_full"])
    field_acres = _acres_in(field_box, acre_crs)
    assert abs(sum_acres - field_acres) < 0.5, (sum_acres, field_acres)


# ---------------------------------------------------------------------------
# CRS contract — G1 intersection EPSG:4326, G2 acreage field UTM
# ---------------------------------------------------------------------------
def test_crs_contract_intersection_and_acreage(zones, field_box):
    zone_geometries, acre_crs = zones
    assert acre_crs == EXPECTED_UTM  # field UTM derived in the polygonize path

    mupolys = [("MU_MONONA", field_box, 5.0)]
    res = zone_mukey_tolerance_rows(zone_geometries, mupolys, {"Critical": 3.0})
    # auto-derived UTM when acre_crs not supplied
    assert res["intersect_crs"] == "EPSG:4326"
    assert res["acre_crs"] == EXPECTED_UTM


def test_acreage_not_degree_area(zones, field_box):
    """Guard against silent regression to degree-area: a ~28-acre half-field
    overlap must read in acres (tens), not the ~3e-6 a degree-area would give."""
    zone_geometries, acre_crs = zones
    mupolys = [("MU_MONONA", field_box, 5.0)]
    res = zone_mukey_tolerance_rows(zone_geometries, mupolys, {"Critical": 3.0}, acre_crs=acre_crs)
    for r in res["rows_full"]:
        assert r["overlap_acres"] > 1.0, r  # degree-area would be << 1e-3


# ---------------------------------------------------------------------------
# (b) multi map unit T=4 (low-T) + T=6 — flag / roll / sub-floor behaviour
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def multi_unit_result(zones):
    zone_geometries, acre_crs = zones
    midlon = WEST + 25 * PX
    left = box(WEST, NORTH - N * PX, midlon, NORTH)          # MU_LOWT  T=4
    right = box(midlon, NORTH - N * PX, WEST + N * PX, NORTH)  # MU_HIGHT T=6
    tiny = box(WEST, NORTH - 4 * PX, WEST + 4 * PX, NORTH)     # MU_SUBFLR T=4, <0.5 ac
    mupolys = [("MU_LOWT", left, 4.0), ("MU_HIGHT", right, 6.0), ("MU_SUBFLR", tiny, 4.0)]
    a_by_zone = {"Critical": 5.0, "Low": 0.5}
    res = zone_mukey_tolerance_rows(zone_geometries, mupolys, a_by_zone, acre_crs=acre_crs)
    return res, acre_crs, tiny


def test_critical_low_t_row_is_flagged(multi_unit_result):
    res, _, _ = multi_unit_result
    flagged = res["rows_flagged"]
    assert len(flagged) == 1
    row = flagged[0]
    assert row["risk_zone"] == "Critical"
    assert row["mukey"] == "MU_LOWT"
    assert row["soil_T"] == 4.0
    assert row["within_t_zone"] is False
    assert row["overlap_acres"] >= 0.5
    assert 10.0 < row["overlap_acres"] < 20.0  # sane quarter-field acreage


def test_subfloor_row_rolled_not_flagged(multi_unit_result):
    res, acre_crs, tiny = multi_unit_result
    assert _acres_in(tiny, acre_crs) < 0.5  # genuinely sub-floor
    full_mukeys = {r["mukey"] for r in res["rows_full"]}
    flagged_mukeys = {r["mukey"] for r in res["rows_flagged"]}
    assert "MU_SUBFLR" in full_mukeys           # present in full table
    assert "MU_SUBFLR" not in flagged_mukeys    # but rolled, not surfaced


def test_within_tolerance_row_rolled(multi_unit_result):
    res, _, _ = multi_unit_result
    # Critical x MU_HIGHT (T=6): 5 <= 6 -> within True -> must be rolled, not flagged
    crit_high = [r for r in res["rows_full"]
                 if r["risk_zone"] == "Critical" and r["mukey"] == "MU_HIGHT"]
    assert len(crit_high) == 1
    assert crit_high[0]["within_t_zone"] is True
    assert all(r["mukey"] != "MU_HIGHT" or r["risk_zone"] != "Critical"
               for r in res["rows_flagged"])


def test_rolled_summary_accounts_remaining_rows(multi_unit_result):
    res, _, _ = multi_unit_result
    n_full = len(res["rows_full"])
    n_flagged = len(res["rows_flagged"])
    assert res["rolled_summary"]["n_rows"] == n_full - n_flagged
    rolled_acres = sum(
        r["overlap_acres"] for r in res["rows_full"] if r not in res["rows_flagged"]
    )
    assert abs(res["rolled_summary"]["overlap_acres"] - round(rolled_acres, 2)) < 0.05


# ---------------------------------------------------------------------------
# (c) failure injection -> graceful degrade, no crash
# ---------------------------------------------------------------------------
def test_none_inputs_degrade_to_empty():
    res = zone_mukey_tolerance_rows(None, None, {})
    assert res["rows_full"] == []
    assert res["rows_flagged"] == []
    assert res["rolled_summary"] == {"n_rows": 0, "overlap_acres": 0.0}


def test_empty_mupolygons_degrade_to_empty(zones):
    zone_geometries, acre_crs = zones
    res = zone_mukey_tolerance_rows(zone_geometries, [], {"Critical": 3.0}, acre_crs=acre_crs)
    assert res["rows_full"] == []
    assert res["rolled_summary"]["n_rows"] == 0


# ---------------------------------------------------------------------------
# (d) G3 — invalid (bowtie) repaired/retained; zero-area sliver skipped
# ---------------------------------------------------------------------------
def test_repair_helper_behaviour():
    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])  # self-intersecting
    sliver = Polygon([(0, 0), (1, 0), (0, 0)])          # degenerate, zero-area
    assert not bowtie.is_valid
    repaired = _repair(bowtie)
    assert repaired is not None and repaired.is_valid

    # _repair guarantees valid + non-empty; a degenerate zero-area polygon may
    # collapse to a valid line, but it carries no area, so it contributes no
    # acreage downstream (the `acres <= 0` guard in the main loop drops it —
    # exercised end-to-end in test_g3_invalid_and_sliver_in_pipeline).
    repaired_sliver = _repair(sliver)
    assert repaired_sliver is None or repaired_sliver.area == 0

    assert _repair(None) is None
    assert _repair(Polygon()) is None                    # empty -> dropped


def test_g3_invalid_and_sliver_in_pipeline(zones, field_box):
    zone_geometries, acre_crs = zones
    bowtie = Polygon([(WEST, NORTH), (WEST + N * PX, NORTH - N * PX),
                      (WEST + N * PX, NORTH), (WEST, NORTH - N * PX)])
    sliver = Polygon([(WEST, NORTH), (WEST + N * PX, NORTH), (WEST, NORTH)])
    mupolys = [("MU_BOWTIE", bowtie, 5.0), ("MU_SLIVER", sliver, 5.0),
               ("MU_GOOD", field_box, 5.0)]
    # Must not raise on the bad slivers.
    res = zone_mukey_tolerance_rows(
        zone_geometries, mupolys, {"Critical": 3.0, "Low": 1.0}, acre_crs=acre_crs
    )
    mukeys = {r["mukey"] for r in res["rows_full"]}
    assert "MU_SLIVER" not in mukeys     # zero-area never contributes
    assert "MU_GOOD" in mukeys           # valid unit retained
    assert len(res["rows_full"]) > 0     # run completed with output


def test_utm_epsg_helper():
    # Iowa (~ -95 lon, +41 lat) is UTM zone 15N.
    assert _utm_epsg_for_lonlat(-95.35, 41.75) == "EPSG:32615"
    # Southern hemisphere uses the 327xx band.
    assert _utm_epsg_for_lonlat(-58.0, -34.6) == "EPSG:32721"


# ---------------------------------------------------------------------------
# Live SDA reconciliation — real Monona field (K=0.37, T=5). Auto-skips offline.
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _sda_reachable():
    try:
        r = requests.post(
            soil_data.SDA_ENDPOINT,
            data={"query": "SELECT TOP 1 mukey FROM mapunit",
                  "format": "JSON+COLUMNNAME+METADATA"},
            timeout=8,
        )
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _sda_reachable(), reason="SDA endpoint unreachable (offline/CI)")
def test_live_monona_k_t_and_acreage_reconciliation():
    # Per-map-unit component-weighted K/T for the Monona reference set.
    per = soil_data.component_weighted(
        soil_data.fetch_soil_components(["626747", "626748", "780777"])
    )
    assert per, "SDA returned no components for Monona reference mukeys"
    for mk in ("626747", "626748", "780777"):
        assert per[mk]["t"] == pytest.approx(5.0)
        assert per[mk]["k"] == pytest.approx(0.37, abs=0.02)

    # Acreage reconciliation against real SSURGO polygons clipped to a field box.
    aoi = box(-95.352, 41.747, -95.344, 41.752)
    mupolys_raw = soil_data.fetch_mupolygons(aoi)
    assert mupolys_raw, "SDA returned no mupolygons for the AOI"
    mu_per = soil_data.component_weighted(
        soil_data.fetch_soil_components(sorted({mk for mk, _ in mupolys_raw}))
    )
    mupolygons = [(mk, g, mu_per.get(mk, {}).get("t")) for mk, g in mupolys_raw]

    # Whole AOI as a single Critical zone; overlaps must tile the AOI.
    zone_geometries = {4: aoi}
    res = zone_mukey_tolerance_rows(zone_geometries, mupolygons, {"Critical": 3.0})
    assert res["acre_crs"] == EXPECTED_UTM
    assert res["intersect_crs"] == "EPSG:4326"

    sum_acres = sum(r["overlap_acres"] for r in res["rows_full"])
    aoi_acres = _acres_in(aoi, EXPECTED_UTM)
    # SSURGO fully tiles the AOI, so clipped overlaps reconcile to ~field acres.
    assert abs(sum_acres - aoi_acres) / aoi_acres < 0.02, (sum_acres, aoi_acres)

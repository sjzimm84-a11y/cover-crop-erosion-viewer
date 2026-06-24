"""
soil_data.py
------------
USDA Soil Data Access (SDA) area-weighted soil property module.

Foundation module (Phase 1): queries SDA for SSURGO map-unit / component /
surface-horizon data within a field boundary and returns AREA-WEIGHTED
K-factor (kwfact) and T-value (tfact), plus drainage-class attributes for a
future N-leaching pathway.

This module is intentionally standalone. It is NOT yet wired into scoring.py,
report_generator.py, or export_utils.py — that integration is Phase 2.

SDA contract (verified 2026-06-11 against the live service)
-----------------------------------------------------------
    Endpoint : https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest
    POST form: {"query": <sql>, "format": "JSON+COLUMNNAME+METADATA"}
    Response : data["Table"][0]  = column headers
               data["Table"][1]  = column type metadata  (skipped)
               data["Table"][2:] = data rows
    No auth, no key.

    NOTE on format: the original spec named the format "JSON+COLUMNNAME" while
    also describing Table[1] as a types row to skip. Those are inconsistent on
    the live service: the plain "JSON+COLUMNNAME" variant returns headers at
    Table[0] and DATA beginning at Table[1] (no metadata row), whereas the
    Table[0]=headers / Table[1]=types / Table[2:]=data layout the spec
    describes is the "JSON+COLUMNNAME+METADATA" variant. This module uses
    "+METADATA" so the documented row offsets hold exactly. SDA returns an
    empty body (not JSON) when a query matches zero rows.

Verified SSURGO column locations
--------------------------------
    component table : mukey, cokey, comppct_r, drainagecl, hydgrp, tfact,
                      majcompflag
    chorizon table  : hzdept_r (horizon top depth), kwfact (K factor, whole
                      soil), kffact (K factor, rock-free), ksat_r (saturated
                      hydraulic conductivity)
    Surface horizon = the horizon with the minimum hzdept_r for a component.

Spatial area weighting
----------------------
    Per-mukey area fractions are computed in shapely (deterministic, testable):
    intersecting mupolygon geometries are fetched from SDA as WGS84 WKT and
    each is intersected with the field boundary. Because all polygons of a
    single field lie at essentially one latitude, the longitude/latitude
    degree-area distortion is a constant factor across every polygon and
    cancels out of the normalized fractions, so raw planar shapely areas yield
    correct relative fractions without reprojection.

    shapely (>=2.0) and requests (>=2.31) are already project dependencies;
    no new dependency is introduced.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import pandas as pd
import requests
from shapely import wkt as shapely_wkt
from shapely.geometry.base import BaseGeometry

_log = logging.getLogger(__name__)

SDA_ENDPOINT = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"
_SDA_FORMAT = "JSON+COLUMNNAME+METADATA"
_HTTP_TIMEOUT = 60


# ---------------------------------------------------------------------------
# 1. Low-level POST wrapper
# ---------------------------------------------------------------------------
def query_sda(sql: str) -> list[dict]:
    """Run a Tabular query against USDA Soil Data Access and return rows as dicts.

    POSTs ``sql`` to the SDA REST endpoint with format
    ``JSON+COLUMNNAME+METADATA``, then parses the response: ``Table[0]`` holds
    the column headers, ``Table[1]`` holds column type metadata (skipped), and
    ``Table[2:]`` are the data rows. Each data row is zipped with the headers
    into a ``dict``.

    Args:
        sql: A T-SQL query string accepted by SDA.

    Returns:
        A list of ``{column_name: value}`` dicts, one per data row. Returns an
        empty list when SDA returns an empty body (zero-row match) or fewer
        than three table rows (i.e. no data rows).

    Raises:
        requests.HTTPError: If SDA responds with a non-2xx status code.
    """
    resp = requests.post(
        SDA_ENDPOINT,
        data={"query": sql, "format": _SDA_FORMAT},
        timeout=_HTTP_TIMEOUT,
    )
    resp.raise_for_status()

    # SDA returns an empty body (not JSON) when the query matches no rows.
    if not resp.text.strip():
        return []

    data = resp.json()
    table = data.get("Table")
    if not table or len(table) < 3:
        return []

    headers = table[0]
    # table[1] is column type metadata — intentionally skipped.
    return [dict(zip(headers, row)) for row in table[2:]]


# ---------------------------------------------------------------------------
# 2. Map-unit -> component -> surface-horizon fetch
# ---------------------------------------------------------------------------
def fetch_soil_components(mukeys: list[str]) -> pd.DataFrame:
    """Fetch major-component soil attributes (surface horizon) for map units.

    Joins ``mapunit`` -> ``component`` -> ``chorizon`` for the given map units,
    keeping only major components (``majcompflag = 'Yes'``) and, for each
    component, the surface horizon (minimum ``hzdept_r``). Pulls per component:
    ``mukey``, ``cokey``, ``comppct_r``, ``drainagecl``, ``hydgrp``, component
    ``tfact``, and surface-horizon ``kwfact`` and ``ksat_r``.

    Args:
        mukeys: SSURGO map unit keys (as strings).

    Returns:
        A ``pandas.DataFrame`` with columns ``mukey``, ``cokey``,
        ``comppct_r``, ``drainagecl``, ``hydgrp``, ``tfact``, ``kwfact``,
        ``ksat_r``. Numeric columns (``comppct_r``, ``tfact``, ``kwfact``,
        ``ksat_r``) are coerced to floats with unparseable values as ``NaN``.
        Returns an empty DataFrame (with those columns) if ``mukeys`` is empty
        or SDA returns no rows. Duplicate rows per ``cokey`` (possible when a
        component has two horizons sharing the minimum depth) are de-duplicated,
        keeping the first.
    """
    cols = ["mukey", "musym", "cokey", "comppct_r", "drainagecl",
            "hydgrp", "tfact", "kwfact", "ksat_r"]
    if not mukeys:
        return pd.DataFrame(columns=cols)

    in_list = ",".join(f"'{m}'" for m in mukeys)
    # musym is a mapunit-level attribute (one value per mukey); selecting it from
    # the already-joined mapunit table is functionally dependent on mu.mukey and
    # does NOT change row cardinality, so the component-level comppct_r weighting
    # below is unaffected.
    sql = f"""
        SELECT mu.mukey, mu.musym, c.cokey, c.comppct_r, c.drainagecl, c.hydgrp,
               c.tfact, ch.kwfact, ch.ksat_r
        FROM mapunit mu
        JOIN component c ON mu.mukey = c.mukey
        JOIN chorizon ch ON c.cokey = ch.cokey
        WHERE mu.mukey IN ({in_list})
          AND c.majcompflag = 'Yes'
          AND ch.hzdept_r = (
              SELECT MIN(ch2.hzdept_r)
              FROM chorizon ch2
              WHERE ch2.cokey = c.cokey
          )
    """
    rows = query_sda(sql)
    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows, columns=cols)
    df = df.drop_duplicates(subset="cokey", keep="first").reset_index(drop=True)

    for numeric_col in ("comppct_r", "tfact", "kwfact", "ksat_r"):
        df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")

    # Trim whitespace from categorical text (SDA pads e.g. 'No ').
    for text_col in ("drainagecl", "hydgrp", "musym"):
        df[text_col] = df[text_col].astype("string").str.strip()

    return df


# ---------------------------------------------------------------------------
# 3. Component aggregation within a map unit (weighted by comppct_r)
# ---------------------------------------------------------------------------
def component_weighted(df: pd.DataFrame) -> dict:
    """Aggregate components to one record per map unit, weighted by comppct_r.

    Within each ``mukey``, computes the ``comppct_r``-weighted mean of
    ``kwfact`` (K), ``tfact`` (T) and ``ksat_r``. For each weighted property,
    only components with a non-null value contribute and the weights are
    renormalized over those components. Drainage class and hydrologic group are
    carried from the dominant component (largest ``comppct_r``).

    Args:
        df: A DataFrame as returned by :func:`fetch_soil_components`.

    Returns:
        A dict keyed by ``mukey``. Each value is a dict with keys ``k``, ``t``,
        ``ksat`` (weighted floats, or ``None`` if no component had a value),
        ``drainagecl`` and ``hydgrp`` (dominant-component strings, or ``None``),
        and ``comppct_total`` (sum of major-component percentages). Returns an
        empty dict for an empty DataFrame.
    """
    result: dict = {}
    if df is None or df.empty:
        return result

    def _weighted_mean(sub: pd.DataFrame, value_col: str) -> Optional[float]:
        valid = sub[["comppct_r", value_col]].dropna()
        weight_sum = valid["comppct_r"].sum()
        if valid.empty or weight_sum <= 0:
            return None
        return float((valid[value_col] * valid["comppct_r"]).sum() / weight_sum)

    for mukey, sub in df.groupby("mukey"):
        dominant = sub.loc[sub["comppct_r"].idxmax()] if sub["comppct_r"].notna().any() else sub.iloc[0]
        # musym is mapunit-level (constant across this group's component rows);
        # carried through verbatim, it does not participate in the weighting.
        _musym = dominant["musym"] if "musym" in sub.columns else None
        result[str(mukey)] = {
            "k": _weighted_mean(sub, "kwfact"),
            "t": _weighted_mean(sub, "tfact"),
            "ksat": _weighted_mean(sub, "ksat_r"),
            "musym": (None if pd.isna(_musym) else str(_musym)),
            "drainagecl": (None if pd.isna(dominant["drainagecl"])
                           else str(dominant["drainagecl"])),
            "hydgrp": (None if pd.isna(dominant["hydgrp"])
                       else str(dominant["hydgrp"])),
            "comppct_total": float(sub["comppct_r"].sum(skipna=True)),
        }
    return result


# ---------------------------------------------------------------------------
# 4. Area-weighted aggregation across map units within the field boundary
# ---------------------------------------------------------------------------
def area_weighted_soil(
    boundary_geom: Optional[BaseGeometry],
    mukey_weights: dict[str, float],
    per_mukey: Optional[dict] = None,
) -> dict:
    """Combine per-map-unit soil properties into field area-weighted values.

    Given the area fraction of each map unit within the field boundary, returns
    the area-weighted K-factor, T-value and ksat, plus an area-fraction
    breakdown by drainage class. Per-map-unit properties come from
    :func:`fetch_soil_components` + :func:`component_weighted`; pass a
    precomputed ``per_mukey`` mapping to avoid the network call (keeps this
    function pure and unit-testable).

    Map units with no usable soil data are dropped and the remaining area
    fractions are renormalized, so the returned weights always sum to 1.0 over
    the map units that actually contributed.

    Args:
        boundary_geom: The field boundary (WGS84 shapely geometry). Reserved
            for future use (e.g. absolute acreage); the area weighting uses the
            precomputed ``mukey_weights`` and does not require it.
        mukey_weights: ``{mukey: area_fraction}`` within the boundary; fractions
            are expected to sum to ~1.0 but are renormalized defensively.
        per_mukey: Optional precomputed output of :func:`component_weighted`.
            When ``None``, soil data is fetched from SDA for the map units in
            ``mukey_weights``.

    Returns:
        A dict with keys:
            ``k_factor``           area-weighted kwfact (or ``None``)
            ``t_value``            area-weighted tfact (or ``None``)
            ``ksat_r``             area-weighted ksat_r (or ``None``)
            ``hydgrp``             dominant (largest-area) hydrologic group
                                   (or ``None``)
            ``drainage_fraction``  ``{drainage_class: area_fraction}``
            ``n_mukeys``           number of contributing map units
            ``mukey_detail``       per-mukey ``{k, t, ksat, drainagecl,
                                   hydgrp, area_fraction}`` (renormalized)
    """
    if per_mukey is None:
        df = fetch_soil_components(list(mukey_weights.keys()))
        per_mukey = component_weighted(df)

    # Keep only map units we have both a weight and soil data for, then
    # renormalize the area fractions over that contributing set.
    contributing = {
        mk: float(w)
        for mk, w in mukey_weights.items()
        if mk in per_mukey and w and w > 0
    }
    total_weight = sum(contributing.values())

    empty = {
        "k_factor": None,
        "t_value": None,
        "ksat_r": None,
        "hydgrp": None,
        "drainage_fraction": {},
        "n_mukeys": 0,
        "mukey_detail": {},
    }
    if total_weight <= 0:
        return empty

    acc = {"k": 0.0, "t": 0.0, "ksat": 0.0}
    acc_weight = {"k": 0.0, "t": 0.0, "ksat": 0.0}
    drainage_fraction: dict[str, float] = defaultdict(float)
    hydgrp_fraction: dict[str, float] = defaultdict(float)
    mukey_detail: dict = {}

    for mukey, raw_weight in contributing.items():
        frac = raw_weight / total_weight
        props = per_mukey[mukey]

        for key in ("k", "t", "ksat"):
            value = props.get(key)
            if value is not None:
                acc[key] += value * frac
                acc_weight[key] += frac

        drainage = props.get("drainagecl") or "Unknown"
        drainage_fraction[drainage] += frac

        # hydgrp is categorical (A/B/C/D); the area-weighted representative is
        # the hydrologic group covering the largest area fraction.
        hg = props.get("hydgrp")
        if hg:
            hydgrp_fraction[hg] += frac

        mukey_detail[mukey] = {
            "k": props.get("k"),
            "t": props.get("t"),
            "ksat": props.get("ksat"),
            "drainagecl": props.get("drainagecl"),
            "hydgrp": props.get("hydgrp"),
            "area_fraction": round(frac, 6),
        }

    def _finalize(key: str) -> Optional[float]:
        # Renormalize over the area that actually had a value for this property.
        return round(acc[key] / acc_weight[key], 4) if acc_weight[key] > 0 else None

    # Dominant (largest-area-fraction) hydrologic group; None if none reported.
    dominant_hydgrp = (max(hydgrp_fraction, key=hydgrp_fraction.get)
                       if hydgrp_fraction else None)

    return {
        "k_factor": _finalize("k"),
        "t_value": _finalize("t"),
        "ksat_r": _finalize("ksat"),
        "hydgrp": dominant_hydgrp,
        "drainage_fraction": {k: round(v, 4) for k, v in drainage_fraction.items()},
        "n_mukeys": len(contributing),
        "mukey_detail": mukey_detail,
    }


# ---------------------------------------------------------------------------
# Spatial helpers — per-mukey area fractions via shapely intersection
# ---------------------------------------------------------------------------
def fetch_mupolygons(boundary_geom: BaseGeometry) -> list[tuple[str, BaseGeometry]]:
    """Fetch SSURGO map-unit polygons intersecting a boundary as shapely geoms.

    Queries the SDA ``mupolygon`` table for every polygon that intersects the
    boundary's WKT (WGS84) and returns each polygon's ``mukey`` paired with its
    shapely geometry. The boundary is sent to SDA only as a coarse intersection
    filter; exact area fractions are computed downstream in shapely.

    Args:
        boundary_geom: Field boundary as a WGS84 (EPSG:4326) shapely geometry.

    Returns:
        A list of ``(mukey, geometry)`` tuples. Invalid geometries are repaired
        with ``buffer(0)``; any that remain empty/unparseable are skipped.
    """
    aoi_wkt = boundary_geom.wkt
    sql = f"""
        SELECT mukey, mupolygongeo.STAsText() AS wkt
        FROM mupolygon
        WHERE mupolygongeo.STIntersects(
            geometry::STGeomFromText('{aoi_wkt}', 4326)) = 1
    """
    polygons: list[tuple[str, BaseGeometry]] = []
    for row in query_sda(sql):
        try:
            geom = shapely_wkt.loads(row["wkt"])
        except Exception:  # noqa: BLE001 - skip any unparseable WKT
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        if not geom.is_empty:
            polygons.append((str(row["mukey"]), geom))
    return polygons


def mukey_area_fractions(boundary_geom: BaseGeometry) -> dict[str, float]:
    """Compute per-mukey area fractions of a boundary using shapely.

    Fetches intersecting map-unit polygons (:func:`fetch_mupolygons`),
    intersects each with the boundary, sums the intersection areas per
    ``mukey``, and normalizes so the fractions sum to 1.0. All spatial math is
    done in shapely for determinism; see the module docstring for why planar
    (unprojected) areas give correct fractions at a single field's scale.

    Args:
        boundary_geom: Field boundary as a WGS84 (EPSG:4326) shapely geometry.

    Returns:
        ``{mukey: area_fraction}`` summing to ~1.0, or an empty dict if no
        polygons intersect the boundary.
    """
    areas: dict[str, float] = defaultdict(float)
    for mukey, geom in fetch_mupolygons(boundary_geom):
        inter = geom.intersection(boundary_geom)
        if not inter.is_empty and inter.area > 0:
            areas[mukey] += inter.area

    total = sum(areas.values())
    if total <= 0:
        return {}
    return {mk: a / total for mk, a in areas.items()}


def soil_summary_for_boundary(boundary_geom: BaseGeometry) -> dict:
    """End-to-end area-weighted soil summary for a field boundary.

    Convenience wrapper: derives per-mukey area fractions with
    :func:`mukey_area_fractions`, then returns the area-weighted soil
    properties from :func:`area_weighted_soil`.

    Args:
        boundary_geom: Field boundary as a WGS84 (EPSG:4326) shapely geometry.

    Returns:
        The dict returned by :func:`area_weighted_soil` (see its docstring).
    """
    weights = mukey_area_fractions(boundary_geom)
    return area_weighted_soil(boundary_geom, weights)


def zone_weighted_t(
    zone_geom: BaseGeometry,
    mupolygons: list[tuple[str, BaseGeometry, float]],
) -> dict:
    """Area-weighted T-value for one Risk zone from the map units it overlaps.

    Risk zones (NDVI x slope) and SSURGO map units are different spatial
    partitions that do not nest: a single zone may overlap several map units.
    A zone's T is therefore the area-weighted mean of the T-values of the
    map-unit polygons it overlaps, weighted by the overlap area WITHIN the
    zone. This reuses the same planar shapely intersection and single-latitude
    fraction assumption as :func:`mukey_area_fractions` (see the module
    docstring). SSURGO minimum mapping unit (~1-2 ac) makes zone-level the
    correct resolution; this does not attempt finer-than-source per-pixel T.

    Args:
        zone_geom: The zone geometry (WGS84 shapely), e.g. the polygonized
            union of the zone's pixels.
        mupolygons: Iterable of ``(mukey, geometry, t_value)`` for the candidate
            map units (e.g. :func:`fetch_mupolygons` output joined to each
            map unit's component-weighted T). Polygons that do not overlap the
            zone are ignored; entries whose ``t_value`` is ``None`` are skipped.

    Returns:
        ``{"t_zone_weighted": float|None, "t_min": float|None,
           "t_max": float|None, "mukey_fractions": {mukey: fraction}}`` where
        the fractions sum to ~1.0 over the map units overlapping this zone.
        All T fields are ``None`` and ``mukey_fractions`` empty when the zone
        overlaps no map unit carrying a T-value.
    """
    none_result = {"t_zone_weighted": None, "t_min": None,
                   "t_max": None, "mukey_fractions": {}}
    if zone_geom is None or zone_geom.is_empty:
        return none_result

    areas: dict[str, float] = defaultdict(float)
    t_by_mukey: dict[str, float] = {}
    for mukey, geom, t_value in mupolygons:
        if t_value is None:
            continue
        inter = geom.intersection(zone_geom)
        if not inter.is_empty and inter.area > 0:
            mk = str(mukey)
            areas[mk] += inter.area
            t_by_mukey[mk] = float(t_value)

    total = sum(areas.values())
    if total <= 0:
        return none_result

    fractions = {mk: a / total for mk, a in areas.items()}
    t_zone_weighted = sum(fractions[mk] * t_by_mukey[mk] for mk in fractions)
    present_t = [t_by_mukey[mk] for mk in fractions]
    return {
        "t_zone_weighted": round(t_zone_weighted, 4),
        "t_min": min(present_t),
        "t_max": max(present_t),
        "mukey_fractions": {mk: round(f, 6) for mk, f in fractions.items()},
    }


def _repair(geom: Optional[BaseGeometry]) -> Optional[BaseGeometry]:
    """Return a valid, non-empty geometry or None (G3).

    Tries shapely ``make_valid`` first, then ``buffer(0)``; any geometry that
    stays empty or cannot be repaired returns None so the caller can skip it
    without raising. A bad sliver must never crash the run.
    """
    if geom is None or geom.is_empty:
        return None
    if geom.is_valid:
        return geom
    try:
        from shapely.validation import make_valid
        repaired = make_valid(geom)
    except Exception:  # noqa: BLE001
        repaired = None
    if repaired is None or repaired.is_empty or not repaired.is_valid:
        try:
            repaired = geom.buffer(0)
        except Exception:  # noqa: BLE001
            return None
    if repaired is None or repaired.is_empty or not repaired.is_valid:
        return None
    return repaired


def _utm_epsg_for_lonlat(lon: float, lat: float) -> str:
    """UTM EPSG string for a lon/lat (projected CRS for acreage, G2)."""
    zone = int((lon + 180.0) // 6.0) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def zone_mukey_tolerance_rows(
    zone_geometries: dict,
    mupolygons: list[tuple[str, BaseGeometry, Optional[float]]],
    a_by_zone: Optional[dict] = None,
    *,
    zone_labels: Optional[dict] = None,
    slope_by_zone: Optional[dict] = None,
    overlap_floor_acres: float = 0.5,
    acre_crs: Optional[str] = None,
    k_by_mukey: Optional[dict] = None,
    musym_by_mukey: Optional[dict] = None,
    k_field: Optional[float] = None,
) -> dict:
    """Per-(risk-zone x map-unit) soil-tolerance rows with projected acreage.

    Intersects each Risk zone geometry with every SSURGO map-unit polygon and
    emits one row per ``(zone, mukey)`` pair (polygon fragments of the same map
    unit in the same zone are POOLED — their overlap acres are summed into a
    single row). This is the Plan-B readable surface on top of the Phase-3
    zone-T machinery: it does not change any RUSLE/T computation.

    Per-soil A (rescale identity, not a recompute): the supplied ``a_by_zone``
    value is the zone soil loss ``A = R·K_field·LS_zone·C_zone`` (computed with
    the FIELD area-weighted K). When ``k_by_mukey`` and ``k_field`` (>0) are
    provided, each row's A is rescaled to that map unit's own erodibility:
    ``A_row = a_zone * (K_mukey / K_field)`` = ``R·K_mukey·LS_zone·C_zone`` —
    LS and C are zone (slope/NDVI) properties and are NOT re-derived per soil.
    ``A/T`` and severity (Tech Guide v1.8 bands) are derived from ``A_row`` and
    the map unit's own T. When the K inputs are absent the row A falls back to
    the unscaled zone A (backward-compatible).

    Geometric guards:
        * G1 - intersection is performed in the CRS the inputs are supplied in,
          which the caller MUST set to EPSG:4326 (both ``zone_geometries`` and
          ``mupolygons`` in WGS84). Fraction/area math here reuses the
          single-latitude planar assumption documented at module top.
        * G2 - ``overlap_acres`` is computed in a projected CRS (auto-selected
          UTM zone of the field, or the supplied ``acre_crs``), NEVER in WGS84
          degrees.
        * G3 - inputs and intersection outputs pass through :func:`_repair`
          (make_valid / buffer(0)); empty, zero-area or unrepairable slivers are
          skipped so no geometry op can crash the run.

    Args:
        zone_geometries: ``{zone_val(int 1-4): shapely geometry}`` in WGS84.
        mupolygons: ``[(mukey, geometry, t_value), ...]`` in WGS84; ``t_value``
            may be ``None``. Multiple entries with the same ``mukey`` are pooled.
        a_by_zone: ``{zone_val OR zone_label: A_current}`` zone RUSLE soil loss
            (computed with the field area-weighted K).
        zone_labels: ``{zone_val: label}``; defaults to Low/Moderate/High/Critical.
        slope_by_zone: ``{zone_val OR zone_label: mean_slope_pct}`` per-zone mean
            slope %, surfaced verbatim as the row's ``mean_slope_pct`` (a zone
            property, not re-derived per soil). ``None``/absent → ``None`` column.
        overlap_floor_acres: rows below this POOLED overlap acreage are rolled
            into the summary rather than surfaced as flagged.
        acre_crs: projected CRS for acreage; auto-derived UTM when ``None``.
        k_by_mukey: ``{mukey: K}`` per-soil erodibility, for the per-soil A
            rescale. When absent, A is the unscaled zone A (backward-compatible).
        musym_by_mukey: ``{mukey: musym}`` map-unit symbols for display.
        k_field: the field area-weighted K used in ``a_by_zone``; required (>0)
            for the per-soil rescale.

    Returns:
        ``{"rows_full", "rows_flagged", "rolled_summary", "intersect_crs",
           "acre_crs"}``, with ``rows_full`` sorted by ``a_over_t`` descending
        (worst first). ``rows_flagged`` are rows where ``within_t_zone is False
        AND overlap_acres >= overlap_floor_acres``; every other row feeds
        ``rolled_summary``. Each row: ``risk_zone, zone_val, mukey, musym,
        soil_T, K_mukey, mean_slope_pct, a_zone, a_over_t, severity,
        within_t_zone, overlap_acres, geometry`` — where ``K_mukey`` is the map
        unit's own erodibility (the rescale numerator, now surfaced),
        ``mean_slope_pct`` is the overlapping zone's mean slope %, ``a_zone`` is
        the per-soil A_row and ``geometry`` is
        the WGS84 zone∩mukey intersection (shapely; ``None`` if unavailable),
        provided for map rendering only — it never enters the table or math.
    """
    from shapely.ops import transform as _shp_transform
    from pyproj import Transformer

    labels = zone_labels or {1: "Low", 2: "Moderate", 3: "High", 4: "Critical"}
    a_by_zone      = a_by_zone or {}
    slope_by_zone  = slope_by_zone or {}
    k_by_mukey     = k_by_mukey or {}
    musym_by_mukey = musym_by_mukey or {}
    _rescale_ok    = k_field is not None and k_field > 0

    def _severity(ratio: Optional[float]) -> Optional[str]:
        # Technical Guide v1.8 bands (verbatim, with the documented <= edges).
        if ratio is None:
            return None
        if ratio <= 1.0:
            return "Within tolerable limit"
        if ratio <= 2.0:
            return "Near tolerable limit"
        if ratio <= 5.0:
            return "Exceeds tolerable limit"
        return "Significantly exceeds limit"

    empty = {
        "rows_full": [], "rows_flagged": [],
        "rolled_summary": {"n_rows": 0, "overlap_acres": 0.0},
        "intersect_crs": "EPSG:4326", "acre_crs": acre_crs,
    }
    if not zone_geometries or not mupolygons:
        return empty

    # Pre-repair the map-unit polygons once (G3).
    repaired_mu = []
    for mk, geom, t_val in mupolygons:
        rg = _repair(geom)
        if rg is not None:
            repaired_mu.append((str(mk), rg, t_val))
    if not repaired_mu:
        return empty

    # Acreage projection (G2): auto-select the field UTM zone from a zone centroid.
    if acre_crs is None:
        try:
            ref = next(iter(zone_geometries.values()))
            c = _repair(ref)
            c = (c if c is not None else ref).centroid
            acre_crs = _utm_epsg_for_lonlat(c.x, c.y)
        except Exception:  # noqa: BLE001
            acre_crs = "EPSG:5070"  # CONUS Albers equal-area fallback
    _to_proj = Transformer.from_crs("EPSG:4326", acre_crs, always_xy=True).transform

    # Accumulate one row per (zone_val, mukey); pool fragment acres into it.
    agg: dict = {}
    for zone_val, zgeom in zone_geometries.items():
        zg = _repair(zgeom)
        if zg is None:
            continue
        label  = labels.get(zone_val)
        a_zone = a_by_zone.get(zone_val, a_by_zone.get(label))
        slope_zone = slope_by_zone.get(zone_val, slope_by_zone.get(label))

        for mk, mgeom, t_val in repaired_mu:
            try:
                inter = _repair(mgeom.intersection(zg))
            except Exception:  # noqa: BLE001 - a bad sliver must never crash
                continue
            if inter is None:
                continue
            try:
                acres = _shp_transform(_to_proj, inter).area / 4046.8564224
            except Exception:  # noqa: BLE001
                continue
            if acres <= 0:
                continue

            key = (zone_val, mk)
            if key not in agg:
                # Per-soil A: rescale the zone A by this map unit's own K
                # (A_row = R·K_mukey·LS_zone·C_zone). LS/C are NOT re-derived.
                k_mukey = k_by_mukey.get(mk)
                if _rescale_ok and k_mukey is not None and a_zone is not None:
                    a_row = float(a_zone) * (float(k_mukey) / float(k_field))
                else:
                    a_row = float(a_zone) if a_zone is not None else None

                ratio  = (a_row / t_val) if (a_row is not None and t_val) else None
                within = (bool(a_row <= t_val)
                          if (a_row is not None and t_val is not None) else None)
                agg[key] = {
                    "risk_zone":     label,
                    "zone_val":      int(zone_val),
                    "mukey":         mk,
                    "musym":         musym_by_mukey.get(mk),
                    "soil_T":        t_val,
                    "K_mukey":       round(float(k_mukey), 4) if k_mukey is not None else None,
                    "mean_slope_pct": round(float(slope_zone), 1) if slope_zone is not None else None,
                    "a_zone":        round(a_row, 2) if a_row is not None else None,
                    "a_over_t":      round(ratio, 2) if ratio is not None else None,
                    "severity":      _severity(ratio),
                    "within_t_zone": within,
                    "overlap_acres": 0.0,
                    "geometry":      None,   # WGS84 zone∩mukey union (for map outline)
                }
            agg[key]["overlap_acres"] += acres
            # Retain the actual intersection geometry (WGS84), unioning the
            # fragments that share this (zone_val, mukey) key. Used only for
            # map rendering — never enters the tolerance/acreage math above.
            _g = agg[key]["geometry"]
            agg[key]["geometry"] = inter if _g is None else _repair(_g.union(inter))

    rows_full = list(agg.values())
    for r in rows_full:
        r["overlap_acres"] = round(r["overlap_acres"], 2)
    # Worst first: sort by A/T descending (None ratios sink to the bottom).
    rows_full.sort(key=lambda r: (r["a_over_t"] if r["a_over_t"] is not None else -1.0),
                   reverse=True)

    rows_flagged = []
    rolled = {"n_rows": 0, "overlap_acres": 0.0}
    for r in rows_full:
        if r["within_t_zone"] is False and r["overlap_acres"] >= overlap_floor_acres:
            rows_flagged.append(r)
        else:
            rolled["n_rows"] += 1
            rolled["overlap_acres"] += r["overlap_acres"]
    rolled["overlap_acres"] = round(rolled["overlap_acres"], 2)

    return {
        "rows_full": rows_full,
        "rows_flagged": rows_flagged,
        "rolled_summary": rolled,
        "intersect_crs": "EPSG:4326",
        "acre_crs": acre_crs,
    }


# ---------------------------------------------------------------------------
# Validation / manual test harness
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from shapely.geometry import box

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Reference set: Shelby County (areasymbol IA165) Monona map units
    # 100D2 / 100E2 / 100F2. These resolve to mukeys 626747 / 626748 / 780777.
    REFERENCE = {"100D2": "626747", "100E2": "626748", "100F2": "780777"}
    ref_mukeys = list(REFERENCE.values())

    print("=" * 72)
    print("SDA soil_data.py validation - Shelby County, IA (Monona reference set)")
    print("=" * 72)

    # --- Per-map-unit component-weighted K and T -------------------------
    comp_df = fetch_soil_components(ref_mukeys)
    per_mukey = component_weighted(comp_df)

    print("\n[1] Per-map-unit component-weighted properties")
    print(f"    {'musym':<7}{'mukey':<9}{'K (kwfact)':<12}"
          f"{'T (tfact)':<11}{'ksat':<7}{'drainage':<14}hydgrp")
    for musym, mukey in REFERENCE.items():
        p = per_mukey.get(mukey, {})
        print(f"    {musym:<7}{mukey:<9}"
              f"{(p.get('k')):<12}{(p.get('t')):<11}{(p.get('ksat')):<7}"
              f"{str(p.get('drainagecl')):<14}{p.get('hydgrp')}")

    # --- Area-weighted result for an example area split ------------------
    # Example field map-unit area fractions (must sum to 1.0).
    example_weights = {"626747": 0.50, "626748": 0.30, "780777": 0.20}
    aw_example = area_weighted_soil(None, example_weights, per_mukey=per_mukey)

    print("\n[2] Area-weighted soil (example fractions 0.50 / 0.30 / 0.20)")
    print(f"    area fractions      : {example_weights}")
    print(f"    area-weighted K     : {aw_example['k_factor']}")
    print(f"    area-weighted T     : {aw_example['t_value']}")
    print(f"    area-weighted ksat  : {aw_example['ksat_r']}")
    print(f"    drainage breakdown  : {aw_example['drainage_fraction']}")

    # --- Full spatial pipeline against a real AOI in Shelby County -------
    # ~700 m box over Monona soils near Defiance/Westphalia, Shelby County.
    aoi = box(-95.352, 41.747, -95.344, 41.752)
    print("\n[3] Full spatial pipeline (shapely intersection of mupolygons)")
    print(f"    AOI bbox (WGS84)    : {aoi.bounds}")
    frac = mukey_area_fractions(aoi)
    print(f"    intersecting mukeys : {len(frac)}")
    for mk, f in sorted(frac.items(), key=lambda kv: -kv[1]):
        print(f"      mukey {mk:<9} area fraction {f:.4f}")

    aw_spatial = soil_summary_for_boundary(aoi)
    print(f"\n    FINAL area-weighted K : {aw_spatial['k_factor']}")
    print(f"    FINAL area-weighted T : {aw_spatial['t_value']}")
    print(f"    FINAL area-weighted ksat : {aw_spatial['ksat_r']}")
    print(f"    drainage breakdown    : {aw_spatial['drainage_fraction']}")
    print("\nCompare the per-map-unit and area-weighted K above against your "
          "current\nWeb Soil Survey output for this field.")

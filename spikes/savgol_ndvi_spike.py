"""
savgol_ndvi_spike.py
====================
FEASIBILITY SPIKE — Savitzky-Golay smoothing of a Sentinel-2 NDVI time series
vs. the current median-composite approach, for Bin Field (2), Shelby County IA.

This is a standalone spike. It does NOT modify src/, app.py, or the scoring
pipeline. It *reuses* production code read-only:

  * src.io_utils.load_boundary_file        -> boundary handling (production)
  * src.gee_ndvi_utils                      -> GEE auth, S2 collection, cloud mask
  * src.scoring._lookup_c_factor / IOWA_C_FACTOR_TABLE -> the C-factor decision bins

Question under test
-------------------
Does a Savitzky-Golay reconstruction of the full per-scene NDVI time series agree
with the current cloud-filtered *median composite* field-mean NDVI — specifically,
do the two land in the SAME IOWA_C_FACTOR_TABLE bin — while surviving cloud gaps
that currently trigger QC caveats (0-scene windows, <50% valid pixels)?

Glass-box contract
------------------
Every smoothed value traces to: (real per-scene field-mean observations)
-> (linear interpolation onto a daily grid, stated) -> (local least-squares
polynomial fit of order `polyorder` over a `window` of days, stated).
No black-box gap-fill. For every daily value we also record the distance in days
to the nearest REAL observation, which is the honesty metric for cloud gaps.

Savitzky-Golay implementation
-----------------------------
scipy is unavailable in this environment (its DLL is blocked by an Application
Control policy), so S-G is implemented directly in numpy. For an interior point
with a full symmetric window this is *identical* to scipy.signal.savgol_filter
with mode='interp'; at the series edges each window shrinks to a one-sided fit
(the same behavior scipy's 'interp' edge handling approximates). The value is the
intercept of a degree-`polyorder` polynomial least-squares fit to the daily
values in the window, i.e. its value at the window centre.

Run
---
    .venv/Scripts/python.exe spikes/savgol_ndvi_spike.py

Outputs (spikes/output/):
    bin_field_scenes.csv     per-scene date, field-mean NDVI, valid-pixel fraction
    bin_field_daily.csv      daily grid: linear fill, gap-distance, S-G per param
    savgol_ndvi_spike.png    time series + composites + smoothed curves
    findings.json            machine-readable summary used to write the report
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# --- make the project importable so we reuse production code, read-only --------
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import ee  # noqa: E402
from shapely.geometry import mapping  # noqa: E402

from src.io_utils import load_boundary_file  # production boundary handling
from src.gee_ndvi_utils import (  # production GEE auth + collection id
    init_gee_from_streamlit_secrets,
    S2_COLLECTION,
)
from src.scoring import _lookup_c_factor, IOWA_C_FACTOR_TABLE  # decision bins

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

BOUNDARY_ZIP = os.path.join(PROJ, "Bin Field (2).zip")

# Cover-crop season: fall establishment (post-2025-harvest) -> spring termination.
SEASON_START = datetime(2025, 9, 15)
SEASON_END   = datetime(2026, 5, 31)

# Composite windows the current pipeline would actually use.
#   W_YOY   = the production MAR_APR early-season window (gee_ndvi_utils.MAR_APR_*)
#   W_TERM  = a spring-termination month composite
#   W_NARROW= a deliberately narrow window to stress the cloud-gap / 0-1 scene case
COMPOSITE_WINDOWS = {
    "YoY Mar15-Apr20":   (datetime(2026, 3, 15), datetime(2026, 4, 21)),
    "Termination Apr":   (datetime(2026, 4, 1),  datetime(2026, 5, 1)),
}
NARROW_WINDOW = ("Narrow 7-day Apr", datetime(2026, 4, 10), datetime(2026, 4, 17))

# SCL classes treated as cloud/shadow (identical to gee_ndvi_utils._field_cloud_fraction)
SCL_BAD = [3, 8, 9, 10, 11]
# Production keeps a scene in the composite when field cloud/shadow fraction < 0.30
PROD_FIELD_CLOUD_MAX = 0.30
# QC valid-pixel disclaimer threshold used by the app (<50% valid -> caveat)
QC_VALID_MIN = 0.50


# =============================================================================
# Savitzky-Golay (glass-box, numpy-only)
# =============================================================================
def savgol_glassbox(y: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Local least-squares polynomial smoothing (Savitzky-Golay).

    `y` is assumed on a uniform (daily) grid. For each index i, fit a degree
    `polyorder` polynomial to the points in [i-half, i+half] (clipped at the
    edges) and return its value at i (the polynomial intercept in centred
    coordinates). With a full symmetric window this equals the classic S-G
    convolution; at edges the window shrinks to a one-sided fit.
    """
    if window % 2 == 0:
        raise ValueError("window must be odd")
    if polyorder >= window:
        raise ValueError("polyorder must be < window")
    y = np.asarray(y, dtype=float)
    n = len(y)
    half = window // 2
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        idx = np.arange(lo, hi)
        x = (idx - i).astype(float)                 # centre on i
        # Not enough points to fit this order -> fall back to local mean.
        p = min(polyorder, len(idx) - 1)
        A = np.vander(x, p + 1, increasing=True)    # cols x^0 .. x^p
        coef, *_ = np.linalg.lstsq(A, y[idx], rcond=None)
        out[i] = coef[0]                            # value at x = 0
    return out


def c_bin_label(ndvi: float) -> str:
    """Human label for the IOWA_C_FACTOR_TABLE bin an NDVI falls in."""
    for (lo, hi) in IOWA_C_FACTOR_TABLE:
        if lo <= ndvi < hi:
            return f"[{lo:.2f},{hi:.2f})"
    return "out-of-range"


# =============================================================================
# GEE extraction
# =============================================================================
def get_aoi(boundary_gdf) -> "ee.Geometry":
    geom = boundary_gdf.to_crs(4326).geometry.iloc[0]
    return ee.Geometry(mapping(geom))


def pull_scene_series(aoi: "ee.Geometry", start: datetime, end: datetime) -> pd.DataFrame:
    """Per-scene field-mean NDVI (clear pixels only) + valid-pixel fraction.

    Reuses the production cloud definition (SCL_BAD). No CLOUDY_PIXEL_PERCENTAGE
    pre-filter here on purpose: we want EVERY acquisition over the field so gaps
    and low-valid-fraction scenes are visible, then we threshold in analysis.
    """
    scl_bad = ee.List(SCL_BAD)

    def per_image(img):
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        scl = img.select("SCL")
        clear = scl.remap(scl_bad, ee.List([0] * len(SCL_BAD)), 1).rename("clear")
        clear_frac = clear.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=20, maxPixels=int(1e8)
        ).get("clear")
        mean_ndvi = (
            ndvi.updateMask(clear)
            .reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi,
                          scale=10, maxPixels=int(1e8))
            .get("NDVI")
        )
        # Feature (not image.set + aggregate_array): aggregate_array DROPS nulls,
        # which desyncs columns when a fully-clouded scene has null mean_ndvi.
        # A FeatureCollection.getInfo() keeps every row aligned, nulls as None.
        return ee.Feature(None, {
            "date": img.date().format("YYYY-MM-dd"),
            "mean_ndvi": mean_ndvi,
            "valid_frac": clear_frac,
            "scene_cloud_pct": img.get("CLOUDY_PIXEL_PERCENTAGE"),
        })

    coll = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        .map(per_image)
    )
    feats = ee.FeatureCollection(coll).getInfo()["features"]
    recs = [f["properties"] for f in feats]
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    for c in ("mean_ndvi", "valid_frac", "scene_cloud_pct"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df[["date", "mean_ndvi", "valid_frac", "scene_cloud_pct"]]
    df = df.sort_values("date").reset_index(drop=True)
    # Collapse same-day duplicate granules (two tiles) to a valid-fraction-weighted mean.
    df = _collapse_same_day(df)
    return df


def _collapse_same_day(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for day, g in df.groupby(df["date"].dt.date):
        vf = g["valid_frac"].fillna(0.0)
        w = vf.to_numpy()
        mn = g["mean_ndvi"].to_numpy(dtype=float)
        ok = np.isfinite(mn) & (w > 0)
        if ok.any():
            mean_ndvi = float(np.average(mn[ok], weights=w[ok]))
        else:
            mean_ndvi = float("nan")
        rows.append({
            "date": pd.Timestamp(day),
            "mean_ndvi": mean_ndvi,
            "valid_frac": float(vf.max()),
            "scene_cloud_pct": float(g["scene_cloud_pct"].min()),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def median_composite_ndvi(aoi: "ee.Geometry", start: datetime, end: datetime) -> dict:
    """Replicate the production median composite field-mean NDVI for a window.

    Mirrors gee_ndvi_utils.fetch_ndvi_for_field: CLOUDY_PIXEL_PERCENTAGE<80,
    field cloud/shadow fraction<0.30, then collection.median(), field-mean over
    the polygon. Returns composite mean NDVI, scene count, and the median of the
    contributing scene dates (the composite's effective date).
    """
    scl_bad = ee.List(SCL_BAD)

    def tag_cloud(img):
        scl = img.select("SCL")
        bad = scl.remap(scl_bad, ee.List([1] * len(SCL_BAD)), 0).rename("bad")
        frac = bad.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=20, maxPixels=int(1e8)
        ).get("bad")
        return img.set("field_cloud_fraction", frac)

    base = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
        .map(tag_cloud)
        .filter(ee.Filter.lt("field_cloud_fraction", PROD_FIELD_CLOUD_MAX))
    )
    count = base.size().getInfo()
    result = {"window_start": start, "window_end": end, "scene_count": count,
              "composite_ndvi": None, "effective_date": None,
              "contributing_dates": []}
    if count == 0:
        return result

    def add_ndvi(img):
        return img.addBands(img.normalizedDifference(["B8", "B4"]).rename("NDVI"))

    comp = base.map(add_ndvi).select("NDVI").median().clip(aoi)
    mean_ndvi = comp.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, scale=10, maxPixels=int(1e8)
    ).get("NDVI").getInfo()

    dates = base.aggregate_array("system:time_start").getInfo()
    dts = sorted(datetime.utcfromtimestamp(d / 1000) for d in dates)
    eff = dts[len(dts) // 2] if dts else None

    result["composite_ndvi"] = float(mean_ndvi) if mean_ndvi is not None else None
    result["effective_date"] = eff
    result["contributing_dates"] = [d.strftime("%Y-%m-%d") for d in dts]
    return result


# =============================================================================
# Daily reconstruction
# =============================================================================
def build_daily(df_obs: pd.DataFrame, start: datetime, end: datetime,
                valid_min: float) -> pd.DataFrame:
    """Daily grid with linear-interpolated NDVI + gap-distance to nearest real obs.

    Only observations with valid_frac >= valid_min and finite NDVI are treated as
    'real' anchor points. Interior gaps are linearly interpolated (stated);
    daily points outside the observed span are left NaN (no extrapolation).
    """
    obs = df_obs[(df_obs["valid_frac"] >= valid_min) & np.isfinite(df_obs["mean_ndvi"])]
    obs = obs.sort_values("date")
    days = pd.date_range(start, end, freq="D")
    grid = pd.DataFrame({"date": days})
    grid["doy"] = np.arange(len(grid))

    obs_days = (obs["date"] - pd.Timestamp(start)).dt.days.to_numpy()
    obs_vals = obs["mean_ndvi"].to_numpy(dtype=float)

    x = grid["doy"].to_numpy(dtype=float)
    # Linear interpolation only within [first_obs, last_obs]; NaN outside.
    lin = np.interp(x, obs_days, obs_vals, left=np.nan, right=np.nan)
    grid["linear_fill"] = lin

    # Real-observation flag + gap distance (days to nearest real obs).
    is_real = np.isin(grid["doy"].to_numpy(), obs_days)
    grid["is_real_obs"] = is_real
    if len(obs_days):
        gapdist = np.min(np.abs(x[:, None] - obs_days[None, :]), axis=1)
    else:
        gapdist = np.full(len(x), np.nan)
    grid["gap_days_to_obs"] = gapdist
    grid.attrs["obs_days"] = obs_days
    grid.attrs["obs_vals"] = obs_vals
    return grid


def smooth_grid(grid: pd.DataFrame, window: int, polyorder: int) -> np.ndarray:
    """S-G on the linear-filled daily series, restricted to the observed span."""
    y = grid["linear_fill"].to_numpy(dtype=float)
    mask = np.isfinite(y)
    out = np.full(len(y), np.nan)
    if mask.sum() > polyorder + 1:
        w = min(window, mask.sum() if mask.sum() % 2 == 1 else mask.sum() - 1)
        if w % 2 == 0:
            w -= 1
        w = max(w, polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3)
        out[mask] = savgol_glassbox(y[mask], w, polyorder)
    return out


def value_at(grid: pd.DataFrame, col: str, when: datetime):
    """Value of a daily column at a given date (exact grid lookup)."""
    hit = grid.index[grid["date"] == pd.Timestamp(when.date())]
    if len(hit) == 0:
        return None
    v = grid.loc[hit[0], col]
    return float(v) if np.isfinite(v) else None


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 72)
    print("SAVITZKY-GOLAY NDVI SPIKE — Bin Field (2), Shelby County IA")
    print("=" * 72)

    boundary = load_boundary_file(BOUNDARY_ZIP)
    acres = boundary.to_crs(boundary.estimate_utm_crs()).area.sum() / 4046.8564224
    print(f"Boundary loaded: {acres:.1f} ac  ({BOUNDARY_ZIP})")

    init_gee_from_streamlit_secrets()
    print("GEE initialized.")
    aoi = get_aoi(boundary)

    # ---- 1. Full per-scene time series -------------------------------------
    print(f"\nPulling per-scene NDVI {SEASON_START.date()} -> {SEASON_END.date()} ...")
    scenes = pull_scene_series(aoi, SEASON_START, SEASON_END)
    scenes.to_csv(os.path.join(OUT, "bin_field_scenes.csv"), index=False)
    n_total = len(scenes)
    n_usable = int(((scenes["valid_frac"] >= QC_VALID_MIN) &
                    np.isfinite(scenes["mean_ndvi"])).sum())
    n_prod = int(((scenes["valid_frac"] >= (1 - PROD_FIELD_CLOUD_MAX)) &
                  np.isfinite(scenes["mean_ndvi"])).sum())
    print(f"  scenes returned:            {n_total}")
    print(f"  usable (>= {QC_VALID_MIN:.0%} valid px):    {n_usable}")
    print(f"  production-grade (>=70% clr): {n_prod}")
    print(scenes.to_string(index=False))

    # Gap analysis on usable observations
    usable = scenes[(scenes["valid_frac"] >= QC_VALID_MIN) &
                    np.isfinite(scenes["mean_ndvi"])].sort_values("date")
    gaps = usable["date"].diff().dt.days.dropna().astype(int)
    max_gap = int(gaps.max()) if len(gaps) else None
    print(f"\n  max consecutive gap between usable scenes: {max_gap} days")

    # ---- 2. Daily reconstruction + S-G grid --------------------------------
    grid = build_daily(scenes, SEASON_START, SEASON_END, QC_VALID_MIN)
    param_grid = [(w, p) for w in (11, 21, 31, 41) for p in (2, 3)]
    for (w, p) in param_grid:
        grid[f"sg_w{w}_p{p}"] = smooth_grid(grid, w, p)
    grid.to_csv(os.path.join(OUT, "bin_field_daily.csv"), index=False)

    # ---- 3. Composite vs smoothed, at composite effective date -------------
    print("\n" + "-" * 72)
    print("COMPOSITE vs SAVITZKY-GOLAY  (decision criterion = same C-factor bin)")
    print("-" * 72)
    comparisons = []
    windows = dict(COMPOSITE_WINDOWS)
    for name, (s, e) in windows.items():
        comp = median_composite_ndvi(aoi, s, e)
        row = {"window": name, "scene_count": comp["scene_count"],
               "composite_ndvi": comp["composite_ndvi"],
               "effective_date": comp["effective_date"].strftime("%Y-%m-%d")
               if comp["effective_date"] else None}
        if comp["composite_ndvi"] is None or comp["effective_date"] is None:
            print(f"\n[{name}] no composite (0 scenes)")
            comparisons.append(row)
            continue
        cnd = comp["composite_ndvi"]
        row["composite_c"] = _lookup_c_factor(cnd)
        row["composite_bin"] = c_bin_label(cnd)
        print(f"\n[{name}]  scenes={comp['scene_count']}  eff_date={row['effective_date']}")
        print(f"  composite NDVI = {cnd:.3f}  -> C={row['composite_c']}  bin {row['composite_bin']}")
        row["sg"] = {}
        for (w, p) in param_grid:
            sv = value_at(grid, f"sg_w{w}_p{p}", comp["effective_date"])
            if sv is None:
                continue
            same = c_bin_label(sv) == row["composite_bin"]
            row["sg"][f"w{w}_p{p}"] = {
                "ndvi": sv, "delta": sv - cnd,
                "c": _lookup_c_factor(sv), "bin": c_bin_label(sv),
                "same_bin": same,
            }
            flag = "OK same-bin" if same else "*** DIFFERENT BIN ***"
            print(f"    S-G w={w:2d} p={p}: NDVI={sv:.3f}  d={sv-cnd:+.3f}  "
                  f"C={_lookup_c_factor(sv)}  bin {c_bin_label(sv)}  {flag}")
        comparisons.append(row)

    # ---- 4. Cloud-gap stress case ------------------------------------------
    print("\n" + "-" * 72)
    print("CLOUD-GAP STRESS  (narrow window that returns 0-1 scenes)")
    print("-" * 72)
    nname, ns, ne = NARROW_WINDOW
    narrow = median_composite_ndvi(aoi, ns, ne)
    mid = ns + (ne - ns) / 2
    sg_at_mid = {f"w{w}_p{p}": value_at(grid, f"sg_w{w}_p{p}", mid)
                 for (w, p) in param_grid}
    gap_at_mid = value_at(grid, "gap_days_to_obs", mid)
    print(f"[{nname}] {ns.date()} -> {ne.date()}")
    print(f"  production composite scenes in window: {narrow['scene_count']} "
          f"(composite NDVI={narrow['composite_ndvi']})")
    print(f"  nearest real obs is {gap_at_mid} day(s) from window midpoint {mid.date()}")
    for k, v in sg_at_mid.items():
        if v is not None:
            print(f"  S-G {k}: NDVI={v:.3f}  C={_lookup_c_factor(v)}  bin {c_bin_label(v)}")

    # Defensibility rule-of-thumb: largest gap-distance still inside observed span
    span = grid[np.isfinite(grid["linear_fill"])]
    worst_interp = float(span["gap_days_to_obs"].max()) if len(span) else None
    print(f"\n  worst interpolation distance inside observed span: {worst_interp} days")

    # ---- 5. Persist machine-readable findings + plot -----------------------
    findings = {
        "field": "Bin Field (2), Shelby County IA",
        "acres": round(float(acres), 1),
        "season": [SEASON_START.strftime("%Y-%m-%d"), SEASON_END.strftime("%Y-%m-%d")],
        "scene_count_total": n_total,
        "scene_count_usable_50pct": n_usable,
        "scene_count_prod_70pct": n_prod,
        "max_gap_days_usable": max_gap,
        "worst_interp_days_in_span": worst_interp,
        "composite_windows": [
            {k: (v.strftime("%Y-%m-%d") if isinstance(v, datetime) else v)
             for k, v in row.items() if k != "sg"} | {"sg": row.get("sg", {})}
            for row in comparisons
        ],
        "narrow_window": {
            "name": nname, "start": ns.strftime("%Y-%m-%d"),
            "end": ne.strftime("%Y-%m-%d"),
            "prod_scene_count": narrow["scene_count"],
            "prod_composite_ndvi": narrow["composite_ndvi"],
            "gap_days_at_midpoint": gap_at_mid,
            "sg_at_midpoint": {k: v for k, v in sg_at_mid.items()},
        },
    }
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(findings, f, indent=2, default=str)
    print(f"\nWrote findings.json + CSVs to {OUT}")

    _plot(scenes, grid, comparisons, param_grid)
    print("Wrote savgol_ndvi_spike.png")
    return findings


def _plot(scenes, grid, comparisons, param_grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(12, 6))
    # C-factor bin bands
    for (lo, hi), c in IOWA_C_FACTOR_TABLE.items():
        ax.axhspan(lo, hi, alpha=0.06, color="gray")
        ax.text(grid["date"].iloc[2], (lo + hi) / 2, f"C={c}", fontsize=7,
                va="center", color="gray")

    usable = scenes[(scenes["valid_frac"] >= QC_VALID_MIN) & np.isfinite(scenes["mean_ndvi"])]
    lowval = scenes[(scenes["valid_frac"] < QC_VALID_MIN) & np.isfinite(scenes["mean_ndvi"])]
    ax.scatter(usable["date"], usable["mean_ndvi"], s=45, c="black", zorder=5,
               label=f">= {QC_VALID_MIN:.0%} valid px (real obs)")
    ax.scatter(lowval["date"], lowval["mean_ndvi"], s=35, facecolors="none",
               edgecolors="red", zorder=5, label="< 50% valid px (QC-caveat)")

    for (w, p) in [(11, 3), (31, 3)]:
        ax.plot(grid["date"], grid[f"sg_w{w}_p{p}"], lw=1.8,
                label=f"S-G window={w}d, poly={p}")

    for row in comparisons:
        if row.get("composite_ndvi") is not None:
            d = pd.Timestamp(row["effective_date"])
            ax.scatter([d], [row["composite_ndvi"]], marker="D", s=90,
                       color="tab:orange", zorder=6)
            ax.annotate(f"{row['window']}\ncomposite {row['composite_ndvi']:.3f}",
                        (d, row["composite_ndvi"]), fontsize=7,
                        xytext=(6, 10), textcoords="offset points")

    ax.set_ylabel("Field-mean NDVI")
    ax.set_title("Bin Field (2) — per-scene NDVI, S-G reconstruction, vs median composites")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylim(0, 0.9)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "savgol_ndvi_spike.png"), dpi=130)


if __name__ == "__main__":
    main()

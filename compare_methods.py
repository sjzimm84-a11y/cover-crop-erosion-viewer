"""
compare_methods.py
------------------
Side-by-side comparison of the old 6-bin NDVI lookup × residue multiplier
C-factor model versus the new continuous exponential C-factor model.

Outputs:
    compare_methods_output.csv  — full grid comparison
    (also prints summary table to stdout)

Parameters used:
    NDVI   : 0.00 – 0.80 in steps of 0.05
    Slope  : 2, 5, 9, 15 %  (representative Iowa range)
    R      : 175.0 MJ·mm/ha·hr·yr  (standard Iowa, NRCS FOTG)
    K      : 0.37 t·ha·hr/ha·MJ·mm  (Monona silt loam representative)
    T      : 5 t/ac/yr
    P      : 1.0 (no practice factor)

Intended audience: Weston Dittmer (Shelby County NRCS) and Mark Licht (ISU)
for pre-calibration review, May 2026.

Run from the project root:
    python compare_methods.py
"""

import csv
import sys
import numpy as np

sys.path.insert(0, ".")

from src.scoring import (
    _lookup_c_factor,
    _continuous_c_factor,
    _analytical_ls_factor,
    RESIDUE_ADJUSTMENTS,
    CONTINUOUS_C_PARAMS,
    RESIDUE_OPTIONS,
    CONCERN_THRESHOLDS,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
NDVI_VALUES  = [round(v, 2) for v in np.arange(0.00, 0.81, 0.05)]
SLOPE_VALUES = [2, 5, 9, 15]

R_FACTOR = 175.0
K_FACTOR = 0.37   # Monona silt loam representative
T_VALUE  = 5      # t/ac/yr

# Exclude "Unknown" — it duplicates conventional tillage parameters
RESIDUE_SYSTEMS = [r for r in RESIDUE_OPTIONS
                   if r != "Unknown — not recorded (conservative default)"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _old_c(ndvi: float, residue_system: str) -> float:
    """Original model: 6-bin lookup × residue multiplier."""
    return _lookup_c_factor(ndvi) * RESIDUE_ADJUSTMENTS.get(residue_system, 1.0)


def _new_c(ndvi: float, residue_system: str) -> float:
    """New model: continuous exponential decay."""
    return _continuous_c_factor(ndvi, residue_system)


def _concern(risk_index: float) -> str:
    for level, threshold in CONCERN_THRESHOLDS.items():
        if risk_index < threshold:
            return level
    return "Critical"


def _soil_loss(c: float, ls: float) -> float:
    """A = R × K × LS × C × P (P = 1.0), result in t/ac/yr."""
    return R_FACTOR * K_FACTOR * ls * c


def _pct_change(old: float, new: float) -> float:
    if old == 0:
        return float("nan")
    return round((new - old) / old * 100, 1)


# ---------------------------------------------------------------------------
# Build rows
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "residue_system",
    "ndvi",
    "slope_pct",
    "ls_factor",
    "old_c_factor",
    "new_c_factor",
    "c_pct_change",
    "old_risk_index",
    "new_risk_index",
    "risk_pct_change",
    "old_soil_loss_t_ac_yr",
    "new_soil_loss_t_ac_yr",
    "soil_loss_pct_change",
    "old_concern",
    "new_concern",
    "concern_changed",
]

rows = []
for residue in RESIDUE_SYSTEMS:
    for slope in SLOPE_VALUES:
        ls = float(_analytical_ls_factor(float(slope)))
        for ndvi in NDVI_VALUES:
            c_old  = _old_c(ndvi, residue)
            c_new  = _new_c(ndvi, residue)
            ri_old = c_old * ls
            ri_new = c_new * ls
            sl_old = _soil_loss(c_old, ls)
            sl_new = _soil_loss(c_new, ls)
            con_old = _concern(ri_old)
            con_new = _concern(ri_new)
            rows.append({
                "residue_system":        residue,
                "ndvi":                  ndvi,
                "slope_pct":             slope,
                "ls_factor":             round(ls, 4),
                "old_c_factor":          round(c_old, 4),
                "new_c_factor":          round(c_new, 4),
                "c_pct_change":          _pct_change(c_old, c_new),
                "old_risk_index":        round(ri_old, 4),
                "new_risk_index":        round(ri_new, 4),
                "risk_pct_change":       _pct_change(ri_old, ri_new),
                "old_soil_loss_t_ac_yr": round(sl_old, 2),
                "new_soil_loss_t_ac_yr": round(sl_new, 2),
                "soil_loss_pct_change":  _pct_change(sl_old, sl_new),
                "old_concern":           con_old,
                "new_concern":           con_new,
                "concern_changed":       "YES" if con_old != con_new else "",
            })

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------
out_path = "compare_methods_output.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {out_path}\n")

# ---------------------------------------------------------------------------
# Print summary table — concern-level shifts by residue system
# ---------------------------------------------------------------------------
print("=" * 72)
print("CONCERN LEVEL SHIFTS  (old model -> new model)")
print(f"  R={R_FACTOR}, K={K_FACTOR}, T={T_VALUE} t/ac/yr")
print("=" * 72)

for residue in RESIDUE_SYSTEMS:
    subset = [r for r in rows if r["residue_system"] == residue and r["concern_changed"]]
    short  = residue.split("(")[0].strip()
    print(f"\n  {short}")
    if not subset:
        print("    No concern-level changes across tested NDVI × slope grid.")
    else:
        for r in subset:
            print(
                f"    NDVI={r['ndvi']:.2f}  slope={r['slope_pct']:2d}%  "
                f"{r['old_concern']:8s} -> {r['new_concern']:8s}  "
                f"(C: {r['old_c_factor']:.3f}->{r['new_c_factor']:.3f}  "
                f"SL: {r['old_soil_loss_t_ac_yr']:.1f}->{r['new_soil_loss_t_ac_yr']:.1f} t/ac)"
            )

# ---------------------------------------------------------------------------
# Print C-factor comparison at key NDVI values (slope=5% for reference)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("C-FACTOR COMPARISON AT SLOPE=5%  (LS={:.3f})".format(
    float(_analytical_ls_factor(5.0))
))
print(f"  {'Residue System':<45}  {'NDVI':>5}  {'Old C':>6}  {'New C':>6}  {'Delta':>7}")
print("-" * 72)
for residue in RESIDUE_SYSTEMS:
    short = residue.split("(")[0].strip()
    for ndvi in [0.00, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80]:
        r = next(x for x in rows
                 if x["residue_system"] == residue
                 and x["ndvi"] == ndvi
                 and x["slope_pct"] == 5)
        pct = r["c_pct_change"]
        pct_str = f"{pct:+.0f}%" if not (isinstance(pct, float) and np.isnan(pct)) else "  n/a"
        print(f"  {short:<45}  {ndvi:>5.2f}  {r['old_c_factor']:>6.3f}  "
              f"{r['new_c_factor']:>6.3f}  {pct_str:>7}")
    print()

print("Done.")

"""
generate_figures.py
-------------------
Generates all technical reference figures for the CoverMap documentation.
Run once from the project root:  python docs/generate_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.gridspec import GridSpec

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

BLUE  = "#1f6aa5"
DARK  = "#1f2328"
GRAY  = "#d0d7de"


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — System data-flow diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig_dataflow():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_facecolor("#f6f8fa")
    fig.patch.set_facecolor("#f6f8fa")

    def box(x, y, w, h, label, sublabel="", color="#dbeafe", textcolor=DARK, fontsize=8.5):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.08", linewidth=1.2,
            edgecolor="#6b7280", facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.12 if sublabel else 0), label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=textcolor, wrap=True)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.2, sublabel,
                    ha="center", va="center", fontsize=6.8,
                    color="#6b7280", style="italic")

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.4))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx+0.05, my+0.12, label, fontsize=6.5,
                    color="#374151", ha="center")

    # INPUT column
    box(0.2, 5.4, 2.2, 0.9, "Field Boundary", "(GeoJSON / Shapefile)", "#fef9c3")
    box(0.2, 4.0, 2.2, 0.9, "Sentinel-2 L2A", "(Google Earth Engine)", "#dbeafe")
    box(0.2, 2.6, 2.2, 0.9, "Iowa 3-m DEM", "(Iowa DNR / USGS 10-m)", "#dcfce7")
    box(0.2, 1.2, 2.2, 0.9, "USDA SSURGO", "(Web Soil Survey API)", "#f3e8ff")
    box(0.2, 0.1, 2.2, 0.8, "FCC Census API", "(County lookup → R)", "#fee2e2")

    # PROCESSING column
    box(4.0, 5.1, 2.4, 1.2, "NDVI Composite", "Cloud-masked median\nSentinel-2 bands 4 & 8", "#dbeafe")
    box(4.0, 3.5, 2.4, 1.2, "Slope Computation", "np.gradient → % slope\nUnit detection (cm→m)", "#dcfce7")
    box(4.0, 1.9, 2.4, 1.2, "K & T Lookup", "K-factor: SDA mukey API\nT-value: IOWA_T_VALUES dict", "#f3e8ff")
    box(4.0, 0.1, 2.4, 1.2, "R-factor Lookup", "FCC county name →\nR=150 or R=175", "#fee2e2")

    # SCORING column
    box(7.4, 4.3, 2.4, 1.3, "C-factor", "NDVI → Iowa RUSLE\nlookup table\n× residue multiplier", "#fef9c3")
    box(7.4, 2.7, 2.4, 1.3, "LS-factor", "Slope % → lookup table\n(per-pixel + mean field)", "#fef9c3")
    box(7.4, 1.1, 2.4, 1.3, "RUSLE A\n= R·K·LS·C", "Field-mean estimate\nvs soil loss tolerance T", "#fef9c3")

    # OUTPUT column
    box(10.4, 5.0, 2.3, 1.1, "Risk Index Map", "C×LS per pixel\n4 zone classes", "#e0f2fe")
    box(10.4, 3.5, 2.3, 1.1, "NDVI / Slope\nOverlay Maps", "Folium interactive\nlayers", "#e0f2fe")
    box(10.4, 2.0, 2.3, 1.1, "Erosion Concern\nLevel + Advisory", "Low / Moderate\nHigh / Critical", "#dcfce7")
    box(10.4, 0.6, 2.3, 1.1, "PDF Report", "2-page: maps + tables\nEQIP documentation", "#e0f2fe")

    # arrows — inputs to processing
    arrow(2.4, 5.85, 4.0, 5.7)
    arrow(2.4, 4.45, 4.0, 4.0)   # sentinel → NDVI
    arrow(2.4, 4.45, 4.0, 5.5)   # also to NDVI
    arrow(2.4, 3.05, 4.0, 4.1)   # DEM → slope
    arrow(2.4, 1.65, 4.0, 2.5)   # SSURGO → K&T
    arrow(2.4, 0.5,  4.0, 0.7)   # FCC → R

    # processing to scoring
    arrow(6.4, 5.7,  7.4, 5.0)   # NDVI → C
    arrow(6.4, 4.1,  7.4, 4.9)   # slope → C (via NDVI, indirect)
    arrow(6.4, 4.1,  7.4, 3.35)  # slope → LS
    arrow(6.4, 2.5,  7.4, 1.8)   # K&T → A
    arrow(6.4, 0.7,  7.4, 1.5)   # R → A

    # scoring to output
    arrow(9.8, 5.0,  10.4, 5.55)
    arrow(9.8, 3.35, 10.4, 4.05)
    arrow(9.8, 1.75, 10.4, 2.55)
    arrow(9.8, 1.75, 10.4, 1.15)

    ax.text(6.5, 6.75, "CoverMap — System Data Flow",
            ha="center", fontsize=13, fontweight="bold", color=DARK)

    col_labels = ["INPUTS", "PRE-PROCESSING", "RUSLE SCORING", "OUTPUTS"]
    xs = [1.3, 5.2, 8.6, 11.55]
    for xl, lab in zip(xs, col_labels):
        ax.text(xl, 6.45, lab, ha="center", fontsize=9,
                fontweight="bold", color="#6b7280")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_dataflow.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("fig1_dataflow.png  ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — NDVI to C-factor step function
# ─────────────────────────────────────────────────────────────────────────────
def fig_ndvi_cfactor():
    fig, ax = plt.subplots(figsize=(9, 5))

    breakpoints = [
        (0.00, 0.15, 0.90, "Failed stand\n(~bare soil)"),
        (0.15, 0.20, 0.75, "Inadequate\n(<1,000 kg/ha)"),
        (0.20, 0.35, 0.45, "Marginal\n(1,000–2,500 kg/ha)"),
        (0.35, 0.50, 0.20, "Adequate\n(>2,500 kg/ha)"),
        (0.50, 0.65, 0.08, "Good stand"),
        (0.65, 1.00, 0.03, "Excellent\n(canopy saturation)"),
    ]
    colors_bp = ["#ef4444", "#f97316", "#facc15", "#86efac", "#4ade80", "#16a34a"]

    ndvi_vals  = []
    c_vals     = []

    for (lo, hi, c, _) in breakpoints:
        ndvi_vals += [lo, hi]
        c_vals    += [c,  c]

    for i, (lo, hi, c, label) in enumerate(breakpoints):
        ax.axhspan(c - 0.002, c + 0.002, alpha=0.0)
        ax.fill_betweenx([0, c], lo, hi, alpha=0.18, color=colors_bp[i])
        ax.hlines(c, lo, hi, colors=colors_bp[i], linewidths=3.5)
        if i < len(breakpoints) - 1:
            ax.vlines(hi, c, breakpoints[i+1][2], colors="#9ca3af",
                     linewidths=1.2, linestyles="--")
        mx = (lo + hi) / 2
        ax.text(mx, c + 0.025, f"C = {c:.2f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=colors_bp[i])
        ax.text(mx, -0.06, label, ha="center", va="top",
                fontsize=7, color=DARK, style="italic",
                multialignment="center")

    # NRCS 340 minimum line
    ax.axvline(0.20, color="#2563eb", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.text(0.21, 0.78, "NRCS 340\nminimum\ndetection\nthreshold", fontsize=7,
            color="#2563eb", va="top")

    ax.axvline(0.25, color="#7c3aed", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.text(0.26, 0.55, "NRCS 340\n~1,500 kg/ha\nbiomass target\n(NDVI ≈ 0.25)", fontsize=7,
            color="#7c3aed", va="top")

    ax.set_xlabel("NDVI (Normalized Difference Vegetation Index)", fontsize=10)
    ax.set_ylabel("RUSLE C-factor  (0 = full cover,  1 = bare soil)", fontsize=10)
    ax.set_title("Iowa RUSLE C-factor Lookup Table — NDVI Step Function\n"
                 "Source: Laflen & Roose (1998); ISU Extension PM-1209; NRCS Practice Code 340",
                 fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.15, 1.02)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_facecolor("#f9fafb")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_ndvi_cfactor.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("fig2_ndvi_cfactor.png  ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Slope to LS-factor lookup
# ─────────────────────────────────────────────────────────────────────────────
def fig_ls_factor():
    fig, ax = plt.subplots(figsize=(9, 5))

    bins = [
        (0,   2,  0.2,  "Flat\n0–2%"),
        (2,   4,  0.5,  "Nearly\nflat 2–4%"),
        (4,   6,  1.0,  "Gentle\n4–6%"),
        (6,   9,  1.8,  "Moderate\n6–9%"),
        (9,  12,  2.8,  "Rolling\n9–12%"),
        (12, 20,  4.5,  "Steep\n12–20%"),
        (20, 30,  7.0,  "Very steep\n>20%"),
    ]
    palette = ["#3b82f6","#60a5fa","#facc15","#fb923c","#f97316","#ef4444","#991b1b"]
    xs  = [0]
    for (lo, hi, ls, label) in bins:
        xs.append(xs[-1] + (min(hi, 30) - lo))

    for i, (lo, hi, ls, label) in enumerate(bins):
        width = min(hi, 30) - lo
        bar = ax.bar(xs[i] + width/2, ls, width=width * 0.92,
                     color=palette[i], edgecolor="white", linewidth=1.2, alpha=0.9)
        ax.text(xs[i] + width/2, ls + 0.08, f"LS = {ls}", ha="center",
                fontsize=9, fontweight="bold", color=DARK)
        ax.text(xs[i] + width/2, -0.35, label, ha="center",
                fontsize=7.5, color=DARK, multialignment="center")

    # NRCS typical concern line
    ax.axhline(1.0, color="#dc2626", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(28, 1.08, "LS = 1.0\n(NRCS concern\nbaseline 4–6%)", fontsize=7,
            color="#dc2626", ha="right")

    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in [0,2,4,6,9,12,20,30]], fontsize=8)
    ax.set_xlabel("Slope (%)", fontsize=10)
    ax.set_ylabel("LS-factor (dimensionless)", fontsize=10)
    ax.set_title("CoverMap Simplified LS-factor Lookup Table\n"
                 "Calibrated to NRCS RUSLE2 Iowa slope-class thresholds\n"
                 "Note: slope length not explicitly computed — field advisory approximation",
                 fontsize=10, fontweight="bold")
    ax.set_ylim(-0.6, 8.0)
    ax.set_xlim(-0.5, 30.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_facecolor("#f9fafb")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_ls_factor.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("fig3_ls_factor.png  ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — C × LS Risk Index zone matrix
# ─────────────────────────────────────────────────────────────────────────────
def fig_risk_matrix():
    c_vals  = [0.03, 0.08, 0.20, 0.45, 0.75, 0.90]
    ls_vals = [0.2,  0.5,  1.0,  1.8,  2.8,  4.5,  7.0]

    c_labels  = ["0.03\n(Excellent)", "0.08\n(Good)", "0.20\n(Adequate)",
                 "0.45\n(Marginal)",  "0.75\n(Low)",  "0.90\n(Bare)"]
    ls_labels = ["0.2\n(0–2%)", "0.5\n(2–4%)", "1.0\n(4–6%)",
                 "1.8\n(6–9%)", "2.8\n(9–12%)","4.5\n(12–20%)","7.0\n(>20%)"]

    grid = np.array([[c * ls for ls in ls_vals] for c in c_vals])

    zone_grid = np.where(grid < 0.3, 1,
                np.where(grid < 0.7, 2,
                np.where(grid < 1.5, 3, 4)))

    cmap  = ListedColormap(["#22c55e", "#facc15", "#f97316", "#ef4444"])
    norm  = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(zone_grid, cmap=cmap, norm=norm, aspect="auto",
                   origin="lower")

    for i in range(len(c_vals)):
        for j in range(len(ls_vals)):
            score = grid[i, j]
            zone  = zone_grid[i, j]
            txt_color = "white" if zone in (3, 4) else DARK
            ax.text(j, i, f"{score:.3f}", ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=txt_color)

    ax.set_xticks(range(len(ls_vals)))
    ax.set_xticklabels(ls_labels, fontsize=8.5)
    ax.set_yticks(range(len(c_vals)))
    ax.set_yticklabels(c_labels, fontsize=8.5)
    ax.set_xlabel("LS-factor  (slope class)", fontsize=10)
    ax.set_ylabel("C-factor  (NDVI class)", fontsize=10)
    ax.set_title("Risk Index Matrix  (C × LS)\n"
                 "Cell values = pixel risk score; color = zone classification",
                 fontsize=11, fontweight="bold")

    patches = [
        mpatches.Patch(color="#22c55e", label="Zone 1 — Low  (< 0.3)"),
        mpatches.Patch(color="#facc15", label="Zone 2 — Moderate  (0.3 – 0.7)"),
        mpatches.Patch(color="#f97316", label="Zone 3 — High  (0.7 – 1.5)"),
        mpatches.Patch(color="#ef4444", label="Zone 4 — Critical  (≥ 1.5)"),
    ]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.01, 1),
              fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_risk_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("fig4_risk_matrix.png  ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Iowa R-factor county zones
# ─────────────────────────────────────────────────────────────────────────────
def fig_iowa_rfactor():
    r175 = sorted([
        "Appanoose", "Cedar", "Clinton", "Davis", "Des Moines", "Delaware",
        "Dubuque", "Henry", "Iowa", "Jackson", "Jefferson", "Johnson",
        "Jones", "Keokuk", "Lee", "Linn", "Louisa", "Mahaska",
        "Monroe", "Muscatine", "Scott", "Van Buren", "Wapello",
        "Washington", "Wayne",
    ])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5),
                             gridspec_kw={"width_ratios": [2, 1]})

    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_facecolor("#e0f2fe")
    ax.axis("off")

    # Schematic Iowa outline
    iowa_x = [1, 9, 9, 7.5, 6, 4, 2.5, 1, 1]
    iowa_y = [2, 2, 8, 8.5, 9, 9, 8.5, 8, 2]
    ax.fill(iowa_x, iowa_y, color="#dbeafe", edgecolor="#1e40af", linewidth=2)

    # Shade SE region
    se_x = [4.5, 9, 9, 7.5, 6, 4.5]
    se_y = [2,   2, 5, 5,   5, 2  ]
    ax.fill(se_x, se_y, color="#fca5a5", alpha=0.7, edgecolor="#dc2626", linewidth=1.5)

    ax.text(3.2, 6.2, "R = 150", fontsize=16, fontweight="bold", color="#1e40af", ha="center")
    ax.text(3.2, 5.4, "Standard Iowa zone\n(remaining 74 counties)", fontsize=8,
            color="#1e40af", ha="center")
    ax.text(6.8, 3.4, "R = 175", fontsize=14, fontweight="bold", color="#991b1b", ha="center")
    ax.text(6.8, 2.7, "SE Iowa\n(25 counties)", fontsize=8, color="#991b1b", ha="center")

    ax.set_title("Iowa Annual Erosivity (R-factor) Zones\n"
                 "Source: NRCS RUSLE2 Iowa State File; ISU Extension PM-1209",
                 fontsize=10, fontweight="bold")

    # County list on right panel
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_facecolor("#fef2f2")
    ax2.text(0.5, 1.0, "R = 175  Counties  (SE Iowa)", ha="center", va="top",
             fontsize=9.5, fontweight="bold", color="#991b1b",
             transform=ax2.transAxes)
    col1 = r175[:13]
    col2 = r175[13:]
    for i, name in enumerate(col1):
        ax2.text(0.05, 0.93 - i * 0.065, f"• {name}", fontsize=8,
                 color=DARK, transform=ax2.transAxes)
    for i, name in enumerate(col2):
        ax2.text(0.55, 0.93 - i * 0.065, f"• {name}", fontsize=8,
                 color=DARK, transform=ax2.transAxes)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_iowa_rfactor.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("fig5_iowa_rfactor.png  ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Soil loss tolerance (T-value) status bands
# ─────────────────────────────────────────────────────────────────────────────
def fig_tvalue_bands():
    fig, ax = plt.subplots(figsize=(10, 4.5))

    t_value = 5.0
    bands = [
        (0,      t_value * 1.00, "#dcfce7", "Within T  (A/T ≤ 1.0)",    "green"),
        (t_value * 1.00, t_value * 1.25, "#fef9c3", "Near T  (1.0 < A/T ≤ 1.25)", "#ca8a04"),
        (t_value * 1.25, t_value * 2.00, "#fee2e2", "Exceeds T  (1.25 < A/T ≤ 2.0)", "#b91c1c"),
        (t_value * 2.00, t_value * 3.50, "#fecaca", "Critical  (A/T > 2.0)",         "#7f1d1d"),
    ]

    for lo, hi, color, label, tc in bands:
        ax.barh(0, hi - lo, left=lo, height=0.5, color=color,
                edgecolor="#9ca3af", linewidth=1)
        cx = (lo + hi) / 2
        ax.text(cx, 0.3, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=tc)
        ax.text(cx, -0.25, f"{lo:.1f}–{hi:.1f} t/ac/yr", ha="center",
                fontsize=7.5, color="#374151")

    ax.axvline(t_value, color="#1d4ed8", linewidth=2.5, linestyle="--")
    ax.text(t_value + 0.05, 0.6, f"T = {t_value} t/ac/yr\n(soil loss tolerance)", fontsize=8,
            color="#1d4ed8", va="center")

    example_a = 6.5
    ax.annotate("", xy=(example_a, 0.02), xytext=(example_a, -0.45),
                arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=2))
    ax.text(example_a + 0.1, -0.5, f"Example: A = {example_a} t/ac/yr\n"
            f"A/T = {example_a/t_value:.2f}× → Exceeds T",
            fontsize=7.5, color="#7c3aed")

    ax.set_xlim(0, 18)
    ax.set_ylim(-0.75, 0.85)
    ax.set_xlabel("Estimated Annual Soil Loss A  (tons/acre/year)", fontsize=10)
    ax.set_title(f"Soil Loss Tolerance (T-value) Status Bands\n"
                 f"Example: T = {t_value} t/ac/yr (Iowa default NRCS SSURGO)",
                 fontsize=10, fontweight="bold")
    ax.set_yticks([])
    ax.set_facecolor("#f9fafb")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_tvalue_bands.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("fig6_tvalue_bands.png  ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Slope computation pipeline
# ─────────────────────────────────────────────────────────────────────────────
def fig_slope_pipeline():
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.5))

    rng = np.random.default_rng(42)
    dem_raw = rng.integers(38000, 43000, (30, 30)).astype(float)
    dem_m   = dem_raw / 100.0
    dem_m[dem_m == 0] = np.nan

    x_res = y_res = 3.0
    dz_dx, dz_dy = np.gradient(dem_m, x_res, y_res)
    slope_pct = np.hypot(dz_dx, dz_dy) * 100
    slope_pct = np.clip(slope_pct, 0, 100)

    ls_arr = np.where(slope_pct < 2,  0.2,
             np.where(slope_pct < 4,  0.5,
             np.where(slope_pct < 6,  1.0,
             np.where(slope_pct < 9,  1.8,
             np.where(slope_pct < 12, 2.8,
             np.where(slope_pct < 20, 4.5, 7.0))))))

    titles = [
        "Step 1: Raw DEM\n(Iowa 3-m, UInt16 cm units)",
        "Step 2: Elevation → meters\n(÷ 100 for Iowa DEM)",
        "Step 3: Slope % via\nnp.gradient (UTM meters)",
        "Step 4: LS-factor\n(lookup table)",
    ]
    data   = [dem_raw, dem_m, slope_pct, ls_arr]
    cmaps  = ["viridis", "terrain", "RdYlBu_r", "YlOrRd"]
    labels = ["UInt16 raw", "meters", "percent slope", "LS value"]

    for ax, arr, title, cmap, clabel in zip(axes, data, titles, cmaps, labels):
        im = ax.imshow(arr, cmap=cmap, aspect="auto")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(clabel, fontsize=7)
        cb.ax.tick_params(labelsize=6)
        ax.set_title(title, fontsize=8.5, fontweight="bold")
        ax.axis("off")

    fig.suptitle("DEM → Slope → LS-factor Processing Pipeline\n"
                 "Iowa 3-m DEM stores elevation as UInt16 centimeters; "
                 "unit detection converts to meters before gradient",
                 fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_slope_pipeline.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("fig7_slope_pipeline.png  ✓")


if __name__ == "__main__":
    fig_dataflow()
    fig_ndvi_cfactor()
    fig_ls_factor()
    fig_risk_matrix()
    fig_iowa_rfactor()
    fig_tvalue_bands()
    fig_slope_pipeline()
    print(f"\nAll figures saved → {OUT}")

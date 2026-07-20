"""
report_generator.py
-------------------
Single-page PDF field summary report for CoverMap.
Designed for farmer, CCA, and NRCS audiences.

Layout:
  - Header: branding, field ID, CCA credentials, report date
  - Map section: 3-zone NDVI management zone map (PNG embedded)
  - Metrics table: NDVI mean, slope, C-factor, erosion concern
  - Zone breakdown: acres by management zone
  - NRCS recommendation: plain-English action text
  - Footer: data sources, NDVI collection date, methodology note

Dependencies: reportlab (already in requirements via PDF skill)
"""

import io
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from src.qc_utils import qc_signals, valid_tier

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, PageBreak, Flowable,
)
from reportlab.pdfgen import canvas as rl_canvas

# ---------------------------------------------------------------------------
# Brand colors matching app UI
# ---------------------------------------------------------------------------
DARK_BG     = colors.HexColor("#0e1117")
BLUE_ACCENT = colors.HexColor("#58a6ff")
GOLD        = colors.HexColor("#f0c040")
ORANGE      = colors.HexColor("#F97316")   # low cover
STEEL_BLUE  = colors.HexColor("#38BDF8")   # marginal
YELLOW      = colors.HexColor("#FACC15")   # good cover
GREEN_BADGE = colors.HexColor("#1a7f37")
RED_BADGE   = colors.HexColor("#cf222e")
AMBER_BADGE = colors.HexColor("#9a6700")
LIGHT_GRAY  = colors.HexColor("#f6f8fa")
MID_GRAY    = colors.HexColor("#d0d7de")
TEXT_DARK   = colors.HexColor("#1f2328")

# Zone colors matching app map exactly
ZONE_COLORS = {
    "Low cover":    "#F97316",
    "Marginal":     "#38BDF8",
    "Good cover":   "#FACC15",
}

CONCERN_BADGE_COLOR = {
    "Low":      GREEN_BADGE,
    "Moderate": AMBER_BADGE,
    "High":     RED_BADGE,
    "Critical": colors.HexColor("#6e1c1c"),
}

# Disclaimer placed beneath every absolute soil-loss (A, t/ac/yr) value in the
# CCA and 45Z reports (the producer report shows % reduction only, no A values).
# Wording is fixed — see Technical Guide §7.7 (RUSLE2 validation).
A_VALUE_DISCLAIMER = (
    "Soil loss estimates reflect CoverMap's field advisory model. Values may run "
    "2–4× above RUSLE2 on backslope positions due to a fixed slope-length exponent "
    "(m=0.5, §7.7). Use for relative risk ranking and % erosion reduction, not "
    "quantitative determination."
)

# One-line validation provenance for the CCA report footer only.
CCA_VALIDATION_PROVENANCE = (
    "K-factor validated against RUSLE2 v2.7.1 (Shelby County Monona, K=0.37/0.49). "
    "LS and C divergence documented in Tech Guide §7.7. % erosion reduction is the "
    "most defensible comparative metric."
)


# ---------------------------------------------------------------------------
# Map image generator
# ---------------------------------------------------------------------------

def generate_zone_map_image(
    ndvi_array: np.ndarray,
    ndvi_threshold: float = 0.20,
    width_px: int = 600,
    height_px: int = 400,
    array_shape: tuple = None,
) -> bytes:
    """
    Generate a 3-zone NDVI map PNG for embedding in PDF.
    Returns PNG bytes.
    """
    marginal_upper = ndvi_threshold + 0.15

    # Build RGB image
    rgb = np.zeros((*ndvi_array.shape, 3), dtype=np.uint8)

    low_mask      = (~np.isnan(ndvi_array)) & (ndvi_array < ndvi_threshold)
    marginal_mask = (~np.isnan(ndvi_array)) & (ndvi_array >= ndvi_threshold) & (ndvi_array < marginal_upper)
    good_mask     = (~np.isnan(ndvi_array)) & (ndvi_array >= marginal_upper)
    nodata_mask   = np.isnan(ndvi_array)

    rgb[low_mask]      = [249, 115,  22]   # orange
    rgb[marginal_mask] = [ 56, 189, 248]   # steel blue
    rgb[good_mask]     = [250, 204,  21]   # yellow
    rgb[nodata_mask]   = [240, 240, 240]   # light gray for nodata

    if array_shape is not None:
        _r, _c = array_shape[0], array_shape[1]
        _asp = _c / max(_r, 1)
        _fw = width_px / 100
        _fh = max(_fw / _asp, 1.0)
        _figsize = (_fw, _fh)
    else:
        _figsize = (width_px / 100, height_px / 100)
    fig, ax = plt.subplots(1, 1, figsize=_figsize, dpi=100)
    ax.imshow(rgb, aspect="auto")
    ax.axis("off")

    # Legend patches
    patches = [
        mpatches.Patch(color="#F97316", label=f"Low cover (NDVI < {ndvi_threshold:.2f})"),
        mpatches.Patch(color="#38BDF8", label=f"Marginal ({ndvi_threshold:.2f}–{marginal_upper:.2f})"),
        mpatches.Patch(color="#FACC15", label=f"Good cover (NDVI > {marginal_upper:.2f})"),
    ]
    ax.legend(
        handles=patches,
        loc="lower left",
        fontsize=8,
        framealpha=0.85,
        edgecolor="#cccccc",
    )

    fig.tight_layout(pad=0.2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_slope_map_image(
    slope_array: np.ndarray,
    width_px: int = 600,
    height_px: int = 400,
    array_shape: tuple = None,
) -> bytes:
    """Generate slope map PNG using absolute NRCS thresholds."""
    slope_clean = slope_array.copy().astype(float)
    slope_clean[slope_clean <= -9999] = np.nan

    SLOPE_MIN, SLOPE_MAX = 0.0, 15.0
    slope_norm = np.where(
        np.isnan(slope_clean),
        np.nan,
        np.clip((slope_clean - SLOPE_MIN) / (SLOPE_MAX - SLOPE_MIN), 0.0, 1.0),
    )
    slope_norm_safe = np.where(np.isnan(slope_norm), 0.0, slope_norm)

    if array_shape is not None:
        _r, _c = array_shape[0], array_shape[1]
        _asp = _c / max(_r, 1)
        _fw = width_px / 100
        _fh = max(_fw / _asp, 1.0)
        _figsize = (_fw, _fh)
    else:
        _figsize = (width_px / 100, height_px / 100)
    fig, ax = plt.subplots(1, 1, figsize=_figsize, dpi=100)
    img = ax.imshow(
        slope_norm_safe,
        cmap="RdYlBu_r",
        aspect="auto",
        vmin=0, vmax=1,
    )

    # Mask nodata
    mask_img = np.where(np.isnan(slope_norm), 1.0, np.nan)
    ax.imshow(mask_img, cmap="gray", aspect="auto", alpha=0.5)
    ax.axis("off")

    cbar = fig.colorbar(img, ax=ax, orientation="vertical",
                        fraction=0.03, pad=0.02)
    cbar.set_label("Slope (%)", fontsize=8)
    cbar.set_ticks([0, 0.4, 0.8, 1.0])
    cbar.set_ticklabels(["0%", "6%", "12%", "15%+"])

    fig.tight_layout(pad=0.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_risk_zone_map_image(
    risk_zone_array: np.ndarray,
    width_px: int = 600,
    height_px: int = 400,
    array_shape: tuple = None,
) -> bytes:
    """Generate Risk Index zone map PNG (zones 1–4) for embedding in PDF."""
    _ZONE_RGB = {
        1: [ 34, 197,  94],   # green  — Low
        2: [250, 204,  21],   # yellow — Moderate
        3: [249, 115,  22],   # orange — High
        4: [239,  68,  68],   # red    — Critical
    }
    h, w = risk_zone_array.shape
    rgb = np.full((h, w, 3), 240, dtype=np.uint8)   # light gray = nodata
    for val, color in _ZONE_RGB.items():
        m = risk_zone_array == val
        rgb[m] = color

    if array_shape is not None:
        _r, _c = array_shape[0], array_shape[1]
        _asp = _c / max(_r, 1)
        _fw = width_px / 100
        _fh = max(_fw / _asp, 1.0)
        _figsize = (_fw, _fh)
    else:
        _figsize = (width_px / 100, height_px / 100)
    fig, ax = plt.subplots(1, 1, figsize=_figsize, dpi=100)
    ax.imshow(rgb, aspect="auto")
    ax.axis("off")
    ax.set_title("Erosion Risk Index Zones (C\u00d7LS)", fontsize=9, pad=4)

    patches = [
        mpatches.Patch(color=[c / 255 for c in [239,  68,  68]], label="Critical (\u22651.5)"),
        mpatches.Patch(color=[c / 255 for c in [249, 115,  22]], label="High (0.7\u20131.5)"),
        mpatches.Patch(color=[c / 255 for c in [250, 204,  21]], label="Moderate (0.3\u20130.7)"),
        mpatches.Patch(color=[c / 255 for c in [ 34, 197,  94]], label="Low (<0.3)"),
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=8,
              framealpha=0.85, edgecolor="#cccccc")

    fig.tight_layout(pad=0.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_yoy_ndvi_chart_image(
    yoy_rows: List[Dict[str, Any]],
    width_in: float = 5.5,
    height_in: float = 2.4,
    dpi: int = 300,
) -> bytes:
    """
    Year-over-year early-season NDVI bar chart PNG for PDF embedding.

    Matplotlib replica of the app's Plotly YoY chart (kaleido is not a
    project dependency, so the Plotly figure cannot be exported directly).
    Same data rows as the app chart, same RdYlGn coloring by NDVI value,
    year labels on the axis and mean NDVI as data labels — light theme to
    match the printed report.
    """
    years  = [r["Year"] for r in yoy_rows]
    values = [r["Mean NDVI"] for r in yoy_rows]

    cmap = plt.get_cmap("RdYlGn")
    vmin, vmax = min(values), max(values)
    span = (vmax - vmin) or 1.0
    bar_colors = [cmap((v - vmin) / span) for v in values]

    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    bars = ax.bar([str(y) for y in years], values, color=bar_colors,
                  edgecolor="#d0d7de", linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.annotate(
            f"{v:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 2), textcoords="offset points",
            ha="center", va="bottom", fontsize=7, color="#1f2328",
        )
    ax.set_ylabel("Mean NDVI", fontsize=8)
    ax.set_ylim(0, max(values) * 1.18)
    ax.tick_params(labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Zone acreage calculator
# ---------------------------------------------------------------------------

def calculate_zone_acres(
    ndvi_array: np.ndarray,
    ndvi_threshold: float,
    acres_per_pixel: float = (10.0 ** 2) / 4046.86,
) -> Dict[str, float]:
    """Calculate acres per zone from pixel counts."""

    marginal_upper = ndvi_threshold + 0.15
    valid = ~np.isnan(ndvi_array)

    low      = np.sum((ndvi_array < ndvi_threshold) & valid)
    marginal = np.sum((ndvi_array >= ndvi_threshold) & (ndvi_array < marginal_upper) & valid)
    good     = np.sum((ndvi_array >= marginal_upper) & valid)
    total    = low + marginal + good

    return {
        "Low cover":  round(low      * acres_per_pixel, 1),
        "Marginal":   round(marginal * acres_per_pixel, 1),
        "Good cover": round(good     * acres_per_pixel, 1),
        "Total":      round(total    * acres_per_pixel, 1),
    }


# ---------------------------------------------------------------------------
# Main PDF builder
# ---------------------------------------------------------------------------

# CCA REPORT — full output with EQIP checklist, CCA initials, soil loss, signature
def generate_field_report(
    # Field info
    field_name: str,
    farm_name: str,
    county: str,
    # Data
    ndvi_array: np.ndarray,
    slope_array: np.ndarray,
    ndvi_stats: Dict[str, float],
    slope_stats: Dict[str, float],
    risk_result: Dict[str, Any],
    zone_summary: Any,
    risk_zone_array: Optional[np.ndarray] = None,
    zone_counts: Optional[Dict[str, int]] = None,
    # Settings
    ndvi_threshold: float = 0.20,
    slope_threshold: float = 6.0,
    # Dates
    ndvi_date_from: Optional[str] = None,
    ndvi_date_to: Optional[str] = None,
    ndvi_scene_date: Optional[str] = None,
    report_date: Optional[str] = None,
    dem_source: str = "Iowa 3-meter Digital Elevation Model (Iowa DNR)",
    # CCA info
    cca_name: str = "Stephen Zimmerman, CCA MS",
    cca_contact: str = "Ankeny, IA | Ag Research Scientist",
    # Optional field detail
    termination_date: Optional[str] = None,
    previous_crop: Optional[str] = None,
    soil_series: Optional[str] = None,
    soil_k_factor: Optional[str] = None,
    residue_system: Optional[str] = None,
    soil_loss_result: Optional[Dict[str, Any]] = None,
    r_factor: float = 150.0,
    r_factor_note: Optional[str] = None,
    acres_per_pixel: float = (10.0 ** 2) / 4046.86,
    scene_count: Optional[int] = None,
) -> bytes:
    """
    Generate single-page PDF field summary report.
    Returns PDF bytes ready for st.download_button.
    """
    if report_date is None:
        report_date = datetime.now().strftime("%B %d, %Y")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.4*inch,
        bottomMargin=0.4*inch,
    )

    styles = getSampleStyleSheet()
    story  = []

    # --- Styles ---
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Normal"],
        fontSize=18,
        textColor=TEXT_DARK,
        fontName="Helvetica-Bold",
        spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#57606a"),
        fontName="Helvetica",
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Normal"],
        fontSize=11,
        textColor=TEXT_DARK,
        fontName="Helvetica-Bold",
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9,
        textColor=TEXT_DARK,
        fontName="Helvetica",
        leading=13,
    )
    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontSize=7.5,
        textColor=colors.HexColor("#57606a"),
        fontName="Helvetica",
        leading=11,
    )

    # -----------------------------------------------------------------------
    # HEADER
    # -----------------------------------------------------------------------
    header_data = [[
        Paragraph(f"<b>CoverMap</b>", title_style),
        Paragraph(
            f"<b>{cca_name}</b><br/>{cca_contact}",
            ParagraphStyle("Right", parent=body_style, alignment=TA_RIGHT)
        ),
    ]]
    header_table = Table(header_data, colWidths=[4*inch, 3*inch])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=2,
                            color=BLUE_ACCENT, spaceAfter=2))
    story.append(Paragraph("CCA Field Documentation Report", subtitle_style))
    story.append(Spacer(1, 4))

    # Field info row
    ndvi_date_str = ""
    if ndvi_date_from and ndvi_date_to:
        ndvi_date_str = f"NDVI: {ndvi_date_from} – {ndvi_date_to}"
    elif ndvi_date_to:
        ndvi_date_str = f"NDVI collected: {ndvi_date_to}"

    _soil_display = "Not available"
    if soil_series and soil_series not in ("Not available", "Unknown"):
        _soil_display = (
            f"{soil_series} — K-factor: {soil_k_factor}"
            if soil_k_factor and soil_k_factor != "N/A"
            else soil_series
        )

    # Derive previous crop display — "Not recorded" only when residue system is also unknown
    _prev_crop_display = (
        previous_crop if previous_crop
        else ("Not recorded"
              if not residue_system or "Unknown" in residue_system
              else "See previous crop / tillage")
    )

    field_data = [
        [
            Paragraph(f"<b>Field:</b> {field_name}", body_style),
            Paragraph(f"<b>Farm:</b> {farm_name}", body_style),
            Paragraph(f"<b>County:</b> {county}", body_style),
            Paragraph(f"<b>Report Date:</b> {report_date}", body_style),
        ],
        [
            Paragraph(f"<b>Previous crop:</b> {_prev_crop_display}", body_style),
            Paragraph(f"<b>Termination date:</b> {termination_date or '⏳ Pending — document at termination'}", body_style),
            Paragraph(f"<b>Dominant soil series:</b> {_soil_display}", body_style),
            Paragraph(f"<b>Previous crop / tillage:</b> {residue_system or 'Not recorded'}", body_style),
        ],
    ]
    field_table = Table(field_data, colWidths=[1.75*inch]*4)
    field_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_GRAY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("BOX",           (0, 0), (-1, -1), 0.5, MID_GRAY),
        ("LINEBELOW",     (0, 0), (-1, 0),  0.3, MID_GRAY),
    ]))
    story.append(field_table)

    if ndvi_date_str:
        story.append(Paragraph(
            f"<i>{ndvi_date_str} &nbsp;|&nbsp; DEM: {dem_source}</i>",
            small_style
        ))
    story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # PAGE 1 — MAPS
    # -----------------------------------------------------------------------
    story.append(Paragraph("Field Risk Maps", section_style))

    map_label_style = ParagraphStyle(
        "MapLabel", parent=body_style,
        alignment=TA_CENTER, fontSize=8, fontName="Helvetica-Bold"
    )
    _marginal_upper = ndvi_threshold + 0.15

    # Dynamic aspect ratio from array shape
    _rows, _cols = ndvi_array.shape
    _aspect = _cols / max(_rows, 1)

    # Risk Index map — full width, aspect-corrected
    _risk_pdf_w = 7.0 * inch
    _risk_pdf_h = min(_risk_pdf_w / _aspect, 3.8 * inch)
    if risk_zone_array is not None:
        risk_png = generate_risk_zone_map_image(
            risk_zone_array, array_shape=risk_zone_array.shape)
        risk_img = RLImage(io.BytesIO(risk_png), width=_risk_pdf_w, height=_risk_pdf_h)
        story.append(risk_img)
        story.append(Paragraph(
            "Erosion Risk Index Zones (C\u00d7LS) \u2014 pixel-level RUSLE risk classification",
            map_label_style,
        ))
        story.append(Spacer(1, 8))

    # NDVI + Slope — side by side, aspect-corrected
    _map_pdf_w = 3.4 * inch
    _map_pdf_h = min(_map_pdf_w / _aspect, 2.8 * inch)
    ndvi_png  = generate_zone_map_image(ndvi_array, ndvi_threshold,
                                        array_shape=ndvi_array.shape)
    slope_png = generate_slope_map_image(slope_array,
                                         array_shape=slope_array.shape)
    map_w = _map_pdf_w
    map_h = _map_pdf_h
    ndvi_img  = RLImage(io.BytesIO(ndvi_png),  width=map_w, height=map_h)
    slope_img = RLImage(io.BytesIO(slope_png), width=map_w, height=map_h)

    maps_table = Table(
        [[ndvi_img, slope_img]],
        colWidths=[map_w + 0.1 * inch, map_w + 0.1 * inch],
    )
    maps_table.setStyle(TableStyle([
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(maps_table)

    labels_data = [[
        Paragraph(
            f"NDVI Cover Quality \u2014 Low (<{ndvi_threshold:.2f}) / Marginal / Good (>{_marginal_upper:.2f})",
            map_label_style,
        ),
        Paragraph("Terrain Slope (% gradient) \u2014 Flat / Moderate / Steep", map_label_style),
    ]]
    labels_table = Table(
        labels_data,
        colWidths=[map_w + 0.1 * inch, map_w + 0.1 * inch],
    )
    labels_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(labels_table)

    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # PAGE 2
    # -----------------------------------------------------------------------

    # --- CoverMap Advisory ---
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=MID_GRAY, spaceAfter=4))
    story.append(Paragraph("CoverMap Advisory & Recommendation", section_style))

    concern     = risk_result.get("concern_level", "N/A")
    concern_col = CONCERN_BADGE_COLOR.get(concern, TEXT_DARK)

    concern_badge_style = ParagraphStyle(
        "ConcernBadge", parent=body_style,
        fontSize=10, fontName="Helvetica-Bold", textColor=concern_col,
    )
    story.append(Paragraph(f"Erosion Concern: {concern}", concern_badge_style))

    rec_text = risk_result.get("recommendation", "No recommendation available.")
    rec_bg   = {
        "Low":      colors.HexColor("#dcfce7"),
        "Moderate": colors.HexColor("#fef9c3"),
        "High":     colors.HexColor("#fee2e2"),
        "Critical": colors.HexColor("#fecaca"),
    }.get(concern, LIGHT_GRAY)

    rec_data = [[Paragraph(rec_text, body_style)]]
    rec_table = Table(rec_data, colWidths=[7 * inch])
    rec_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), rec_bg),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("BOX",           (0, 0), (-1, -1), 0.5, MID_GRAY),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 8))

    # --- NDVI Zone Summary ---
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=MID_GRAY, spaceAfter=4))
    story.append(Paragraph("Cover Crop Stand \u2014 NDVI Zone Summary", section_style))

    zone_acres  = calculate_zone_acres(ndvi_array, ndvi_threshold, acres_per_pixel=acres_per_pixel)
    total_acres = zone_acres.get("Total", 1)

    ndvi_low_label  = f"Low Cover  (NDVI < {ndvi_threshold:.2f})"
    ndvi_mid_label  = (
        f"Marginal  (NDVI {ndvi_threshold:.2f}"
        f"\u2013{_marginal_upper:.2f})")
    ndvi_good_label = f"Good Cover  (NDVI > {_marginal_upper:.2f})"

    ndvi_zone_rows = [["Zone", "Acres", "% of Field"]]
    ndvi_zone_bg   = []
    for i, (zone_key, label, bg) in enumerate([
        ("Low cover",  ndvi_low_label,  colors.HexColor("#FEE8D5")),
        ("Marginal",   ndvi_mid_label,  colors.HexColor("#E0F2FE")),
        ("Good cover", ndvi_good_label, colors.HexColor("#FEF9C3")),
    ], start=1):
        acres = zone_acres.get(zone_key, 0)
        pct   = acres / total_acres * 100 if total_acres > 0 else 0
        ndvi_zone_rows.append([label, f"{acres:.1f}", f"{pct:.0f}%"])
        ndvi_zone_bg.append(("BACKGROUND", (0, i), (-1, i), bg))
    ndvi_zone_rows.append(["Total", f"{total_acres:.1f}", "100%"])

    ndvi_zone_table = Table(ndvi_zone_rows, colWidths=[3.2 * inch, 1.0 * inch, 1.0 * inch])
    ndvi_zone_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  BLUE_ACCENT),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("BACKGROUND",    (0, 4), (-1, 4),  LIGHT_GRAY),
        ("FONTNAME",      (0, 4), (-1, 4),  "Helvetica-Bold"),
        ("ALIGN",         (1, 0), (2, -1),  "CENTER"),
        ("GRID",          (0, 0), (-1, -1), 0.3, MID_GRAY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ] + ndvi_zone_bg))
    story.append(ndvi_zone_table)
    story.append(Spacer(1, 8))

    # --- Risk Index Zone Summary ---
    if zone_counts and sum(zone_counts.values()) > 0:
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=MID_GRAY, spaceAfter=4))
        story.append(Paragraph("Erosion Risk Zone Summary (C\u00d7LS)", section_style))

        px_area_acres = acres_per_pixel
        total_px = sum(zone_counts.values())

        ri_config = [
            (4, "Critical Risk", "#EF4444", "#fecaca", "> 1.5"),
            (3, "High Risk",     "#F97316", "#FEE8D5", "0.7\u20131.5"),
            (2, "Moderate Risk", "#FACC15", "#FEF9C3", "0.3\u20130.7"),
            (1, "Low Risk",      "#22C55E", "#dcfce7", "< 0.3"),
        ]
        ri_rows = [["Zone", "C\u00d7LS Range", "Acres", "% of Field"]]
        for val, label, _, bg, thresh in ri_config:
            count = zone_counts.get(val, 0)
            acres = count * px_area_acres
            pct   = count / total_px * 100 if total_px > 0 else 0
            ri_rows.append([label, thresh, f"{acres:.1f}", f"{pct:.0f}%"])

        ri_style = TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  BLUE_ACCENT),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
            ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
            ("GRID",          (0, 0), (-1, -1), 0.3, MID_GRAY),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ])
        bg_map = {4: "#fecaca", 3: "#FEE8D5", 2: "#FEF9C3", 1: "#dcfce7"}
        for idx, (val, _, _, _, _) in enumerate(ri_config, start=1):
            ri_style.add(
                "BACKGROUND",
                (0, idx), (-1, idx),
                colors.HexColor(bg_map[val]),
            )

        ri_table = Table(
            ri_rows,
            colWidths=[1.8 * inch, 1.2 * inch, 1.0 * inch, 1.0 * inch],
        )
        ri_table.setStyle(ri_style)
        story.append(ri_table)
        story.append(Paragraph(
            "<i>Risk Index = C-factor (NDVI) \u00d7 "
            "LS-factor (slope). Boundary-masked field pixels only. "
            "Critical >1.5 \u00b7 High 0.7\u20131.5 "
            "\u00b7 Moderate 0.3\u20130.7 \u00b7 Low <0.3</i>",
            small_style,
        ))
        story.append(Spacer(1, 6))

    # --- Amber disclaimer ---
    _img_date_label = ndvi_scene_date or ndvi_date_to or "unknown"
    disclaimer_text = (
        f"NDVI imagery dated {_img_date_label}. Field conditions may have changed since "
        f"image capture. This report documents satellite-observed conditions only."
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=body_style,
        fontSize=8, textColor=colors.HexColor("#92400e"),
    )
    disc_data = [[Paragraph(disclaimer_text, disclaimer_style)]]
    disc_table = Table(disc_data, colWidths=[7 * inch])
    disc_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#fef3c7")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("BOX",           (0, 0), (-1, -1), 0.5, colors.HexColor("#f59e0b")),
    ]))
    story.append(disc_table)
    story.append(Spacer(1, 8))

    # -----------------------------------------------------------------------
    # EQIP PRE-VERIFICATION CHECKLIST
    # -----------------------------------------------------------------------
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=MID_GRAY, spaceAfter=4))
    story.append(Paragraph("Cover Crop Stand Assessment \u2014 Satellite Documentation", section_style))

    ndvi_mean_val  = ndvi_stats.get("mean", 0.0)
    biomass_kgha   = max(0.0, (ndvi_mean_val - 0.10) / 0.40 * 3500)
    biomass_lbac   = biomass_kgha * 0.891
    biomass_low    = max(0, round(biomass_lbac * 0.6 / 50) * 50)
    biomass_high   = round(biomass_lbac * 1.4 / 50) * 50
    valid_px       = ndvi_array[~np.isnan(ndvi_array)]
    pct_above_020  = (np.sum(valid_px > 0.20) / valid_px.size * 100) if valid_px.size > 0 else 0.0
    image_date_str = ndvi_scene_date if ndvi_scene_date else (ndvi_date_to if ndvi_date_to else "Upload date unknown")

    cover_status = (
        f"\u2705 NDVI {ndvi_mean_val:.3f} \u2014 cover crop confirmed"
        if ndvi_mean_val > 0.20 else
        f"\u26a0\ufe0f NDVI {ndvi_mean_val:.3f} \u2014 inadequate cover"
    )
    ground_cover_status = (
        "\u2705 Estimated adequate cover zones based on NDVI threshold \u2014 field verification recommended"
        if pct_above_020 > 50 else
        "\u26a0\ufe0f Estimated adequate cover zones below 50% of field \u2014 field verification recommended"
    )
    _term_status = termination_date if termination_date else "\u23f3 Pending \u2014 document at termination"

    _SAT_VER = "Satellite\nVerified"
    _CCA_REQ = ""

    eqip_data = [
        ["Requirement", "Data Source", "Status", "CCA\nInitials"],
        ["Cover crop present",   "Sentinel-2 NDVI > 0.20",  cover_status,        _SAT_VER],
        ["Field boundary",       "Operator provided",        "Verify against FSA CLU records", _CCA_REQ],
        ["Image date",           "GEE metadata",             image_date_str,      _SAT_VER],
        ["Estimated biomass",    "NDVI proxy",               f"~{biomass_low}\u2013{biomass_high} lb/acre (\u00b140% NDVI proxy)", _SAT_VER],
        ["30% ground cover",     "NDVI threshold",           ground_cover_status, _SAT_VER],
        ["Seeding rate",         "Field records required",   "\U0001f4cb CCA to verify on-site", _CCA_REQ],
        ["Species confirmation", "Field records required",   "\U0001f4cb CCA to verify on-site", _CCA_REQ],
        ["Termination date",     "Field records required",   _term_status,        _CCA_REQ],
        ["Cooperator signature", "Physical form required",   "\U0001f4cb Required for EQIP submission", _CCA_REQ],
    ]

    eqip_col_w = [1.6 * inch, 1.6 * inch, 2.8 * inch, 0.8 * inch]
    eqip_table = Table(
        [[Paragraph(str(cell), body_style) for cell in row] for row in eqip_data],
        colWidths=eqip_col_w,
    )
    eqip_style = TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  BLUE_ACCENT),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 8.0),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GRAY, colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.3, MID_GRAY),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",          (3, 0), (3, -1),  "CENTER"),
        ("FONTSIZE",       (3, 1), (3, -1),  8),
    ])
    for _r in [1, 3, 4, 5]:    # satellite-verified rows — green label
        eqip_style.add("TEXTCOLOR", (3, _r), (3, _r), colors.HexColor("#166534"))
    for _r in [2, 6, 7, 8, 9]: # CCA-required rows — underline for initials
        eqip_style.add("LINEBELOW", (3, _r), (3, _r), 0.75, colors.HexColor("#555555"))
    eqip_table.setStyle(eqip_style)
    story.append(eqip_table)
    story.append(Paragraph(
        "<i>Satellite Verified = confirmed by CoverMap remote sensing analysis | "
        "blank line = CCA field verification and initials required before EQIP "
        "submission per NRCS Practice Code 340</i>",
        small_style,
    ))
    story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # COVER CROP METRICS
    # -----------------------------------------------------------------------
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=MID_GRAY, spaceAfter=4))
    story.append(Paragraph("Field Level Results", section_style))

    _c_adj     = risk_result.get("c_factor", 0)
    _c_base    = risk_result.get("c_factor_baseline", _c_adj)
    _c_pct     = int((_c_base - _c_adj) / _c_base * 100) if _c_base > 0 else 0
    _c_display = f"{_c_adj:.3f} ({_c_pct}% reduction vs. baseline)"
    _c_label   = "C-Factor (exp. model)"

    # QC signals (Signal 1 valid-pixel tier) via the shared helper so this
    # field-metrics table matches the app top box and the 45Z report exactly.
    _qc_cca = qc_signals(ndvi_array, scene_count=scene_count,
                         mean_ndvi=ndvi_stats.get("mean", 0.0))

    metrics = [
        ["Metric", "Value"],
        ["NDVI Mean",         f"{ndvi_stats.get('mean', 0):.3f}"],
        ["NDVI Range",        f"{ndvi_stats.get('min', 0):.3f} \u2013 {ndvi_stats.get('max', 0):.3f}"],
        ["Slope Mean (%)",    f"{slope_stats.get('mean', 0):.1f}%"],
        [_c_label,            _c_display],
        ["Risk Index (C\u00d7LS)", f"{risk_result.get('rusle_score', 0):.3f}"],
        ["Erosion Concern",   concern],
        ["Valid Pixels (QC)",
         f"{_qc_cca['valid_pct']:.0f}% \u2014 {_qc_cca['valid_phrase']}"],
    ]

    met_style = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  BLUE_ACCENT),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT_GRAY, colors.white]),
        ("ALIGN",         (1, 0), (1, -1),  "CENTER"),
        ("GRID",          (0, 0), (-1, -1), 0.3, MID_GRAY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("TEXTCOLOR",     (1, 6), (1, 6),   concern_col),
        ("FONTNAME",      (1, 6), (1, 6),   "Helvetica-Bold"),
    ])
    met_table = Table(metrics, colWidths=[1.8 * inch, 1.8 * inch])
    met_table.setStyle(met_style)
    story.append(met_table)
    # QC Signals 2 (single-scene) and 3 (saturation) \u2014 conditional advisory lines.
    _qc_amber = ParagraphStyle("QCAmber", parent=small_style,
                               textColor=colors.HexColor("#92400e"))
    for _qc_line in (_qc_cca["single_scene"], _qc_cca["saturation"]):
        if _qc_line:
            story.append(Paragraph(f"<i>QC: {_qc_line}</i>", _qc_amber))
    story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # EROSION REDUCTION BY RISK ZONE
    # -----------------------------------------------------------------------
    _zes = risk_result.get("zone_erosion_summary", [])
    if _zes:
        # A legitimate weighted 0.0 (e.g. NDVI at/below the living-cover
        # baseline in every zone) must stay 0.0 — `sum(...) or None` coerced
        # it to None and crashed the division below. Matches app.py:712-723.
        _saved_vals_r = [z["a_saved_zone"] * z["area_fraction"]
                         for z in _zes if z.get("a_saved_zone") is not None]
        _base_vals_r  = [z["a_baseline_zone"] * z["area_fraction"]
                         for z in _zes if z.get("a_baseline_zone") is not None]
        _cur_vals_r   = [z["a_current_zone"] * z["area_fraction"]
                         for z in _zes if z.get("a_current_zone") is not None]
        _a_saved_weighted_r    = sum(_saved_vals_r) if _saved_vals_r else None
        _a_baseline_weighted_r = sum(_base_vals_r)  if _base_vals_r  else None
        _a_current_weighted_r  = sum(_cur_vals_r)   if _cur_vals_r   else None
        _pct_reduction_weighted_r = (
            (_a_saved_weighted_r / _a_baseline_weighted_r) * 100
            if (_a_baseline_weighted_r and _a_saved_weighted_r is not None)
            else None
        )
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=MID_GRAY, spaceAfter=4))
        story.append(Paragraph("Erosion Reduction by Risk Zone", section_style))
        _zone_hdr = [
            "Risk Zone", "Zone Area\n(%)", "Mean Slope\n(%)", "Mean NDVI", "C-factor", "Mean LS",
            "Est. Soil Loss\n(t/ac/yr)", "Est. Reduction\n(%)",
            "Est. Soil Saved\n(t/ac/yr)",
        ]
        _zone_rows = [_zone_hdr]
        for _z in _zes:
            _a_cur = f"{_z['a_current_zone']:.1f}" if _z["a_current_zone"] is not None else "K unavail."
            _pct   = f"{_z['pct_reduction']:.0f}%" if _z["pct_reduction"] is not None else "N/A"
            _a_sav = f"{_z['a_saved_zone']:.1f}"   if _z["a_saved_zone"] is not None else "N/A"
            _zone_rows.append([
                _z["zone_label"],
                f"{_z['area_fraction']*100:.0f}%",
                f"{_z['mean_slope_pct']:.1f}%",
                f"{_z['mean_ndvi']:.3f}",
                f"{_z['c_adj']:.3f}",
                f"{_z['mean_ls']:.2f}",
                _a_cur,
                _pct,
                _a_sav,
            ])
        _footer_label_style = ParagraphStyle(
            "FooterLabel", parent=styles["Normal"],
            fontSize=8, fontName="Helvetica-Bold", textColor=colors.white,
        )
        _zone_rows.append([
            Paragraph("Field (area-weighted)", _footer_label_style),
            "100%",
            "—", "—", "—", "—",
            f"{_a_current_weighted_r:.1f}" if _a_current_weighted_r is not None else "—",
            f"{_pct_reduction_weighted_r:.1f}%" if _pct_reduction_weighted_r is not None else "—",
            f"{_a_saved_weighted_r:.1f}"         if _a_saved_weighted_r       is not None else "—",
        ])
        _footer_row_idx = len(_zone_rows) - 1
        _zt = Table(
            _zone_rows,
            colWidths=[1.0*inch, 0.55*inch, 0.65*inch, 0.65*inch, 0.60*inch, 0.60*inch,
                       0.90*inch, 0.80*inch, 1.25*inch],
        )
        _zt_style = TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),           BLUE_ACCENT),
            ("TEXTCOLOR",     (0, 0), (-1, 0),           colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),           "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1),          8.0),
            ("ROWBACKGROUNDS",(0, 1), (-1, _footer_row_idx - 1), [LIGHT_GRAY, colors.white]),
            ("FONTNAME",      (0, 1), (0, -1),           "Helvetica-Bold"),
            ("ALIGN",         (0, 0), (-1, -1),          "CENTER"),
            ("GRID",          (0, 0), (-1, -1),          0.3, MID_GRAY),
            ("BOTTOMPADDING", (0, 0), (-1, -1),          4),
            ("TOPPADDING",    (0, 0), (-1, -1),          4),
            ("LEFTPADDING",   (0, 0), (-1, -1),          4),
            ("BACKGROUND",    (0, _footer_row_idx), (-1, _footer_row_idx), colors.HexColor("#1f2937")),
            ("TEXTCOLOR",     (0, _footer_row_idx), (-1, _footer_row_idx), colors.white),
            ("FONTNAME",      (0, _footer_row_idx), (-1, _footer_row_idx), "Helvetica-Bold"),
        ])
        _zone_color_map = {
            "Low":      GREEN_BADGE,
            "Moderate": AMBER_BADGE,
            "High":     RED_BADGE,
            "Critical": colors.HexColor("#6e1c1c"),
        }
        for _ri, _z in enumerate(_zes, start=1):
            _zt_style.add("TEXTCOLOR", (0, _ri), (0, _ri),
                          _zone_color_map.get(_z["zone_label"], TEXT_DARK))
        _zt.setStyle(_zt_style)
        story.append(_zt)
        story.append(Paragraph(
            "<i>Reduction percentages reflect per-zone C-factor from the piecewise exponential NDVI model. "
            "Absolute soil loss differs by zone due to LS variation. ±10 pt uncertainty on reduction percentage.</i>",
            small_style,
        ))
        story.append(Paragraph(
            f"<i>{A_VALUE_DISCLAIMER}</i>",
            ParagraphStyle("AValueDisclaimer", parent=small_style,
                           textColor=colors.HexColor("#92400e")),
        ))
        story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # SOIL-LOSS TOLERANCE BY RISK ZONE × SOIL  (per-map-unit A/T, FULL list)
    # Adjacent to "Erosion Reduction by Risk Zone" (both zone-level erosion
    # content). Severity is read verbatim from rows_full — the SAME source the
    # app table uses — so wording matches exactly. reportlab splits long tables
    # natively at row boundaries; repeatRows=1 repeats the header on each page.
    # Omitted entirely on WSS-fallback runs where zone_mukey_tolerance is absent.
    # -----------------------------------------------------------------------
    _zmt_rows = (risk_result.get("zone_mukey_tolerance") or {}).get("rows_full") or []
    if _zmt_rows:
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=MID_GRAY, spaceAfter=4))
        story.append(Paragraph("Soil-Loss Tolerance by Risk Zone × Soil", section_style))

        # Sort A/T descending, None ratios last (matches the app "show all" view).
        _zmt_sorted = sorted(
            _zmt_rows,
            key=lambda r: (r["a_over_t"] if r.get("a_over_t") is not None else -1.0),
            reverse=True,
        )
        _tol_data = [[
            "Risk Zone", "Soil (musym)", "T", "A/T", "Severity", "Acres", "A (t/ac/yr)",
        ]]
        for _r in _zmt_sorted:
            _aot = _r.get("a_over_t")
            _ac  = _r.get("overlap_acres")
            _az  = _r.get("a_zone")
            _tv  = _r.get("soil_T")
            _tol_data.append([
                _r.get("risk_zone", ""),
                str(_r.get("musym") or _r.get("mukey") or ""),
                (f"{_tv:g}"      if _tv  is not None else ""),
                (f"{_aot:.1f}×" if _aot is not None else ""),
                _r.get("severity") or "",
                (f"{_ac:.2f}"    if _ac  is not None else ""),
                (f"{_az:.1f}"    if _az  is not None else ""),
            ])
        _tol_tbl = Table(
            _tol_data,
            colWidths=[1.0*inch, 1.0*inch, 0.5*inch, 0.65*inch, 2.25*inch,
                       0.8*inch, 0.85*inch],
            repeatRows=1,
        )
        _tol_style = TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  BLUE_ACCENT),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8.0),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT_GRAY, colors.white]),
            ("FONTNAME",      (0, 1), (0, -1),  "Helvetica-Bold"),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("GRID",          (0, 0), (-1, -1), 0.3, MID_GRAY),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ])
        # Color the Severity cell by Tech Guide tier (mirrors the zone table's
        # per-zone label coloring; the wording itself is unchanged).
        _sev_color = {
            "Within tolerable limit":      GREEN_BADGE,
            "Near tolerable limit":        AMBER_BADGE,
            "Exceeds tolerable limit":     RED_BADGE,
            "Significantly exceeds limit": colors.HexColor("#6e1c1c"),
        }
        for _ri, _r in enumerate(_zmt_sorted, start=1):
            _sc = _sev_color.get(_r.get("severity"))
            if _sc is not None:
                _tol_style.add("TEXTCOLOR", (4, _ri), (4, _ri), _sc)
        _tol_tbl.setStyle(_tol_style)
        story.append(_tol_tbl)
        story.append(Paragraph(
            "<i>A = modeled soil loss (R·K·LS·C) per soil map unit; A/T compares it "
            "to that soil's tolerance T. All zone–soil combinations shown, sorted by A/T "
            "descending. Simplified RUSLE advisory estimate (±10 pt uncertainty) — not a "
            "substitute for a site-specific RUSLE2 run or official NRCS determination.</i>",
            small_style,
        ))
        story.append(Paragraph(
            f"<i>{A_VALUE_DISCLAIMER}</i>",
            ParagraphStyle("AValueDisclaimer", parent=small_style,
                           textColor=colors.HexColor("#92400e")),
        ))
        story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # ESTIMATED SOIL LOSS vs. SOIL LOSS TOLERANCE
    # -----------------------------------------------------------------------
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=MID_GRAY, spaceAfter=4))
    story.append(Paragraph("Estimated Soil Loss vs. Soil Loss Tolerance", section_style))

    if soil_loss_result and soil_loss_result.get("status_code") != "unavailable":
        _sl          = soil_loss_result.get("soil_loss_tons_ac_yr", 0)
        _tv          = soil_loss_result.get("t_value", 5)
        _rt          = soil_loss_result.get("ratio_to_t", 0)
        _sc          = soil_loss_result.get("status_code", "over_t")
        _status_text = soil_loss_result.get("conservation_status", "")

        sl_metrics = [
            ["Est. Soil Loss (A)", "Soil Loss Tolerance (T)", "Ratio to T", "Status"],
            [f"{_sl:.1f} t/ac/yr", f"{_tv} t/ac/yr", f"{_rt:.2f}\u00d7", _status_text],
        ]
        # T-value thresholds: <=1x within_t, 1-2x near_t,
        # 2-5x over_t, >5x critical_t
        # Source: Iowa NRCS FOTG / RUSLE advisory thresholds
        _status_bg = {
            "within_t":   colors.HexColor("#dcfce7"),
            "near_t":     colors.HexColor("#fef9c3"),
            "over_t":     colors.HexColor("#fee2e2"),
            "critical_t": colors.HexColor("#fecaca"),
        }.get(_sc, LIGHT_GRAY)
        _status_fg = {
            "within_t":   GREEN_BADGE,
            "near_t":     AMBER_BADGE,
            "over_t":     RED_BADGE,
            "critical_t": colors.HexColor("#6e1c1c"),
        }.get(_sc, TEXT_DARK)

        sl_table = Table(
            [[Paragraph(str(cell), body_style) for cell in row] for row in sl_metrics],
            colWidths=[1.5 * inch, 1.5 * inch, 1.0 * inch, 3.0 * inch],
        )
        sl_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  BLUE_ACCENT),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
            ("BACKGROUND",    (0, 1), (-1, 1),  _status_bg),
            ("TEXTCOLOR",     (3, 1), (3, 1),   _status_fg),
            ("FONTNAME",      (3, 1), (3, 1),   "Helvetica-Bold"),
            ("ALIGN",         (0, 0), (2, -1),  "CENTER"),
            ("GRID",          (0, 0), (-1, -1), 0.3, MID_GRAY),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ]))
        story.append(sl_table)
        _r_note = r_factor_note or f"R={r_factor:.0f} (erosivity index)"
        story.append(Paragraph(
            f"<i>RUSLE R-factor: {_r_note} | "
            f"A = R \u00d7 K \u00d7 LS \u00d7 C (P=1.0). Simplified RUSLE estimate for advisory "
            f"use only \u2014 not a substitute for a site-specific RUSLE2 run or official "
            f"NRCS determination.</i>",
            small_style,
        ))
        story.append(Paragraph(
            "<i>Field-average estimate using mean slope and dominant soil series. "
            "Steep backslope units likely exceed this estimate significantly. "
            "RUSLE2 analysis of the dominant critical area will produce higher values "
            "for the same field. See RUSLE2 comparison note in Technical Guide.</i>",
            ParagraphStyle("SoilLossContext", parent=small_style,
                           textColor=colors.HexColor("#92400e")),
        ))
        story.append(Paragraph(
            f"<i>{A_VALUE_DISCLAIMER}</i>",
            ParagraphStyle("AValueDisclaimer", parent=small_style,
                           textColor=colors.HexColor("#92400e")),
        ))
    else:
        story.append(Paragraph(
            "Soil loss estimate unavailable \u2014 K-factor not returned from USDA "
            "Web Soil Survey for this field location.",
            body_style,
        ))
    story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # CCA FIELD VERIFICATION NOTES
    # -----------------------------------------------------------------------
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=MID_GRAY, spaceAfter=4))
    story.append(Paragraph("CCA Field Verification Notes", section_style))

    notes_style = ParagraphStyle(
        "NotesLine", parent=body_style,
        fontSize=8, leading=18, textColor=colors.HexColor("#57606a"),
    )
    rule_line = "_" * 110
    notes_content = [
        Paragraph(rule_line, notes_style),
        Paragraph(rule_line, notes_style),
        Paragraph(rule_line, notes_style),
        Paragraph(rule_line, notes_style),
        Spacer(1, 6),
        Paragraph(
            "[ ] I have reviewed this satellite assessment and confirm it accurately "
            "represents field conditions to the best of my knowledge.",
            body_style,
        ),
        Spacer(1, 8),
        Paragraph(
            "CCA Signature: ___________________________  Initials: _______  Date: _______________",
            body_style,
        ),
        Spacer(1, 4),
        Paragraph(f"Printed Name: {cca_name}", body_style),
    ]
    notes_block = Table([[col] for col in notes_content], colWidths=[7 * inch])
    notes_block.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_GRAY),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("BOX",           (0, 0), (-1, -1), 0.5, MID_GRAY),
    ]))
    story.append(notes_block)
    story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # FOOTER
    # -----------------------------------------------------------------------
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=MID_GRAY, spaceAfter=3))

    footer_lines = [
        f"NDVI Source: Sentinel-2 via Google Earth Engine ({ndvi_date_str}) | "
        f"DEM: {dem_source} | Slope: computed in UTM meters (EPSG:26915)",
        f"Data provenance — DEM source: {dem_source}  |  "
        f"R-factor source: {r_factor_note or 'R=%.0f' % r_factor}",
        "C-Factor methodology: Continuous exponential model \u2014 "
        "C(NDVI) = floor + (intercept \u2212 floor) \u00d7 exp(\u2212k \u00d7 NDVI). "
        "Parameters derived from published RUSLE2 value ranges for Iowa cropland; residue system "
        "context is baked into per-system intercept, floor, and k parameters. "
        "Calibration against RUSLE2 Iowa State File runs in progress (Shelby County NRCS, W. Dittmer, 2026). "
        "Parameters subject to revision. This report is advisory only and does not constitute an official NRCS determination.",
        CCA_VALIDATION_PROVENANCE,
        f"CoverMap CCA Report \u00b7 {cca_name} \u00b7 Sentinel-2 via Google Earth Engine \u00b7 Iowa RUSLE C-factor calibration \u00b7 {report_date}",
    ]
    for line in footer_lines:
        story.append(Paragraph(line, small_style))

    # Build PDF
    doc.build(story)
    buf.seek(0)
    return buf.read()


def generate_producer_report(
    # Field info
    field_name: str,
    farm_name: str,
    county: str,
    # Data
    ndvi_array: np.ndarray,
    slope_array: np.ndarray,
    ndvi_stats: Dict[str, float],
    slope_stats: Dict[str, float],
    risk_result: Dict[str, Any],
    zone_summary: Any,
    risk_zone_array: Optional[np.ndarray] = None,
    zone_counts: Optional[Dict[str, int]] = None,
    # Settings
    ndvi_threshold: float = 0.20,
    slope_threshold: float = 6.0,
    # Dates
    ndvi_date_from: Optional[str] = None,
    ndvi_date_to: Optional[str] = None,
    ndvi_scene_date: Optional[str] = None,
    report_date: Optional[str] = None,
    dem_source: str = "Iowa 3-meter Digital Elevation Model (Iowa DNR)",
    # CCA info
    cca_name: str = "Stephen Zimmerman, CCA MS",
    cca_contact: str = "Ankeny, IA | Ag Research Scientist",
    # Optional field detail
    termination_date: Optional[str] = None,
    previous_crop: Optional[str] = None,
    soil_series: Optional[str] = None,
    soil_k_factor: Optional[str] = None,
    residue_system: Optional[str] = None,
    soil_loss_result: Optional[Dict[str, Any]] = None,
    r_factor: float = 150.0,
    r_factor_note: Optional[str] = None,
    acres_per_pixel: float = (10.0 ** 2) / 4046.86,
    scene_count: Optional[int] = None,
    yoy_rows: Optional[List[Dict[str, Any]]] = None,
    cdl_rotation: Optional[List[Dict[str, Any]]] = None,
) -> bytes:
    """
    Simplified single-page PDF field summary for producers.
    Returns PDF bytes ready for st.download_button.

    yoy_rows: optional [{"Year": 2023, "Mean NDVI": 0.300}, ...] from
    gee_ndvi_utils.fetch_yoy_ndvi_rows — when present, a year-over-year
    early-season NDVI chart is embedded after the stand assessment section.

    cdl_rotation: optional rows from cdl_utils.get_cdl_rotation_rows — when
    present, a "Recent Crop Rotation (USDA CDL)" table is inserted after the
    field-setup metadata, plus a CDL provenance footer note.
    """
    if report_date is None:
        report_date = datetime.now().strftime("%B %d, %Y")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.4*inch,
        bottomMargin=0.4*inch,
    )

    styles = getSampleStyleSheet()
    story  = []

    # --- Styles ---
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Normal"],
        fontSize=18,
        textColor=TEXT_DARK,
        fontName="Helvetica-Bold",
        spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#57606a"),
        fontName="Helvetica",
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Normal"],
        fontSize=11,
        textColor=TEXT_DARK,
        fontName="Helvetica-Bold",
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9,
        textColor=TEXT_DARK,
        fontName="Helvetica",
        leading=13,
    )
    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontSize=7.5,
        textColor=colors.HexColor("#57606a"),
        fontName="Helvetica",
        leading=11,
    )

    # -----------------------------------------------------------------------
    # HEADER
    # -----------------------------------------------------------------------
    header_data = [[
        Paragraph(f"<b>CoverMap</b>", title_style),
        Paragraph(
            f"<b>{cca_name}</b><br/>{cca_contact}",
            ParagraphStyle("Right", parent=body_style, alignment=TA_RIGHT)
        ),
    ]]
    header_table = Table(header_data, colWidths=[4*inch, 3*inch])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=2,
                            color=BLUE_ACCENT, spaceAfter=2))
    story.append(Paragraph("Field Cover Crop Assessment", subtitle_style))
    story.append(Spacer(1, 4))

    # Field info row
    ndvi_date_str = ""
    if ndvi_date_from and ndvi_date_to:
        ndvi_date_str = f"NDVI: {ndvi_date_from} – {ndvi_date_to}"
    elif ndvi_date_to:
        ndvi_date_str = f"NDVI collected: {ndvi_date_to}"

    _soil_display = "Not available"
    if soil_series and soil_series not in ("Not available", "Unknown"):
        _soil_display = (
            f"{soil_series} — K-factor: {soil_k_factor}"
            if soil_k_factor and soil_k_factor != "N/A"
            else soil_series
        )

    _prev_crop_display = (
        previous_crop if previous_crop
        else ("Not recorded"
              if not residue_system or "Unknown" in residue_system
              else "See previous crop / tillage")
    )

    field_data = [
        [
            Paragraph(f"<b>Field:</b> {field_name}", body_style),
            Paragraph(f"<b>Farm:</b> {farm_name}", body_style),
            Paragraph(f"<b>County:</b> {county}", body_style),
            Paragraph(f"<b>Report Date:</b> {report_date}", body_style),
        ],
        [
            Paragraph(f"<b>Previous crop:</b> {_prev_crop_display}", body_style),
            Paragraph(f"<b>Termination date:</b> {termination_date or '⏳ Pending — document at termination'}", body_style),
            Paragraph(f"<b>Dominant soil series:</b> {_soil_display}", body_style),
            Paragraph(f"<b>Previous crop / tillage:</b> {residue_system or 'Not recorded'}", body_style),
        ],
    ]
    field_table = Table(field_data, colWidths=[1.75*inch]*4)
    field_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_GRAY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("BOX",           (0, 0), (-1, -1), 0.5, MID_GRAY),
        ("LINEBELOW",     (0, 0), (-1, 0),  0.3, MID_GRAY),
    ]))
    story.append(field_table)

    if ndvi_date_str:
        story.append(Paragraph(
            f"<i>{ndvi_date_str} &nbsp;|&nbsp; DEM: {dem_source}</i>",
            small_style
        ))
    story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # RECENT CROP ROTATION (USDA CDL) — after field-setup metadata
    # -----------------------------------------------------------------------
    if cdl_rotation:
        story.append(Paragraph("Recent Crop Rotation (USDA CDL)", section_style))

        rot_rows: List[List[Any]] = [["Year", "Label", "Dominant Class", "Confidence"]]
        _rot_flagged = False
        for r in cdl_rotation:
            if r.get("boundary_warning"):
                _rot_flagged = True
            _conf = (
                f"{r['dominant_pct']:.0f}% pixel share"
                if r.get("dominant_pct") is not None else "—"
            )
            rot_rows.append([
                str(r.get("year", "")),
                Paragraph(r.get("label") or "—", body_style),
                r.get("dominant_class") or "—",
                _conf,
            ])

        rot_table = Table(
            rot_rows,
            colWidths=[0.7 * inch, 3.1 * inch, 1.7 * inch, 1.5 * inch],
        )
        rot_table.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0),  BLUE_ACCENT),
            ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",       (0, 0), (-1, -1), 8.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GRAY, colors.white]),
            ("ALIGN",          (0, 0), (0, -1),  "CENTER"),
            ("ALIGN",          (3, 0), (3, -1),  "CENTER"),
            ("GRID",           (0, 0), (-1, -1), 0.3, MID_GRAY),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
            ("TOPPADDING",     (0, 0), (-1, -1), 4),
            ("LEFTPADDING",    (0, 0), (-1, -1), 6),
            ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(rot_table)
        if _rot_flagged:
            story.append(Paragraph(
                "<i>⚠ Non-agricultural pixels exceed 10% of the field for at "
                "least one year — verify the field boundary excludes roads, "
                "waterways, and building sites.</i>",
                ParagraphStyle("RotFlag", parent=small_style,
                               textColor=colors.HexColor("#92400e")),
            ))
        story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # PAGE 1 — MAPS
    # -----------------------------------------------------------------------
    story.append(Paragraph("Field Risk Maps", section_style))

    map_label_style = ParagraphStyle(
        "MapLabel", parent=body_style,
        alignment=TA_CENTER, fontSize=8, fontName="Helvetica-Bold"
    )
    _marginal_upper = ndvi_threshold + 0.15

    _rows, _cols = ndvi_array.shape
    _aspect = _cols / max(_rows, 1)

    _risk_pdf_w = 7.0 * inch
    _risk_pdf_h = min(_risk_pdf_w / _aspect, 3.8 * inch)
    if risk_zone_array is not None:
        risk_png = generate_risk_zone_map_image(
            risk_zone_array, array_shape=risk_zone_array.shape)
        risk_img = RLImage(io.BytesIO(risk_png), width=_risk_pdf_w, height=_risk_pdf_h)
        story.append(risk_img)
        story.append(Paragraph(
            "Erosion Risk Index Zones (C×LS) — pixel-level RUSLE risk classification",
            map_label_style,
        ))
        story.append(Spacer(1, 8))

    _map_pdf_w = 3.4 * inch
    _map_pdf_h = min(_map_pdf_w / _aspect, 2.8 * inch)
    ndvi_png  = generate_zone_map_image(ndvi_array, ndvi_threshold,
                                        array_shape=ndvi_array.shape)
    slope_png = generate_slope_map_image(slope_array,
                                         array_shape=slope_array.shape)
    map_w = _map_pdf_w
    map_h = _map_pdf_h
    ndvi_img  = RLImage(io.BytesIO(ndvi_png),  width=map_w, height=map_h)
    slope_img = RLImage(io.BytesIO(slope_png), width=map_w, height=map_h)

    maps_table = Table(
        [[ndvi_img, slope_img]],
        colWidths=[map_w + 0.1 * inch, map_w + 0.1 * inch],
    )
    maps_table.setStyle(TableStyle([
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(maps_table)

    labels_data = [[
        Paragraph(
            f"NDVI Cover Quality — Low (<{ndvi_threshold:.2f}) / Marginal / Good (>{_marginal_upper:.2f})",
            map_label_style,
        ),
        Paragraph("Terrain Slope (% gradient) — Flat / Moderate / Steep", map_label_style),
    ]]
    labels_table = Table(
        labels_data,
        colWidths=[map_w + 0.1 * inch, map_w + 0.1 * inch],
    )
    labels_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(labels_table)

    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # PAGE 2
    # -----------------------------------------------------------------------

    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=MID_GRAY, spaceAfter=4))
    story.append(Paragraph("CoverMap Advisory & Recommendation", section_style))

    concern     = risk_result.get("concern_level", "N/A")
    concern_col = CONCERN_BADGE_COLOR.get(concern, TEXT_DARK)

    concern_badge_style = ParagraphStyle(
        "ConcernBadge", parent=body_style,
        fontSize=10, fontName="Helvetica-Bold", textColor=concern_col,
    )
    story.append(Paragraph(f"Erosion Concern: {concern}", concern_badge_style))

    rec_text = risk_result.get("recommendation", "No recommendation available.")
    rec_bg   = {
        "Low":      colors.HexColor("#dcfce7"),
        "Moderate": colors.HexColor("#fef9c3"),
        "High":     colors.HexColor("#fee2e2"),
        "Critical": colors.HexColor("#fecaca"),
    }.get(concern, LIGHT_GRAY)

    rec_data = [[Paragraph(rec_text, body_style)]]
    rec_table = Table(rec_data, colWidths=[7 * inch])
    rec_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), rec_bg),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("BOX",           (0, 0), (-1, -1), 0.5, MID_GRAY),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 8))

    # --- NDVI Zone Summary ---
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=MID_GRAY, spaceAfter=4))
    story.append(Paragraph("Cover Crop Stand — NDVI Zone Summary", section_style))

    zone_acres  = calculate_zone_acres(ndvi_array, ndvi_threshold, acres_per_pixel=acres_per_pixel)
    total_acres = zone_acres.get("Total", 1)

    ndvi_low_label  = f"Low Cover  (NDVI < {ndvi_threshold:.2f})"
    ndvi_mid_label  = (
        f"Marginal  (NDVI {ndvi_threshold:.2f}"
        f"–{_marginal_upper:.2f})")
    ndvi_good_label = f"Good Cover  (NDVI > {_marginal_upper:.2f})"

    ndvi_zone_rows = [["Zone", "Acres", "% of Field"]]
    ndvi_zone_bg   = []
    for i, (zone_key, label, bg) in enumerate([
        ("Low cover",  ndvi_low_label,  colors.HexColor("#FEE8D5")),
        ("Marginal",   ndvi_mid_label,  colors.HexColor("#E0F2FE")),
        ("Good cover", ndvi_good_label, colors.HexColor("#FEF9C3")),
    ], start=1):
        acres = zone_acres.get(zone_key, 0)
        pct   = acres / total_acres * 100 if total_acres > 0 else 0
        ndvi_zone_rows.append([label, f"{acres:.1f}", f"{pct:.0f}%"])
        ndvi_zone_bg.append(("BACKGROUND", (0, i), (-1, i), bg))
    ndvi_zone_rows.append(["Total", f"{total_acres:.1f}", "100%"])

    ndvi_zone_table = Table(ndvi_zone_rows, colWidths=[3.2 * inch, 1.0 * inch, 1.0 * inch])
    ndvi_zone_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  BLUE_ACCENT),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("BACKGROUND",    (0, 4), (-1, 4),  LIGHT_GRAY),
        ("FONTNAME",      (0, 4), (-1, 4),  "Helvetica-Bold"),
        ("ALIGN",         (1, 0), (2, -1),  "CENTER"),
        ("GRID",          (0, 0), (-1, -1), 0.3, MID_GRAY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ] + ndvi_zone_bg))
    story.append(ndvi_zone_table)
    story.append(Spacer(1, 8))

    # --- Risk Index Zone Summary ---
    if zone_counts and sum(zone_counts.values()) > 0:
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=MID_GRAY, spaceAfter=4))
        story.append(Paragraph("Erosion Risk Zone Summary (C×LS)", section_style))

        px_area_acres = acres_per_pixel
        total_px = sum(zone_counts.values())

        ri_config = [
            (4, "Critical Risk", "#EF4444", "#fecaca", "> 1.5"),
            (3, "High Risk",     "#F97316", "#FEE8D5", "0.7–1.5"),
            (2, "Moderate Risk", "#FACC15", "#FEF9C3", "0.3–0.7"),
            (1, "Low Risk",      "#22C55E", "#dcfce7", "< 0.3"),
        ]
        ri_rows = [["Zone", "C×LS Range", "Acres", "% of Field"]]
        for val, label, _, bg, thresh in ri_config:
            count = zone_counts.get(val, 0)
            acres = count * px_area_acres
            pct   = count / total_px * 100 if total_px > 0 else 0
            ri_rows.append([label, thresh, f"{acres:.1f}", f"{pct:.0f}%"])

        ri_style = TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  BLUE_ACCENT),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
            ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
            ("GRID",          (0, 0), (-1, -1), 0.3, MID_GRAY),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ])
        bg_map = {4: "#fecaca", 3: "#FEE8D5", 2: "#FEF9C3", 1: "#dcfce7"}
        for idx, (val, _, _, _, _) in enumerate(ri_config, start=1):
            ri_style.add(
                "BACKGROUND",
                (0, idx), (-1, idx),
                colors.HexColor(bg_map[val]),
            )

        ri_table = Table(
            ri_rows,
            colWidths=[1.8 * inch, 1.2 * inch, 1.0 * inch, 1.0 * inch],
        )
        ri_table.setStyle(ri_style)
        story.append(ri_table)
        story.append(Paragraph(
            "<i>Risk Index = C-factor (NDVI) × "
            "LS-factor (slope). Boundary-masked field pixels only. "
            "Critical >1.5 · High 0.7–1.5 "
            "· Moderate 0.3–0.7 · Low <0.3</i>",
            small_style,
        ))
        story.append(Spacer(1, 6))

    # --- Amber disclaimer ---
    _img_date_label = ndvi_scene_date or ndvi_date_to or "unknown"
    disclaimer_text = (
        f"NDVI imagery dated {_img_date_label}. Field conditions may have changed since "
        f"image capture. This report documents satellite-observed conditions only."
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=body_style,
        fontSize=8, textColor=colors.HexColor("#92400e"),
    )
    disc_data = [[Paragraph(disclaimer_text, disclaimer_style)]]
    disc_table = Table(disc_data, colWidths=[7 * inch])
    disc_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#fef3c7")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("BOX",           (0, 0), (-1, -1), 0.5, colors.HexColor("#f59e0b")),
    ]))
    story.append(disc_table)
    story.append(Spacer(1, 8))

    # -----------------------------------------------------------------------
    # COVER CROP STAND ASSESSMENT (simplified — no CCA initials column)
    # -----------------------------------------------------------------------
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=MID_GRAY, spaceAfter=4))
    story.append(Paragraph("Cover Crop Stand Assessment — Satellite Documentation", section_style))

    ndvi_mean_val  = ndvi_stats.get("mean", 0.0)
    biomass_kgha   = max(0.0, (ndvi_mean_val - 0.10) / 0.40 * 3500)
    biomass_lbac   = biomass_kgha * 0.891
    biomass_low    = max(0, round(biomass_lbac * 0.6 / 50) * 50)
    biomass_high   = round(biomass_lbac * 1.4 / 50) * 50
    valid_px       = ndvi_array[~np.isnan(ndvi_array)]
    pct_above_020  = (np.sum(valid_px > 0.20) / valid_px.size * 100) if valid_px.size > 0 else 0.0
    image_date_str = ndvi_scene_date if ndvi_scene_date else (ndvi_date_to if ndvi_date_to else "Upload date unknown")

    cover_status = (
        f"✅ NDVI {ndvi_mean_val:.3f} — cover crop confirmed"
        if ndvi_mean_val > 0.20 else
        f"⚠️ NDVI {ndvi_mean_val:.3f} — inadequate cover"
    )
    ground_cover_status = (
        "✅ Estimated adequate cover zones based on NDVI threshold — field verification recommended"
        if pct_above_020 > 50 else
        "⚠️ Estimated adequate cover zones below 50% of field — field verification recommended"
    )

    prod_eqip_data = [
        ["Requirement", "Data Source", "Status"],
        ["Cover crop present", "Sentinel-2 NDVI > 0.20",  cover_status],
        ["Image date",         "GEE metadata",             image_date_str],
        ["Estimated biomass",  "NDVI proxy",               f"~{biomass_low}–{biomass_high} lb/acre (±40% NDVI proxy)"],
        ["30% ground cover",   "NDVI threshold",           ground_cover_status],
    ]

    prod_eqip_col_w = [1.8 * inch, 1.8 * inch, 3.2 * inch]
    prod_eqip_table = Table(
        [[Paragraph(str(cell), body_style) for cell in row] for row in prod_eqip_data],
        colWidths=prod_eqip_col_w,
    )
    prod_eqip_table.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  BLUE_ACCENT),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 8.0),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GRAY, colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.3, MID_GRAY),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(prod_eqip_table)
    story.append(Paragraph(
        "<i>Satellite-verified cover crop status. "
        "Field verification recommended before termination.</i>",
        small_style,
    ))
    story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # YEAR-OVER-YEAR EARLY-SEASON NDVI (report-only chart)
    # -----------------------------------------------------------------------
    if yoy_rows:
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=MID_GRAY, spaceAfter=4))
        story.append(Paragraph(
            "Year-over-Year Early-Season NDVI (March–April)", section_style))
        yoy_png = generate_yoy_ndvi_chart_image(yoy_rows)
        story.append(RLImage(io.BytesIO(yoy_png),
                             width=5.5 * inch, height=2.4 * inch))
        story.append(Paragraph(
            "<i>Mean field NDVI for the March 15 – April 20 early-season window "
            "of each year, Sentinel-2 via Google Earth Engine.</i>",
            small_style,
        ))
        story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # FIELD LEVEL RESULTS
    # -----------------------------------------------------------------------
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=MID_GRAY, spaceAfter=4))
    story.append(Paragraph("Field Level Results", section_style))

    _c_adj     = risk_result.get("c_factor", 0)
    _c_base    = risk_result.get("c_factor_baseline", _c_adj)
    _c_pct     = int((_c_base - _c_adj) / _c_base * 100) if _c_base > 0 else 0
    _c_display = f"{_c_adj:.3f} ({_c_pct}% reduction vs. baseline)"
    _c_label   = "C-Factor (exp. model)"

    # QC signals (Signal 1 valid-pixel tier) via the shared helper so this
    # field-metrics table matches the app top box and the 45Z report exactly.
    _qc_prod = qc_signals(ndvi_array, scene_count=scene_count,
                          mean_ndvi=ndvi_stats.get("mean", 0.0))

    metrics = [
        ["Metric", "Value"],
        ["NDVI Mean",         f"{ndvi_stats.get('mean', 0):.3f}"],
        ["NDVI Range",        f"{ndvi_stats.get('min', 0):.3f} – {ndvi_stats.get('max', 0):.3f}"],
        ["Slope Mean (%)",    f"{slope_stats.get('mean', 0):.1f}%"],
        [_c_label,            _c_display],
        ["Risk Index (C×LS)", f"{risk_result.get('rusle_score', 0):.3f}"],
        ["Erosion Concern",   concern],
        ["Valid Pixels (QC)",
         f"{_qc_prod['valid_pct']:.0f}% — {_qc_prod['valid_phrase']}"],
    ]

    met_style = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  BLUE_ACCENT),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT_GRAY, colors.white]),
        ("ALIGN",         (1, 0), (1, -1),  "CENTER"),
        ("GRID",          (0, 0), (-1, -1), 0.3, MID_GRAY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("TEXTCOLOR",     (1, 6), (1, 6),   concern_col),
        ("FONTNAME",      (1, 6), (1, 6),   "Helvetica-Bold"),
    ])
    met_table = Table(metrics, colWidths=[1.8 * inch, 1.8 * inch])
    met_table.setStyle(met_style)
    story.append(met_table)
    # QC Signals 2 (single-scene) and 3 (saturation) — conditional advisory lines.
    _qc_amber_p = ParagraphStyle("QCAmberP", parent=small_style,
                                 textColor=colors.HexColor("#92400e"))
    for _qc_line in (_qc_prod["single_scene"], _qc_prod["saturation"]):
        if _qc_line:
            story.append(Paragraph(f"<i>QC: {_qc_line}</i>", _qc_amber_p))
    story.append(Spacer(1, 8))

    # --- Cover Crop Erosion Reduction ---
    _zes_p = risk_result.get("zone_erosion_summary", [])
    if _zes_p:
        # 0.0 savings is a real value, not "unavailable" — see CCA-report note.
        _saved_vals_p = [z["a_saved_zone"] * z["area_fraction"]
                         for z in _zes_p if z.get("a_saved_zone") is not None]
        _base_vals_p  = [z["a_baseline_zone"] * z["area_fraction"]
                         for z in _zes_p if z.get("a_baseline_zone") is not None]
        _a_saved_weighted_p    = sum(_saved_vals_p) if _saved_vals_p else None
        _a_baseline_weighted_p = sum(_base_vals_p)  if _base_vals_p  else None
        _pct_reduction_weighted_p = (
            (_a_saved_weighted_p / _a_baseline_weighted_p) * 100
            if (_a_baseline_weighted_p and _a_saved_weighted_p is not None)
            else None
        )
        if _pct_reduction_weighted_p is not None:
            # Producer report shows the % reduction only — absolute A values
            # (t/ac/yr baseline/saved) are CCA/45Z-tier detail.
            cc_red_rows_p = [
                ["Metric", "Value"],
                [
                    "Est. Cover Crop Erosion Reduction",
                    f"{_pct_reduction_weighted_p:.1f}%",
                ],
            ]
            cc_red_table_p = Table(
                cc_red_rows_p,
                colWidths=[2.5 * inch, 2.0 * inch],
            )
            cc_red_table_p.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0),  BLUE_ACCENT),
                ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
                ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT_GRAY, colors.white]),
                ("FONTNAME",      (0, 1), (0, -1),  "Helvetica-Bold"),
                ("ALIGN",         (1, 0), (1, -1),  "CENTER"),
                ("GRID",          (0, 0), (-1, -1), 0.3, MID_GRAY),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING",    (0, 0), (-1, -1), 4),
                ("LEFTPADDING",   (0, 0), (-1, -1), 6),
                # Headline the % erosion reduction (row 1) as the primary metric.
                ("BACKGROUND",    (0, 1), (-1, 1),  colors.HexColor("#dcfce7")),
                ("FONTSIZE",      (1, 1), (1, 1),   14),
                ("FONTNAME",      (1, 1), (1, 1),   "Helvetica-Bold"),
                ("TEXTCOLOR",     (1, 1), (1, 1),   GREEN_BADGE),
                ("TOPPADDING",    (0, 1), (-1, 1),  6),
                ("BOTTOMPADDING", (0, 1), (-1, 1),  6),
            ]))
            story.append(cc_red_table_p)
            story.append(Paragraph(
                "<i>Estimates based on RUSLE C-factor methodology. C-factor derived from piecewise "
                "exponential NDVI model. ±10 pt uncertainty on reduction percentage.</i>",
                small_style,
            ))
            story.append(Spacer(1, 8))

    # -----------------------------------------------------------------------
    # FOOTER
    # -----------------------------------------------------------------------
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=MID_GRAY, spaceAfter=3))

    footer_lines = [
        f"NDVI Source: Sentinel-2 via Google Earth Engine ({ndvi_date_str}) | "
        f"DEM: {dem_source} | Slope: computed in UTM meters (EPSG:26915)",
        f"Data provenance — DEM source: {dem_source}  |  "
        f"R-factor source: {r_factor_note or 'R=%.0f' % r_factor}",
        "C-Factor methodology: piecewise exponential NDVI model — "
        "continuous C-factor differentiated by residue system. "
        "This report is advisory only and does not constitute an official NRCS determination.",
        f"CoverMap Field Report · {cca_name} · Sentinel-2 via Google Earth Engine · Iowa RUSLE C-factor calibration · {report_date}",
    ]
    if yoy_rows:
        footer_lines.insert(-1, (
            "NDVI window: March 15 – April 20. Chosen to capture rye "
            "biomass-accumulation phase between dormancy and typical "
            "pre-termination."
        ))
    if cdl_rotation:
        footer_lines.insert(-1, (
            "Crop rotation derived from USDA NASS Cropland Data Layer (30 m "
            "resolution). Corn/soybean accuracy in Iowa historically 95–98% "
            "(NASS published accuracy assessments). Not CoverMap-validated."
        ))
    for line in footer_lines:
        story.append(Paragraph(line, small_style))

    # Build PDF
    doc.build(story)
    buf.seek(0)
    return buf.read()


# ===========================================================================
# 45Z COVER CROP VERIFICATION PACKAGE
# ---------------------------------------------------------------------------
# Third report type (alongside the CCA and producer reports above, which are
# FROZEN — this section does not touch them). Verifier-facing audit workpaper
# implementing docs/45z_package_layout_v1.md against 7 CFR § 2100.052(c).
#
# Layout (3 physical pages; the wireframe's "Page 4" is the repeating footer):
#   Page 1 — § 2100.052(c) establishment evidence (callout + evidence table +
#            NDVI zone map + zone summary + seeding-gap disclosure)
#   Page 2 — Supplemental CCA field advisory, walled off from (c) evidence
#            (risk-index map + risk zone summary + field metrics + slope map)
#   Page 3 — § 2100.052(b) producer records as fillable PDF (AcroForm) fields,
#            producer attestation + signature line, regulatory-basis footer
#
# Rendering only — no scoring logic. Reuses the shared map/acreage helpers and
# color constants above. Producer § 2100.052(b) fields are AcroForm widgets so
# a producer can type or print-and-hand-write them; empty producer_inputs -> a
# blank fillable form, populated producer_inputs -> pre-filled values.
# ===========================================================================

# Producer records rendered on Page 3, sourced from the 45Z crosswalk Section 5
# gap list. Method options match the wireframe's Page 3 checkbox groups.
_45Z_SEEDING_METHODS     = ["Drilled", "Broadcast", "Aerial", "Interseeded"]
_45Z_TERMINATION_METHODS = ["Winter kill", "Herbicide", "Roller crimper", "Mowing"]


class _ProducerRecordsForm(Flowable):
    """Raw-canvas Page 3 fillable § 2100.052(b) producer-records block.

    Draws real PDF AcroForm widgets (text fields + checkboxes) via the live
    canvas so the fields are typeable in any PDF viewer and print cleanly for
    hand entry. ``producer_inputs`` pre-fills values; an empty/omitted dict
    yields a blank form. Field names are unique within the document.
    """

    _ROW_H = 0.62 * inch   # vertical space per record row

    def __init__(self, width, producer_inputs=None):
        super().__init__()
        self.width  = width
        self.pi     = producer_inputs or {}
        # 7 record rows + a little breathing room top/bottom.
        self.height = self._ROW_H * 7 + 0.15 * inch

    # -- small drawing helpers (operate in the flowable's local coord space) --
    def _checkbox(self, name, x, y, checked, label):
        # relative=True -> coords honor the Platypus flowable's canvas transform
        # (acroForm widgets are otherwise placed in absolute page coordinates).
        size = 10
        self.canv.acroForm.checkbox(
            name=name, x=x, y=y - size + 2, size=size, relative=True,
            checked=bool(checked), buttonStyle="check",
            borderWidth=0.75, borderColor=MID_GRAY, fillColor=colors.white,
            textColor=TEXT_DARK, forceBorder=True,
        )
        self.canv.setFont("Helvetica", 8)
        self.canv.setFillColor(TEXT_DARK)
        self.canv.drawString(x + size + 3, y - size + 4, label)
        return x + size + 5 + self.canv.stringWidth(label, "Helvetica", 8) + 12

    def _textfield(self, name, x, y, w, value, h=13):
        self.canv.acroForm.textfield(
            name=name, x=x, y=y - h + 1, width=w, height=h, relative=True,
            value=value or "", fontSize=8, fontName="Helvetica",
            borderWidth=0.75, borderColor=MID_GRAY,
            fillColor=colors.HexColor("#fbfcfd"), textColor=TEXT_DARK,
            borderStyle="underlined", forceBorder=True,
        )

    def _cite(self, cite, label, y):
        self.canv.setFont("Helvetica-Bold", 8)
        self.canv.setFillColor(colors.HexColor("#57606a"))
        self.canv.drawString(0, y - 9, cite)
        self.canv.setFont("Helvetica", 8.5)
        self.canv.setFillColor(TEXT_DARK)
        self.canv.drawString(0.62 * inch, y - 9, label)

    def draw(self):
        c  = self.canv
        pi = self.pi
        top = self.height
        row = self._ROW_H
        entry_x = 3.55 * inch   # left edge of the producer-entry column

        # Faint top rule.
        c.setStrokeColor(MID_GRAY)
        c.setLineWidth(0.4)
        c.line(0, top, self.width, top)

        y = top - 0.14 * inch

        # (b)(1) — seed purchase/receipt on file
        self._cite("(b)(1)", "Seed purchase / receipt on file", y)
        _rx = self._checkbox("b1_receipt_yes", entry_x, y,
                             pi.get("seed_receipt_on_file") == "Yes", "Yes")
        _rx = self._checkbox("b1_receipt_no", _rx, y,
                             pi.get("seed_receipt_on_file") == "No", "No")
        c.setFont("Helvetica", 8)
        c.setFillColor(TEXT_DARK)
        c.drawString(_rx, y - 9, "retention date:")
        self._textfield("b1_retention_date",
                        _rx + c.stringWidth("retention date:", "Helvetica", 8) + 4,
                        y, 0.95 * inch, pi.get("seed_retention_date"))
        y -= row

        # (b)(3) — seeding date
        self._cite("(b)(3)", "Cover crop seeding date", y)
        self._textfield("b3_seeding_date", entry_x, y, 1.6 * inch,
                        pi.get("seeding_date"))
        y -= row

        # (b)(3) — seeding method
        self._cite("(b)(3)", "Seeding method", y)
        _mx = entry_x
        _sel_method = pi.get("seeding_method")
        for _m in _45Z_SEEDING_METHODS:
            _mx = self._checkbox(f"b3_method_{_m.lower()}", _mx, y,
                                 _sel_method == _m, _m)
        y -= row

        # (b)(3) — seeding rate
        self._cite("(b)(3)", "Seeding rate (lb/ac)", y)
        self._textfield("b3_seeding_rate", entry_x, y, 1.2 * inch,
                        pi.get("seeding_rate"))
        y -= row

        # (b)(6) — termination date
        self._cite("(b)(6)", "Cover crop termination date", y)
        self._textfield("b6_termination_date", entry_x, y, 1.6 * inch,
                        pi.get("termination_date"))
        y -= row

        # (b)(6) — termination method
        self._cite("(b)(6)", "Termination method", y)
        _tx = entry_x
        _sel_term = pi.get("termination_method")
        for _t in _45Z_TERMINATION_METHODS:
            _tx = self._checkbox(f"b6_term_{_t.split()[0].lower()}", _tx, y,
                                 _sel_term == _t, _t)
        y -= row

        # (b)(8) — grazing prior to termination
        self._cite("(b)(8)", "Grazing occurred prior to termination", y)
        _gx = self._checkbox("b8_grazing_no", entry_x, y,
                             pi.get("grazing_occurred") == "No", "No")
        _gx = self._checkbox("b8_grazing_yes", _gx, y,
                             pi.get("grazing_occurred") == "Yes",
                             "Yes — georeferenced photos attached")
        y -= row

        # Row separators.
        c.setStrokeColor(colors.HexColor("#e5e9ee"))
        c.setLineWidth(0.3)
        for _i in range(1, 7):
            _yy = top - 0.14 * inch - row * _i + row - 0.02 * inch
            c.line(0, _yy, self.width, _yy)


def generate_45z_verification_report(
    # Field info
    field_name: str,
    farm_name: str,
    county: str,
    # Data
    ndvi_array: np.ndarray,
    slope_array: np.ndarray,
    ndvi_stats: Dict[str, float],
    slope_stats: Dict[str, float],
    risk_result: Dict[str, Any],
    zone_summary: Any,
    risk_zone_array: Optional[np.ndarray] = None,
    zone_counts: Optional[Dict[str, int]] = None,
    # Settings
    ndvi_threshold: float = 0.20,
    slope_threshold: float = 6.0,
    # Dates
    ndvi_date_from: Optional[str] = None,
    ndvi_date_to: Optional[str] = None,
    ndvi_scene_date: Optional[str] = None,
    report_date: Optional[str] = None,
    dem_source: str = "Iowa 3-meter Digital Elevation Model (Iowa DNR)",
    # CCA info
    cca_name: str = "Stephen Zimmerman, CCA MS",
    cca_contact: str = "Ankeny, IA | Ag Research Scientist",
    # Optional field detail
    termination_date: Optional[str] = None,
    previous_crop: Optional[str] = None,
    soil_series: Optional[str] = None,
    soil_k_factor: Optional[str] = None,
    residue_system: Optional[str] = None,
    soil_loss_result: Optional[Dict[str, Any]] = None,
    r_factor: float = 150.0,
    r_factor_note: Optional[str] = None,
    acres_per_pixel: float = (10.0 ** 2) / 4046.86,
    # ---- 45Z-specific ----
    producer_inputs: Optional[Dict[str, Any]] = None,
    management_unit_id: Optional[str] = None,
    boundary_source: Optional[str] = None,
    scene_count: Optional[int] = None,
    ndvi_scene_from: Optional[str] = None,
    ndvi_scene_to: Optional[str] = None,
    valid_pixel_fraction: Optional[float] = None,
    report_id: Optional[str] = None,
) -> bytes:
    """Generate the 45Z Cover Crop Verification Package PDF (verifier-facing).

    Three physical pages per docs/45z_package_layout_v1.md. Rendering only —
    reuses the same data structures the CCA/producer reports consume, plus
    45Z-specific fields. ``producer_inputs`` (see _ProducerRecordsForm) pre-
    fills the Page 3 § 2100.052(b) fillable form; an empty/omitted dict yields
    a blank form. Returns PDF bytes ready for st.download_button.
    """
    if report_date is None:
        report_date = datetime.now().strftime("%B %d, %Y")

    # Report ID: [FarmName]-[FieldName], no timestamp (locked decision #3).
    if report_id is None:
        _fn = (farm_name or "").strip() or "Farm"
        _fd = (field_name or "").strip() or "Field"
        report_id = f"{_fn}-{_fd}"

    # QC signals via the shared helper so the 45Z evidence page matches the app
    # top box and the CCA/producer reports exactly (same formula + phrasing).
    # Signal 1 three-tier phrase, Signal 2 single-scene, Signal 3 saturation.
    # Rendering-only — no gate is enforced here (Signal 4 is blocked upstream).
    _ndvi_mean = ndvi_stats.get("mean", 0.0)
    if scene_count is None:
        scene_count = risk_result.get("scene_count")
    _qc45 = qc_signals(ndvi_array, scene_count=scene_count, mean_ndvi=_ndvi_mean)
    if valid_pixel_fraction is None:
        valid_pixel_fraction = _qc45["valid_pct"]
    _vpf_tier, _vpf_phrase = valid_tier(valid_pixel_fraction)
    _nonnan = ndvi_array[~np.isnan(ndvi_array)]

    # Establishment determination — reuse the existing "cover crop confirmed"
    # gate (field NDVI mean vs. threshold). No new scoring logic.
    _established = _ndvi_mean >= ndvi_threshold

    # Establishment pixel fraction (>= 0.20) — the defensible headline metric,
    # computed the same way the existing reports compute pct_above_020.
    _pct_ge_thresh = (
        float(np.sum(_nonnan >= ndvi_threshold)) / _nonnan.size * 100.0
        if _nonnan.size > 0 else 0.0
    )
    _zone_acres_45  = calculate_zone_acres(ndvi_array, ndvi_threshold,
                                           acres_per_pixel=acres_per_pixel)
    _total_acres_45 = _zone_acres_45.get("Total", 0.0)
    _est_acres      = _pct_ge_thresh / 100.0 * _total_acres_45

    # Scene metadata for the evidence table.
    _scene_range = (
        f"{ndvi_scene_from} – {ndvi_scene_to}"
        if ndvi_scene_from and ndvi_scene_to
        else (ndvi_scene_date or ndvi_scene_to or "—")
    )
    _window_range = (
        f"{ndvi_date_from} – {ndvi_date_to}"
        if ndvi_date_from and ndvi_date_to else (ndvi_date_to or "—")
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.4 * inch,
        bottomMargin=0.4 * inch,
        title=f"45Z Verification Package — {report_id}",
    )

    styles = getSampleStyleSheet()
    story  = []

    # --- Styles ---
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Normal"], fontSize=17,
        textColor=TEXT_DARK, fontName="Helvetica-Bold", spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], fontSize=10,
        textColor=colors.HexColor("#57606a"), fontName="Helvetica",
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Normal"], fontSize=11, textColor=TEXT_DARK,
        fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=9, textColor=TEXT_DARK,
        fontName="Helvetica", leading=13,
    )
    small_style = ParagraphStyle(
        "Small", parent=styles["Normal"], fontSize=7.5,
        textColor=colors.HexColor("#57606a"), fontName="Helvetica", leading=11,
    )
    wall_style = ParagraphStyle(
        "Wall", parent=small_style, fontSize=8.5,
        textColor=colors.HexColor("#57606a"), fontName="Helvetica-Oblique",
        leading=12,
    )

    # -----------------------------------------------------------------------
    # PAGE 1 — HEADER (verifier audience)
    # -----------------------------------------------------------------------
    header_data = [[
        Paragraph("<b>CoverMap</b>", title_style),
        Paragraph(
            f"<b>{cca_name}</b><br/>{cca_contact}",
            ParagraphStyle("Right", parent=body_style, alignment=TA_RIGHT),
        ),
    ]]
    header_table = Table(header_data, colWidths=[4 * inch, 3 * inch])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=2,
                            color=BLUE_ACCENT, spaceAfter=2))
    story.append(Paragraph(
        "45Z Cover Crop Verification Package", subtitle_style))
    story.append(Paragraph(
        f"Report Date: {report_date} &nbsp;·&nbsp; Report ID: {report_id}",
        small_style,
    ))
    story.append(Spacer(1, 4))

    # Field info block
    field_info = [
        [
            Paragraph(f"<b>Field:</b> {field_name}", body_style),
            Paragraph(f"<b>Farm Producer:</b> {farm_name or '—'}", body_style),
            Paragraph(f"<b>County:</b> {county or '—'}", body_style),
        ],
        [
            Paragraph(
                f"<b>Field / management unit identifier:</b> "
                f"{management_unit_id or '— [producer-supplied, § 2100.031(e)(3)(i)]'}",
                body_style,
            ),
            Paragraph(f"<b>Field acreage:</b> {_total_acres_45:.1f} ac", body_style),
            Paragraph(
                f"<b>Boundary source:</b> {boundary_source or '—'}",
                body_style,
            ),
        ],
    ]
    field_table = Table(field_info, colWidths=[2.9 * inch, 2.1 * inch, 2.0 * inch])
    field_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_GRAY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("BOX",           (0, 0), (-1, -1), 0.5, MID_GRAY),
        ("LINEBELOW",     (0, 0), (-1, 0),  0.3, MID_GRAY),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(field_table)
    story.append(Spacer(1, 8))

    # -----------------------------------------------------------------------
    # PAGE 1 — VERIFICATION STATEMENT (headline callout box)
    # -----------------------------------------------------------------------
    _determination = "ESTABLISHMENT CONFIRMED" if _established else \
                     "ESTABLISHMENT: INSUFFICIENT EVIDENCE"
    _det_color = GREEN_BADGE if _established else RED_BADGE
    _callout_title = ParagraphStyle(
        "CalloutTitle", parent=body_style, fontSize=10.5,
        fontName="Helvetica-Bold", textColor=TEXT_DARK,
    )
    _callout_det = ParagraphStyle(
        "CalloutDet", parent=body_style, fontSize=12,
        fontName="Helvetica-Bold", textColor=_det_color,
    )
    _callout_body = ParagraphStyle(
        "CalloutBody", parent=body_style, fontSize=9, leading=13,
    )
    _vpf_line = (
        f"{valid_pixel_fraction:.0f}% — {_vpf_phrase}"
        if valid_pixel_fraction is not None else "—"
    )
    callout_rows = [
        [Paragraph("§ 2100.052(c) COVER CROP ESTABLISHMENT VERIFICATION",
                   _callout_title)],
        [Paragraph(
            "Method used: <b>Remote sensing data</b> "
            "(on-site visit and georeferenced photographs not used)",
            _callout_body)],
        [Paragraph(f"Determination: {_determination}", _callout_det)],
        [Paragraph(
            f"Basis: Sentinel-2 NDVI median composite, "
            f"{scene_count if scene_count else '—'} scenes, {_scene_range}",
            _callout_body)],
        [Paragraph(
            f"Field NDVI mean: <b>{_ndvi_mean:.3f}</b> &nbsp;·&nbsp; "
            f"Pixels ≥ {ndvi_threshold:.2f} threshold: "
            f"<b>{_pct_ge_thresh:.0f}%</b> ({_est_acres:.1f} of {_total_acres_45:.1f} ac)",
            _callout_body)],
        [Paragraph(
            f"Valid pixel fraction: <b>{_vpf_line}</b> "
            f"(≥ 75% required for report generation)",
            _callout_body)],
    ]
    # QC Signals 2 (single-scene) and 3 (saturation) — conditional, shown in the
    # § 2100.052(c) callout only when triggered (Bin Field triggers neither).
    if _qc45["single_scene"]:
        callout_rows.append([Paragraph(
            f"<b>Single-scene composite:</b> {_qc45['single_scene']}",
            _callout_body)])
    if _qc45["saturation"]:
        callout_rows.append([Paragraph(
            f"<b>Saturation:</b> {_qc45['saturation']}", _callout_body)])
    callout = Table(callout_rows, colWidths=[7 * inch])
    callout.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#f0f6ff")),
        ("BOX",           (0, 0), (-1, -1), 1.2, BLUE_ACCENT),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (0, 0),   8),
        ("TOPPADDING",    (0, 1), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -2), 2),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
        ("LINEBELOW",     (0, 0), (-1, 0),  0.4, colors.HexColor("#c5d9f5")),
    ]))
    story.append(callout)
    story.append(Paragraph(
        "<i>Binary determination. Numbers state the case. Not an NRCS "
        "determination — verifier-facing evidence under § 2100.052(c).</i>",
        small_style,
    ))
    story.append(Spacer(1, 8))

    # -----------------------------------------------------------------------
    # PAGE 1 — § 2100.052(c) ESTABLISHMENT EVIDENCE table
    # -----------------------------------------------------------------------
    story.append(Paragraph("§ 2100.052(c) Establishment Evidence", section_style))

    _c_adj  = risk_result.get("c_factor", 0)
    evidence_rows = [
        ["Element", "Value", "Source"],
        ["Compositing method", "Pixel-wise temporal median", "Tech Guide §2.1"],
        ["Compositing window", _window_range, "User-defined spring window"],
        ["Scenes composited", str(scene_count) if scene_count else "—", "GEE metadata"],
        ["Scene date range", _scene_range, "GEE metadata"],
        ["Cloud filter — scene", "<80% cloud cover", "Two-stage filter"],
        ["Cloud filter — pixel",
         "SCL 3, 8, 9, 10, 11 excluded; scenes >30% cloud/shadow excluded",
         "Two-stage filter"],
        ["Valid pixel definition",
         "NDVI > 0.05 (excludes water, cloud shadow, saturated bare soil)",
         "Tech Guide QC"],
        ["Valid pixel fraction",
         f"{_vpf_line} (≥ 75% required)",
         "Pipeline gate"],
        ["NDVI mean", f"{_ndvi_mean:.3f}", "Field-mean, boundary-masked"],
        ["NDVI range",
         f"{ndvi_stats.get('min', 0):.3f} – {ndvi_stats.get('max', 0):.3f}",
         "Pixel min/max"],
        [f"Pixels ≥ {ndvi_threshold:.2f} (establishment threshold)",
         f"{_pct_ge_thresh:.0f}%", "Zone summary"],
        ["Data platform", "Sentinel-2 L2A via Google Earth Engine", "Provenance"],
    ]
    _evi_cell = ParagraphStyle("EviCell", parent=body_style, fontSize=7.5,
                               leading=9)
    _evi_hdr  = ParagraphStyle("EviHdr", parent=_evi_cell,
                               fontName="Helvetica-Bold", textColor=colors.white)
    evidence_table = Table(
        [[Paragraph(str(c), _evi_hdr if _ri == 0 else _evi_cell) for c in row]
         for _ri, row in enumerate(evidence_rows)],
        colWidths=[2.0 * inch, 3.1 * inch, 1.9 * inch],
    )
    evidence_table.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  BLUE_ACCENT),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 7.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GRAY, colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.3, MID_GRAY),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 1.5),
        ("TOPPADDING",     (0, 0), (-1, -1), 1.5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(evidence_table)
    story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # PAGE 1 — NDVI COVER QUALITY zone map + zone summary
    # -----------------------------------------------------------------------
    story.append(Paragraph("NDVI Cover Quality", section_style))
    _rows_a, _cols_a = ndvi_array.shape
    _aspect_a = _cols_a / max(_rows_a, 1)
    _marginal_upper = ndvi_threshold + 0.15
    _ndvi_map_w = 3.1 * inch
    _ndvi_map_h = min(_ndvi_map_w / _aspect_a, 2.1 * inch)
    ndvi_png = generate_zone_map_image(ndvi_array, ndvi_threshold,
                                       array_shape=ndvi_array.shape)
    ndvi_map_img = RLImage(io.BytesIO(ndvi_png),
                           width=_ndvi_map_w, height=_ndvi_map_h)

    _zt_total = _zone_acres_45.get("Total", 1) or 1
    _zt_cell = ParagraphStyle("ZtCell", parent=body_style, fontSize=7.5,
                              leading=9)
    ndvi_zone_rows = [["Zone", "Acres", "% Fld"]]
    _zbg = []
    for _i, (_zk, _lbl, _bg) in enumerate([
        ("Good cover", f"Good Cover (>{_marginal_upper:.2f})",
         colors.HexColor("#FEF9C3")),
        ("Marginal",   f"Marginal ({ndvi_threshold:.2f}–{_marginal_upper:.2f})",
         colors.HexColor("#E0F2FE")),
        ("Low cover",  f"Low Cover (<{ndvi_threshold:.2f})",
         colors.HexColor("#FEE8D5")),
    ], start=1):
        _ac  = _zone_acres_45.get(_zk, 0)
        _pct = _ac / _zt_total * 100 if _zt_total > 0 else 0
        ndvi_zone_rows.append([_lbl, f"{_ac:.1f}", f"{_pct:.0f}%"])
        _zbg.append(("BACKGROUND", (0, _i), (-1, _i), _bg))
    ndvi_zone_rows.append(["Total", f"{_zt_total:.1f}", "100%"])
    ndvi_zone_table = Table(
        [[Paragraph(str(c), _zt_cell) for c in row] for row in ndvi_zone_rows],
        colWidths=[1.9 * inch, 0.7 * inch, 0.7 * inch])
    ndvi_zone_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  BLUE_ACCENT),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("BACKGROUND",    (0, 4), (-1, 4),  LIGHT_GRAY),
        ("FONTNAME",      (0, 4), (-1, 4),  "Helvetica-Bold"),
        ("ALIGN",         (1, 0), (2, -1),  "CENTER"),
        ("GRID",          (0, 0), (-1, -1), 0.3, MID_GRAY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ] + _zbg))
    # Map (left) and zone summary (right) side by side to conserve height.
    ndvi_combo = Table([[ndvi_map_img, ndvi_zone_table]],
                       colWidths=[_ndvi_map_w + 0.2 * inch, 3.6 * inch])
    ndvi_combo.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",        (0, 0), (0, 0),   "CENTER"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(ndvi_combo)
    story.append(Spacer(1, 4))

    # Seeding-gap disclosure (locked wording from the wireframe).
    _low_ac  = _zone_acres_45.get("Low cover", 0.0)
    story.append(Paragraph(
        f"<b>Note on the {(_low_ac / _zt_total * 100 if _zt_total else 0):.0f}% "
        f"low-cover fraction.</b> Verifier will see this; it is not hidden. "
        f"Low-cover pixels reflect an actual seeding gap (e.g., aerial applicator "
        f"coverage limitation), not a data artifact or masking issue. The "
        f"establishment threshold ({_pct_ge_thresh:.0f}% of field pixels ≥ "
        f"{ndvi_threshold:.2f} NDVI) supports § 2100.052(c) confirmation on "
        f"the {_est_acres:.1f} seeded acres. Producer may need to exclude the "
        f"{_low_ac:.1f} low-cover acres from claimed CI credit acreage in the "
        f"Biofuel Feedstock Report per § 2100.031(e).",
        small_style,
    ))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # PAGE 2 — SUPPLEMENTAL FIELD ADVISORY (walled off from (c) evidence)
    # -----------------------------------------------------------------------
    story.append(Paragraph("Supplemental Field Advisory", section_style))
    story.append(Paragraph(
        "The following section is CCA field advisory. It is <b>not</b> part of "
        "the § 2100.052(c) verification evidence on Page 1. RUSLE-derived "
        "erosion values are for relative ranking and % reduction only. See Tech "
        "Guide §7.7.",
        wall_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1.0,
                            color=colors.HexColor("#c5d9f5"), spaceBefore=4,
                            spaceAfter=6))

    # Risk Index + Terrain Slope maps, side by side (both retained per wireframe).
    _map_lbl = ParagraphStyle("MapLabel2", parent=body_style,
                              alignment=TA_CENTER, fontSize=8,
                              fontName="Helvetica-Bold")
    _adv_map_w = 3.35 * inch
    _adv_map_h = min(_adv_map_w / _aspect_a, 2.4 * inch)
    slope_png = generate_slope_map_image(slope_array, array_shape=slope_array.shape)
    slope_img = RLImage(io.BytesIO(slope_png), width=_adv_map_w, height=_adv_map_h)
    if risk_zone_array is not None:
        risk_png = generate_risk_zone_map_image(
            risk_zone_array, array_shape=risk_zone_array.shape)
        risk_img = RLImage(io.BytesIO(risk_png), width=_adv_map_w, height=_adv_map_h)
        adv_maps = Table(
            [[risk_img, slope_img],
             [Paragraph("Erosion Risk Index Zones (C×LS)", _map_lbl),
              Paragraph("Terrain Slope (% gradient)", _map_lbl)]],
            colWidths=[_adv_map_w + 0.1 * inch, _adv_map_w + 0.1 * inch],
        )
        adv_maps.setStyle(TableStyle([
            ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(adv_maps)
        story.append(Spacer(1, 6))
    else:
        story.append(slope_img)
        story.append(Paragraph("Terrain Slope (% gradient)", _map_lbl))
        story.append(Spacer(1, 6))

    # Risk zone summary
    if zone_counts and sum(zone_counts.values()) > 0:
        story.append(Paragraph("Erosion Risk Zone Summary (C×LS)", section_style))
        _total_px = sum(zone_counts.values())
        _ri_cfg = [
            (4, "Critical", "> 1.5",   "#fecaca"),
            (3, "High",     "0.7–1.5", "#FEE8D5"),
            (2, "Moderate", "0.3–0.7", "#FEF9C3"),
            (1, "Low",      "< 0.3",   "#dcfce7"),
        ]
        _ri_rows = [["Zone", "C×LS Range", "Acres", "% of Field"]]
        for _val, _lbl, _thr, _bg in _ri_cfg:
            _cnt = zone_counts.get(_val, 0)
            _ac  = _cnt * acres_per_pixel
            _pc  = _cnt / _total_px * 100 if _total_px > 0 else 0
            _ri_rows.append([_lbl, _thr, f"{_ac:.1f}", f"{_pc:.0f}%"])
        _ri_tbl = Table(_ri_rows,
                        colWidths=[1.8 * inch, 1.2 * inch, 1.0 * inch, 1.0 * inch])
        _ri_style = TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  BLUE_ACCENT),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
            ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
            ("GRID",          (0, 0), (-1, -1), 0.3, MID_GRAY),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ])
        for _idx, (_val, _, _, _bg) in enumerate(_ri_cfg, start=1):
            _ri_style.add("BACKGROUND", (0, _idx), (-1, _idx), colors.HexColor(_bg))
        _ri_tbl.setStyle(_ri_style)
        story.append(_ri_tbl)
        story.append(Spacer(1, 6))

    # Field-level metrics
    story.append(Paragraph("Field-Level Metrics", section_style))
    _c_base = risk_result.get("c_factor_baseline", _c_adj)
    _c_pct  = int((_c_base - _c_adj) / _c_base * 100) if _c_base > 0 else 0
    _zes = risk_result.get("zone_erosion_summary", [])
    _a_saved_w = _a_base_w = _a_cur_w = None
    if _zes:
        # 0.0 savings is a real value, not "unavailable" — see CCA-report note.
        _saved_vals_45 = [z["a_saved_zone"] * z["area_fraction"]
                          for z in _zes if z.get("a_saved_zone") is not None]
        _base_vals_45  = [z["a_baseline_zone"] * z["area_fraction"]
                          for z in _zes if z.get("a_baseline_zone") is not None]
        _cur_vals_45   = [z["a_current_zone"] * z["area_fraction"]
                          for z in _zes if z.get("a_current_zone") is not None]
        _a_saved_w = sum(_saved_vals_45) if _saved_vals_45 else None
        _a_base_w  = sum(_base_vals_45)  if _base_vals_45  else None
        _a_cur_w   = sum(_cur_vals_45)   if _cur_vals_45   else None
    _pct_red_w = ((_a_saved_w / _a_base_w * 100)
                  if (_a_base_w and _a_saved_w is not None) else None)
    metrics_rows = [
        ["Metric", "Value"],
        ["NDVI mean", f"{_ndvi_mean:.3f}"],
        ["Slope mean", f"{slope_stats.get('mean', 0):.1f}%"],
        [f"C-factor ({residue_system or 'residue system'})",
         f"{_c_adj:.3f}"],
        ["C-factor reduction vs. baseline", f"{_c_pct}%"],
        ["Risk Index (C×LS), field mean",
         f"{risk_result.get('rusle_score', 0):.3f}"],
    ]
    if _pct_red_w is not None:
        metrics_rows.append(["Estimated % erosion reduction (cover vs. bare)",
                             f"{_pct_red_w:.1f}%"])
    if _a_base_w is not None:
        metrics_rows.append(["Baseline soil loss estimate (no cover)",
                             f"{_a_base_w:.1f} t/ac/yr"])
    if _a_saved_w is not None:
        metrics_rows.append(["Estimated soil saved (area-weighted)",
                             f"{_a_saved_w:.1f} t/ac/yr"])
    _met_tbl = Table(metrics_rows, colWidths=[3.6 * inch, 2.0 * inch])
    _met_tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  BLUE_ACCENT),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GRAY, colors.white]),
        ("ALIGN",          (1, 0), (1, -1),  "CENTER"),
        ("GRID",           (0, 0), (-1, -1), 0.3, MID_GRAY),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
    ]))
    story.append(_met_tbl)
    story.append(Paragraph(
        f"<i>{A_VALUE_DISCLAIMER}</i>",
        ParagraphStyle("AValueDisc45", parent=small_style,
                       textColor=colors.HexColor("#92400e"))))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # PAGE 3 — § 2100.052(b) PRODUCER RECORDS (fillable AcroForm)
    # -----------------------------------------------------------------------
    story.append(Paragraph(
        "§ 2100.052(b) Producer Records", section_style))
    story.append(Paragraph(
        "The fields below are producer-supplied § 2100.052(b) records. "
        "CoverMap does not generate this content. Producer signature at bottom "
        "certifies these records exist and are retained for 5 years per "
        "§ 2100.052(b). Fields are fillable in a PDF viewer or may be "
        "printed and completed by hand.",
        wall_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1.0,
                            color=colors.HexColor("#c5d9f5"), spaceBefore=4,
                            spaceAfter=6))
    story.append(_ProducerRecordsForm(7 * inch, producer_inputs=producer_inputs))
    story.append(Spacer(1, 10))

    # Producer attestation + signature block (print / sign / scan convention).
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=MID_GRAY, spaceAfter=6))
    story.append(Paragraph(
        "I certify the § 2100.052(b) records above and the underlying "
        "documentation are true, accurate, and retained for 5 years per 7 CFR "
        "§ 2100.052(b).",
        body_style,
    ))
    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "Producer signature: _________________________&nbsp;&nbsp;&nbsp;"
        "Date: __________",
        body_style,
    ))
    story.append(Spacer(1, 12))

    # -----------------------------------------------------------------------
    # FOOTER (retained provenance verbatim + two new regulatory-basis lines)
    # -----------------------------------------------------------------------
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=MID_GRAY, spaceAfter=3))
    footer_lines = [
        f"NDVI Source: Sentinel-2 via Google Earth Engine ({_window_range}) | "
        f"DEM: {dem_source} | Slope: computed in UTM meters (EPSG:26915)",
        f"Data provenance — DEM source: {dem_source}  |  "
        f"R-factor source: {r_factor_note or 'R=%.0f' % r_factor}",
        "Regulatory basis: 7 CFR Part 2100, § 2100.052(c), effective July 29, "
        "2026 (91 FR 39334, Docket USDA-2024-0003, RIN 0503-AA82).",
        "Attachment framing: Supports Biofuel Feedstock Report per 7 CFR "
        "§ 2100.031(e). Attached as cover crop practice evidence under "
        "§ 2100.052(c). Not a substitute for USDA FD-CIC calculation "
        "documentation required under § 2100.031(e)(3)(i).",
        f"CoverMap 45Z Verification Package · {cca_name} · "
        f"Report ID {report_id} · {report_date}",
    ]
    for line in footer_lines:
        story.append(Paragraph(line, small_style))

    # Build PDF
    doc.build(story)
    buf.seek(0)
    return buf.read()

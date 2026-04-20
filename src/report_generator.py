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

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, PageBreak,
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


# ---------------------------------------------------------------------------
# Map image generator
# ---------------------------------------------------------------------------

def generate_zone_map_image(
    ndvi_array: np.ndarray,
    ndvi_threshold: float = 0.20,
    width_px: int = 600,
    height_px: int = 400,
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

    fig, ax = plt.subplots(1, 1, figsize=(width_px/100, height_px/100), dpi=100)
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

    fig, ax = plt.subplots(1, 1, figsize=(width_px/100, height_px/100), dpi=100)
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

    fig, ax = plt.subplots(1, 1, figsize=(width_px / 100, height_px / 100), dpi=100)
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


# ---------------------------------------------------------------------------
# Zone acreage calculator
# ---------------------------------------------------------------------------

def calculate_zone_acres(
    ndvi_array: np.ndarray,
    ndvi_threshold: float,
    pixel_size_m: float = 10.0,
) -> Dict[str, float]:
    """Calculate acres per zone from pixel counts."""
    m2_per_pixel = pixel_size_m ** 2
    acres_per_pixel = m2_per_pixel / 4046.86

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
                            color=BLUE_ACCENT, spaceAfter=6))

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

    field_data = [
        [
            Paragraph(f"<b>Field:</b> {field_name}", body_style),
            Paragraph(f"<b>Farm:</b> {farm_name}", body_style),
            Paragraph(f"<b>County:</b> {county}", body_style),
            Paragraph(f"<b>Report Date:</b> {report_date}", body_style),
        ],
        [
            Paragraph(f"<b>Previous crop:</b> {previous_crop or 'Not recorded'}", body_style),
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

    # Risk Index map — full width
    if risk_zone_array is not None:
        risk_png = generate_risk_zone_map_image(risk_zone_array)
        risk_img = RLImage(io.BytesIO(risk_png), width=7.0 * inch, height=3.4 * inch)
        story.append(risk_img)
        story.append(Paragraph(
            "Erosion Risk Index Zones (C\u00d7LS) \u2014 pixel-level RUSLE risk classification",
            map_label_style,
        ))
        story.append(Spacer(1, 8))

    # NDVI + Slope — side by side
    ndvi_png  = generate_zone_map_image(ndvi_array, ndvi_threshold)
    slope_png = generate_slope_map_image(slope_array)
    map_w = 3.4 * inch
    map_h = 2.2 * inch
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

    zone_acres  = calculate_zone_acres(ndvi_array, ndvi_threshold)
    total_acres = zone_acres.get("Total", 1)
    zone_display_rows = [
        ("Low cover",  f"Low cover (NDVI < {ndvi_threshold:.2f})",             colors.HexColor("#FEE8D5")),
        ("Marginal",   f"Marginal ({ndvi_threshold:.2f}\u2013{_marginal_upper:.2f})", colors.HexColor("#E0F2FE")),
        ("Good cover", f"Good cover (NDVI > {_marginal_upper:.2f})",            colors.HexColor("#FEF9C3")),
    ]
    ndvi_zone_rows = [["Zone", "Acres", "% Field"]]
    ndvi_zone_bg  = []
    for i, (key, label, bg) in enumerate(zone_display_rows, start=1):
        acres = zone_acres.get(key, 0)
        pct   = acres / total_acres * 100 if total_acres > 0 else 0
        ndvi_zone_rows.append([label, f"{acres:.1f}", f"{pct:.0f}%"])
        ndvi_zone_bg.append(("BACKGROUND", (0, i), (-1, i), bg))
    ndvi_zone_rows.append(["Total", f"{total_acres:.1f}", "100%"])

    ndvi_zone_table = Table(ndvi_zone_rows, colWidths=[3.5 * inch, 1.0 * inch, 1.0 * inch])
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
    if zone_counts:
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=MID_GRAY, spaceAfter=4))
        story.append(Paragraph("Erosion Risk Zone Summary (C\u00d7LS)", section_style))

        RISK_ZONE_DEFS = [
            (4, "Critical risk (\u22651.5)",   colors.HexColor("#fecaca")),
            (3, "High risk (0.7\u20131.5)",    colors.HexColor("#fee2e2")),
            (2, "Moderate risk (0.3\u20130.7)", colors.HexColor("#fef9c3")),
            (1, "Low risk (<0.3)",             colors.HexColor("#dcfce7")),
        ]
        total_zone_px   = sum(zone_counts.values()) or 1
        risk_zone_rows  = [["Zone", "Pixels", "% Field"]]
        risk_zone_bg    = []
        actual_row      = 1
        for val, label, bg in RISK_ZONE_DEFS:
            px = zone_counts.get(val, 0)
            if px == 0:
                continue
            pct = px / total_zone_px * 100
            risk_zone_rows.append([label, str(px), f"{pct:.0f}%"])
            risk_zone_bg.append(("BACKGROUND", (0, actual_row), (-1, actual_row), bg))
            actual_row += 1

        risk_zone_t = Table(risk_zone_rows, colWidths=[3.5 * inch, 1.0 * inch, 1.0 * inch])
        risk_zone_t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  BLUE_ACCENT),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
            ("ALIGN",         (1, 0), (2, -1),  "CENTER"),
            ("GRID",          (0, 0), (-1, -1), 0.3, MID_GRAY),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ] + risk_zone_bg))
        story.append(risk_zone_t)
        story.append(Spacer(1, 8))

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

    eqip_data = [
        ["Requirement", "Data Source", "Status"],
        ["Cover crop present",   "Sentinel-2 NDVI > 0.20",  cover_status],
        ["Field boundary",       "Operator provided",        "Verify against FSA CLU records"],
        ["Image date",           "GEE metadata",             image_date_str],
        ["Estimated biomass",    "NDVI proxy",               f"~{biomass_low}\u2013{biomass_high} lb/acre (\u00b140% NDVI proxy)"],
        ["30% ground cover",     "NDVI threshold",           ground_cover_status],
        ["Seeding rate",         "Field records required",   "\U0001f4cb CCA to verify on-site"],
        ["Species confirmation", "Field records required",   "\U0001f4cb CCA to verify on-site"],
        ["Termination date",     "Field records required",   _term_status],
        ["Cooperator signature", "Physical form required",   "\U0001f4cb Required for EQIP submission"],
    ]

    eqip_col_w = [1.8 * inch, 1.9 * inch, 3.3 * inch]
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
    ])
    eqip_table.setStyle(eqip_style)
    story.append(eqip_table)
    story.append(Paragraph(
        "<i>Remote sensing confirms spatial cover crop presence. "
        "Seeding rate, species, and termination compliance require CCA field "
        "verification per NRCS Practice Code 340.</i>",
        small_style,
    ))
    story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # COVER CROP METRICS
    # -----------------------------------------------------------------------
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=MID_GRAY, spaceAfter=4))
    story.append(Paragraph("Cover Crop Metrics", section_style))

    _c_adj   = risk_result.get("c_factor", 0)
    _c_raw   = risk_result.get("c_factor_unadjusted", _c_adj)
    _c_mult  = risk_result.get("residue_multiplier", 1.0)
    if _c_mult < 1.0:
        _c_display = f"{_c_raw:.3f} \u2192 {_c_adj:.3f} (\u00d7{_c_mult:.2f} residue)"
        _c_label   = "C-Factor (adj.)"
    else:
        _c_display = f"{_c_adj:.3f} (no residue adj.)"
        _c_label   = "C-Factor (RUSLE)"

    metrics = [
        ["Metric", "Value"],
        ["NDVI Mean",         f"{ndvi_stats.get('mean', 0):.3f}"],
        ["NDVI Range",        f"{ndvi_stats.get('min', 0):.3f} \u2013 {ndvi_stats.get('max', 0):.3f}"],
        ["Slope Mean (%)",    f"{slope_stats.get('mean', 0):.1f}%"],
        [_c_label,            _c_display],
        ["Risk Index (C\u00d7LS)", f"{risk_result.get('rusle_score', 0):.3f}"],
        ["Erosion Concern",   concern],
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
        _r_note = r_factor_note or f"R={r_factor:.0f} (Iowa erosivity index)"
        story.append(Paragraph(
            f"<i>Iowa R-factor: {_r_note} | "
            f"A = R \u00d7 K \u00d7 LS \u00d7 C (P=1.0). Simplified RUSLE estimate for advisory "
            f"use only \u2014 not a substitute for a site-specific RUSLE2 run or official "
            f"NRCS determination.</i>",
            small_style,
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
        "C-Factor methodology: Iowa RUSLE lookup table \u2014 "
        "Laflen & Roose (1998), ISU Extension PM-1209. "
        "This report is advisory only and does not constitute an official NRCS determination.",
        "C-Factor residue adjustment: CoverMap applies a research-based multiplier to the NDVI-derived "
        "C-factor to account for crop residue protection not captured by satellite imagery. "
        "Multipliers are calibrated to Iowa tillage system residue levels per ISU Extension PM-1901 "
        "and NRCS RUSLE2 Iowa State File guidance. This adjustment is an agronomic estimate requiring "
        "validation against site-specific RUSLE2 runs. Default multiplier = 1.00 (no adjustment) "
        "when tillage system is unknown.",
        f"CoverMap \u00b7 {cca_name} \u00b7 Sentinel-2 via Google Earth Engine \u00b7 Iowa RUSLE C-factor calibration \u00b7 {report_date}",
    ]
    for line in footer_lines:
        story.append(Paragraph(line, small_style))

    # Build PDF
    doc.build(story)
    buf.seek(0)
    return buf.read()

"""
report_generator.py
-------------------
Single-page PDF field summary report for Cover Crop Erosion Viewer.
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
    Image as RLImage, HRFlowable,
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
    # Settings
    ndvi_threshold: float,
    slope_threshold: float,
    # Dates
    ndvi_date_from: Optional[str] = None,
    ndvi_date_to: Optional[str] = None,
    report_date: Optional[str] = None,
    dem_source: str = "Iowa 3m WCS",
    # CCA info
    cca_name: str = "Stephen Zimmerman, CCA MS",
    cca_contact: str = "Ankeny, IA | Ag Research Scientist",
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
        Paragraph(f"<b>Cover Crop Erosion Viewer</b>", title_style),
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

    field_data = [[
        Paragraph(f"<b>Field:</b> {field_name}", body_style),
        Paragraph(f"<b>Farm:</b> {farm_name}", body_style),
        Paragraph(f"<b>County:</b> {county}", body_style),
        Paragraph(f"<b>Report Date:</b> {report_date}", body_style),
    ]]
    field_table = Table(field_data, colWidths=[1.75*inch]*4)
    field_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), LIGHT_GRAY),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [LIGHT_GRAY]),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("BOX", (0,0), (-1,-1), 0.5, MID_GRAY),
    ]))
    story.append(field_table)

    if ndvi_date_str:
        story.append(Paragraph(
            f"<i>{ndvi_date_str} &nbsp;|&nbsp; DEM: {dem_source}</i>",
            small_style
        ))
    story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # MAPS — side by side
    # -----------------------------------------------------------------------
    story.append(Paragraph("Management Zone Map", section_style))

    ndvi_png  = generate_zone_map_image(ndvi_array, ndvi_threshold)
    slope_png = generate_slope_map_image(slope_array)

    map_w = 3.4 * inch
    map_h = 2.3 * inch

    ndvi_img  = RLImage(io.BytesIO(ndvi_png),  width=map_w, height=map_h)
    slope_img = RLImage(io.BytesIO(slope_png), width=map_w, height=map_h)

    map_label_style = ParagraphStyle(
        "MapLabel", parent=body_style,
        alignment=TA_CENTER, fontSize=8, fontName="Helvetica-Bold"
    )

    maps_data = [[
        [ndvi_img,  Paragraph("NDVI Cover Quality Zones",  map_label_style)],
        [slope_img, Paragraph("Terrain Slope (% gradient)", map_label_style)],
    ]]
    maps_table = Table(
        [[ndvi_img, slope_img]],
        colWidths=[map_w + 0.1*inch, map_w + 0.1*inch],
    )
    maps_table.setStyle(TableStyle([
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(maps_table)

    # Map labels
    labels_data = [[
        Paragraph("NDVI Cover Quality Zones", map_label_style),
        Paragraph("Terrain Slope (% gradient)", map_label_style),
    ]]
    labels_table = Table(
        labels_data,
        colWidths=[map_w + 0.1*inch, map_w + 0.1*inch],
    )
    labels_table.setStyle(TableStyle([
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ]))
    story.append(labels_table)
    story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # METRICS + ZONE TABLE — side by side
    # -----------------------------------------------------------------------
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=MID_GRAY, spaceAfter=4))

    concern      = risk_result.get("concern_level", "N/A")
    concern_col  = CONCERN_BADGE_COLOR.get(concern, TEXT_DARK)
    zone_acres   = calculate_zone_acres(ndvi_array, ndvi_threshold)

    # Metrics table
    metrics = [
        ["Metric", "Value"],
        ["NDVI Mean",          f"{ndvi_stats.get('mean', 0):.3f}"],
        ["NDVI Range",         f"{ndvi_stats.get('min',0):.3f} – {ndvi_stats.get('max',0):.3f}"],
        ["Slope Mean (%)",     f"{slope_stats.get('mean',0):.1f}%"],
        ["C-Factor (RUSLE)",   f"{risk_result.get('c_factor', 0):.3f}"],
        ["RUSLE C x LS Score", f"{risk_result.get('rusle_score', 0):.3f}"],
        ["Erosion Concern",    concern],
    ]

    met_style = TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  BLUE_ACCENT),
        ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [LIGHT_GRAY, colors.white]),
        ("ALIGN",         (1,0), (1,-1),  "CENTER"),
        ("GRID",          (0,0), (-1,-1), 0.3, MID_GRAY),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        # Color the concern row
        ("TEXTCOLOR",     (1,6), (1,6),   concern_col),
        ("FONTNAME",      (1,6), (1,6),   "Helvetica-Bold"),
    ])

    met_table = Table(metrics, colWidths=[1.5*inch, 1.1*inch])
    met_table.setStyle(met_style)

    # Zone acres table
    zone_rows = [["Zone", "Acres", "% Field"]]
    total_acres = zone_acres.get("Total", 1)
    zone_color_map = {
        "Low cover":  colors.HexColor("#FEE8D5"),
        "Marginal":   colors.HexColor("#E0F2FE"),
        "Good cover": colors.HexColor("#FEF9C3"),
    }
    for zone in ["Low cover", "Marginal", "Good cover"]:
        acres = zone_acres.get(zone, 0)
        pct   = acres / total_acres * 100 if total_acres > 0 else 0
        zone_rows.append([zone, f"{acres:.1f}", f"{pct:.0f}%"])
    zone_rows.append(["Total", f"{total_acres:.1f}", "100%"])

    zone_t_style = TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  BLUE_ACCENT),
        ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8.5),
        ("BACKGROUND",    (0,1), (-1,1),  zone_color_map["Low cover"]),
        ("BACKGROUND",    (0,2), (-1,2),  zone_color_map["Marginal"]),
        ("BACKGROUND",    (0,3), (-1,3),  zone_color_map["Good cover"]),
        ("BACKGROUND",    (0,4), (-1,4),  LIGHT_GRAY),
        ("FONTNAME",      (0,4), (-1,4),  "Helvetica-Bold"),
        ("ALIGN",         (1,0), (2,-1),  "CENTER"),
        ("GRID",          (0,0), (-1,-1), 0.3, MID_GRAY),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
    ])
    zone_table = Table(zone_rows, colWidths=[1.1*inch, 0.7*inch, 0.7*inch])
    zone_table.setStyle(zone_t_style)

    # Side by side
    combined = Table(
        [[met_table, Spacer(0.2*inch, 1), zone_table]],
        colWidths=[2.7*inch, 0.2*inch, 2.6*inch],
    )
    combined.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story.append(combined)
    story.append(Spacer(1, 8))

    # -----------------------------------------------------------------------
    # NRCS RECOMMENDATION
    # -----------------------------------------------------------------------
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=MID_GRAY, spaceAfter=4))
    story.append(Paragraph("NRCS Advisory & Recommendation", section_style))

    rec_text = risk_result.get("recommendation", "No recommendation available.")
    rec_bg   = {
        "Low":      colors.HexColor("#dcfce7"),
        "Moderate": colors.HexColor("#fef9c3"),
        "High":     colors.HexColor("#fee2e2"),
        "Critical": colors.HexColor("#fecaca"),
    }.get(concern, LIGHT_GRAY)

    rec_data = [[Paragraph(rec_text, body_style)]]
    rec_table = Table(rec_data, colWidths=[7*inch])
    rec_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), rec_bg),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("BOX",           (0,0), (-1,-1), 0.5, MID_GRAY),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # EQIP PRE-VERIFICATION CHECKLIST
    # -----------------------------------------------------------------------
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=MID_GRAY, spaceAfter=4))
    story.append(Paragraph("EQIP Pre-Verification Report", section_style))

    ndvi_mean_val   = ndvi_stats.get("mean", 0.0)
    biomass_kgha    = max(0.0, (ndvi_mean_val - 0.10) / 0.40 * 3500)
    valid_px        = ndvi_array[~np.isnan(ndvi_array)]
    pct_above_020   = (np.sum(valid_px > 0.20) / valid_px.size * 100) if valid_px.size > 0 else 0.0
    image_date_str  = ndvi_date_to if ndvi_date_to else "Upload date unknown"

    cover_status = (
        f"\u2705 NDVI {ndvi_mean_val:.3f} \u2014 cover crop confirmed"
        if ndvi_mean_val > 0.20 else
        f"\u26a0\ufe0f NDVI {ndvi_mean_val:.3f} \u2014 inadequate cover"
    )
    ground_cover_status = (
        f"\u2705 {pct_above_020:.0f}% of field above NDVI 0.20"
        if pct_above_020 > 50 else
        f"\u26a0\ufe0f Only {pct_above_020:.0f}% of field above NDVI 0.20"
    )

    eqip_data = [
        ["Requirement", "Data Source", "Status"],
        ["Cover crop present",   "Sentinel-2 NDVI > 0.20",  cover_status],
        ["Spatial distribution", "Zone map attached",        "\u2705 Zone map attached"],
        ["Image date",           "GEE metadata",             image_date_str],
        ["Estimated biomass",    "NDVI proxy",               f"~{biomass_kgha:.0f} kg/ha estimated"],
        ["30% ground cover",     "NDVI threshold",           ground_cover_status],
        ["Seeding rate",         "Field records required",   "\U0001f4cb CCA to verify on-site"],
        ["Species confirmation", "Field records required",   "\U0001f4cb CCA to verify on-site"],
        ["Termination date",     "Not yet applicable",       "\u23f3 Pending \u2014 document at termination"],
        ["Cooperator signature", "Physical form required",   "\U0001f4cb Required for EQIP submission"],
    ]

    eqip_col_w = [1.8*inch, 1.9*inch, 3.3*inch]
    eqip_table = Table(
        [[Paragraph(str(cell), body_style) for cell in row] for row in eqip_data],
        colWidths=eqip_col_w,
    )
    eqip_style = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  BLUE_ACCENT),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.0),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT_GRAY, colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.3, MID_GRAY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
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
    # FOOTER
    # -----------------------------------------------------------------------
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=MID_GRAY, spaceAfter=3))

    footer_lines = [
        f"NDVI Source: Sentinel-2 via Google Earth Engine ({ndvi_date_str}) | "
        f"DEM: {dem_source} | Slope: computed in UTM meters (EPSG:26915)",
        "C-Factor methodology: Iowa RUSLE lookup table — "
        "Laflen & Roose (1998), ISU Extension PM-1209. "
        "This report is advisory only and does not constitute an official NRCS determination.",
        f"Generated by Cover Crop Erosion Viewer | {cca_name} | {report_date}",
    ]
    for line in footer_lines:
        story.append(Paragraph(line, small_style))

    # Build PDF
    doc.build(story)
    buf.seek(0)
    return buf.read()

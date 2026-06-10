"""
Convert covermap_explainer_v2.html to a professional PDF using Playwright/Chromium.
Injects print CSS to control page breaks and preserve all styling.
"""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

HTML_FILE = Path(__file__).parent / "covermap_explainer_v2.html"
PDF_FILE  = Path(__file__).parent / "covermap_explainer_v2.pdf"

PRINT_CSS = """
/* ── Print overrides ── */
@page {
  size: letter;
  margin: 18mm 15mm 18mm 15mm;
}

/* Hide nav and lightbox overlay in print */
nav, .lb-overlay, .lb-close { display: none !important; }

/* Remove sticky positioning */
* { position: static !important; }
nav { display: none !important; }

/* Hero: keep on one block, allow background */
.hero {
  -webkit-print-color-adjust: exact !important;
  print-color-adjust: exact !important;
  padding: 48px 24px 40px !important;
  break-inside: avoid;
}

/* Force all backgrounds to print */
body, section, .hero, .formula-block, .card, .zone-card,
.stat-box, .warning-box, .limitation, th, tr:nth-child(even) td,
.pipe-step, .report-img-wrap .img-caption, .badge {
  -webkit-print-color-adjust: exact !important;
  print-color-adjust: exact !important;
}

/* Section spacing for print */
section { padding: 36px 0 !important; }
.inner  { max-width: 100% !important; }

/* Keep cards together */
.card { break-inside: avoid; }

/* Keep zone grid together */
.zone-grid { break-inside: avoid; }
.zone-card { break-inside: avoid; }

/* Keep stat rows together */
.stat-row { break-inside: avoid; }
.stat-box  { break-inside: avoid; }

/* Keep formula blocks together */
.formula-block { break-inside: avoid; }

/* Keep warning box together */
.warning-box { break-inside: avoid; }

/* Keep pipeline diagram together */
.pipeline { break-inside: avoid; }

/* Keep table rows from splitting */
table { break-inside: auto; }
tr    { break-inside: avoid; }
thead { display: table-header-group; }

/* Keep images with their captions */
.report-img-wrap { break-inside: avoid; }

/* Section headings stay with content */
h2, h3, h4 { break-after: avoid; }
.sec-label  { break-after: avoid; }
.sec-title  { break-after: avoid; }
.sec-lead   { break-after: avoid; }

/* Two-col: stack for print to avoid splitting */
.two-col {
  grid-template-columns: 1fr !important;
  gap: 16px !important;
}

/* References section */
.references li { break-inside: avoid; }

/* Font sizes: slightly smaller for print legibility */
body { font-size: 10pt !important; }
.hero h1 { font-size: 22pt !important; }
.sec-title { font-size: 15pt !important; }
.formula-block { font-size: 8pt !important; }

/* Ensure images scale to page width */
img { max-width: 100% !important; height: auto !important; }
"""


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # Load the HTML file
        url = HTML_FILE.as_uri()
        print(f"Loading: {url}")
        await page.goto(url, wait_until="networkidle")

        # Inject print CSS
        await page.add_style_tag(content=PRINT_CSS)

        # Wait briefly for any layout reflow
        await page.wait_for_timeout(800)

        # Export to PDF
        print(f"Exporting PDF to: {PDF_FILE}")
        await page.pdf(
            path=str(PDF_FILE),
            format="Letter",
            print_background=True,
            margin={
                "top":    "18mm",
                "bottom": "18mm",
                "left":   "15mm",
                "right":  "15mm",
            },
            display_header_footer=True,
            header_template="""
                <div style="
                  width:100%; font-size:7pt;
                  font-family:'Segoe UI',sans-serif;
                  color:#1d5c38; text-align:right;
                  padding-right:15mm; padding-top:6mm;
                  border-bottom:1px solid #ddd8cc;
                ">CoverMap — Satellite-Assisted Cover Crop Erosion Risk Assessment</div>
            """,
            footer_template="""
                <div style="
                  width:100%; font-size:7pt;
                  font-family:'Segoe UI',sans-serif;
                  color:#5a5750; text-align:center;
                  padding-bottom:6mm;
                ">Page <span class='pageNumber'></span> of <span class='totalPages'></span></div>
            """,
        )

        await browser.close()
        print(f"Done! PDF saved to: {PDF_FILE}")
        print(f"File size: {PDF_FILE.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    asyncio.run(main())

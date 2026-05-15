"""Generate covermap_explainer.html — self-contained, all images embedded."""

import base64

def b64img(path, mime="image/png"):
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"

FIG3  = b64img("figure3_c_factor_curves.png")
RPT1  = b64img("_report_p1.png")
RPT2  = b64img("_report_p2.png")
RPT3  = b64img("_report_p3.png")

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CoverMap — Satellite-Assisted Cover Crop Erosion Risk Assessment</title>
<style>
/* ── Reset ── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html {{ scroll-behavior: smooth; font-size: 16px; }}

/* ── Tokens ── */
:root {{
  --green:    #1d5c38;
  --green-lt: #2e7d4f;
  --gold:     #b5882b;
  --gold-lt:  #e8c96d;
  --bg:       #f7f5f0;
  --bg-alt:   #ffffff;
  --border:   #ddd8cc;
  --text:     #2c2a26;
  --muted:    #5a5750;
  --c-low:    #2e8b57;
  --c-mod:    #c8940a;
  --c-high:   #c0632a;
  --c-crit:   #a01c1c;
  --nav-h:    56px;
}}

/* ── Base ── */
body {{
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.7;
}}
h1, h2, h3, h4 {{
  font-family: Georgia, 'Times New Roman', serif;
  color: var(--green);
  line-height: 1.25;
}}
p {{ margin-bottom: 1em; }}
a {{ color: var(--green); }}
strong {{ color: var(--text); font-weight: 600; }}
em {{ color: var(--muted); }}

/* ── Nav ── */
nav {{
  position: sticky; top: 0; z-index: 100;
  background: var(--green);
  height: var(--nav-h);
  display: flex; align-items: center;
  padding: 0 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,.25);
}}
nav .brand {{
  font-family: Georgia, serif;
  font-size: 1.15rem;
  color: #fff;
  font-weight: bold;
  margin-right: 32px;
  white-space: nowrap;
}}
nav ul {{
  list-style: none;
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}}
nav ul li a {{
  color: rgba(255,255,255,.82);
  text-decoration: none;
  font-size: .78rem;
  padding: 4px 10px;
  border-radius: 3px;
  transition: background .15s;
  white-space: nowrap;
}}
nav ul li a:hover {{
  background: rgba(255,255,255,.18);
  color: #fff;
}}

/* ── Hero ── */
.hero {{
  background: linear-gradient(145deg, #12402a 0%, #1d5c38 55%, #2e7d4f 100%);
  color: white;
  padding: 72px 24px 64px;
  text-align: center;
}}
.hero h1 {{
  color: white;
  font-size: 2.6rem;
  margin-bottom: .4em;
  letter-spacing: -.5px;
}}
.hero .tagline {{
  font-size: 1.18rem;
  color: rgba(255,255,255,.85);
  max-width: 680px;
  margin: 0 auto 1.4em;
  line-height: 1.55;
}}
.hero .author {{
  font-size: .88rem;
  color: var(--gold-lt);
  letter-spacing: .5px;
}}
.hero .badges {{
  display: flex;
  justify-content: center;
  gap: 12px;
  margin-top: 28px;
  flex-wrap: wrap;
}}
.badge {{
  background: rgba(255,255,255,.12);
  border: 1px solid rgba(255,255,255,.3);
  color: rgba(255,255,255,.9);
  padding: 5px 14px;
  border-radius: 20px;
  font-size: .8rem;
}}

/* ── Sections ── */
section {{
  padding: 64px 24px;
}}
section.alt {{ background: var(--bg-alt); }}
.inner {{
  max-width: 960px;
  margin: 0 auto;
}}
.sec-label {{
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--gold);
  margin-bottom: .5em;
}}
.sec-title {{
  font-size: 1.85rem;
  margin-bottom: .6em;
}}
.sec-lead {{
  font-size: 1.06rem;
  color: var(--muted);
  max-width: 720px;
  margin-bottom: 2em;
  line-height: 1.65;
}}

/* ── Pipeline diagram ── */
.pipeline {{
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0;
  margin: 2em 0;
  justify-content: center;
}}
.pipe-step {{
  background: var(--green);
  color: white;
  border-radius: 8px;
  padding: 14px 18px;
  text-align: center;
  min-width: 110px;
  font-size: .82rem;
  line-height: 1.35;
}}
.pipe-step .pipe-icon {{
  font-size: 1.4rem;
  margin-bottom: 4px;
}}
.pipe-step strong {{
  display: block;
  color: var(--gold-lt);
  font-size: .78rem;
  margin-bottom: 2px;
  font-family: Georgia, serif;
}}
.pipe-arrow {{
  color: var(--green);
  font-size: 1.5rem;
  padding: 0 4px;
  flex-shrink: 0;
}}

/* ── Two-col layout ── */
.two-col {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 32px;
  align-items: start;
  margin: 1.5em 0;
}}
@media (max-width: 680px) {{
  .two-col {{ grid-template-columns: 1fr; }}
  nav ul {{ display: none; }}
}}

/* ── Cards ── */
.card {{
  background: var(--bg-alt);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 24px 28px;
  margin-bottom: 20px;
}}
.card h3 {{
  font-size: 1.1rem;
  margin-bottom: .5em;
}}
.card.green-top {{ border-top: 4px solid var(--green); }}
.card.gold-top  {{ border-top: 4px solid var(--gold);  }}

/* ── Formula blocks ── */
.formula-block {{
  background: #1a2e1e;
  color: #a8d5b0;
  font-family: 'Courier New', monospace;
  font-size: .92rem;
  padding: 20px 24px;
  border-radius: 8px;
  line-height: 1.9;
  margin: 1.5em 0;
  overflow-x: auto;
}}
.formula-block .comment {{
  color: #5d8a65;
  font-style: italic;
}}
.formula-block .hl {{ color: var(--gold-lt); font-weight: bold; }}

/* ── Tables ── */
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: .88rem;
  margin: 1.5em 0;
}}
th {{
  background: var(--green);
  color: white;
  padding: 9px 12px;
  text-align: left;
  font-weight: 600;
  font-size: .82rem;
  letter-spacing: .3px;
}}
td {{
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}}
tr:nth-child(even) td {{ background: #f3f0ea; }}
tr:last-child td {{ border-bottom: none; }}

/* ── Risk zone chips ── */
.zone-grid {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin: 2em 0;
}}
@media (max-width: 680px) {{ .zone-grid {{ grid-template-columns: repeat(2,1fr); }} }}
.zone-card {{
  border-radius: 10px;
  padding: 20px 16px;
  text-align: center;
  color: white;
}}
.zone-card.low      {{ background: var(--c-low);  }}
.zone-card.moderate {{ background: var(--c-mod);  }}
.zone-card.high     {{ background: var(--c-high); }}
.zone-card.critical {{ background: var(--c-crit); }}
.zone-card .z-label {{
  font-family: Georgia, serif;
  font-size: 1.05rem;
  font-weight: bold;
  margin-bottom: 4px;
}}
.zone-card .z-range {{
  font-size: .78rem;
  opacity: .9;
  font-family: monospace;
  margin-bottom: 8px;
}}
.zone-card .z-desc {{
  font-size: .78rem;
  opacity: .85;
  line-height: 1.35;
}}

/* ── Inline stat boxes ── */
.stat-row {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px,1fr));
  gap: 14px;
  margin: 1.5em 0;
}}
.stat-box {{
  background: var(--bg-alt);
  border: 1px solid var(--border);
  border-left: 5px solid var(--green);
  border-radius: 6px;
  padding: 14px 16px;
}}
.stat-box .stat-val {{
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--green);
  font-family: Georgia, serif;
  line-height: 1;
  margin-bottom: 4px;
}}
.stat-box .stat-lbl {{
  font-size: .78rem;
  color: var(--muted);
  line-height: 1.3;
}}

/* ── Report images ── */
.report-img-wrap {{
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 4px 16px rgba(0,0,0,.12);
  margin: 1.5em 0;
}}
.report-img-wrap img {{
  width: 100%;
  display: block;
}}
.report-img-wrap .img-caption {{
  background: var(--green);
  color: rgba(255,255,255,.9);
  font-size: .78rem;
  padding: 7px 14px;
  font-style: italic;
}}

/* ── Warning / calibration box ── */
.warning-box {{
  background: #fff8e1;
  border: 2px solid #f5a623;
  border-left: 6px solid #e67e00;
  border-radius: 8px;
  padding: 24px 28px;
  margin: 2em 0;
}}
.warning-box .warn-title {{
  font-family: Georgia, serif;
  font-size: 1.05rem;
  color: #7a4500;
  margin-bottom: .6em;
  display: flex;
  align-items: center;
  gap: 8px;
}}
.warning-box p {{
  color: #5a3a00;
  font-size: .92rem;
  margin-bottom: .6em;
}}
.warning-box p:last-child {{ margin-bottom: 0; }}

/* ── Footnote / limitation ── */
.limitation {{
  background: #f0f0f0;
  border-left: 4px solid #999;
  padding: 10px 16px;
  border-radius: 0 6px 6px 0;
  font-size: .83rem;
  color: var(--muted);
  margin: 1em 0;
}}

/* ── References ── */
.references ol {{
  padding-left: 1.3em;
}}
.references li {{
  font-size: .86rem;
  color: var(--muted);
  margin-bottom: .55em;
  line-height: 1.5;
}}

/* ── Divider ── */
.divider {{
  border: none;
  border-top: 1px solid var(--border);
  margin: 2.5em 0;
}}

/* ── Highlight text ── */
.callout {{
  background: #eaf4ec;
  border-left: 4px solid var(--green);
  padding: 12px 18px;
  border-radius: 0 6px 6px 0;
  margin: 1.5em 0;
  font-size: .93rem;
  color: #1a3d22;
}}

/* ── Footer ── */
footer {{
  background: #111;
  color: rgba(255,255,255,.55);
  text-align: center;
  padding: 28px 24px;
  font-size: .78rem;
  line-height: 1.7;
}}
</style>
</head>
<body>

<!-- ═══════════════════════════════════════════ NAV -->
<nav>
  <div class="brand">CoverMap</div>
  <ul>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#ndvi">NDVI</a></li>
    <li><a href="#baseline">Residue Baseline</a></li>
    <li><a href="#cfactor">C-Factor</a></li>
    <li><a href="#lsfactor">LS-Factor</a></li>
    <li><a href="#riskzones">Risk Zones</a></li>
    <li><a href="#report">Field Report</a></li>
    <li><a href="#soilloss">Soil Loss</a></li>
    <li><a href="#calibration">Calibration</a></li>
    <li><a href="#references">References</a></li>
  </ul>
</nav>

<!-- ═══════════════════════════════════════════ HERO -->
<div class="hero">
  <h1>CoverMap</h1>
  <p class="tagline">Satellite-assisted cover crop erosion risk assessment for Iowa cropland — documenting stand quality and RUSLE risk across field slope positions.</p>
  <p class="author">Stephen Zimmerman, CCA MS &nbsp;·&nbsp; Ag Research Scientist &nbsp;·&nbsp; Ankeny, Iowa</p>
  <div class="badges">
    <span class="badge">Sentinel-2 · 10 m</span>
    <span class="badge">Google Earth Engine</span>
    <span class="badge">RUSLE C × LS</span>
    <span class="badge">Iowa NRCS FOTG</span>
    <span class="badge">EQIP Practice 340</span>
  </div>
</div>

<!-- ═══════════════════════════════════════════ OVERVIEW -->
<section id="overview" class="alt">
<div class="inner">
  <p class="sec-label">Overview</p>
  <h2 class="sec-title">What CoverMap Does</h2>
  <p class="sec-lead">CoverMap turns freely available satellite imagery into field-scale erosion risk maps and advisory reports — giving CCAs and agronomists a defensible, spatially explicit picture of cover crop stand quality before termination decisions are made.</p>

  <div class="two-col">
    <div>
      <h3 style="margin-bottom:.5em">The Problem It Solves</h3>
      <p>EQIP Practice 340 and NRCS payment verification require documented evidence of cover crop establishment. Visual scouting from a field road captures a fraction of a field's variability — particularly on rolling Midwest topography where north-facing backslopes and south-facing summit areas perform entirely differently. A field that looks adequate at the road edge may be at critical erosion risk on the 12–18% backslope 400 feet in.</p>
      <p>CoverMap replaces point-in-time, location-biased scouting with wall-to-wall 10 m resolution satellite assessment every spring. Every pixel in the field boundary gets a cover quality rating and an erosion risk classification — not just the areas a consultant can physically reach.</p>
    </div>
    <div>
      <h3 style="margin-bottom:.5em">Who It Is Built For</h3>
      <ul style="padding-left:1.2em; margin-bottom:1em; line-height:2">
        <li>Certified Crop Advisers managing EQIP cover crop contracts</li>
        <li>NRCS field offices documenting Practice 340 compliance</li>
        <li>Iowa State Extension agronomists evaluating cover crop programs</li>
        <li>Farmers who want to know where their cover crop is and is not protecting their soil</li>
      </ul>
      <p>All outputs are generated from publicly accessible satellite data (Sentinel-2 via Google Earth Engine) and publicly available soil databases (SSURGO). No proprietary data sources are required.</p>
    </div>
  </div>

  <hr class="divider">
  <h3 style="margin-bottom:1em">Workflow at a Glance</h3>
  <div class="pipeline">
    <div class="pipe-step">
      <div class="pipe-icon">🛰</div>
      <strong>Sentinel-2 L2A</strong>
      10 m surface reflectance<br>spring window composite
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step">
      <div class="pipe-icon">📡</div>
      <strong>NDVI</strong>
      Per-pixel NDVI<br>cloud-masked
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step">
      <div class="pipe-icon">🌱</div>
      <strong>C-Factor</strong>
      Piecewise exponential<br>by residue system
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step">
      <div class="pipe-icon">⛰</div>
      <strong>LS-Factor</strong>
      Iowa 3 m DEM<br>McCool et al. 1987
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step">
      <div class="pipe-icon">🗺</div>
      <strong>Risk Index</strong>
      C × LS per pixel<br>4-zone classification
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step">
      <div class="pipe-icon">📋</div>
      <strong>Field Report</strong>
      CCA advisory<br>+ RUSLE soil loss
    </div>
  </div>
</div>
</section>

<!-- ═══════════════════════════════════════════ NDVI -->
<section id="ndvi">
<div class="inner">
  <p class="sec-label">Remote Sensing</p>
  <h2 class="sec-title">NDVI — Quantifying Cover Density from Space</h2>
  <p class="sec-lead">The Normalized Difference Vegetation Index exploits the distinct spectral signatures of photosynthetically active tissue to estimate the density of green plant cover at 10 m resolution across an entire field in a single image acquisition.</p>

  <div class="two-col">
    <div>
      <h3>The Physics</h3>
      <p>Chlorophyll a and b strongly absorb incident radiation in the red band (650–680 nm) for photosynthesis, while the spongy mesophyll cell structure of healthy leaves scatters and reflects strongly in the near-infrared (NIR, 800–900 nm). Bare soil and crop residue lack this differential — they reflect both bands at similar, moderate levels. NDVI amplifies this contrast:</p>
      <div class="formula-block">
<span class="hl">NDVI = (NIR − Red) / (NIR + Red)</span>

<span class="comment">Scale: −1 to +1
Bare soil / residue:  0.05 – 0.18
Sparse cover:         0.18 – 0.30
Adequate stand:       0.30 – 0.50
Dense canopy:         0.50 – 0.80+</span>
      </div>
      <p>Sentinel-2 Band 8 (842 nm NIR) and Band 4 (665 nm Red) are used. Both are native 10 m resolution — no pan-sharpening or resampling required. The Level-2A product delivers atmospherically corrected surface reflectance, removing most aerosol and water vapor effects before NDVI computation.</p>
    </div>
    <div>
      <h3>Imagery Acquisition</h3>
      <p>CoverMap queries Google Earth Engine for all Sentinel-2 L2A scenes intersecting the field boundary within a user-defined spring window (default March 15 – May 15). The Sentinel-2 constellation revisits Iowa approximately every 5 days under clear conditions.</p>
      <p><strong>Cloud masking:</strong> The ESA Scene Classification Layer (SCL) is applied at the pixel level to remove cloud, cloud shadow, and saturated pixels before compositing. Scenes with fewer than 50% valid pixels trigger a reliability warning; fewer than 75% valid pixels trigger a quality advisory.</p>
      <p><strong>Temporal composite:</strong> Valid pixels across all scenes in the window are summarized as a temporal median — reducing noise from single-date anomalies and partial cloud cover. The composite represents integrated canopy conditions across the observation window, not a single snapshot.</p>
      <div class="limitation">
        <strong>Known limitation:</strong> NDVI saturates above approximately 3,500 kg/ha dry biomass (~3,100 lb/acre — the national cereal rye database mean). CoverMap cannot differentiate between a good and an excellent stand above this threshold. Additionally, an NDVI reading of 0.25 in late March (dormant rye) reflects different biomass than the same reading in early May (active growth). No growing degree day adjustment is currently applied.
      </div>
    </div>
  </div>

  <h3 style="margin-top:1.5em; margin-bottom:.8em">NDVI Cover Quality Zone Classification</h3>
  <table>
    <thead><tr><th>NDVI Range</th><th>Zone</th><th>Biomass Estimate (cereal rye)</th><th>NRCS Practice 340 Context</th></tr></thead>
    <tbody>
      <tr><td>&lt; 0.20</td><td><span style="color:var(--c-crit);font-weight:600">Low Cover</span></td><td>&lt; 900 lb/acre</td><td>Below minimum stand threshold; inadequate erosion protection</td></tr>
      <tr><td>0.20 – 0.35</td><td><span style="color:var(--c-mod);font-weight:600">Marginal</span></td><td>900 – 2,200 lb/acre</td><td>Borderline; may meet minimum on favorable soils; field verification recommended</td></tr>
      <tr><td>&gt; 0.35</td><td><span style="color:var(--c-low);font-weight:600">Good Cover</span></td><td>&gt; 2,200 lb/acre</td><td>Exceeds NRCS 340 stand requirement; effective erosion control likely</td></tr>
    </tbody>
  </table>
  <p style="font-size:.83rem; color:var(--muted)">Biomass estimates derived from Huddell et al. (2024) national cereal rye NDVI-biomass database (n = 5,695 field observations, mean 3,428 kg/ha). Uncertainty ±40% at the field scale.</p>
</div>
</section>

<!-- ═══════════════════════════════════════════ RESIDUE BASELINE -->
<section id="baseline" class="alt">
<div class="inner">
  <p class="sec-label">Model Foundation</p>
  <h2 class="sec-title">The Residue Baseline — What NDVI Actually Sees in Spring</h2>
  <p class="sec-lead">A critical step in translating satellite NDVI to erosion-relevant C-factor is distinguishing the living cover crop signal from the background NIR reflectance of crop residue — a distinction the raw NDVI value alone cannot make.</p>

  <div class="two-col">
    <div>
      <h3>The Problem with Raw NDVI</h3>
      <p>In March and April, an Iowa field's NDVI reading is not a pure function of living cover crop biomass. Corn stover at 80% surface coverage reflects NIR at levels that drive NDVI readings of 0.12–0.18 independently of any cover crop presence. A conventionally tilled field with no cover crop planted but residue present may produce NDVI near 0.10–0.14. Treating that reflectance as evidence of "some cover" and applying a corresponding C-factor reduction would overstate erosion protection.</p>
      <p>Any C-factor model that begins awarding living-cover credit at NDVI = 0 implicitly assumes that the entire NDVI signal originates from living vegetation — an assumption that fails in Iowa spring conditions across every tillage system.</p>
    </div>
    <div>
      <h3>The Universal NDVI Baseline: 0.185</h3>
      <p>CoverMap establishes a residue signal threshold below which no living-cover credit is awarded. Below NDVI = 0.185, C-factor equals the residue system's intercept — the erosion rate for that tillage system's residue alone, with no established cover crop. Above the threshold, C-factor decays exponentially as living plant biomass adds measurable canopy protection on top of the residue layer.</p>
      <div class="card green-top" style="margin-top:1em">
        <h3>Baseline Derivation</h3>
        <table style="margin:.5em 0">
          <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
          <tbody>
            <tr><td>Observations</td><td>15 Iowa fields</td></tr>
            <tr><td>Counties</td><td>5 Iowa counties</td></tr>
            <tr><td>Tillage systems</td><td>No-till, conservation, conventional</td></tr>
            <tr><td>Previous crops</td><td>Corn and soybean</td></tr>
            <tr><td>Imagery</td><td>Sentinel-2 L2A, March–April 2026</td></tr>
            <tr><td>Mean bare/residue NDVI</td><td>0.179</td></tr>
            <tr><td>Standard deviation</td><td>0.013</td></tr>
            <tr><td><strong>Threshold (mean + 0.5 SD)</strong></td><td><strong>0.185</strong></td></tr>
          </tbody>
        </table>
        <p style="font-size:.82rem; color:var(--muted); margin:0">No statistically significant difference in residue NDVI baseline was found across residue type or tillage system — supporting a single universal threshold.</p>
      </div>
    </div>
  </div>

  <div class="callout">
    <strong>Agronomic interpretation:</strong> An Iowa field with mean NDVI of 0.18 in April is not demonstrating cover crop stand — it is demonstrating residue. The cover crop may be present but too sparse or dormant to register above the soil + residue background. CoverMap's 0.185 threshold prevents that residue signal from being credited as erosion protection the cover crop has not yet earned.
  </div>
</div>
</section>

<!-- ═══════════════════════════════════════════ C-FACTOR -->
<section id="cfactor">
<div class="inner">
  <p class="sec-label">RUSLE Component 1</p>
  <h2 class="sec-title">C-Factor — Piecewise Exponential Model by Residue System</h2>
  <p class="sec-lead">The cover-management factor C encapsulates all vegetation and tillage effects on erosion relative to a standard bare fallow (C = 1.0). CoverMap parameterizes C as a piecewise exponential function of NDVI, differentiated by the CCA-selected residue system.</p>

  <h3>Model Formula</h3>
  <div class="formula-block">
<span class="comment">Residue-only zone (no living-cover credit):</span>
<span class="hl">C(NDVI) = intercept</span>                              <span class="comment">where NDVI ≤ 0.185</span>

<span class="comment">Living-cover zone (exponential decay):</span>
<span class="hl">C(NDVI) = floor + (intercept − floor) × exp(−k × (NDVI − 0.185))</span>   <span class="comment">where NDVI &gt; 0.185</span>

<span class="comment">Parameters by residue system:</span>
  intercept  =  C at NDVI = 0  (residue protection only, no cover crop)
  floor      =  C asymptote at high NDVI  (maximum cover crop benefit)
  k          =  decay constant  (higher = faster benefit per unit NDVI)
  </div>

  <div class="two-col" style="margin-top:1.5em">
    <div>
      <h3>Parameter Table</h3>
      <table>
        <thead><tr><th>Residue System</th><th>intercept</th><th>floor</th><th>k</th></tr></thead>
        <tbody>
          <tr><td>No-till corn (~80% residue)</td><td>0.05</td><td>0.005</td><td>8</td></tr>
          <tr><td>No-till soybeans (fragile residue)</td><td>0.10</td><td>0.015</td><td>7</td></tr>
          <tr><td>Conservation tillage (&gt;30% residue)</td><td>0.25</td><td>0.050</td><td>6</td></tr>
          <tr><td>Conventional tillage (&lt;30% residue)</td><td>0.45</td><td>0.100</td><td>5</td></tr>
          <tr><td>Unknown (conservative default)</td><td>0.45</td><td>0.100</td><td>5</td></tr>
        </tbody>
      </table>
      <p style="font-size:.82rem;color:var(--muted)">Parameters are initial estimates based on published RUSLE2 value ranges. See Calibration Status section.</p>

      <h3 style="margin-top:1.2em">Reading the Parameters</h3>
      <p><strong>Intercept</strong> is the C-factor for a field with that tillage system and zero living cover crop. No-till corn at intercept = 0.05 reflects that 80% corn stover coverage achieves substantial erosion protection independent of cover crop establishment — a well-documented field observation. Conventional tillage at 0.45 reflects near bare-soil conditions with minimal residue, consistent with NRCS RUSLE2 guidance.</p>
      <p><strong>Floor</strong> is the best achievable C under near-canopy-saturation NDVI for that system. No-till corn reaches a floor of 0.005 — effectively trace erosion relative to bare soil. Conventional tillage floor = 0.10 reflects irreducible risk from tillage-induced surface roughness and reduced aggregate stability even under dense living cover.</p>
      <p><strong>k</strong> governs response speed. No-till corn (k = 8) decays fastest — even modest green cover on top of a strong residue base yields large proportional C reductions. Conventional tillage (k = 5) responds more slowly because each unit of NDVI gain must overcome a larger initial C.</p>
    </div>
    <div>
      <h3>C-Factor Curves by Residue System</h3>
      <div class="report-img-wrap">
        <img src="{FIG3}" alt="Figure 3: Piecewise exponential C-factor curves">
        <div class="img-caption">Figure 3. Piecewise exponential C-factor curves. The flat region left of the dashed line (NDVI ≤ 0.185) awards no living-cover credit — C equals the residue-system intercept. Above 0.185, C decays exponentially toward the floor asymptote. Pre-calibration parameters.</div>
      </div>
      <div class="callout">
        <strong>Why the curves don't start at zero:</strong> Even with no cover crop, crop residue and tillage-induced roughness reduce erosion relative to a bare tilled reference. The intercept encodes that protection. The cover crop's job is to push C from the intercept down toward the floor — the additional reduction visible in the decaying portion of each curve.
      </div>
    </div>
  </div>
</div>
</section>

<!-- ═══════════════════════════════════════════ LS-FACTOR -->
<section id="lsfactor" class="alt">
<div class="inner">
  <p class="sec-label">RUSLE Component 2</p>
  <h2 class="sec-title">LS-Factor — Terrain Amplification of Erosion Risk</h2>
  <p class="sec-lead">The LS-factor is the topographic multiplier on erosion. It is why a cover crop at NDVI 0.25 on a 3% sideslope is not the same erosion story as the same cover crop on the 14% backslope 200 feet upslope.</p>

  <div class="two-col">
    <div>
      <h3>The McCool Formula</h3>
      <p>CoverMap computes S-factor using the continuous McCool et al. (1987) piecewise formula, which replaced the earlier 7-step USLE LS table with a smooth function calibrated to natural slope gradients:</p>
      <div class="formula-block">
θ = arctan(slope% / 100)

<span class="comment">S-factor (steepness):</span>
S = 10.8 × sin(θ) + 0.03    <span class="comment">slope &lt; 9%</span>
S = 16.8 × sin(θ) − 0.50    <span class="comment">slope ≥ 9%</span>
S ≥ 0.03                     <span class="comment">floor prevents zero on flat ground</span>

<span class="comment">L-factor (slope length, λ = 100 m assumed):</span>
L = (100 / 22.13) ^ m
  m = 0.2  (slope &lt; 1%)
  m = 0.3  (1% ≤ slope &lt; 3%)
  m = 0.4  (3% ≤ slope &lt; 5%)
  m = 0.5  (slope ≥ 5%)

<span class="hl">LS = L × S</span>  <span class="comment">(computed at every 10 m pixel)</span>
      </div>
      <div class="limitation">
        <strong>Slope length assumption:</strong> True RUSLE2 LS requires horizontal distance from flow origin to concentration point — a flow-accumulation computation. CoverMap uses a fixed 100 m slope length for all pixels, consistent with typical Iowa row-crop slope segment geometry. This introduces conservatism on long uniform slopes (&gt;200 m) and may underestimate LS on short steep concentrated-flow segments. Appropriate for field advisory use; not a substitute for formal NRCS RUSLE2 analysis.
      </div>
    </div>
    <div>
      <h3>LS Values at 100 m Slope Length</h3>
      <table>
        <thead><tr><th>Slope (%)</th><th>LS-Factor</th><th>Risk Amplification vs. 4%</th></tr></thead>
        <tbody>
          <tr><td>1</td><td>0.14</td><td>0.14×</td></tr>
          <tr><td>2</td><td>0.28</td><td>0.28×</td></tr>
          <tr><td>4</td><td>0.73</td><td>1.0× (reference)</td></tr>
          <tr><td>6</td><td>1.21</td><td>1.7×</td></tr>
          <tr><td>9</td><td>2.06</td><td>2.8×</td></tr>
          <tr><td>12</td><td>2.91</td><td>4.0×</td></tr>
          <tr><td>15</td><td>3.75</td><td>5.1×</td></tr>
          <tr><td>18</td><td>4.59</td><td>6.3×</td></tr>
        </tbody>
      </table>
      <div class="callout" style="margin-top:1em">
        <strong>Why per-pixel LS matters:</strong> A field with mean slope of 8% and a 16% backslope generates LS-factors of 1.8 and 4.3 respectively. Applying the mean slope to the whole field understates backslope risk by more than 2×. CoverMap computes LS independently at every 10 m pixel using the Iowa DNR 3 m LiDAR-derived DEM, then resamples to match the Sentinel-2 NDVI grid before multiplying with C-factor.
      </div>
    </div>
  </div>
</div>
</section>

<!-- ═══════════════════════════════════════════ RISK ZONES -->
<section id="riskzones">
<div class="inner">
  <p class="sec-label">Classification</p>
  <h2 class="sec-title">Risk Index and Zone Classification</h2>
  <p class="sec-lead">The per-pixel Risk Index is the product of C-factor and LS-factor — a dimensionless erosion potential that CoverMap classifies into four advisory zones. Every 10 m pixel in the field boundary receives an independent zone assignment based on its own NDVI and slope values.</p>

  <div class="formula-block" style="margin-bottom:2em">
<span class="hl">Risk Index = C(NDVI, residue system) × LS(slope%)</span>

<span class="comment">Computed at every 10 m pixel within the field boundary mask.
Higher values = higher erosion potential from the combination of poor cover and steep terrain.</span>
  </div>

  <div class="zone-grid">
    <div class="zone-card low">
      <div class="z-label">Low Risk</div>
      <div class="z-range">Risk Index &lt; 0.3</div>
      <div class="z-desc">Cover crop providing effective protection. Erosion rate well below tolerable loss for most Iowa soil types under normal spring rainfall.</div>
    </div>
    <div class="zone-card moderate">
      <div class="z-label">Moderate Risk</div>
      <div class="z-range">0.3 – 0.7</div>
      <div class="z-desc">Variable protection. Stand may be adequate on gentler ground but marginal on steeper slope positions. Field verification recommended prior to termination.</div>
    </div>
    <div class="zone-card high">
      <div class="z-label">High Risk</div>
      <div class="z-range">0.7 – 1.5</div>
      <div class="z-desc">Insufficient cover for slope conditions. These pixels are generating measurable soil loss above tolerable rates. Cover crop termination should be delayed if possible.</div>
    </div>
    <div class="zone-card critical">
      <div class="z-label">Critical Risk</div>
      <div class="z-range">Risk Index &gt; 1.5</div>
      <div class="z-desc">Severe erosion potential. Combination of poor cover and steep slope generates risk comparable to bare conventional tillage on moderate terrain. Immediate agronomic attention warranted.</div>
    </div>
  </div>

  <div class="callout">
    <strong>Field-average vs. per-pixel — why the distinction matters:</strong> A 48-acre Shelby County field with mean NDVI 0.25 and mean slope 8% might appear Moderate overall. But within that same field, the 16-acre backslope unit at NDVI 0.15 and slope 14% generates Critical-level Risk Index values — and that backslope is where Iowa loses the most topsoil per unit area per rain event. CoverMap's per-pixel approach surfaces that spatial heterogeneity rather than averaging it away.
  </div>
</div>
</section>

<!-- ═══════════════════════════════════════════ FIELD REPORT -->
<section id="report" class="alt">
<div class="inner">
  <p class="sec-label">Example Output</p>
  <h2 class="sec-title">Reading a CoverMap Field Report — Bin Field, Shelby County</h2>
  <p class="sec-lead">The following three-page report was generated for a conservation tillage field in Shelby County, Iowa using April 2026 Sentinel-2 imagery. Walk through each section to understand what CoverMap measures and how to interpret the advisory output.</p>

  <h3 style="margin-bottom:.6em">Page 1 — Field Maps and Metadata</h3>
  <div class="report-img-wrap">
    <img src="{RPT1}" alt="CoverMap Report Page 1 — Bin Field">
    <div class="img-caption">Page 1: Field identification, satellite acquisition metadata, and three spatially co-registered maps — Erosion Risk Index, NDVI Cover Quality, and Terrain Slope. All maps share the same 10 m pixel grid and field boundary mask.</div>
  </div>

  <div class="two-col" style="margin-top:1em">
    <div class="card green-top">
      <h3>Erosion Risk Index Map</h3>
      <p>The primary output. Each pixel is colored by its C × LS Risk Index value — the combination of cover density (from NDVI) and slope steepness (from the Iowa 3 m DEM). Red and orange pixels identify where cover crop stand is insufficient relative to the terrain's erosion amplification. This is the map that answers: <em>"Where is this field most vulnerable right now?"</em></p>
    </div>
    <div class="card green-top">
      <h3>NDVI and Slope Maps</h3>
      <p>The two component maps shown alongside the Risk Index allow the user to decompose the risk driver at any location. A High Risk pixel that is red on the NDVI map (low cover) but not particularly steep indicates a cover crop stand failure problem. A High Risk pixel that is green on the NDVI map but red on the slope map indicates a terrain-driven problem that better cover crop management can only partially offset.</p>
    </div>
  </div>

  <hr class="divider">
  <h3 style="margin-bottom:.6em">Page 2 — Advisory, Zone Tables, and Field Metrics</h3>
  <div class="report-img-wrap">
    <img src="{RPT2}" alt="CoverMap Report Page 2 — Bin Field">
    <div class="img-caption">Page 2: Erosion concern level with rationale, NDVI zone summary, Risk Index zone summary, and stand documentation checklist for EQIP Practice 340 compliance review.</div>
  </div>

  <div class="stat-row" style="margin-top:1.5em">
    <div class="stat-box"><div class="stat-val">0.251</div><div class="stat-lbl">Mean field NDVI<br>Apr 5–20, 2026</div></div>
    <div class="stat-box"><div class="stat-val">12.0%</div><div class="stat-lbl">Mean field slope<br>Iowa DNR 3m DEM</div></div>
    <div class="stat-box"><div class="stat-val">0.185</div><div class="stat-lbl">C-Factor<br>26% below baseline</div></div>
    <div class="stat-box"><div class="stat-val">0.592</div><div class="stat-lbl">Risk Index (C×LS)<br>Moderate concern</div></div>
    <div class="stat-box"><div class="stat-val">27%</div><div class="stat-lbl">Field area in<br>Low Cover zone</div></div>
    <div class="stat-box"><div class="stat-val">38%</div><div class="stat-lbl">Field area in<br>High or Critical zone</div></div>
  </div>

  <div class="two-col" style="margin-top:.5em">
    <div>
      <h3>NDVI Zone Breakdown</h3>
      <table>
        <thead><tr><th>Zone</th><th>Acres</th><th>% of Field</th></tr></thead>
        <tbody>
          <tr><td style="color:var(--c-crit);font-weight:600">Low Cover (NDVI &lt; 0.20)</td><td>12.9</td><td>27%</td></tr>
          <tr><td style="color:var(--c-mod);font-weight:600">Marginal (0.20–0.35)</td><td>31.9</td><td>66%</td></tr>
          <tr><td style="color:var(--c-low);font-weight:600">Good Cover (&gt; 0.35)</td><td>3.6</td><td>7%</td></tr>
          <tr style="font-weight:600"><td>Total</td><td>48.3</td><td>100%</td></tr>
        </tbody>
      </table>
      <p style="font-size:.82rem; color:var(--muted); margin-top:.5em">66% of the field is in the Marginal zone — NDVI readings above the residue baseline but below the Good Cover threshold. These pixels have living cover crop present but below the biomass density that provides reliable erosion control on steep ground.</p>
    </div>
    <div>
      <h3>Risk Index Zone Breakdown</h3>
      <table>
        <thead><tr><th>Zone</th><th>Acres</th><th>% of Field</th></tr></thead>
        <tbody>
          <tr><td style="color:var(--c-crit);font-weight:600">Critical (&gt; 1.5)</td><td>0.7</td><td>2%</td></tr>
          <tr><td style="color:var(--c-high);font-weight:600">High (0.7–1.5)</td><td>17.2</td><td>36%</td></tr>
          <tr><td style="color:var(--c-mod);font-weight:600">Moderate (0.3–0.7)</td><td>21.0</td><td>43%</td></tr>
          <tr><td style="color:var(--c-low);font-weight:600">Low (&lt; 0.3)</td><td>9.4</td><td>19%</td></tr>
        </tbody>
      </table>
      <p style="font-size:.82rem; color:var(--muted); margin-top:.5em">38% of field area classifies as High or Critical risk. These 18 acres — concentrated on the steeper backslope units — are generating the majority of this field's erosion losses despite representing only a fraction of its total acreage.</p>
    </div>
  </div>

  <hr class="divider">
  <h3 style="margin-bottom:.6em">Page 3 — Erosion Reduction Estimates</h3>
  <div class="report-img-wrap">
    <img src="{RPT3}" alt="CoverMap Report Page 3 — Bin Field">
    <div class="img-caption">Page 3: RUSLE-based soil loss estimates, per-zone erosion reduction breakdown, comparison to soil loss tolerance (T-value), and data source citations.</div>
  </div>
</div>
</section>

<!-- ═══════════════════════════════════════════ SOIL LOSS -->
<section id="soilloss">
<div class="inner">
  <p class="sec-label">RUSLE Estimation</p>
  <h2 class="sec-title">Soil Loss Estimation — Bin Field Walkthrough</h2>
  <p class="sec-lead">When SSURGO K-factor data are available, CoverMap estimates absolute annual soil loss using simplified RUSLE. The calculation uses field-mean values for a representative estimate — it is not a substitute for formal NRCS RUSLE2 analysis on the critical slope area.</p>

  <div class="formula-block">
<span class="hl">A = R × K × LS × C × P</span>

R  =  175.0  MJ·mm/ha·hr·yr  <span class="comment">Shelby County · Iowa NRCS FOTG Figure 2 (September 2002)</span>
K  =    0.32  t·ha·hr/ha·MJ·mm  <span class="comment">Monona silt loam · SSURGO (SDM Data Access API)</span>
LS =    5.24  (dimensionless)   <span class="comment">mean field LS · Iowa DNR 3m DEM · McCool et al. 1987</span>
C  =   0.185  (dimensionless)   <span class="comment">cover crop C · piecewise exponential model (this report)</span>
P  =    1.0                      <span class="comment">no support practice factor applied</span>

<span class="hl">A_current  = 175 × 0.32 × 5.24 × 0.185 × 1.0  ≈  54.3 t/ac/yr</span>
<span class="hl">A_baseline = 175 × 0.32 × 5.24 × 0.250 × 1.0  ≈  46.0 t/ac/yr</span>

<span class="comment">C_baseline = intercept for conservation tillage (NDVI = 0) = 0.250</span>
<span class="comment">Soil saved = A_baseline − A_current = 10.8 t/ac/yr  (23.4% reduction)</span>
<span class="comment">T-value (Monona) = 5 t/ac/yr  —  field is 9.2× above tolerable soil loss</span>
  </div>

  <div class="two-col">
    <div>
      <h3>Interpreting the Numbers</h3>
      <p>The 46 t/ac/yr baseline estimate — what this field would lose without any cover crop under conservation tillage — reflects Shelby County's challenging combination: R = 175 (higher erosivity than northwest Iowa's R = 150), Monona's relatively high K, and a 12% mean slope driving LS above 5.0. These are among the most erosive conditions in Iowa.</p>
      <p>The cover crop achieves 23.4% reduction — meaningful protection, but the field still generates estimated soil loss more than 9× above the Monona T-value of 5 t/ac/yr. This is the agronomically honest message of a CoverMap report: <em>cover crops help, but on 12% Monona slopes, they are one tool in what needs to be a layered erosion management system.</em></p>
    </div>
    <div>
      <h3>How CoverMap Compares to RUSLE2</h3>
      <table>
        <thead><tr><th>Parameter</th><th>RUSLE2</th><th>CoverMap</th></tr></thead>
        <tbody>
          <tr><td>Slope area</td><td>Critical slope segment</td><td>Field mean</td></tr>
          <tr><td>Slope length</td><td>Flow accumulation</td><td>100 m assumed</td></tr>
          <tr><td>C-factor</td><td>Daily through crop stages</td><td>Single NDVI composite</td></tr>
          <tr><td>LS precision</td><td>High (GIS-derived)</td><td>Moderate (fixed L)</td></tr>
          <tr><td>Typical result</td><td>Higher (critical area)</td><td>Lower (field average)</td></tr>
        </tbody>
      </table>
      <p style="font-size:.82rem; color:var(--muted); margin-top:.6em">CoverMap estimates will almost always be lower than RUSLE2 for the same field. RUSLE2 focuses on the dominant critical slope area; CoverMap averages across the entire field including flat toeslopes. Both are correct for what they measure. CoverMap reports include this disclaimer explicitly.</p>
    </div>
  </div>
</div>
</section>

<!-- ═══════════════════════════════════════════ CALIBRATION -->
<section id="calibration" class="alt">
<div class="inner">
  <p class="sec-label">Model Status</p>
  <h2 class="sec-title">Calibration Status and Known Limitations</h2>

  <div class="warning-box">
    <div class="warn-title">⚠ Pre-Calibration Model — Advisory Use Only</div>
    <p><strong>CoverMap's piecewise exponential C-factor parameters are initial estimates</strong> based on published RUSLE2 value ranges for Iowa cropland. They have not yet been validated against formal RUSLE2 Iowa State File runs for each residue system scenario.</p>
    <p><strong>Planned calibration:</strong> Comparison against RUSLE2 Iowa State File outputs for Monona silt loam map units (100D2, 100E2, 100F2) is in progress in coordination with Shelby County NRCS (W. Dittmer) and ISU Extension Agronomy (M. Licht). Parameters are subject to revision following this comparison.</p>
    <p><strong>Practical implication:</strong> Treat soil loss estimates and erosion reduction percentages as order-of-magnitude advisory figures — directionally correct but carrying an estimated ±10 percentage point uncertainty on reduction estimates. Do not use CoverMap outputs for formal NRCS conservation planning, payment calculations, or compliance determination. This report is advisory only and does not constitute an official NRCS determination.</p>
  </div>

  <h3 style="margin-top:1.5em; margin-bottom:.8em">Summary of Known Limitations</h3>
  <table>
    <thead><tr><th>Limitation</th><th>Priority</th><th>Planned Resolution</th></tr></thead>
    <tbody>
      <tr><td>C-factor parameters not RUSLE2-validated</td><td style="color:var(--c-crit);font-weight:600">HIGH</td><td>RUSLE2 comparison (Dittmer/Licht, 2026)</td></tr>
      <tr><td>Fixed 100 m slope length — no flow accumulation</td><td style="color:var(--c-high);font-weight:600">MODERATE</td><td>GIS-derived slope length (Phase 2)</td></tr>
      <tr><td>NDVI biomass proxy ±40% uncertainty</td><td style="color:var(--c-high);font-weight:600">MODERATE</td><td>Inherent sensor limitation; disclosed in reports</td></tr>
      <tr><td>No GDD adjustment for image date variation</td><td style="color:var(--c-mod);font-weight:600">MODERATE</td><td>NOAA GDD API integration planned</td></tr>
      <tr><td>NDVI cannot distinguish species or weeds</td><td style="color:var(--c-mod);font-weight:600">MODERATE</td><td>Field verification recommended before termination</td></tr>
      <tr><td>R-factor limited to 2-zone Iowa lookup</td><td style="color:var(--c-low);font-weight:600">LOW</td><td>County-level R from NRCS FOTG sufficient for advisory</td></tr>
    </tbody>
  </table>
</div>
</section>

<!-- ═══════════════════════════════════════════ REFERENCES -->
<section id="references">
<div class="inner references">
  <p class="sec-label">Literature</p>
  <h2 class="sec-title">References</h2>
  <ol>
    <li>Laflen, J.M. &amp; Roose, E.J. (1998). Methodologies for assessment of soil degradation due to water erosion. <em>Soil Degradation in the United States</em>, CRC Press. <em>Basis for Iowa NDVI–C-factor calibration ranges.</em></li>
    <li>McCool, D.K., Brown, L.C., Foster, G.R., Mutchler, C.K., &amp; Meyer, L.D. (1987). Revised slope steepness factor for the Universal Soil Loss Equation. <em>Transactions of the ASAE</em>, 30(5), 1387–1396. <em>Source of S-factor piecewise formula used in LS computation.</em></li>
    <li>Renard, K.G., Foster, G.R., Weesies, G.A., McCool, D.K., &amp; Yoder, D.C. (1997). <em>Predicting Soil Erosion by Water: A Guide to Conservation Planning with the Revised Universal Soil Loss Equation.</em> USDA Agriculture Handbook 703.</li>
    <li>Huddell, A.M., et al. (2024). National synthesis of cereal rye cover crop biomass and performance. <em>Nature Sustainability</em>. n = 5,695 field observations, mean 3,428 kg/ha. <em>Basis for NDVI–biomass estimates.</em></li>
    <li>Iowa NRCS. (2002). <em>FOTG Section I — USLE Erosion Prediction, Figure 2: Rainfall Factors</em>. Updated electronically September 2002. <em>Source for Iowa R-factor two-zone lookup (R = 150 NW Iowa; R = 175 remainder).</em></li>
    <li>USDA NRCS. (2023). <em>Conservation Practice Standard 340 — Cover Crop</em>. Iowa Supplement. <em>Stand documentation and minimum biomass requirements referenced throughout.</em></li>
    <li>ISU Extension. (2021). <em>PM-1209 — Cover Crop Management in Iowa</em>. Iowa State University Extension and Outreach. <em>C-factor context for Iowa cover crop species.</em></li>
    <li>European Space Agency. (2022). <em>Sentinel-2 Level-2A Product Definition</em>. ESA-EOPG-CSCOP-TN-0002. <em>Spectral band definitions, atmospheric correction, SCL cloud masking.</em></li>
    <li>Dittmer, W. (2026, in progress). <em>CoverMap C-Factor Calibration Against RUSLE2 Iowa State File — Monona Silt Loam, Shelby County</em>. USDA NRCS Shelby County Field Office. <em>Planned validation of exponential model parameters.</em></li>
    <li>USDA NRCS. (2024). <em>Web Soil Survey — SSURGO Database</em>. Soil Data Access API (SDMDataAccess.nrcs.usda.gov). <em>Source for K-factor (kwfact) and soil series identification.</em></li>
  </ol>
</div>
</section>

<!-- ═══════════════════════════════════════════ FOOTER -->
<footer>
  <strong style="color:rgba(255,255,255,.75)">CoverMap</strong> &nbsp;·&nbsp;
  Stephen Zimmerman, CCA MS &nbsp;·&nbsp; Ag Research Scientist &nbsp;·&nbsp; Ankeny, Iowa<br>
  Sentinel-2 imagery via Google Earth Engine &nbsp;·&nbsp; Iowa RUSLE C-factor methodology &nbsp;·&nbsp; Iowa DNR 3m DEM<br><br>
  Advisory only — does not constitute an official NRCS determination. Pre-calibration model parameters subject to revision.
</footer>

</body>
</html>"""

with open("covermap_explainer.html", "w", encoding="utf-8") as f:
    f.write(HTML)

size_kb = len(HTML.encode("utf-8")) // 1024
print(f"Written: covermap_explainer.html  ({size_kb} KB)")

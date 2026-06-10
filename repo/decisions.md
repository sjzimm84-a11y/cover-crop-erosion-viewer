# CoverMap Intake Form — Build Decisions

Decisions, tradeoffs, and gaps encountered while building the intake form that were not addressed in the spec.

---

## 1. File Upload Handling: Metadata Only

**Decision:** The uploaded Field Boundary File is validated for type but not saved to disk. Only file metadata (filename, MIME type, byte size) is captured in the JSON payload.

**Why:** The spec says "package all inputs into a JSON payload," but binary file content cannot be serialized to JSON natively. Options: (a) base64-encode and embed — rejected due to payload bloat; (b) save to disk and include path — rejected since no storage location was specified; (c) metadata only — chosen as the safe, non-destructive default.

**Action needed before production:** Add a storage backend (local disk, S3, Azure Blob, etc.) and include the storage reference in the payload.

---

## 2. File Validation: Extension Only, No Content Inspection

**Decision:** File type is validated by extension (`.geojson`, `.json`, `.zip`, `.kml`). Browser-supplied MIME types and file magic bytes are not checked.

**Why:** Browser MIME types are unreliable and can be spoofed. Extension checking matches the spec's intent and is consistent with how most intake tools operate.

**Limitation:** A user could rename any file with a `.geojson` extension and it would pass. For higher assurance, add server-side content parsing (e.g., attempt `json.load()` on GeoJSON/JSON files). Left as a future enhancement.

---

## 3. File Upload Size Limit: 50 MB

**Decision:** `MAX_CONTENT_LENGTH` is set to 50 MB. Flask returns HTTP 413 automatically if exceeded.

**Why:** Spec is silent on size limits. Spatial files — especially ZIP archives containing shapefiles — can be large. 50 MB covers most single-field boundary files without being unreasonably permissive.

**Action needed:** Adjust if real-world submissions are consistently larger or smaller.

---

## 4. Termination Date Format: ISO 8601 (YYYY-MM-DD)

**Decision:** Using HTML5 `<input type="date">`, which produces YYYY-MM-DD. Server validates against this format.

**Why:** Spec did not specify a date format. ISO 8601 is unambiguous, standard in JSON, and what the HTML5 date picker natively outputs.

---

## 5. Other Outputs: Checkboxes, Not `<select multiple>`

**Decision:** Rendered as three individual checkboxes, not an HTML `<select multiple>` element.

**Why:** For a small fixed set (3 options), checkboxes are significantly more usable — all options are visible simultaneously, no Ctrl+click required, and they work intuitively on mobile. `<select multiple>` is notoriously confusing on touchscreens.

---

## 6. Report Type: Radio Buttons, Not Dropdown

**Decision:** Rendered as radio buttons.

**Why:** Two mutually exclusive options are clearest as radio buttons — both choices are visible at once and the conditional CCA Name logic is more intuitive when the selection is explicit on screen.

---

## 7. CCA Name Field: Hidden When Not Applicable

**Decision:** The CCA Name field is hidden (`display: none`) when Report Type is not CCA. It appears via JavaScript when the CCA radio is selected. Server-side validation enforces the conditional requirement independently of JS state.

**Why:** A disabled-but-visible field creates confusion. Hiding it entirely removes ambiguity. If JavaScript is disabled, the field remains visible and server validation still enforces the rule correctly — no loss of correctness.

---

## 8. Tillage Practice Casing: Aligned to `RESIDUE_OPTIONS` in `src/scoring.py`

**Decision:** Dropdown values match the casing used in the main app's `RESIDUE_ADJUSTMENTS` dict: `No-till`, `Conservation tillage`, `Conventional tillage` (all-lowercase "tillage" throughout).

**Why:** The spec had inconsistent capitalization ("Conventional **T**illage"). The canonical source is `scoring.py`, where the combined labels use "conservation tillage" and "conventional tillage" in all-lowercase. The intake form now uses these same strings so values round-trip cleanly to the scoring model.

---

## 9. `submitted_at` Timestamp Added to Payload

**Decision:** A `submitted_at` field (UTC ISO 8601) is appended to every JSON payload.

**Why:** Not in spec, but intake records almost always need a submission timestamp for logging, auditing, and ordering. This is low-risk and high value for any downstream processing.

---

## 10. CSRF Protection: Not Implemented

**Decision:** No CSRF token on the form submission.

**Why:** Adding Flask-WTF or a manual CSRF implementation adds a dependency and complexity beyond the spec scope. Acceptable for an internal or trusted-network intake form.

**Action needed before public deployment:** Add CSRF protection (Flask-WTF is the standard approach).

---

## 11. POST-Redirect-GET Pattern: Not Used

**Decision:** On successful submission, the success view is rendered directly in the POST response rather than using a redirect.

**Why:** Simpler and keeps the JSON payload visible in the browser for transparency during testing. The tradeoff is that refreshing the success page re-submits the form — fine for internal use, but a redirect (`302 → GET`) would be better for a public-facing deployment.

---

## 12. "Intake Agent" Interpreted as Web Form

**Decision:** Built as a standard web form with server-side validation. No AI agent behavior, automated routing, or notification logic was implemented.

**Why:** The spec details (named fields, validation rules, JSON output) describe a structured intake form. If automated routing, email notifications, or AI-assisted field completion are desired, those would be separate features.

---

## 13. Bootstrap Loaded from CDN

**Decision:** Bootstrap 5.3.2 CSS loaded from jsDelivr CDN.

**Why:** No CSS build toolchain was specified. CDN is zero-config and sufficient for a form UI.

**Limitation:** Requires internet access to render styled correctly. For air-gapped or offline deployments, copy Bootstrap CSS locally.

---

## 14. File Input Cannot Be Repopulated After Validation Error

**Decision:** All text/select/radio/checkbox fields are repopulated from submitted values when the form is re-rendered after a validation error. The file input field is not repopulated.

**Why:** Browser security policy prevents pre-filling `<input type="file">` from server-side code. Users must re-select their file after any validation error. This is a browser constraint, not an implementation choice. Mitigating option: add client-side JS validation for the file field so file errors are caught before submission.

---

## 15. ZIP File Contents: Not Inspected

**Decision:** For `.zip` uploads, only the extension is checked. Archive contents are not validated.

**Why:** Shapefiles are distributed as ZIP archives containing `.shp`, `.dbf`, `.prj`, and other components. Validating ZIP contents would require knowing the expected internal structure — out of scope for an intake form. Recommended as a backend processing step after the file is received and stored.

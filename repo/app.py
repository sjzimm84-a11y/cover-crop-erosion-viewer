from flask import Flask, request, jsonify, render_template
import json
import re
from datetime import datetime, timezone
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

ALLOWED_EXTENSIONS = {"geojson", "json", "zip", "kml"}
VALID_CROPS = ("Corn", "Soybeans")
VALID_TILLAGE = ("No-till", "Conservation tillage", "Conventional tillage")
VALID_REPORT_TYPES = ("Producer", "CCA")
VALID_OTHER_OUTPUTS = {"Multi-Year", "SHP File", "Paper Copy"}

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    errors = {}
    form = request.form

    # Requestor Name
    requestor_name = form.get("requestor_name", "").strip()
    if not requestor_name:
        errors["requestor_name"] = "Requestor Name is required."

    # Email
    email = form.get("email", "").strip()
    if not email:
        errors["email"] = "Email is required."
    elif not _EMAIL_RE.match(email):
        errors["email"] = "Please enter a valid email address."

    # Farm Name (optional)
    farm_name = form.get("farm_name", "").strip() or None

    # Previous Crop
    previous_crop = form.get("previous_crop", "").strip()
    if not previous_crop:
        errors["previous_crop"] = "Previous Crop is required."
    elif previous_crop not in VALID_CROPS:
        errors["previous_crop"] = "Previous Crop must be Corn or Soybeans."

    # Tillage Practice
    tillage_practice = form.get("tillage_practice", "").strip()
    if not tillage_practice:
        errors["tillage_practice"] = "Tillage Practice is required."
    elif tillage_practice not in VALID_TILLAGE:
        errors["tillage_practice"] = "Invalid Tillage Practice selection."

    # Termination Date (optional)
    termination_date_raw = form.get("termination_date", "").strip()
    termination_date = None
    if termination_date_raw:
        try:
            datetime.strptime(termination_date_raw, "%Y-%m-%d")
            termination_date = termination_date_raw
        except ValueError:
            errors["termination_date"] = "Termination Date must be a valid date."

    # Report Type
    report_type = form.get("report_type", "").strip()
    if not report_type:
        errors["report_type"] = "Report Type is required."
    elif report_type not in VALID_REPORT_TYPES:
        errors["report_type"] = "Report Type must be Producer or CCA."

    # CCA Name (required only when Report Type = CCA)
    cca_name = form.get("cca_name", "").strip() or None
    if report_type == "CCA" and not cca_name:
        errors["cca_name"] = "CCA Name is required when Report Type is CCA."

    # Other Outputs (optional multi-select)
    other_outputs = form.getlist("other_outputs")
    invalid_outputs = set(other_outputs) - VALID_OTHER_OUTPUTS
    if invalid_outputs:
        errors["other_outputs"] = (
            f"Invalid output option(s): {', '.join(sorted(invalid_outputs))}."
        )

    # Field Boundary File
    file = request.files.get("field_boundary_file")
    file_meta = None
    if file is None or file.filename == "":
        errors["field_boundary_file"] = "Field Boundary File is required."
    elif not _allowed_file(file.filename):
        errors["field_boundary_file"] = (
            "Unsupported file type. Please upload a GEOJSON, JSON, ZIP, or KML file."
        )
    else:
        safe_name = secure_filename(file.filename)
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)
        file_meta = {
            "filename": safe_name,
            "content_type": file.content_type,
            "size_bytes": file_size,
        }

    if errors:
        return render_template("index.html", errors=errors, form_data=form), 400

    payload = {
        "requestor_name": requestor_name,
        "email": email,
        "farm_name": farm_name,
        "previous_crop": previous_crop,
        "tillage_practice": tillage_practice,
        "termination_date": termination_date,
        "report_type": report_type,
        "cca_name": cca_name if report_type == "CCA" else None,
        "other_outputs": other_outputs,
        "field_boundary_file": file_meta,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }

    print("\n=== CoverMap Intake Submission ===")
    print(json.dumps(payload, indent=2))
    print("==================================\n")

    return render_template("index.html", success=True, payload=payload)


if __name__ == "__main__":
    app.run(debug=True)

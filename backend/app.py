"""
app.py — SoundMatch AI — Flask Application
==========================================
Endpoints:
  GET  /                     → Serve frontend
  POST /api/analyze          → Analyse image + optional crop
  POST /api/recommend        → Full pipeline (analyse + recommend)
  GET  /api/health           → Health check + library stats
  GET  /api/library/stats    → Song count per language
"""

from __future__ import annotations
import io
import os
import sys
import traceback

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Ensure backend package is importable
sys.path.insert(0, os.path.dirname(__file__))

from analyzer   import analyse_image, AnalysisResult
from recommender import recommend, _load_library

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Flask setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "templates")
STATIC_DIR   = os.path.join(os.path.dirname(__file__), "..", "frontend", "static")

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR,
)
CORS(app)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MAX_IMAGE_BYTES   = 15 * 1024 * 1024   # 15 MB
ALLOWED_MIME      = {"image/jpeg", "image/jpg", "image/png",
                     "image/gif", "image/webp", "image/bmp"}
SUPPORTED_LANGS   = {"english", "hindi", "punjabi"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _read_image_file(field_name: str) -> bytes | None:
    """Read and validate an uploaded image field. Returns bytes or None."""
    # Handle JSON body with base64 data
    if request.is_json:
        data = request.get_json()
        if data and field_name in data:
            import base64
            img_str = data[field_name]
            if img_str.startswith("data:"):
                # Remove data URL prefix (e.g., "data:image/jpeg;base64,")
                img_str = img_str.split(",", 1)[1]
            try:
                return base64.b64decode(img_str)
            except Exception:
                return None
        return None
    
    # Handle multipart/form-data file upload
    if field_name not in request.files:
        return None
    f = request.files[field_name]
    if not f or f.filename == "":
        return None
    data = f.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image '{field_name}' exceeds 15 MB limit.")
    return data


def _analysis_to_dict(a: AnalysisResult) -> dict:
    return {
        "mood":            a.mood,
        "secondary_moods": a.secondary_moods,
        "context":         a.context,
        "confidence":      a.confidence,
        "is_cropped":      a.is_cropped,
        "color": {
            "brightness":   a.color.brightness,
            "saturation":   a.color.saturation,
            "contrast":     a.color.contrast,
            "sharpness":    a.color.sharpness,
            "warm_ratio":   a.color.warm_ratio,
            "cool_ratio":   a.color.cool_ratio,
            "dark_ratio":   a.color.dark_ratio,
            "dominant_temp": a.color.dominant_temp,
        },
        "aesthetic": {
            "brightness_score": a.aesthetic.brightness_score,
            "contrast_score":   a.aesthetic.contrast_score,
            "saturation_score": a.aesthetic.saturation_score,
            "sharpness_score":  a.aesthetic.sharpness_score,
            "overall":          a.aesthetic.overall,
            "grade":            a.aesthetic.grade,
        },
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Routes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    POST /api/analyze
    Fields:
      image (file)       — full uploaded image [required]
      crop  (file)       — cropped region      [optional]
    """
    try:
        full_bytes = _read_image_file("image")
        if not full_bytes:
            return jsonify({"success": False, "error": "No image file provided."}), 400
        crop_bytes = _read_image_file("crop")
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 413

    try:
        result = analyse_image(full_bytes, crop_bytes)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 422
    except Exception:
        traceback.print_exc()
        return jsonify({"success": False, "error": "Image analysis failed."}), 500

    return jsonify({"success": True, **_analysis_to_dict(result)})


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """
    POST /api/recommend
    Fields:
      image    (file)    — full uploaded image [required]
      crop     (file)    — cropped region      [optional]
      language (form)    — english|hindi|punjabi [default: english]
      top_n    (form)    — integer >= 7         [default: 7]
    """
    try:
        full_bytes = _read_image_file("image")
        if not full_bytes:
            return jsonify({"success": False, "error": "No image file provided."}), 400
        crop_bytes = _read_image_file("crop")
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 413

    language = "english"
    top_n = 7
    
    # Handle both JSON and form data
    if request.is_json:
        data = request.get_json()
        if data:
            language = data.get("language", "english").lower()
            try:
                top_n = max(7, int(data.get("top_n", 7)))
            except (ValueError, TypeError):
                top_n = 7
    else:
        language = request.form.get("language", "english").lower()
        try:
            top_n = max(7, int(request.form.get("top_n", 7)))
        except (ValueError, TypeError):
            top_n = 7
    
    if language not in SUPPORTED_LANGS:
        language = "english"

    # ── Analysis ──
    try:
        analysis = analyse_image(full_bytes, crop_bytes)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 422
    except Exception:
        traceback.print_exc()
        return jsonify({"success": False, "error": "Image analysis failed."}), 500

    # ── Recommendations ──
    try:
        songs = recommend(analysis, language=language, top_n=top_n)
    except Exception:
        traceback.print_exc()
        return jsonify({"success": False, "error": "Recommendation engine failed."}), 500

    return jsonify({
        "success":         True,
        "language":        language,
        "analysis":        _analysis_to_dict(analysis),
        "recommendations": songs,
    })


@app.route("/api/health")
def api_health():
    stats = {}
    for lang in SUPPORTED_LANGS:
        try:
            stats[lang] = len(_load_library(lang))
        except Exception:
            stats[lang] = 0
    return jsonify({"status": "ok", "library": stats})


@app.route("/api/library/stats")
def api_library_stats():
    stats = {}
    for lang in SUPPORTED_LANGS:
        try:
            lib = _load_library(lang)
            stats[lang] = {
                "count":       len(lib),
                "avg_trend":   round(sum(s.get("trend_score", 0) for s in lib) / max(len(lib), 1), 1),
                "verified":    sum(1 for s in lib if s.get("verified")),
            }
        except Exception:
            stats[lang] = {"count": 0, "avg_trend": 0, "verified": 0}
    return jsonify(stats)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dev server entry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
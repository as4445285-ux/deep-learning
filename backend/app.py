"""
app.py  –  Flask REST API for the CaptionFlow Image Captioning System.

Endpoints
---------
GET  /                  – Serve frontend (index.html)
GET  /health            – Health check
POST /api/caption       – Upload image → returns caption + metadata
GET  /api/history       – Return in-memory caption history (last 50)
DELETE /api/history     – Clear history
"""

import io
import os
import sys
import time
import uuid
import base64
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

# Allow running from project root
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from inference import CaptionPredictor

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(_ROOT, "frontend")
MAX_IMAGE_MB = 10
MODEL_NAME = os.environ.get("CAPTION_MODEL", "Salesforce/blip-image-captioning-base")

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

# In-memory history store (last 50 entries)
_history: list[dict] = []

# Load the pretrained model once at startup
print("[app] Loading BLIP captioning model …")
try:
    predictor = CaptionPredictor(MODEL_NAME)
except Exception as exc:
    predictor = None
    print(f"[app] Model failed to load: {exc}\n")

print("[app] Ready ✔\n") if predictor else print("[app] App ready in degraded mode; model downloads are required if the model is unavailable.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def _allowed(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


def _pil_to_b64(img: Image.Image, max_size: int = 300) -> str:
    """Resize and base64-encode a thumbnail for the history panel."""
    thumb = img.copy()
    thumb.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/health")
def health():
    payload = {
        "status": "ok",
        "model_ready": bool(predictor and predictor.ready),
        "model_name": MODEL_NAME,
        "device": str(getattr(predictor, "device", "unknown")),
        "error": getattr(predictor, "error", None),
        "timestamp": datetime.utcnow().isoformat(),
    }
    return jsonify(payload)


@app.route("/api/caption", methods=["POST"])
def caption_image():
    if predictor is None or not predictor.ready:
        return jsonify({
            "error": "Model is not available. Download the Hugging Face model first, then restart the app.",
            "details": getattr(predictor, "error", None) or "Unknown model loading error.",
        }), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use field name 'image'."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400
    if not _allowed(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 415

    raw = file.read()
    if len(raw) > MAX_IMAGE_MB * 1024 * 1024:
        return jsonify({"error": f"File too large (max {MAX_IMAGE_MB} MB)."}), 413

    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        return jsonify({"error": f"Cannot open image: {exc}"}), 422

    temperature = float(request.form.get("temperature", 1.0))
    temperature = max(0.1, min(2.0, temperature))
    max_len = int(request.form.get("max_len", 25))
    max_len = max(5, min(50, max_len))

    try:
        t0 = time.time()
        caption = predictor.predict_from_pil(pil_image, temperature=temperature, max_len=max_len)
        elapsed = round((time.time() - t0) * 1000, 1)
    except Exception as exc:
        return jsonify({
            "error": "Caption generation failed during inference.",
            "details": str(exc),
        }), 500

    entry = {
        "id": str(uuid.uuid4()),
        "caption": caption,
        "filename": file.filename,
        "width": pil_image.width,
        "height": pil_image.height,
        "elapsed_ms": elapsed,
        "temperature": temperature,
        "demo_mode": False,
        "timestamp": datetime.utcnow().isoformat(),
        "thumbnail": _pil_to_b64(pil_image),
    }

    _history.insert(0, entry)
    if len(_history) > 50:
        _history.pop()

    return jsonify(entry), 200


@app.route("/api/history", methods=["GET"])
def get_history():
    return jsonify(_history), 200


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    _history.clear()
    return jsonify({"message": "History cleared."}), 200


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀  Starting CaptionFlow API on http://localhost:{port}")
    print(f"   Frontend : http://localhost:{port}/")
    print(f"   Health   : http://localhost:{port}/health")
    print(f"   API      : http://localhost:{port}/api/caption\n")
    app.run(host="0.0.0.0", port=port, debug=False)

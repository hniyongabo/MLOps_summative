"""Bean leaf disease web app: dashboard with data insights, prediction page and retraining page.

Run locally:   python app.py           -> http://localhost:7860
Production:    gunicorn app:app        (used by the Dockerfile)
"""
import base64
import io
import os
import threading
import time
import traceback

from flask import Flask, jsonify, render_template, request
from PIL import Image

from src import database
from src.insights import dataset_insights, model_insights
from src.model import load_metadata
from src.prediction import get_model, predict_image
from src.preprocessing import CLASS_NAMES, DISPLAY_NAMES
from src.retrain import ingest_upload, preprocess_db_images, retrain

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB uploads
START_TIME = time.time()
DATA_INSIGHTS = dataset_insights()
MODEL_INSIGHTS = model_insights()
JOB = {"state": "idle", "log": [], "progress": 0, "result": None, "error": None}
JOB_LOCK = threading.Lock()


def status():
    up = int(time.time() - START_TIME)
    return {"status": "online", "uptime": f"{up // 3600}h {up % 3600 // 60}m {up % 60}s",
            "model": load_metadata(), "database": database.stats(), "retraining": JOB["state"]}


# ---------- pages ----------
@app.route("/")
def dashboard():
    return render_template("dashboard.html", s=status(), data=DATA_INSIGHTS, mi=MODEL_INSIGHTS,
                           page="dashboard", v=int(time.time()))


@app.route("/predict")
def predict_page():
    return render_template("predict.html", s=status(), page="predict")


@app.route("/retrain")
def retrain_page():
    return render_template("retrain.html", s=status(), page="retrain", classes=CLASS_NAMES)


# ---------- API ----------
@app.get("/api/status")
def api_status():
    return jsonify(status())


@app.post("/api/predict")
def api_predict():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="Please choose an image file."), 400
    try:
        return jsonify(predict_image(f.read()))
    except Exception as e:
        return jsonify(error=f"Could not read this image: {e}"), 400


@app.post("/api/upload")
def api_upload():
    """Retraining trigger 1: upload a data file and save its images to the database."""
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".zip"):
        return jsonify(error="Please upload a .zip file with folders angular_leaf_spot, bean_rust, healthy."), 400
    try:
        result = ingest_upload(f.filename, f.read())
    except Exception as e:
        return jsonify(error=str(e)), 400
    result["database"] = database.stats()
    return jsonify(result)


def _thumb(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).resize((96, 96)).save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@app.post("/api/preprocess")
def api_preprocess():
    """Retraining trigger 2: preprocess the new images stored in the database."""
    X, y, _ = preprocess_db_images(only_unused=True)
    if not len(y):
        return jsonify(error="No new data in the database. Upload a file first."), 400
    examples = []
    for label in sorted(set(y.tolist())):
        idx = [i for i, v in enumerate(y) if v == label][:3]
        examples += [{"label": DISPLAY_NAMES[CLASS_NAMES[label]], "img": _thumb(X[i])} for i in idx]
    return jsonify({"n_images": int(len(y)), "array_shape": list(X.shape), "dtype": str(X.dtype),
                    "per_class": {DISPLAY_NAMES[c]: int((y == i).sum()) for i, c in enumerate(CLASS_NAMES)},
                    "examples": examples,
                    "steps": ["EXIF rotation", "RGB conversion", "resize to 224x224", "array 0-255 (scaled to -1..1 inside the model)"]})


def _run_retraining():
    global MODEL_INSIGHTS

    def progress(msg, pct=None):
        JOB["log"].append(f"{time.strftime('%H:%M:%S')}  {msg}")
        if pct is not None:
            JOB["progress"] = pct
    try:
        result = retrain(progress)
        if result["promoted"]:
            progress("Loading the new model into the prediction service", 98)
            get_model(reload=True)
        MODEL_INSIGHTS = model_insights()
        JOB.update(state="done", result=result, progress=100)
        progress("Done")
    except Exception as e:
        traceback.print_exc()
        JOB.update(state="error", error=str(e))


@app.post("/api/retrain")
def api_retrain():
    """Retraining trigger 3: start fine-tuning the pre-trained model in the background."""
    with JOB_LOCK:
        if JOB["state"] == "running":
            return jsonify(error="Retraining is already running."), 409
        if database.stats()["unused"] == 0:
            return jsonify(error="There is no new data in the database. Upload a file first."), 400
        JOB.update(state="running", log=[], progress=0, result=None, error=None)
        threading.Thread(target=_run_retraining, daemon=True).start()
    return jsonify(state="running")


@app.get("/api/retrain/status")
def api_retrain_status():
    return jsonify(JOB)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), debug=False, threaded=True)

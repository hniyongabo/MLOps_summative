"""Retraining pipeline.

Trigger 1 - Upload + save to database : `ingest_upload()`        (zip -> SQLite)
Trigger 2 - Preprocessing of new data : `preprocess_db_images()`  (DB bytes -> 224x224 RGB arrays)
Trigger 3 - Retraining                : `retrain()` loads OUR saved model (models/bean_model.keras) as the
                                         pre-trained starting point and fine-tunes it on the new data plus a
                                         sample of the original data (so it does not forget), with early stopping.
"""
import os

import numpy as np

from src import database
from src.model import evaluate, fine_tune, load_metadata, load_model, save_model
from src.preprocessing import CLASS_NAMES, load_image, load_image_folder, read_labelled_zip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
REPLAY_PER_CLASS = 100   # original training images mixed in per class


def ingest_upload(filename, file_bytes):
    """Step 1: read the uploaded zip and store every labelled image in the database."""
    items = read_labelled_zip(file_bytes)
    if not items:
        raise ValueError("No labelled images found. The zip must contain folders named "
                         "angular_leaf_spot, bean_rust and/or healthy with images inside.")
    upload_id = database.save_upload(filename, items)
    counts = {c: 0 for c in CLASS_NAMES}
    for label, _, _ in items:
        counts[CLASS_NAMES[label]] += 1
    return {"upload_id": upload_id, "n_images": len(items), "per_class": counts}


def preprocess_db_images(only_unused=True):
    """Step 2: preprocess images stored in the database (same function as for prediction)."""
    rows = database.fetch_images(only_unused=only_unused)
    if not rows:
        return np.empty((0, 224, 224, 3), np.uint8), np.empty(0, int), []
    X = np.stack([load_image(bytes(r["image"])) for r in rows])
    y = np.array([r["label"] for r in rows], dtype=int)
    return X, y, [r["id"] for r in rows]

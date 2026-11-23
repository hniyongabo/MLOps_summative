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


def retrain(progress=lambda msg, pct=None: None):
    """Step 3: fine-tune the existing (pre-trained) model on the new data."""
    progress("Preprocessing new images from the database", 5)
    X_new, y_new, ids = preprocess_db_images(only_unused=True)
    if len(y_new) == 0:
        raise ValueError("There is no new data in the database. Upload a file first.")

    progress("Loading original training sample, validation and test sets", 15)
    X_old, y_old, _ = load_image_folder(os.path.join(DATA, "train"), max_per_class=REPLAY_PER_CLASS)
    X_val, y_val, _ = load_image_folder(os.path.join(DATA, "validation"))
    X_test, y_test, _ = load_image_folder(os.path.join(DATA, "test"))
    X_prev, y_prev, _ = preprocess_db_images(only_unused=False)   # all uploads (new + earlier ones)
    X_train = np.concatenate([X_old, X_prev]).astype(np.float32)
    y_train = np.concatenate([y_old, y_prev])

    progress("Loading the current model (pre-trained model) and scoring it on the test set", 25)
    model = load_model()
    before = evaluate(model, X_test.astype(np.float32), y_test)

    def on_epoch(epoch, total, logs):
        progress(f"Training epoch {epoch}/{total}: loss {logs.get('loss', 0):.3f}, "
                 f"val_loss {logs.get('val_loss', 0):.3f}, val_accuracy {logs.get('val_accuracy', 0):.3f}",
                 30 + int(55 * epoch / total))

    progress(f"Fine-tuning on {len(y_train)} images ({len(y_new)} new)", 30)
    model, history = fine_tune(model, X_train, y_train, X_val.astype(np.float32), y_val, on_epoch=on_epoch)

    progress("Evaluating the retrained model on the test set", 88)
    after = evaluate(model, X_test.astype(np.float32), y_test)

    old_meta = load_metadata()
    version = f"v{int(str(old_meta.get('version', 'v1')).lstrip('v')) + 1}"
    # Deploy the new model only if it is at least as accurate on the held-out test set
    promoted = after["accuracy"] >= before["accuracy"]
    if promoted:
        progress("Saving and deploying the new model version", 95)
        meta = dict(old_meta, version=version, test_metrics=after,
                    last_retrain={"new_images": int(len(y_new)), "training_images": int(len(y_train)),
                                  "epochs": len(history.get("loss", []))})
        save_model(model, meta)
    database.mark_used(ids)
    epochs = len(history.get("loss", []))
    database.log_run(n_new_images=len(y_new), epochs=epochs, old_accuracy=before["accuracy"],
                     new_accuracy=after["accuracy"], new_f1=after["f1_score"],
                     model_version=version if promoted else old_meta.get("version"), promoted=promoted)
    return {"n_new_images": int(len(y_new)), "n_training_images": int(len(y_train)), "epochs_run": epochs,
            "before": before, "after": after, "promoted": promoted,
            "version": version if promoted else old_meta.get("version"),
            "history": {k: [round(float(v), 4) for v in vals] for k, vals in history.items()}}

"""Model loading/saving, evaluation metrics and fine-tuning (TensorFlow / Keras)."""
import json
import os
import shutil
from datetime import datetime

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)

from src.preprocessing import CLASS_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "bean_model.keras")
META_PATH = os.path.join(MODEL_DIR, "model_metadata.json")
N_CLASSES = len(CLASS_NAMES)


def load_model(path=MODEL_PATH):
    from tensorflow import keras  # imported lazily: TensorFlow is slow to import
    return keras.models.load_model(path)


def predict_proba(model, X, batch_size=32):
    return np.asarray(model.predict(X, batch_size=batch_size, verbose=0))


def evaluate(model, X, y):
    """The evaluation metrics used everywhere in the project."""
    proba = predict_proba(model, X)
    pred = proba.argmax(axis=1)
    labels = list(range(N_CLASSES))
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "loss": round(float(log_loss(y, np.clip(proba, 1e-7, 1), labels=labels)), 4),
        "precision": round(float(precision_score(y, pred, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, average="macro", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
        "per_class_f1": [round(float(v), 4) for v in f1_score(y, pred, average=None, labels=labels, zero_division=0)],
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
    }


def load_metadata():
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    return {}


def save_model(model, metadata, path=MODEL_PATH):
    """Save the model; the previous version is archived in models/archive/."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(path):
        archive = os.path.join(MODEL_DIR, "archive")
        os.makedirs(archive, exist_ok=True)
        old_version = load_metadata().get("version", "unknown")
        shutil.copy(path, os.path.join(archive, f"bean_model_{old_version}.keras"))
    model.save(path)
    metadata = dict(metadata, saved_at=datetime.now().isoformat(timespec="seconds"))
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)


def find_base_model(model):
    """Return the MobileNetV2 feature extractor nested inside our model."""
    for layer in model.layers:
        if hasattr(layer, "layers") and len(layer.layers) > 20:
            return layer
    return None


def fine_tune(model, X_train, y_train, X_val, y_val, max_epochs=8, learning_rate=1e-4,
              patience=3, batch_size=16, on_epoch=None):
    """Continue training an already-trained model on new data.

    - the MobileNetV2 feature extractor stays frozen (fast on CPU, keeps ImageNet features)
    - only the classification head is updated, with a small learning rate (Adam)
    - early stopping on validation loss restores the best weights
    """
    from tensorflow import keras

    base = find_base_model(model)
    if base is not None:
        base.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)]
    if on_epoch:
        callbacks.append(keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: on_epoch(epoch + 1, max_epochs, logs or {})))
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=max_epochs,
                        batch_size=batch_size, shuffle=True, callbacks=callbacks, verbose=0)
    return model, history.history

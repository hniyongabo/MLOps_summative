"""Prediction: one leaf image in -> predicted class + confidence out."""
import threading

import numpy as np

from src.model import load_model, predict_proba
from src.preprocessing import CLASS_NAMES, DISPLAY_NAMES, load_image, to_batch

ADVICE = {
    "angular_leaf_spot": "Fungal disease: angular brown spots bounded by leaf veins. Remove infected leaves, avoid "
                         "working in the field when plants are wet, rotate crops and use clean seed.",
    "bean_rust": "Fungal disease: small reddish-brown powdery pustules on the leaves. Remove infected plant "
                 "debris, space plants for air flow and consider rust-resistant varieties.",
    "healthy": "No disease detected. Keep monitoring the crop regularly.",
}

_model = None
_lock = threading.Lock()


def get_model(reload=False):
    global _model
    with _lock:
        if _model is None or reload:
            _model = load_model()
        return _model


def predict_image(image_source):
    model = get_model()
    proba = predict_proba(model, to_batch([load_image(image_source)]))[0]
    order = np.argsort(proba)[::-1]
    top = CLASS_NAMES[order[0]]
    return {
        "predicted_class": top,
        "display_name": DISPLAY_NAMES[top],
        "confidence": round(float(proba[order[0]]), 4),
        "probabilities": [{"class": CLASS_NAMES[i], "display_name": DISPLAY_NAMES[CLASS_NAMES[i]],
                           "probability": round(float(proba[i]), 4)} for i in order],
        "advice": ADVICE[top],
    }

"""Image preprocessing shared by the notebook, the prediction API and the retraining pipeline.

Every image, whether used for training, prediction or retraining, goes through `load_image()`:
  1. open the file and apply the phone's EXIF rotation (photos taken in portrait stay upright)
  2. convert to RGB (drops transparency / grayscale differences)
  3. resize to 224x224 pixels, the input size of MobileNetV2
  4. return a uint8 array (224, 224, 3)

Pixel scaling to [-1, 1] (what MobileNetV2 expects) is done by a Rescaling layer INSIDE the model,
so it can never be forgotten or done differently at prediction time.
"""
import io
import os
import re
import zipfile

import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = 224
CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]
DISPLAY_NAMES = {"angular_leaf_spot": "Angular leaf spot", "bean_rust": "Bean rust", "healthy": "Healthy"}
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_image(source, size=IMG_SIZE):
    """Path, raw bytes or PIL image -> uint8 RGB array of shape (size, size, 3)."""
    if isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source))
    elif isinstance(source, Image.Image):
        img = source
    else:
        img = Image.open(source)
    img = ImageOps.exif_transpose(img).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def to_batch(images):
    """List of uint8 arrays -> float32 batch (N, 224, 224, 3) with values 0-255."""
    return np.stack(images).astype(np.float32)

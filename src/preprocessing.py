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


def label_from_name(name):
    """Map a folder name to a class index. Accepts 'bean_rust', 'Bean Rust', 'bean-rust' or '1'."""
    key = re.sub(r"[\s\-]+", "_", name.strip().lower())
    if key in CLASS_NAMES:
        return CLASS_NAMES.index(key)
    if key.isdigit() and int(key) < len(CLASS_NAMES):
        return int(key)
    return None


def load_image_folder(folder, max_per_class=None, seed=0):
    """Load <folder>/<class_name>/<images>. Returns X (uint8 N,224,224,3), y (int), file paths."""
    rng = np.random.RandomState(seed)
    X, y, paths = [], [], []
    if not os.path.isdir(folder):
        return np.empty((0, IMG_SIZE, IMG_SIZE, 3), np.uint8), np.empty(0, int), []
    for cls in sorted(os.listdir(folder)):
        label = label_from_name(cls)
        sub = os.path.join(folder, cls)
        if label is None or not os.path.isdir(sub):
            continue
        files = sorted(f for f in os.listdir(sub) if f.lower().endswith(VALID_EXT))
        if max_per_class and len(files) > max_per_class:
            files = list(rng.choice(files, max_per_class, replace=False))
        for f in files:
            p = os.path.join(sub, f)
            X.append(load_image(p))
            y.append(label)
            paths.append(p)
    if not X:
        return np.empty((0, IMG_SIZE, IMG_SIZE, 3), np.uint8), np.empty(0, int), []
    return np.stack(X), np.array(y, dtype=int), paths


def read_labelled_zip(zip_bytes):
    """Read an uploaded .zip whose folders are named after the class
    (angular_leaf_spot/, bean_rust/, healthy/). Returns [(label_index, filename, raw_bytes)].
    The raw bytes are stored in the database; preprocessing happens when the data is used."""
    items = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for name in z.namelist():
            if name.endswith("/") or "__MACOSX" in name or not name.lower().endswith(VALID_EXT):
                continue
            parts = [p for p in name.replace("\\", "/").split("/") if p]
            if len(parts) < 2:
                continue
            label = label_from_name(parts[-2])
            if label is None:
                continue  # skip files that are not inside a class folder
            data = z.read(name)
            try:
                Image.open(io.BytesIO(data)).verify()   # skip corrupt / non-image files
            except Exception:
                continue
            items.append((label, parts[-1], data))
    return items

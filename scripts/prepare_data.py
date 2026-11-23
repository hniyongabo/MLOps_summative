"""Download the iBean dataset and organise it into the project folders (same steps as sections 1-2
of the notebook). Only needed if you want to rebuild data/ locally instead of using the Colab output.

    python scripts/prepare_data.py

Creates:
    data/train/<class>/         75 % of the official train split -> builds the model
    data/retrain_pool/<class>/  25 % of the official train split -> "new data" for retraining
    data/validation/<class>/    official validation split -> early stopping / tuning
    data/test/<class>/          official test split -> final evaluation
All images are resized to 224x224 JPEG.
"""
import os
import shutil
import sys
import urllib.request
import zipfile

from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.preprocessing import CLASS_NAMES, load_image  # noqa: E402

RAW = os.path.join(ROOT, "raw")
DATA = os.path.join(ROOT, "data")
SPLITS = ["train", "validation", "test"]
SEED = 42


def download():
    os.makedirs(RAW, exist_ok=True)
    for split in SPLITS:
        target = os.path.join(RAW, f"{split}_extracted")
        if os.path.isdir(target):
            continue
        zpath = os.path.join(RAW, f"{split}.zip")
        print("downloading", split)
        urllib.request.urlretrieve(f"https://storage.googleapis.com/ibeans/{split}.zip", zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(target)


def collect(split):
    items = []
    for dirpath, _, files in os.walk(os.path.join(RAW, f"{split}_extracted")):
        cls = os.path.basename(dirpath)
        if cls in CLASS_NAMES:
            items += [(os.path.join(dirpath, f), CLASS_NAMES.index(cls))
                      for f in sorted(files) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return items


if __name__ == "__main__":
    download()
    for s in SPLITS:
        print(s, len(collect(s)), "images")

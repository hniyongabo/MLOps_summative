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


def export(items, split):
    for path, label in items:
        folder = os.path.join(DATA, split, CLASS_NAMES[label])
        os.makedirs(folder, exist_ok=True)
        name = os.path.splitext(os.path.basename(path))[0] + ".jpg"
        Image.fromarray(load_image(path)).save(os.path.join(folder, name), quality=92)


def main():
    download()
    raw = {s: collect(s) for s in SPLITS}
    shutil.rmtree(DATA, ignore_errors=True)
    train_items, pool_items = train_test_split(raw["train"], test_size=0.25, random_state=SEED,
                                               stratify=[label for _, label in raw["train"]])
    export(train_items, "train")
    export(pool_items, "retrain_pool")
    export(raw["validation"], "validation")
    export(raw["test"], "test")
    print(f"train={len(train_items)} retrain_pool={len(pool_items)} "
          f"validation={len(raw['validation'])} test={len(raw['test'])}")


if __name__ == "__main__":
    main()

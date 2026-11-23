"""Data-insight charts for the dashboard. Dataset charts use only the image files (no TensorFlow);
model charts use the metrics stored in models/model_metadata.json, so they refresh after retraining."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.model import load_metadata
from src.preprocessing import CLASS_NAMES, DISPLAY_NAMES, VALID_EXT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "static", "insights")
HISTORY = os.path.join(ROOT, "models", "training_history.json")
COLORS = {"angular_leaf_spot": "#B7791F", "bean_rust": "#C0392B", "healthy": "#2F855A"}
LABELS = [DISPLAY_NAMES[c] for c in CLASS_NAMES]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=110, bbox_inches="tight")
    plt.close(fig)


def _files(split, cls):
    folder = os.path.join(DATA, split, cls)
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(VALID_EXT))


def dataset_insights():
    """Charts from the image files. Returns numbers used in the dashboard text."""
    os.makedirs(OUT, exist_ok=True)
    splits = ["train", "retrain_pool", "validation", "test"]
    counts = {s: [len(_files(s, c)) for c in CLASS_NAMES] for s in splits}

    # 1. images per class and split
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    w = 0.2
    for k, s in enumerate(splits):
        ax.bar(np.arange(3) + (k - 1.5) * w, counts[s], w, label=s.replace("_", " "))
    ax.set_xticks(range(3)); ax.set_xticklabels(LABELS); ax.set_ylabel("Images")
    ax.set_title("Images per class and split"); ax.legend(fontsize=8, frameon=False)
    _save(fig, "class_distribution.png")

    # 2. example leaves
    fig, axes = plt.subplots(3, 5, figsize=(8, 5))
    for r, c in enumerate(CLASS_NAMES):
        files = _files("train", c)[:5]
        for k in range(5):
            axes[r, k].axis("off")
            if k < len(files):
                axes[r, k].imshow(Image.open(files[k]))
        axes[r, 0].set_title(DISPLAY_NAMES[c], loc="left", fontweight="bold", color=COLORS[c])
    fig.tight_layout()
    _save(fig, "examples.png")

    # 3. colour analysis: greenness and brightness per class
    green, bright = {}, {}
    for c in CLASS_NAMES:
        g, b = [], []
        for f in _files("train", c):
            px = np.asarray(Image.open(f).convert("RGB").resize((64, 64)), dtype=np.float32).reshape(-1, 3)
            rgb = px.mean(axis=0)
            g.append(rgb[1] / max(rgb.sum(), 1e-6))
            b.append(rgb.mean())
        green[c], bright[c] = g, b
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    for a, data, title in [(ax[0], green, "Greenness (share of green)"), (ax[1], bright, "Brightness (0-255)")]:
        bp = a.boxplot([data[c] or [0] for c in CLASS_NAMES], tick_labels=LABELS, patch_artist=True)
        for patch, c in zip(bp["boxes"], CLASS_NAMES):
            patch.set_facecolor(COLORS[c]); patch.set_alpha(0.55)
        a.set_title(title)
    fig.tight_layout()
    _save(fig, "colour_analysis.png")

    mean_green = {c: float(np.mean(green[c])) if green[c] else 0.0 for c in CLASS_NAMES}
    train_counts = counts["train"]
    return {
        "counts": counts,
        "total": int(sum(sum(v) for v in counts.values())),
        "min_class": int(min(train_counts)), "max_class": int(max(train_counts)),
        "greenest": DISPLAY_NAMES[max(mean_green, key=mean_green.get)],
        "least_green": DISPLAY_NAMES[min(mean_green, key=mean_green.get)],
        "mean_green": {DISPLAY_NAMES[c]: round(v, 3) for c, v in mean_green.items()},
    }


def model_insights():
    """Charts from the live model's stored test metrics (refreshed after retraining)."""
    os.makedirs(OUT, exist_ok=True)
    meta = load_metadata()
    m = meta.get("test_metrics", {})
    out = {}
    if "confusion_matrix" in m:
        cm = np.array(m["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4.4, 3.8))
        ax.imshow(cm, cmap="Greens")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(3)); ax.set_xticklabels(LABELS, fontsize=8)
        ax.set_yticks(range(3)); ax.set_yticklabels(LABELS, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion matrix (test set, {meta.get('version')})")
        _save(fig, "confusion_matrix.png")
        off = cm.copy(); np.fill_diagonal(off, 0)
        t, p = np.unravel_index(off.argmax(), off.shape)
        out["top_confusion"] = (f"{LABELS[t]} mistaken for {LABELS[p]} ({off[t, p]}x)" if off.max() else "no mistakes")
        disease_vs_disease = int(off[0, 1] + off[1, 0])                      # classes 0, 1 = the two diseases
        disease_vs_healthy = int(off[0, 2] + off[1, 2] + off[2, 0] + off[2, 1])
        out["confusion_note"] = (
            f"The two diseases are confused with each other ({disease_vs_disease} errors) more than with healthy leaves "
            f"({disease_vs_healthy}), because both start as small brown lesions." if disease_vs_disease > disease_vs_healthy else
            f"Errors between a disease and a healthy leaf ({disease_vs_healthy}) are at least as common as errors between the "
            f"two diseases ({disease_vs_disease}): early-stage lesions are small and easy to miss.")
    if "per_class_f1" in m:
        f1 = m["per_class_f1"]
        fig, ax = plt.subplots(figsize=(5, 3.3))
        ax.bar(LABELS, f1, color=[COLORS[c] for c in CLASS_NAMES])
        ax.set_ylim(min(0.5, min(f1) - 0.05), 1.0); ax.set_ylabel("F1 score"); ax.set_title("F1 score per class (test set)")
        for i, v in enumerate(f1):
            ax.text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=9)
        _save(fig, "per_class_f1.png")
        out["hardest"] = LABELS[int(np.argmin(f1))]
    if os.path.exists(HISTORY):
        with open(HISTORY) as f:
            h = json.load(f)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        for a, k in zip(ax, ["loss", "accuracy"]):
            a.plot(h[k], label="train"); a.plot(h[f"val_{k}"], label="validation")
            a.axvline(h.get("phase1_epochs", 0) - 0.5, ls="--", color="grey", label="fine-tuning starts")
            a.set_title(f"Training {k}"); a.set_xlabel("Epoch"); a.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        _save(fig, "training_curves.png")
        out["has_history"] = True
    return out

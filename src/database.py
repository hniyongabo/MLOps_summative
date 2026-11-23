"""SQLite database that stores uploaded training images and the history of retraining runs."""
import os
import sqlite3
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get("DB_PATH", os.path.join(ROOT, "database", "mlop.db"))


def connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT, n_images INTEGER, uploaded_at TEXT
        );
        CREATE TABLE IF NOT EXISTS training_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER REFERENCES uploads(id),
            label INTEGER NOT NULL,          -- 0 angular_leaf_spot, 1 bean_rust, 2 healthy
            filename TEXT,
            image BLOB NOT NULL,             -- original uploaded file bytes
            used_for_training INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS retrain_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT, n_new_images INTEGER, epochs INTEGER,
            old_accuracy REAL, new_accuracy REAL, new_f1 REAL,
            model_version TEXT, promoted INTEGER
        );
        """
    )
    return conn


def save_upload(filename, items):
    """Store an uploaded file's images [(label, name, raw bytes)] in the database."""
    conn = connect()
    with conn:
        cur = conn.execute(
            "INSERT INTO uploads (filename, n_images, uploaded_at) VALUES (?, ?, ?)",
            (filename, len(items), datetime.now().isoformat(timespec="seconds")),
        )
        upload_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO training_images (upload_id, label, filename, image) VALUES (?, ?, ?, ?)",
            [(upload_id, label, name, sqlite3.Binary(data)) for label, name, data in items],
        )
    conn.close()
    return upload_id


def fetch_images(only_unused=False):
    conn = connect()
    q = "SELECT id, label, filename, image FROM training_images"
    if only_unused:
        q += " WHERE used_for_training = 0"
    rows = conn.execute(q + " ORDER BY id").fetchall()
    conn.close()
    return rows


def mark_used(ids):
    conn = connect()
    with conn:
        conn.executemany("UPDATE training_images SET used_for_training = 1 WHERE id = ?", [(i,) for i in ids])
    conn.close()

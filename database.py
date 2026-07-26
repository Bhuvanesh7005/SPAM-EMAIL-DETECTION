"""
database.py
------------
Small helper module around plain sqlite3 (no ORM) — stores every prediction
made through the web app so users can see a history of checked messages.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(message, prediction, confidence):
    conn = get_connection()
    conn.execute(
        "INSERT INTO predictions (message, prediction, confidence) VALUES (?, ?, ?)",
        (message, prediction, confidence)
    )
    conn.commit()
    conn.close()


def get_history(limit=50):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_stats():
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]
    spam_count = conn.execute(
        "SELECT COUNT(*) AS c FROM predictions WHERE prediction = 'spam'"
    ).fetchone()["c"]
    conn.close()
    ham_count = total - spam_count
    return {"total": total, "spam": spam_count, "ham": ham_count}


def clear_history():
    conn = get_connection()
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()

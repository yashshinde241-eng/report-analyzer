"""
init_db.py — Smart Triage Priority System
Run this once to initialize the SQLite database for the triage queue.
Usage: python init_db.py
"""

import sqlite3
import os

DB_PATH = "reports.db"


def init_db():
    print(f"[INFO] Connecting to '{DB_PATH}'...")
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Show what columns exist right now (if table already exists)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='reports'")
    existing = cursor.fetchone()
    if existing:
        cursor.execute("PRAGMA table_info(reports)")
        cols = [row[1] for row in cursor.fetchall()]
        print(f"[INFO] Old 'reports' table found with columns: {cols}")
        print("[INFO] Dropping old table...")
        cursor.execute("DROP TABLE reports")
    else:
        print("[INFO] No existing 'reports' table — creating fresh.")

    cursor.execute("""
        CREATE TABLE reports (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            filename       TEXT    NOT NULL,
            status         TEXT    NOT NULL DEFAULT 'Pending'
                               CHECK(status IN ('Pending', 'Analyzed')),
            prediction     TEXT,
            confidence     REAL,
            severity_score REAL,
            timestamp      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    print("[OK] Schema: id, filename, status, prediction, confidence, severity_score, timestamp")
    print(f"[OK] Database ready at '{DB_PATH}'. Start simple_backend.py now.")


if __name__ == "__main__":
    init_db()
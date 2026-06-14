import sqlite3
from datetime import datetime
from contextlib import contextmanager
 
DB_PATH = "defects.db"
 
 
# ── Connection helper ─────────────────────────────────────────
@contextmanager
def _get_conn():
    """Context manager: always closes the connection, even on error."""
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()
 
 
# ── Public API ────────────────────────────────────────────────
def init_db():
    """Creates the database table and indexes if they don't exist."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                frame_no    INTEGER,
                defect_type TEXT,
                confidence  REAL,
                x1          INTEGER,
                y1          INTEGER,
                x2          INTEGER,
                y2          INTEGER
            )
        """)
        # Indexes for faster GROUP BY / ORDER BY queries as DB grows
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_defect_type
            ON detections (defect_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_frame_no
            ON detections (frame_no)
        """)
        conn.commit()
 
 
def save_detection(frame_no, defect_type, confidence, x1, y1, x2, y2):
    """Saves one detected defect to the database."""
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO detections
                (timestamp, frame_no, defect_type, confidence, x1, y1, x2, y2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            frame_no,
            defect_type,
            round(confidence, 3),
            x1, y1, x2, y2
        ))
        conn.commit()
 
 
def fetch_all_detections():
    """Returns all saved detections as a list of dicts, newest first."""
    columns = ["ID", "Timestamp", "Frame", "Defect Type",
               "Confidence", "X1", "Y1", "X2", "Y2"]
    with _get_conn() as conn:
        cursor = conn.execute("SELECT * FROM detections ORDER BY id DESC")
        rows = cursor.fetchall()
    return [dict(zip(columns, row)) for row in rows]
 
 
def get_stats():
    """Returns (total_count, [(defect_type, count), ...]) in one connection."""
    with _get_conn() as conn:
        total   = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        by_type = conn.execute(
            "SELECT defect_type, COUNT(*) FROM detections GROUP BY defect_type"
        ).fetchall()
    return total, by_type
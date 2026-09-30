# observability.py
# 11-12 hafta: Logging va Monitoring
#
# Ikki qatlam:
#   1. Strukturali JSON log fayli (logs/events.jsonl) — debug, audit, keyinchalik tahlil uchun
#   2. SQLite feedback jadvali — "noto'g'ri javobni boshqarish" uchun inson-fikri sig'imi
#
# AI Engineer mindset: model 100% to'g'ri javob bermaydi. Muhimi — buni BILISH va
# o'lchash: qaysi javoblar yomon deb belgilangan, qaysi prompt versiyasida, qachon.

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone

LOG_DIR = os.environ.get("LOG_DIR", "logs")
LOG_FILE = os.path.join(LOG_DIR, "events.jsonl")
FEEDBACK_DB_PATH = os.environ.get("FEEDBACK_DB_PATH", "feedback.db")


def log_event(event_type, **fields):
    """
    Bitta hodisani JSON-lines formatida faylga yozadi (har qator — bitta JSON obyekt).
    Masalan: log_event("request", session_id="abc", latency_ms=842, cache_hit=False)
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "event": event_type,
        **fields,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


class Timer:
    """Kontekst menejeri: bilan bloki ichidagi vaqtni millisekundda o'lchaydi."""

    def __enter__(self):
        self._start = time.perf_counter()
        self.elapsed_ms = None
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = round((time.perf_counter() - self._start) * 1000, 1)


# ---------------------------------------------------------------------------
# FEEDBACK (yomon javoblarni boshqarish)
# ---------------------------------------------------------------------------

@contextmanager
def _conn():
    c = sqlite3.connect(FEEDBACK_DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_feedback_db():
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id TEXT NOT NULL,
                session_id TEXT,
                rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
                comment TEXT,
                prompt_version TEXT,
                created_at REAL NOT NULL
            )"""
        )


def record_feedback(interaction_id, rating, session_id=None, comment=None, prompt_version=None):
    """rating: 'up' yoki 'down'. Har bir javobga (interaction_id orqali) izlanadi."""
    init_feedback_db()
    with _conn() as c:
        c.execute(
            """INSERT INTO feedback (interaction_id, session_id, rating, comment, prompt_version, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (interaction_id, session_id, rating, comment, prompt_version, time.time()),
        )
    log_event("feedback", interaction_id=interaction_id, rating=rating, prompt_version=prompt_version)


def get_feedback_stats(since_seconds=None):
    init_feedback_db()
    where, params = "", ()
    if since_seconds is not None:
        where, params = "WHERE created_at >= ?", (time.time() - since_seconds,)

    with _conn() as c:
        row = c.execute(
            f"""SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN rating = 'up' THEN 1 ELSE 0 END) AS up_count,
                    SUM(CASE WHEN rating = 'down' THEN 1 ELSE 0 END) AS down_count
                FROM feedback {where}""",
            params,
        ).fetchone()

    total = row["total"] or 0
    down = row["down_count"] or 0
    return {
        "total_feedback": total,
        "thumbs_up": row["up_count"] or 0,
        "thumbs_down": down,
        "negative_rate": round(down / total, 3) if total else 0.0,
    }


def get_recent_negative_feedback(limit=20):
    """Yomon baholangan so'nggi javoblar — tahlil/debug uchun."""
    init_feedback_db()
    with _conn() as c:
        rows = c.execute(
            """SELECT interaction_id, session_id, comment, prompt_version, created_at
               FROM feedback WHERE rating = 'down' ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
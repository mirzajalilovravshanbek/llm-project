# cost_tracker.py
# 11-12 hafta: Cost Optimization — token usage va xarajatni hisoblash/kuzatish

import os
import sqlite3
import time
from contextlib import contextmanager

USAGE_DB_PATH = os.environ.get("USAGE_DB_PATH", "usage.db")

# $ / 1M token (2026-yil boshidagi narxlar — yangilanishi mumkin, docs.claude.com'dan tekshiring)
PRICING = {
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-opus-5": {"input": 15.00, "output": 75.00},
}
DEFAULT_PRICING = {"input": 3.00, "output": 15.00}  # Nomalum model uchun taxminiy


def calc_cost(model, input_tokens, output_tokens):
    """Bitta so'rov uchun dollar hisobida taxminiy xarajat."""
    price = PRICING.get(model, DEFAULT_PRICING)
    cost = (input_tokens / 1_000_000) * price["input"] + (output_tokens / 1_000_000) * price["output"]
    return round(cost, 6)


@contextmanager
def _conn():
    c = sqlite3.connect(USAGE_DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db():
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost REAL NOT NULL,
                cache_hit INTEGER NOT NULL DEFAULT 0,
                prompt_version TEXT,
                created_at REAL NOT NULL
            )"""
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_usage_time ON usage_log(created_at)")


def record_usage(session_id, model, input_tokens, output_tokens, cache_hit=False, prompt_version=None):
    """Bitta so'rovni yozadi. Kesh'dan kelgan javob uchun cost = 0 (haqiqiy API chaqirilmagan)."""
    init_db()
    cost = 0.0 if cache_hit else calc_cost(model, input_tokens, output_tokens)
    with _conn() as c:
        c.execute(
            """INSERT INTO usage_log
               (session_id, model, input_tokens, output_tokens, cost, cache_hit, prompt_version, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, model, input_tokens, output_tokens, cost, int(cache_hit), prompt_version, time.time()),
        )
    return cost


def get_stats(since_seconds=None):
    """
    Umumiy statistika: jami so'rovlar, xarajat, o'rtacha token, kesh urish darajasi.
    since_seconds=3600 -> faqat oxirgi 1 soat.
    """
    init_db()
    where = ""
    params = ()
    if since_seconds is not None:
        where = "WHERE created_at >= ?"
        params = (time.time() - since_seconds,)

    with _conn() as c:
        row = c.execute(
            f"""SELECT
                    COUNT(*) AS total_requests,
                    COALESCE(SUM(cost), 0) AS total_cost,
                    COALESCE(SUM(input_tokens), 0) AS total_input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS total_output_tokens,
                    COALESCE(SUM(CASE WHEN cache_hit = 0 THEN input_tokens ELSE 0 END), 0) AS billed_input_tokens,
                    COALESCE(SUM(CASE WHEN cache_hit = 0 THEN output_tokens ELSE 0 END), 0) AS billed_output_tokens,
                    COALESCE(SUM(CASE WHEN cache_hit = 1 THEN input_tokens + output_tokens ELSE 0 END), 0) AS tokens_saved_by_cache,
                    COALESCE(SUM(cache_hit), 0) AS cache_hits
                FROM usage_log {where}""",
            params,
        ).fetchone()

    total = row["total_requests"]
    return {
        "total_requests": total,
        "total_cost_usd": round(row["total_cost"], 4),
        # Haqiqatan Claude API'ga yuborilgan (to'langan) tokenlar:
        "billed_input_tokens": row["billed_input_tokens"],
        "billed_output_tokens": row["billed_output_tokens"],
        # Kesh tufayli API'ga umuman yuborilmagan, lekin foydalanuvchiga xizmat qilingan tokenlar:
        "tokens_saved_by_cache": row["tokens_saved_by_cache"],
        # Eski nomlar (orqaga moslik uchun saqlanadi) — ikkalasini ham (billed + saved) qamraydi:
        "total_input_tokens": row["total_input_tokens"],
        "total_output_tokens": row["total_output_tokens"],
        "cache_hits": row["cache_hits"],
        "cache_hit_rate": round(row["cache_hits"] / total, 3) if total else 0.0,
        "avg_cost_per_request": round(row["total_cost"] / total, 6) if total else 0.0,
    }
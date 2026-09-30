# cache_layer.py
# 11-12 hafta: Caching
#
# Default: SQLite (fayl asosida) — Redis/Docker shart emas.
# Production'da .env ga REDIS_URL qo'shilsa, xuddi shu interfeys Redis orqali ishlaydi
# (celery_worker.py'dagi bilan bir xil naqsh: kod bir xil, transport almashadi).

import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager

CACHE_DB_PATH = os.environ.get("CACHE_DB_PATH", "cache.db")
REDIS_URL = os.environ.get("REDIS_URL")

DEFAULT_TTL_SECONDS = 3600  # 1 soat


def make_cache_key(*, system, messages, model, max_tokens):
    """
    So'rovning barcha muhim qismlarini (system, messages, model) birlashtirib hash qiladi.
    Agar ulardan birontasi o'zgarsa — kalit ham o'zgaradi, eski (mos kelmaydigan) javob
    qaytarilmaydi.
    """
    payload = json.dumps(
        {"system": system, "messages": messages, "model": model, "max_tokens": max_tokens},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# SQLITE BACKEND (default)
# ---------------------------------------------------------------------------

@contextmanager
def _conn():
    c = sqlite3.connect(CACHE_DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def _init_sqlite():
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                hits INTEGER NOT NULL DEFAULT 0
            )"""
        )


class SQLiteCache:
    def __init__(self):
        _init_sqlite()

    def get(self, key):
        with _conn() as c:
            row = c.execute("SELECT value, expires_at, hits FROM cache WHERE key = ?", (key,)).fetchone()
            if row is None:
                return None
            if row["expires_at"] < time.time():
                c.execute("DELETE FROM cache WHERE key = ?", (key,))
                return None
            c.execute("UPDATE cache SET hits = hits + 1 WHERE key = ?", (key,))
            return json.loads(row["value"])

    def set(self, key, value, ttl=DEFAULT_TTL_SECONDS):
        now = time.time()
        with _conn() as c:
            c.execute(
                """INSERT INTO cache (key, value, created_at, expires_at, hits)
                   VALUES (?, ?, ?, ?, 0)
                   ON CONFLICT(key) DO UPDATE SET
                     value=excluded.value, created_at=excluded.created_at,
                     expires_at=excluded.expires_at, hits=0""",
                (key, json.dumps(value, ensure_ascii=False), now, now + ttl),
            )

    def stats(self):
        with _conn() as c:
            row = c.execute(
                "SELECT COUNT(*) AS entries, COALESCE(SUM(hits), 0) AS total_hits FROM cache"
            ).fetchone()
            return {"entries": row["entries"], "total_hits": row["total_hits"]}

    def clear_expired(self):
        with _conn() as c:
            cur = c.execute("DELETE FROM cache WHERE expires_at < ?", (time.time(),))
            return cur.rowcount


# ---------------------------------------------------------------------------
# REDIS BACKEND (ixtiyoriy — REDIS_URL berilsa avtomatik ishlatiladi)
# ---------------------------------------------------------------------------

class RedisCache:
    def __init__(self, url):
        import redis  # pip install redis

        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._hits_prefix = "hits:"

    def get(self, key):
        raw = self._client.get(key)
        if raw is None:
            return None
        self._client.incr(self._hits_prefix + key)
        return json.loads(raw)

    def set(self, key, value, ttl=DEFAULT_TTL_SECONDS):
        self._client.setex(key, ttl, json.dumps(value, ensure_ascii=False))

    def stats(self):
        entries = self._client.dbsize()
        return {"entries": entries, "total_hits": None}  # Redis'da umumiy hit sanog'i shart emas

    def clear_expired(self):
        return 0  # Redis TTL'ni o'zi boshqaradi


def get_cache():
    """REDIS_URL bo'lsa Redis, bo'lmasa SQLite qaytaradi — kodning qolgan qismi bilmasa ham bo'ladi."""
    if REDIS_URL:
        try:
            return RedisCache(REDIS_URL)
        except Exception:
            pass  # Redis mavjud bo'lmasa, SQLite'ga tushamiz
    return SQLiteCache()


# Modul darajasida bitta umumiy instance (import qilingan joyda qayta ishlatiladi)
cache = get_cache()
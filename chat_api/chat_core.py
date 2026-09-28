# chat_core.py
# 9-10 hafta: Chat API uchun umumiy modul
#   - SQLite'da suhbat tarixi (sessions, messages)
#   - Claude API: oddiy (call_claude) va streaming (stream_claude) chaqiruvlar
#
# chat_api.py (FastAPI) va celery_worker.py (Celery) ikkalasi ham shu modulni ishlatadi.

import http.client
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

load_dotenv()

# ANTHROPIC_API_URL — faqat test/proxy uchun o'zgartiriladi, odatda tegmang
API_URL = os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages")
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
DB_PATH = os.environ.get("CHAT_DB_PATH", "chat.db")

# Modelga yuboriladigan oxirgi xabarlar soni (sliding window) — token/narxni cheklaydi
MAX_HISTORY_MESSAGES = 20

DEFAULT_SYSTEM = (
    "Siz foydali va aniq javob beradigan yordamchisiz. "
    "Foydalanuvchi qaysi tilda yozsa, shu tilda javob bering."
)


# ---------------------------------------------------------------------------
# 1. SQLITE — SUHBAT TARIXI
# ---------------------------------------------------------------------------

@contextmanager
def get_conn():
    """Har operatsiya uchun alohida ulanish (thread-safe, oddiy va ishonchli)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db():
    with get_conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                summary TEXT,
                created_at TEXT NOT NULL
            )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)")


def create_session(session_id=None):
    """Sessiya yaratadi. Agar allaqachon mavjud bo'lsa — tegmaydi. ID qaytaradi."""
    session_id = session_id or uuid.uuid4().hex[:12]
    with get_conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO sessions (id, title, summary, created_at) VALUES (?, NULL, NULL, ?)",
            (session_id, now_iso()),
        )
    return session_id


def get_session(session_id):
    with get_conn() as c:
        row = c.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def list_sessions(limit=50):
    with get_conn() as c:
        rows = c.execute(
            """SELECT s.id, s.title, s.summary, s.created_at,
                      (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count
               FROM sessions s ORDER BY s.rowid DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def add_messages(session_id, pairs):
    """pairs: [("user", "matn"), ("assistant", "matn")] — bitta tranzaksiyada yoziladi."""
    ts = now_iso()
    with get_conn() as c:
        c.executemany(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            [(session_id, role, content, ts) for role, content in pairs],
        )


def get_history(session_id):
    """Butun suhbat (vaqt tartibida) — foydalanuvchiga ko'rsatish uchun."""
    with get_conn() as c:
        rows = c.execute(
            "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_recent_messages(session_id, limit=MAX_HISTORY_MESSAGES):
    """Modelga yuboriladigan oxirgi `limit` ta xabar (sliding window)."""
    with get_conn() as c:
        rows = c.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    messages = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    # Claude API birinchi xabar "user" bo'lishini talab qiladi
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    return messages


def set_title(session_id, title):
    with get_conn() as c:
        c.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))


def set_summary(session_id, summary):
    with get_conn() as c:
        c.execute("UPDATE sessions SET summary = ? WHERE id = ?", (summary, session_id))


def delete_session(session_id):
    """Sessiya va uning barcha xabarlarini o'chiradi (CASCADE). O'chirilgan bo'lsa True."""
    with get_conn() as c:
        cur = c.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# 2. CLAUDE API
# ---------------------------------------------------------------------------

def _build_request(messages, system, max_tokens, stream):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY o'rnatilmagan! .env faylini tekshiring.")

    payload = {"model": MODEL, "max_tokens": max_tokens, "messages": messages}
    if system:
        payload["system"] = system
    if stream:
        payload["stream"] = True

    return Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )


def _http_error_message(e):
    return f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')}"


def call_claude(messages, system=None, max_tokens=1000, timeout=60):
    """Oddiy (streaming'siz) chaqiruv — to'liq javob matnini qaytaradi."""
    req = _build_request(messages, system, max_tokens, stream=False)
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        raise RuntimeError(_http_error_message(e))
    except URLError as e:
        raise RuntimeError(f"Ulanish xatosi: {e.reason}")

    parts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
    if not parts:
        raise RuntimeError(f"'text' blok topilmadi: {[b.get('type') for b in data.get('content', [])]}")
    return "\n".join(parts)


def stream_claude(messages, system=None, max_tokens=1000, timeout=120):
    """
    Streaming chaqiruv — generator: har bir kelgan matn bo'lagini (str) yield qiladi.

    Anthropic SSE hodisalari:
      content_block_delta (delta.type == "text_delta") -> matn bo'lagi
      message_stop                                     -> tugadi
      error                                            -> xato
    """
    req = _build_request(messages, system, max_tokens, stream=True)
    try:
        resp = urlopen(req, timeout=timeout)
    except HTTPError as e:
        raise RuntimeError(_http_error_message(e))
    except URLError as e:
        raise RuntimeError(f"Ulanish xatosi: {e.reason}")

    try:
        with resp:
            for raw_line in resp:  # SSE: qator-qator, javob kelishi bilan
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if not data_str:
                    continue

                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")
                if event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        yield delta["text"]
                elif event_type == "error":
                    raise RuntimeError(f"Stream xatosi: {event.get('error')}")
                elif event_type == "message_stop":
                    break
    except (OSError, http.client.HTTPException) as e:
        raise RuntimeError(f"Stream uzildi: {e}")
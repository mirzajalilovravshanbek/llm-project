# celery_worker.py
# 9-10 hafta: Background jobs — Celery
#
# Broker (vazifalar navbati) ikki xil rejimda ishlaydi — KOD BIR XIL, faqat transport almashadi:
#
#   1) FAYL TIZIMI (default) — hech narsa o'rnatish shart emas, navbat `celery_data/` papkasida.
#      O'rganish va lokal ishlab chiqish uchun ideal (Docker/Redis kerak emas).
#
#   2) REDIS — production uchun. .env ga qo'shing:  REDIS_URL=redis://localhost:6379/0
#
# Worker'ni ishga tushirish (alohida terminalda):
#   Linux/Mac : celery -A celery_worker:celery_app worker --loglevel=info
#   Windows   : celery -A celery_worker:celery_app worker --loglevel=info --pool=solo
#               (Windows'da --pool=solo majburiy, aks holda vazifalar ishlamaydi)

import os

from celery import Celery

import chat_core as core

REDIS_URL = os.environ.get("REDIS_URL")  # Bo'sh bo'lsa — fayl tizimi rejimi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("CELERY_DATA_DIR", os.path.join(BASE_DIR, "celery_data"))

if REDIS_URL:
    BROKER_MODE = "redis"
    celery_app = Celery("chat_tasks", broker=REDIS_URL, backend=REDIS_URL)
else:
    BROKER_MODE = "filesystem"
    queue_dir = os.path.join(DATA_DIR, "queue")
    results_dir = os.path.join(DATA_DIR, "results")
    control_dir = os.path.join(DATA_DIR, "control")
    for d in (queue_dir, results_dir, control_dir):
        os.makedirs(d, exist_ok=True)

    # pathlib.as_uri() ishlatamiz, oddiy "file://" + yo'l birlashtirish emas —
    # Windows'da disk harfi (masalan "E:\...") bo'lgan yo'llarni noto'g'ri
    # birlashtirsak, Celery uni "host:port" deb xato talqin qiladi
    # (masalan: "Port could not be cast to integer value as '\\llm-project\\...'").
    from pathlib import Path

    results_uri = Path(results_dir).resolve().as_uri()

    celery_app = Celery(
        "chat_tasks",
        broker="filesystem://",
        backend=results_uri,  # Natijalar ham fayl ko'rinishida saqlanadi
    )
    # Producer (API) va consumer (worker) BIR XIL papkadan foydalanishi shart
    celery_app.conf.broker_transport_options = {
        "data_folder_in": queue_dir,
        "data_folder_out": queue_dir,
        "control_folder": control_dir,
    }

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,      # Natijalar 1 soat saqlanadi
    task_track_started=True,  # STARTED holatini ko'rsatish uchun
)


def broker_available(timeout=1):
    """
    Broker'ga tez tekshiruv. Redis o'chiq bo'lsa, Celery/kombu ~20 soniya qayta ulanishga
    urinadi — API foydalanuvchini shuncha kuttirmasligi uchun avval shu yerda tekshiramiz.
    """
    if BROKER_MODE == "filesystem":
        return os.access(DATA_DIR, os.W_OK)

    try:
        import redis

        client = redis.Redis.from_url(
            REDIS_URL, socket_connect_timeout=timeout, socket_timeout=timeout
        )
        return bool(client.ping())
    except Exception:
        return False


MAX_TRANSCRIPT_CHARS = 30000


def _is_transient(error_text):
    """Qayta urinishga arziydigan xatolar: rate limit, server xatolari, tarmoq."""
    return (
        error_text.startswith("HTTP 429")
        or error_text.startswith("HTTP 5")
        or error_text.startswith("Ulanish xatosi")
        or error_text.startswith("Stream uzildi")
    )


@celery_app.task(name="summarize_session", bind=True)
def summarize_session(self, session_id):
    """Suhbatni xulosalab, sessiya yozuviga saqlaydi."""
    core.init_db()

    history = core.get_history(session_id)
    if not history:
        return {"session_id": session_id, "summary": None, "note": "Suhbat bo'sh"}

    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        transcript = transcript[-MAX_TRANSCRIPT_CHARS:]  # Eng oxirgi qismi

    try:
        summary = core.call_claude(
            [{
                "role": "user",
                "content": "Quyidagi suhbatni 3-5 jumlada xulosalang. Asosiy mavzular va "
                           f"kelishilgan narsalarni ko'rsating.\n\nSUHBAT:\n{transcript}",
            }],
            system="Siz suhbatlarni aniq va qisqa xulosalaydigan yordamchisiz.",
            max_tokens=400,
        )
    except RuntimeError as e:
        if _is_transient(str(e)):
            # Vaqtinchalik xato: 5 soniyadan keyin qayta urinadi (maks. 3 marta)
            raise self.retry(exc=e, countdown=5, max_retries=3)
        raise  # Doimiy xato (masalan noto'g'ri API key) — qayta urinish befoyda

    core.set_summary(session_id, summary)
    return {"session_id": session_id, "summary": summary}
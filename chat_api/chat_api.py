# chat_api.py
# 9-10 hafta task: Chat API
#   - suhbat tarixi bilan ishlaydi (SQLite, session_id orqali)
#   - streaming response (SSE) qo'llab-quvvatlaydi
#   - BackgroundTasks: suhbat sarlavhasini avtomatik yaratish
#   - Celery: suhbatni xulosalash (og'ir vazifa, alohida worker'da; Redis ixtiyoriy)
#
# Ishga tushirish:
#   uvicorn chat_api:app --reload
# Brauzerda: http://127.0.0.1:8000/docs  (Swagger — API'ni shu yerdan sinash mumkin)

import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Path
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

import chat_core as core

SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


@asynccontextmanager
async def lifespan(app: FastAPI):
    core.init_db()  # Server ishga tushganda jadvallarni yaratadi
    yield


app = FastAPI(
    title="Chat API",
    version="1.0.0",
    description="History va streaming qo'llab-quvvatlaydigan Claude chat API",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# 1. MODELLAR (Pydantic — kiruvchi ma'lumot avtomatik tekshiriladi)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    session_id: Optional[str] = Field(
        None, pattern=SESSION_ID_PATTERN,
        description="Bo'sh qoldirilsa yangi suhbat boshlanadi",
    )
    system: Optional[str] = Field(None, max_length=4000, description="Ixtiyoriy system prompt")
    max_tokens: int = Field(1000, ge=1, le=4000)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v):
        if not v.strip():
            raise ValueError("message bo'sh bo'lishi mumkin emas")
        return v


class ChatResponse(BaseModel):
    session_id: str
    reply: str


# ---------------------------------------------------------------------------
# 2. YORDAMCHI FUNKSIYALAR
# ---------------------------------------------------------------------------

def _prepare_conversation(req):
    """Sessiyani topadi/yaratadi va modelga yuboriladigan xabarlar ro'yxatini tuzadi."""
    session_id = core.create_session(req.session_id)
    history = core.get_recent_messages(session_id)
    is_first = len(history) == 0
    messages = history + [{"role": "user", "content": req.message}]
    return session_id, is_first, messages


def _sse(payload):
    """Server-Sent Event formatiga o'rash: 'data: {...}' + bo'sh qator."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _generate_title(session_id, first_message):
    """BackgroundTasks: javob yuborilgandan KEYIN ishlaydi, foydalanuvchini kuttirmaydi."""
    try:
        title = core.call_claude(
            [{
                "role": "user",
                "content": "Quyidagi xabar uchun 3-5 so'zli qisqa sarlavha yozing. "
                           "FAQAT sarlavhaning o'zini, qo'shtirnoqsiz yozing.\n\n"
                           f"Xabar: {first_message[:500]}",
            }],
            max_tokens=30,
        )
        title = title.strip().strip("\"'")[:80]
        if title:
            core.set_title(session_id, title)
    except Exception as e:  # Background vazifa xatosi asosiy javobga ta'sir qilmasin
        print(f"[background] sarlavha yaratib bo'lmadi: {e}")


def _get_celery():
    """Celery ixtiyoriy — o'rnatilmagan bo'lsa, tushunarli xato beradi."""
    try:
        from celery_worker import broker_available, celery_app, summarize_session
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Celery o'rnatilmagan. O'rnating: pip install celery",
        )
    return celery_app, summarize_session, broker_available


# ---------------------------------------------------------------------------
# 3. ENDPOINTLAR — CHAT
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model": core.MODEL}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    """Oddiy (streaming'siz) chat: to'liq javob tayyor bo'lgach qaytadi."""
    session_id, is_first, messages = _prepare_conversation(req)

    try:
        reply = core.call_claude(
            messages, system=req.system or core.DEFAULT_SYSTEM, max_tokens=req.max_tokens
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # Javob muvaffaqiyatli bo'lgandan keyingina tarixga yozamiz
    core.add_messages(session_id, [("user", req.message), ("assistant", reply)])

    if is_first:
        background_tasks.add_task(_generate_title, session_id, req.message)

    return ChatResponse(session_id=session_id, reply=reply)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, background_tasks: BackgroundTasks):
    """Streaming chat (SSE): javob bo'lak-bo'lak, generatsiya bo'layotgan paytda keladi."""
    session_id, is_first, messages = _prepare_conversation(req)

    def event_stream():
        yield _sse({"type": "session", "session_id": session_id})

        parts = []
        try:
            for piece in core.stream_claude(
                messages, system=req.system or core.DEFAULT_SYSTEM, max_tokens=req.max_tokens
            ):
                parts.append(piece)
                yield _sse({"type": "text", "text": piece})
        except RuntimeError as e:
            # Stream allaqachon boshlangan — HTTP status'ni o'zgartirib bo'lmaydi,
            # shuning uchun xatoni hodisa sifatida yuboramiz. Tarixga yozilmaydi.
            yield _sse({"type": "error", "message": str(e)})
            return

        reply = "".join(parts)
        core.add_messages(session_id, [("user", req.message), ("assistant", reply)])

        if is_first:
            background_tasks.add_task(_generate_title, session_id, req.message)

        yield _sse({"type": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # nginx orqasida ham buferlamasligi uchun
        },
    )


# ---------------------------------------------------------------------------
# 4. ENDPOINTLAR — SESSIYALAR
# ---------------------------------------------------------------------------

@app.get("/sessions")
def list_sessions():
    return {"sessions": core.list_sessions()}


@app.get("/sessions/{session_id}/history")
def session_history(session_id: str = Path(..., pattern=SESSION_ID_PATTERN)):
    session = core.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sessiya topilmadi")
    return {"session": session, "messages": core.get_history(session_id)}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str = Path(..., pattern=SESSION_ID_PATTERN)):
    if not core.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Sessiya topilmadi")
    return {"deleted": True, "session_id": session_id}


# ---------------------------------------------------------------------------
# 5. ENDPOINTLAR — CELERY (og'ir vazifa alohida worker'da)
# ---------------------------------------------------------------------------

@app.post("/sessions/{session_id}/summarize", status_code=202)
def summarize(session_id: str = Path(..., pattern=SESSION_ID_PATTERN)):
    """Suhbatni xulosalash vazifasini Celery navbatiga qo'yadi va darhol task_id qaytaradi."""
    if not core.get_session(session_id):
        raise HTTPException(status_code=404, detail="Sessiya topilmadi")

    _, summarize_session, broker_available = _get_celery()

    # Broker (Redis) o'chiq bo'lsa, Celery ~20 soniya qayta urinadi — shuning uchun oldindan tez tekshiramiz
    if not broker_available():
        raise HTTPException(status_code=503, detail="Celery broker'iga ulanib bo'lmadi (Redis ishlayaptimi?)")

    try:
        task = summarize_session.delay(session_id)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Vazifani navbatga qo'yib bo'lmadi ({type(e).__name__})",
        )
    return {"task_id": task.id, "status": "queued"}


@app.get("/tasks/{task_id}")
def task_status(task_id: str = Path(..., pattern=r"^[A-Za-z0-9-]{1,64}$")):
    """Celery vazifasi holati: PENDING / STARTED / RETRY / SUCCESS / FAILURE."""
    celery_app, _, _ = _get_celery()
    try:
        result = celery_app.AsyncResult(task_id)
        payload = {"task_id": task_id, "status": result.status}
        if result.successful():
            payload["result"] = result.result
        elif result.failed():
            payload["error"] = str(result.result)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Celery natija ombori bilan aloqa yo'q ({type(e).__name__})",
        )
    return payload
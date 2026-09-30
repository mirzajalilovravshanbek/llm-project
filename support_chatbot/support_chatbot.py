# support_chatbot.py
# 11-12 hafta — FINAL PROJECT: AI Support Chatbot (RAG bilan)
#
# Barcha internship davomida qurilgan qatlamlarni birlashtiradi:
#   7-8 hafta  -> RAG (recursive chunking, score threshold, Chroma)
#   9-10 hafta -> FastAPI, streaming, chat history
#   11-12 hafta -> caching, cost tracking, prompt versioning, logging, feedback
#
# AI Engineer mindset — bu faylda ko'rinadigan narsalar:
#   - Modelni "chaqirish"dan "product"ga farq: kesh, narx, monitoring, feedback qatlamlari
#   - Noto'g'ri javobni boshqarish: score threshold (hallucination guard) + /feedback endpoint
#   - Cost/latency: har so'rovda o'lchanadi va /metrics orqali ko'rinadi
#
# Ishga tushirish:
#   pip install fastapi uvicorn chromadb sentence-transformers pypdf python-dotenv
#   uvicorn support_chatbot:app --reload

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import cache_layer
import chat_core as core
import cost_tracker
import observability as obs
import prompts

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DIR = "support_kb"
COLLECTION_NAME = "support_docs"
MIN_SCORE_THRESHOLD = 0.25
COMPANY_NAME = "TechNova"


# ---------------------------------------------------------------------------
# 1. RAG KNOWLEDGE BASE (7-8 haftadagi bilan bir xil pattern)
# ---------------------------------------------------------------------------

def chunk_recursive(text, chunk_size=400, separators=None):
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator, remaining = separators[0], separators[1:]
    parts = [p for p in text.split(separator) if p.strip()]
    chunks, current = [], ""
    for part in parts:
        candidate = (current + separator + part) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                chunks.extend(chunk_recursive(part, chunk_size, remaining))
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)
    return chunks


class KnowledgeBase:
    def __init__(self, persist_directory=CHROMA_DIR):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_document(self, file_path, chunk_size=400):
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        else:
            text = path.read_text(encoding="utf-8")

        chunks = chunk_recursive(text, chunk_size=chunk_size)
        source = path.name
        ids = [f"{source}_{i}" for i in range(len(chunks))]
        metas = [{"source": source} for _ in chunks]
        self.collection.add(documents=chunks, ids=ids, metadatas=metas)
        return len(chunks)

    def search(self, query, top_k=4, min_score=MIN_SCORE_THRESHOLD):
        if self.collection.count() == 0:
            return []
        results = self.collection.query(query_texts=[query], n_results=top_k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        matches = []
        for doc, meta, dist in zip(docs, metas, distances):
            score = max(0.0, min(1.0, 1 - dist))
            if score >= min_score:
                matches.append({"chunk": doc, "source": meta.get("source", "unknown"), "score": round(score, 3)})
        return matches


kb = KnowledgeBase()


# ---------------------------------------------------------------------------
# 2. FASTAPI ILOVASI
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    core.init_db()
    cost_tracker.init_db()
    obs.init_feedback_db()
    yield


app = FastAPI(title="AI Support Chatbot", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    interaction_id: str
    rating: str = Field(..., pattern="^(up|down)$")
    comment: Optional[str] = None


# ---------------------------------------------------------------------------
# 3. ASOSIY ENDPOINT — /ask (RAG + cache + cost + prompt version + logging)
# ---------------------------------------------------------------------------

@app.post("/ask")
def ask(req: AskRequest):
    interaction_id = uuid.uuid4().hex[:12]
    session_id = req.session_id or interaction_id

    system_template, prompt_version = prompts.get_prompt("support_chatbot", version="latest")

    # --- 1. RETRIEVAL (kesh kalitidan OLDIN, chunki kontekst javobga ta'sir qiladi) ---
    with obs.Timer() as retrieval_timer:
        relevant_chunks = kb.search(req.question)

    if not relevant_chunks:
        obs.log_event(
            "request", interaction_id=interaction_id, session_id=session_id,
            cache_hit=False, grounded=False, latency_ms=retrieval_timer.elapsed_ms,
        )
        return {
            "interaction_id": interaction_id,
            "session_id": session_id,
            "answer": "Kechirasiz, bu savolga bilim bazamizda mos ma'lumot topilmadi.",
            "sources": [],
            "grounded": False,
            "cache_hit": False,
        }

    context = "\n\n---\n\n".join(f"[Manba: {r['source']}]\n{r['chunk']}" for r in relevant_chunks)
    system_prompt = system_template.format(company_name=COMPANY_NAME)
    user_content = f"KONTEKST:\n{context}\n\nSAVOL: {req.question}"
    messages = [{"role": "user", "content": user_content}]

    # --- 2. CACHE ---
    cache_key = cache_layer.make_cache_key(
        system=system_prompt, messages=messages, model=core.MODEL, max_tokens=800
    )
    cached = cache_layer.cache.get(cache_key)

    if cached is not None:
        cost_tracker.record_usage(
            session_id, core.MODEL, cached["input_tokens"], cached["output_tokens"],
            cache_hit=True, prompt_version=prompt_version,
        )
        obs.log_event(
            "request", interaction_id=interaction_id, session_id=session_id,
            cache_hit=True, grounded=True, cost=0.0,
            latency_ms=retrieval_timer.elapsed_ms, prompt_version=prompt_version,
        )
        return {
            "interaction_id": interaction_id, "session_id": session_id,
            "answer": cached["answer"],
            "sources": [{"source": r["source"], "score": r["score"]} for r in relevant_chunks],
            "grounded": True, "cache_hit": True,
        }

    # --- 3. GENERATION (kesh mos kelmadi) ---
    with obs.Timer() as gen_timer:
        try:
            answer, usage = core.call_claude_with_usage(messages, system=system_prompt, max_tokens=800)
        except RuntimeError as e:
            obs.log_event(
                "error", interaction_id=interaction_id, session_id=session_id,
                error=str(e), prompt_version=prompt_version,
            )
            raise HTTPException(status_code=502, detail=str(e))

    cache_layer.cache.set(cache_key, {
        "answer": answer, "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
    })

    cost = cost_tracker.record_usage(
        session_id, core.MODEL, usage["input_tokens"], usage["output_tokens"],
        cache_hit=False, prompt_version=prompt_version,
    )

    total_latency = retrieval_timer.elapsed_ms + gen_timer.elapsed_ms
    obs.log_event(
        "request", interaction_id=interaction_id, session_id=session_id,
        cache_hit=False, grounded=True, cost=cost,
        retrieval_ms=retrieval_timer.elapsed_ms, generation_ms=gen_timer.elapsed_ms,
        latency_ms=total_latency, prompt_version=prompt_version,
        input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
    )

    return {
        "interaction_id": interaction_id, "session_id": session_id,
        "answer": answer,
        "sources": [{"source": r["source"], "score": r["score"]} for r in relevant_chunks],
        "grounded": True, "cache_hit": False,
    }


# ---------------------------------------------------------------------------
# 4. FEEDBACK — Noto'g'ri javoblarni boshqarish
# ---------------------------------------------------------------------------

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    obs.record_feedback(req.interaction_id, req.rating, comment=req.comment)
    return {"recorded": True}


# ---------------------------------------------------------------------------
# 5. HUJJAT QO'SHISH
# ---------------------------------------------------------------------------

@app.post("/documents")
def add_document(file_path: str):
    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Fayl topilmadi")
    n_chunks = kb.add_document(file_path)
    return {"file": Path(file_path).name, "chunks_added": n_chunks}


# ---------------------------------------------------------------------------
# 6. MONITORING — /metrics
# ---------------------------------------------------------------------------

@app.get("/metrics")
def metrics(since_seconds: Optional[int] = None):
    usage_stats = cost_tracker.get_stats(since_seconds)
    feedback_stats = obs.get_feedback_stats(since_seconds)
    cache_stats = cache_layer.cache.stats()

    return {
        "usage": usage_stats,
        "feedback": feedback_stats,
        "cache": cache_stats,
        "knowledge_base": {"total_chunks": kb.collection.count()},
    }


@app.get("/metrics/negative-feedback")
def negative_feedback(limit: int = 20):
    return {"items": obs.get_recent_negative_feedback(limit)}


@app.get("/health")
def health():
    return {"status": "ok"}
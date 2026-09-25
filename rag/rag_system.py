# rag_system.py
# 5-6 hafta task: Document Search System (RAG)
#
# Oqim: PDF/text yuklash -> chunking -> embedding -> FAISS index -> savol-javob
#
# Embedding: sentence-transformers (local, bepul — Anthropic embedding API bermaydi)
# Vector DB: FAISS (server kerak emas, oson boshlash uchun)
# Generation: Claude API (avvalgi call_claude() patterni)

import os
import json
import pickle
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-5"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Kichik, tez, sifatli (384 o'lcham)
INDEX_DIR = Path("rag_index")


# ---------------------------------------------------------------------------
# 1. CLAUDE API CALL (avvalgi loyihadagi bilan bir xil pattern)
# ---------------------------------------------------------------------------

def call_claude(messages, system=None, max_tokens=800, model=MODEL):
    if not API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY o'rnatilmagan!")

    payload = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system:
        payload["system"] = system

    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
    }

    request = Request(
        API_URL, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )

    try:
        with urlopen(request) as response:
            data = json.loads(response.read().decode())
    except HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()}")
    except URLError as e:
        raise RuntimeError(f"Ulanish xatosi: {e.reason}")

    content = data.get("content", [])
    text_parts = [b["text"] for b in content if b.get("type") == "text"]
    if text_parts:
        return "\n".join(text_parts)

    raise RuntimeError(f"'text' blok topilmadi: {[b.get('type') for b in content]}")


# ---------------------------------------------------------------------------
# 2. DOCUMENT LOADING (PDF yoki .txt)
# ---------------------------------------------------------------------------

def clean_path(raw_path):
    """
    Foydalanuvchi kiritgan yo'lni tozalaydi:
    - "Copy as path" bilan kelgan qo'shtirnoqlarni olib tashlaydi
    - Boshi/oxiridagi bo'sh joylarni kesib tashlaydi
    """
    cleaned = raw_path.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ('"', "'"):
        cleaned = cleaned[1:-1]
    return cleaned.strip()


def load_document(file_path):
    """PDF yoki .txt fayldan matnni o'qiydi."""
    file_path = clean_path(file_path)
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Fayl topilmadi: {file_path}")

    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text

    elif path.suffix.lower() in (".txt", ".md"):
        return path.read_text(encoding="utf-8")

    else:
        raise ValueError(f"Qo'llab-quvvatlanmaydigan format: {path.suffix}")


# ---------------------------------------------------------------------------
# 3. CHUNKING — Matnni kichik bo'laklarga bo'lish
# ---------------------------------------------------------------------------

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Matnni so'zlar bo'yicha bo'laklarga (chunk) bo'ladi.

    chunk_size: har bir chunk taxminan nechta so'zdan iborat
    overlap: chunklar orasidagi qoplanish (kontekst uzilib qolmasligi uchun)
    """
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap  # overlap bilan siljish

    return chunks


# ---------------------------------------------------------------------------
# 4. RAG SYSTEM — Embedding + FAISS + Retrieval + Generation
# ---------------------------------------------------------------------------

class DocumentSearchSystem:
    def __init__(self):
        print(f"⏳ Embedding modeli yuklanmoqda ({EMBEDDING_MODEL_NAME})...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.dimension = self.embedder.get_sentence_embedding_dimension()

        self.index = None          # FAISS index
        self.chunks = []           # Chunk matnlari (index bilan mos tartibda)
        self.sources = []          # Har bir chunk qaysi fayldan kelgani

    # -- Indexing -----------------------------------------------------------

    def add_document(self, file_path, chunk_size=500, overlap=50):
        """Hujjatni yuklab, chunklab, embeddingga aylantirib indexga qo'shadi."""
        print(f"📄 Yuklanmoqda: {file_path}")
        text = load_document(file_path)

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        print(f"   {len(chunks)} ta chunk yaratildi")

        print("   🔢 Embeddinglar hisoblanmoqda...")
        embeddings = self.embedder.encode(chunks, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")

        # Cosine similarity uchun normalizatsiya (IndexFlatIP bilan)
        faiss.normalize_L2(embeddings)

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product = cosine (normalized)

        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self.sources.extend([Path(file_path).name] * len(chunks))

        print(f"   ✅ Index-ga qo'shildi. Jami chunklar: {len(self.chunks)}")

    # -- Retrieval ------------------------------------------------------------

    def search(self, query, top_k=3):
        """Savolga eng mos keladigan top_k chunkni topadi."""
        if self.index is None or len(self.chunks) == 0:
            raise RuntimeError("Hech qanday hujjat qo'shilmagan!")

        query_embedding = self.embedder.encode([query]).astype("float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "chunk": self.chunks[idx],
                "source": self.sources[idx],
                "score": float(score),
            })
        return results

    # -- Generation (RAG) -----------------------------------------------------

    def ask(self, query, top_k=3):
        """
        To'liq RAG oqimi:
        1. Savolga mos chunklarni topish (retrieval)
        2. Topilgan kontekst asosida Claude'dan javob olish (generation)
        """
        relevant_chunks = self.search(query, top_k=top_k)

        if not relevant_chunks:
            return {"answer": "Hujjatlarda tegishli ma'lumot topilmadi.", "sources": []}

        context = "\n\n---\n\n".join(
            f"[Manba: {r['source']}]\n{r['chunk']}" for r in relevant_chunks
        )

        system_prompt = """Siz hujjatlar asosida savollarga javob beradigan yordamchisiz.

QOIDALAR:
1. FAQAT quyida berilgan kontekst asosida javob bering.
2. Agar kontekstda javob yo'q bo'lsa, aniq ayting: "Bu haqda hujjatda ma'lumot topilmadi."
   — hech qachon o'zingiz bilganingizdan qo'shib yubormang (hallucination qilmang).
3. Javobingizda qaysi manbadan (fayl) foydalanganingizni ko'rsating."""

        user_prompt = f"""KONTEKST:
{context}

SAVOL: {query}

Yuqoridagi kontekst asosida javob bering:"""

        answer = call_claude(
            [{"role": "user", "content": user_prompt}],
            system=system_prompt,
        )

        return {
            "answer": answer,
            "sources": [{"source": r["source"], "score": round(r["score"], 3)} for r in relevant_chunks],
        }

    # -- Persistence ------------------------------------------------------------

    def save(self, directory=INDEX_DIR):
        """Index va chunklarni diskga saqlash (keyingi safar qayta yuklamaslik uchun)."""
        directory = Path(directory)
        directory.mkdir(exist_ok=True)

        faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "chunks.pkl", "wb") as f:
            pickle.dump({"chunks": self.chunks, "sources": self.sources}, f)

        print(f"💾 Index saqlandi: {directory}")

    def load(self, directory=INDEX_DIR):
        """Saqlangan indexni diskdan yuklash."""
        directory = Path(directory)

        self.index = faiss.read_index(str(directory / "index.faiss"))
        with open(directory / "chunks.pkl", "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.sources = data["sources"]

        print(f"📂 Index yuklandi: {len(self.chunks)} chunk")


# ---------------------------------------------------------------------------
# 5. DEMO / INTERACTIVE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    rag = DocumentSearchSystem()

    # Agar saqlangan index mavjud bo'lsa, uni yuklaymiz; aks holda yangi hujjat so'raymiz
    if (INDEX_DIR / "index.faiss").exists():
        print("\n📂 Avvalgi index topildi.")
        choice = input("Mavjud indexdan foydalanaymi? (ha/yo'q): ").strip().lower()
        if choice in ("ha", "h", "yes", "y"):
            rag.load()
        else:
            file_path = clean_path(input("Hujjat yo'lini kiriting (.pdf yoki .txt): "))
            rag.add_document(file_path)
            rag.save()
    else:
        file_path = clean_path(input("Hujjat yo'lini kiriting (.pdf yoki .txt): "))
        if not file_path:
            print("❌ Fayl yo'li kiritilmadi. Chiqilmoqda.")
            sys.exit(1)
        rag.add_document(file_path)
        rag.save()

    print("\n" + "=" * 60)
    print("DOCUMENT SEARCH SYSTEM — Savol berish rejimi")
    print("Chiqish uchun 'exit' yozing")
    print("=" * 60)

    while True:
        query = input("\n❓ Savol: ").strip()

        if query.lower() == "exit":
            print("Xayr!")
            break

        if not query:
            continue

        print("\n⏳ Qidirilmoqda va javob tayyorlanmoqda...\n")
        try:
            result = rag.ask(query)
            print(f"💬 Javob:\n{result['answer']}")
            print(f"\n📚 Manbalar: {result['sources']}")
        except RuntimeError as e:
            print(f"❌ Xato: {e}")
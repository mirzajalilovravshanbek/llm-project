# chroma_rag_system.py
# 5-6 hafta task — CHROMA versiyasi
#
# FAISS versiyasidan farqi:
#   - Persistence AVTOMATIK (qo'lda save()/load() yozish shart emas)
#   - Metadata bilan filtrlash mumkin (masalan faqat bitta fayldan qidirish)
#   - ID orqali boshqarish (chunk'ni o'chirish/yangilash oson)
#   - Ichki embedding funksiyasi mavjud (o'zimiz sentence-transformers beramiz)

import os
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-5"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DIR = "chroma_db"          # Chroma o'zi shu papkaga avtomatik saqlaydi
COLLECTION_NAME = "documents"


# ---------------------------------------------------------------------------
# 1. CLAUDE API CALL (avvalgi loyihalardagi bilan bir xil pattern)
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
# 2. FAYL YO'LINI TOZALASH VA YUKLASH (FAISS versiyasi bilan bir xil)
# ---------------------------------------------------------------------------

def clean_path(raw_path):
    """'Copy as path' bilan kelgan qo'shtirnoqlarni va bo'sh joylarni tozalaydi."""
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
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    elif path.suffix.lower() in (".txt", ".md"):
        return path.read_text(encoding="utf-8")

    else:
        raise ValueError(f"Qo'llab-quvvatlanmaydigan format: {path.suffix}")


def chunk_text(text, chunk_size=500, overlap=50):
    """Matnni so'zlar bo'yicha overlap bilan chunklarga bo'ladi."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# 3. CHROMA-ASOSLI DOCUMENT SEARCH SYSTEM
# ---------------------------------------------------------------------------

class ChromaDocumentSearchSystem:
    def __init__(self, persist_directory=CHROMA_DIR, collection_name=COLLECTION_NAME):
        print(f"⏳ Chroma client ishga tushirilmoqda ({persist_directory})...")

        # PersistentClient — avtomatik diskka saqlaydi, qo'lda save() kerak emas
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Chroma o'z ichida embedding funksiyasini chaqiradi (sentence-transformers)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )

        # Agar collection allaqachon mavjud bo'lsa — o'shani oladi, aks holda yaratadi
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},  # cosine similarity ishlatish
        )

        existing = self.collection.count()
        if existing > 0:
            print(f"   📂 Mavjud collection topildi: {existing} ta chunk")

    # -- Indexing -------------------------------------------------------------

    def add_document(self, file_path, chunk_size=500, overlap=50):
        """Hujjatni yuklab, chunklab, Chroma collection'ga qo'shadi."""
        print(f"📄 Yuklanmoqda: {file_path}")
        text = load_document(file_path)

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        print(f"   {len(chunks)} ta chunk yaratildi")

        source_name = Path(clean_path(file_path)).name

        # Har bir chunk uchun unique ID va metadata (Chroma'ning FAISS'dan afzalligi)
        ids = [f"{source_name}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

        print("   🔢 Chroma'ga qo'shilmoqda (embedding avtomatik hisoblanadi)...")
        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas,
        )

        print(f"   ✅ Qo'shildi. Jami chunklar: {self.collection.count()}")
        # Diqqat: qo'lda save() chaqirish SHART EMAS — PersistentClient avtomatik yozadi

    # -- Retrieval + metadata filter -------------------------------------------

    def search(self, query, top_k=3, source_filter=None):
        """
        Savolga eng mos chunklarni topadi.
        source_filter: faqat bitta fayldan qidirish uchun (masalan "notes.txt")
        — bu FAISS'da qo'lda yozish kerak bo'lgan funksiya, Chroma'da tayyor.
        """
        where_clause = {"source": source_filter} if source_filter else None

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_clause,
        )

        matches = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, distance in zip(docs, metas, distances):
            # Chroma "cosine" space'da distance = 1 - cosine_similarity qaytaradi.
            # (hnswlib cosine space shu formuladan foydalanadi, shuning uchun bu aniq)
            similarity = 1 - distance
            # Himoya: floating-point xatolar tufayli -0.0001 kabi qiymatlar chiqmasligi uchun
            similarity = max(0.0, min(1.0, similarity))
            matches.append({
                "chunk": doc,
                "source": meta.get("source", "unknown"),
                "score": round(similarity, 3),
                "raw_distance": round(distance, 3),  # debug uchun asl qiymat
            })

        return matches

    # -- Generation (RAG) -------------------------------------------------------

    def ask(self, query, top_k=3, source_filter=None):
        relevant_chunks = self.search(query, top_k=top_k, source_filter=source_filter)

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
            "sources": [{"source": r["source"], "score": r["score"]} for r in relevant_chunks],
        }

    # -- Boshqaruv (Chroma'ning qo'shimcha imkoniyati) ---------------------------

    def delete_source(self, source_name):
        """Bitta faylga tegishli barcha chunklarni o'chiradi (FAISS'da bu qiyin)."""
        self.collection.delete(where={"source": source_name})
        print(f"🗑️ '{source_name}' fayliga tegishli chunklar o'chirildi")

    def list_sources(self):
        """Collection'dagi barcha noyob fayl nomlarini qaytaradi."""
        all_data = self.collection.get()
        sources = set(m.get("source", "unknown") for m in all_data.get("metadatas", []))
        return sorted(sources)


# ---------------------------------------------------------------------------
# 4. DEMO / INTERACTIVE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    rag = ChromaDocumentSearchSystem()

    if rag.collection.count() > 0:
        print(f"\n📂 Mavjud fayllar: {rag.list_sources()}")
        choice = input("Yangi hujjat qo'shaymi? (ha/yo'q): ").strip().lower()
        if choice in ("ha", "h", "yes", "y"):
            file_path = clean_path(input("Hujjat yo'lini kiriting (.pdf yoki .txt): "))
            rag.add_document(file_path)
    else:
        file_path = clean_path(input("Hujjat yo'lini kiriting (.pdf yoki .txt): "))
        if not file_path:
            print("❌ Fayl yo'li kiritilmadi. Chiqilmoqda.")
            sys.exit(1)
        rag.add_document(file_path)

    print("\n" + "=" * 60)
    print("CHROMA DOCUMENT SEARCH SYSTEM — Savol berish rejimi")
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
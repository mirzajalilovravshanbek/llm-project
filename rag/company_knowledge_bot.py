# company_knowledge_bot.py
# 7-8 hafta task: Mini RAG App — Company Knowledge Bot / Docs Assistant
#
# Yangiliklar (5-6 haftadagi versiyadan farqi):
#   - Recursive chunking (paragraf -> jumla -> so'z, ma'noni saqlab qolish uchun)
#   - Score threshold (past-relevance chunklarni avtomatik rad etish)
#   - Ko'p hujjat qo'llab-quvvatlash (butun papkani yuklash)
#   - Source citation majburiy formatda
#   - Session tarixi (oldingi savol-javoblarni eslab qolish)

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
CHROMA_DIR = "knowledge_db"
COLLECTION_NAME = "company_docs"
MIN_SCORE_THRESHOLD = 0.25  # Bundan past score'li chunklar "mos emas" deb hisoblanadi


# ---------------------------------------------------------------------------
# 1. CLAUDE API CALL
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
# 2. FAYLLARNI O'QISH
# ---------------------------------------------------------------------------

def clean_path(raw_path):
    cleaned = raw_path.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ('"', "'"):
        cleaned = cleaned[1:-1]
    return cleaned.strip()


def load_document(file_path):
    path = Path(clean_path(file_path))

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


def load_folder(folder_path):
    """Papkadagi barcha .txt/.md/.pdf fayllarni topadi."""
    folder = Path(clean_path(folder_path))
    if not folder.is_dir():
        raise NotADirectoryError(f"Papka topilmadi: {folder_path}")

    supported = (".txt", ".md", ".pdf")
    files = [f for f in folder.iterdir() if f.suffix.lower() in supported]
    return files


# ---------------------------------------------------------------------------
# 3. RECURSIVE CHUNKING
# ---------------------------------------------------------------------------

def chunk_recursive(text, chunk_size=400, separators=None):
    """
    Paragraf -> jumla -> so'z tartibida bo'lish.
    Fixed-size'dan farqi: jumla/paragraf o'rtasida kesmaslikka harakat qiladi.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    if not separators:
        # Oxirgi chora: qattiq bo'lish
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator = separators[0]
    remaining_separators = separators[1:]

    parts = [p for p in text.split(separator) if p.strip()]
    chunks = []
    current = ""

    for part in parts:
        candidate = (current + separator + part) if current else part

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                chunks.extend(chunk_recursive(part, chunk_size, remaining_separators))
                current = ""
            else:
                current = part

    if current:
        chunks.append(current)

    return chunks


# ---------------------------------------------------------------------------
# 4. KNOWLEDGE BOT
# ---------------------------------------------------------------------------

class CompanyKnowledgeBot:
    def __init__(self, company_name="Kompaniya", persist_directory=CHROMA_DIR):
        self.company_name = company_name
        print(f"⏳ Bilim bazasi ishga tushirilmoqda...")

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        self.chat_history = []  # Suhbat konteksti uchun

        existing = self.collection.count()
        if existing > 0:
            print(f"   📂 Mavjud bilim bazasi: {existing} chunk, fayllar: {self.list_sources()}")

    # -- Indexing ---------------------------------------------------------------

    def add_document(self, file_path, chunk_size=400):
        text = load_document(file_path)
        chunks = chunk_recursive(text, chunk_size=chunk_size)

        source_name = Path(clean_path(file_path)).name
        ids = [f"{source_name}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

        self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        print(f"   ✅ '{source_name}': {len(chunks)} chunk qo'shildi")

    def add_folder(self, folder_path, chunk_size=400):
        """Papkadagi barcha qo'llab-quvvatlanadigan fayllarni qo'shadi."""
        files = load_folder(folder_path)
        if not files:
            print("   ⚠️ Papkada mos fayl topilmadi (.txt, .md, .pdf)")
            return

        print(f"   📁 {len(files)} ta fayl topildi")
        for f in files:
            try:
                self.add_document(str(f), chunk_size=chunk_size)
            except Exception as e:
                print(f"   ❌ '{f.name}' qo'shishda xato: {e}")

    def list_sources(self):
        all_data = self.collection.get()
        sources = set(m.get("source", "unknown") for m in all_data.get("metadatas", []))
        return sorted(sources)

    # -- Retrieval (score threshold bilan) -----------------------------------------

    def search(self, query, top_k=4, min_score=MIN_SCORE_THRESHOLD):
        results = self.collection.query(query_texts=[query], n_results=top_k)

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        matches = []
        for doc, meta, distance in zip(docs, metas, distances):
            similarity = max(0.0, min(1.0, 1 - distance))
            matches.append({
                "chunk": doc,
                "source": meta.get("source", "unknown"),
                "score": round(similarity, 3),
            })

        # Score threshold bo'yicha filtrlash — bu 5-6 haftadagi muammoni hal qiladi
        relevant = [m for m in matches if m["score"] >= min_score]
        return relevant, matches  # (filtrlangan, hammasi debug uchun)

    # -- Generation ---------------------------------------------------------------

    def ask(self, query, top_k=4, use_history=True):
        relevant_chunks, all_chunks = self.search(query, top_k=top_k)

        if not relevant_chunks:
            best_score = all_chunks[0]["score"] if all_chunks else 0
            return {
                "answer": (
                    f"Kechirasiz, bu savolga bilim bazamizda yetarlicha mos "
                    f"ma'lumot topilmadi (eng yaqin moslik: {best_score:.2f}, "
                    f"chegara: {MIN_SCORE_THRESHOLD}). Boshqacha so'rab ko'ring "
                    f"yoki tegishli hujjatni bazaga qo'shing."
                ),
                "sources": [],
                "grounded": False,
            }

        context = "\n\n---\n\n".join(
            f"[Manba: {r['source']}]\n{r['chunk']}" for r in relevant_chunks
        )

        system_prompt = f"""Siz {self.company_name} kompaniyasining ichki hujjatlari
bo'yicha yordamchisiz (docs assistant).

QOIDALAR:
1. FAQAT quyida berilgan kontekstga asoslaning.
2. Agar javob kontekstda to'liq yo'q bo'lsa, aniq ayting: "Bu ma'lumot 
   hujjatlarda topilmadi" — hech qachon o'zingizdan qo'shib yubormang.
3. Har bir faktdan keyin manba faylini ko'rsating, masalan: (Manba: notes.txt)
4. Agar bir nechta manba qarama-qarshi bo'lsa, buni aniq ayting.
5. Qisqa, aniq va tushunarli javob bering."""

        messages = []
        if use_history:
            messages.extend(self.chat_history[-6:])  # Oxirgi 3 juft savol-javob

        messages.append({
            "role": "user",
            "content": f"KONTEKST:\n{context}\n\nSAVOL: {query}",
        })

        answer = call_claude(messages, system=system_prompt)

        if use_history:
            self.chat_history.append({"role": "user", "content": query})
            self.chat_history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "sources": [{"source": r["source"], "score": r["score"]} for r in relevant_chunks],
            "grounded": True,
        }

    def reset_conversation(self):
        """Suhbat tarixini tozalash (bilim bazasi saqlanadi)."""
        self.chat_history = []


# ---------------------------------------------------------------------------
# 5. INTERACTIVE CLI
# ---------------------------------------------------------------------------

def print_banner(company_name):
    print("\n" + "=" * 60)
    print(f"  🤖 {company_name} — DOCS ASSISTANT")
    print("=" * 60)
    print("Buyruqlar: /add <fayl>, /addfolder <papka>, /sources, /reset, /exit\n")


if __name__ == "__main__":
    company_name = input("Kompaniya nomi (Enter = 'Kompaniya'): ").strip() or "Kompaniya"
    bot = CompanyKnowledgeBot(company_name=company_name)

    if bot.collection.count() == 0:
        print("\n📥 Bilim bazasi bo'sh. Hujjat qo'shish kerak.")
        choice = input("Bitta fayl (1) yoki butun papka (2)? ").strip()
        if choice == "2":
            folder = input("Papka yo'li: ").strip()
            bot.add_folder(folder)
        else:
            file_path = input("Fayl yo'li: ").strip()
            bot.add_document(file_path)

    print_banner(company_name)

    while True:
        query = input("❓ Savol (yoki buyruq): ").strip()

        if not query:
            continue

        if query.lower() in ("/exit", "exit"):
            print("Xayr!")
            break

        elif query.startswith("/add "):
            file_path = query[5:].strip()
            try:
                bot.add_document(file_path)
            except Exception as e:
                print(f"❌ Xato: {e}")
            continue

        elif query.startswith("/addfolder "):
            folder_path = query[11:].strip()
            try:
                bot.add_folder(folder_path)
            except Exception as e:
                print(f"❌ Xato: {e}")
            continue

        elif query == "/sources":
            print(f"📚 Bazadagi fayllar: {bot.list_sources()}")
            continue

        elif query == "/reset":
            bot.reset_conversation()
            print("🔄 Suhbat tarixi tozalandi (bilim bazasi saqlanib qoldi)")
            continue

        print("\n⏳ Qidirilmoqda...\n")
        try:
            result = bot.ask(query)
            print(f"💬 Javob:\n{result['answer']}")
            if result["sources"]:
                print(f"\n📚 Manbalar: {result['sources']}")
            print()
        except RuntimeError as e:
            print(f"❌ Xato: {e}\n")
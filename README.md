# LLM/AI Internship — 1–8 Hafta Loyihasi

Ushbu repository Claude (Anthropic) API asosida LLM fundamentals'dan to'liq RAG (Retrieval-Augmented Generation) tizimigacha bo'lgan 2 oylik internship dasturi davomida yaratilgan skriptlarni o'z ichiga oladi.

Barcha skriptlar `urllib` (Python standart kutubxonasi) orqali Anthropic API'ga murojaat qiladi — `anthropic` SDK ataylab ishlatilmagan, chunki ba'zi Windows kompyuterlarida `pydantic_core` bilan bog'liq DLL import xatolari uchraydi. Bu yondashuv o'sha muammoni butunlay chetlab o'tadi.

---

## 📁 To'liq Loyiha Strukturasi

```
llm-project/
│
├── .env                        # API key shu yerda (git-ga qo'shilmaydi!)
├── .gitignore
├── README.md                   # Shu fayl
│
├── hello_claude.py             # [1-2 hafta] Eng sodda test skript
├── summarizer.py                # [1-2 hafta] Matn qisqartiruvchi
├── qa_bot.py                    # [1-2 hafta] Interactive savol-javob bot
│
├── review_analyzer.py           # [3-4 hafta] Sentiment/JSON tahlil (boshlang'ich)
├── feature_extractor.py         # [3-4 hafta] Sentiment+category, validation, self-correction
│
├── rag_system.py                # [5-6 hafta] RAG — FAISS versiyasi
├── chroma_rag_system.py         # [5-6 hafta] RAG — Chroma versiyasi (metadata filter bilan)
│
└── company_knowledge_bot.py     # [7-8 hafta] Production-ready RAG app (recursive chunking,
                                  #             score threshold, ko'p hujjat, chat history)
```

---

## ⚙️ Umumiy O'rnatish

### 1. Python

Python 3.10+ kerak:
```bash
python --version
```

### 2. Barcha kerakli paketlar

```bash
pip install python-dotenv chromadb sentence-transformers pypdf faiss-cpu numpy
```

> Agar `numpy`/`faiss`/`sentence-transformers` o'rnatishda **"Политика управления приложениями" / DLL load failed** xatosi chiqsa — bu Windows'ning korporativ xavfsizlik siyosati (WDAC/AppLocker). Yechimlar: (a) IT bo'limidan Python `venv` papkasini ruxsat etilganlar ro'yxatiga qo'shishni so'rash, (b) WSL (Linux subsystem) ishlatish, (c) Google Colab kabi cloud muhitda ishlash.

### 3. API Key olish

1. https://console.anthropic.com → **API Keys** → yangi key yaratish (`sk-ant-...`)
2. **Plans & Billing** bo'limida hisobga kredit qo'shing (minimal $5)

### 4. `.env` faylini yaratish

Loyiha papkasida `.env` fayli:
```
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxxx
```

**Qoidalar:** qo'shtirnoqsiz, `=` atrofida bo'sh joysiz, `.gitignore`ga qo'shilgan bo'lishi shart.

### 5. Ishlatilgan Model

```
claude-sonnet-5
```

---

# 1-OY: AI FUNDAMENTALS VA LLM BILAN ISHLASH

## 1–2 Hafta: LLM Basics va Python Stack

**Mavzular:** LLM nima (GPT, open-source modellar), token/embedding/context window, zero-shot/few-shot/system prompt, Python stack (requests, asyncio, JSON).

### `hello_claude.py`
Eng sodda test — API ulanishini tekshirish uchun.
```bash
python hello_claude.py
```

### `summarizer.py`
Matnni 2-3 jumlada qisqartiradi (system prompt orqali rol berilgan).
```bash
python summarizer.py
```

### `qa_bot.py`
Interactive savol-javob bot, suhbat tarixini (`conversation_history`) eslab qoladi.
```bash
python qa_bot.py
# Savol: Python nima?
# ...
# Savol: exit
```

---

## 3–4 Hafta: Prompt Engineering va Evaluation

**Mavzular:** Role-based prompting, structured output (JSON), hallucination, output validation.

### `review_analyzer.py`
Sentiment tahlilining boshlang'ich versiyasi — review matndan JSON qaytaradi.

### `feature_extractor.py`
To'liq production-pattern: **role-based prompting** (CX tahlilchi roli) + **ikki qatlamli JSON himoyasi** (`extract_json()`) + **hallucination guard** (faqat matndagi ma'lumot, `"unknown"` fallback) + **schema validation** (`validate_output()`: required fields, enum, range) + **self-correction** (validatsiya muvaffaqiyatsiz bo'lsa, model xatolarini ko'rsatib qayta so'raladi).

```bash
python feature_extractor.py
```

**Natija** (`review_analysis_results.json`ga saqlanadi):
```json
{
  "success": true,
  "data": {
    "sentiment": "Mixed",
    "category": "Delivery/Shipping",
    "confidence": 85,
    "rating": 3,
    "key_phrases": ["kechiktirilgan buyurtma", "2 soat kech"]
  },
  "validation_errors": []
}
```

---

# 2-OY: RAG VA VECTOR DATABASE

## 5–6 Hafta: Embedding va Vector DB

**Mavzular:** Embedding tushunchasi, similarity search (cosine similarity), Vector DB (Qdrant/FAISS/Chroma taqqoslash).

> **Muhim:** Anthropic embedding API bermaydi (faqat text generation). Shuning uchun embedding uchun **local, bepul** `sentence-transformers` (`all-MiniLM-L6-v2`, 384 o'lcham) ishlatiladi.

### Vector DB Taqqoslash

| Xususiyat | FAISS | Chroma | Qdrant |
|---|---|---|---|
| Turi | Kutubxona | Kutubxona + avtomatik persistence | To'liq server |
| Persistence | Qo'lda saqlash | Avtomatik | Avtomatik |
| Metadata filter | Yo'q | Bor | Bor (kuchli) |
| Qulaylik | O'rtacha | Eng oson | O'rtacha-murakkab |
| Ishlatish holati | Kichik lokal loyiha | Kichik-o'rta loyiha, o'rganish | Production, katta hajm |

### `rag_system.py` — FAISS versiyasi

To'liq oqim: PDF/txt yuklash → chunking (fixed-size, overlap bilan) → embedding → FAISS `IndexFlatIP` (cosine) → retrieval → Claude orqali generation. Index `save()`/`load()` bilan qo'lda diskka yoziladi.

```bash
pip install faiss-cpu sentence-transformers pypdf numpy
python rag_system.py
```

### `chroma_rag_system.py` — Chroma versiyasi

Xuddi shu oqim, lekin: **avtomatik persistence** (`PersistentClient`), **metadata filter** (`source_filter="notes.txt"`), **fayl bo'yicha o'chirish** (`delete_source()`), **fayllar ro'yxati** (`list_sources()`).

```bash
pip install chromadb sentence-transformers pypdf python-dotenv
python chroma_rag_system.py
```

**Muhim dars (amalda sinalgan):** Chunk hajmi katta bo'lsa (masalan butun kichik hujjat 1 ta 500-so'zlik chunkka sig'ib ketsa), cosine similarity "suyultiriladi" — savol relevant bo'lsa ham score past chiqadi (masalan `0.347`). Kichikroq va aniqroq chunklar (masalan 100-400 so'z) bu muammoni kamaytiradi — 7-8 haftada bu **recursive chunking** bilan hal qilindi.

---

## 7–8 Hafta: RAG System (Production-Ready)

**Mavzular:** To'liq RAG pipeline (chunking → embedding → retrieval → generation), LangChain/LlamaIndex (ixtiyoriy, taqqoslash uchun ko'rib chiqildi).

### Chunking Strategiyalari

| Strategiya | Tezlik | Sifat | Holat |
|---|---|---|---|
| Fixed-size | Tez | O'rtacha | `rag_system.py`, `chroma_rag_system.py`da ishlatilgan |
| **Recursive** | O'rtacha | Yaxshi | `company_knowledge_bot.py`da ishlatilgan (tavsiya etiladi) |
| Semantic | Sekin | Eng yaxshi | Amalga oshirilmagan (katta hujjatlar uchun kerak bo'lganda) |

### Retrieval Yaxshilash

- **Score threshold** — past-relevance chunklarni avtomatik rad etish (`MIN_SCORE_THRESHOLD = 0.25`), model "topilmadi" deb javob berishga majburlanadi (hallucination oldini olish)
- **MMR (Maximal Marginal Relevance)** — kontseptual darajada ko'rib chiqildi (diversity uchun), amalga oshirilmagan

### `company_knowledge_bot.py` — Yakuniy Task: Mini RAG App

Barcha o'rganilgan konsepsiyalarni birlashtiruvchi **Company Knowledge Bot / Docs Assistant**:

| Xususiyat | Tavsif |
|---|---|
| **Recursive chunking** | Paragraf → jumla → so'z tartibida, ma'noni saqlab qolish uchun |
| **Score threshold** | Mos kelmaydigan javoblarni avtomatik "topilmadi" deb belgilaydi |
| **Ko'p hujjat** | `/addfolder` — butun papkani bir vaqtda yuklash |
| **Source citation** | Har bir javobda qaysi fayldan olinganini ko'rsatadi |
| **Chat history** | Oxirgi 3 savol-javobni eslaydi — follow-up savollar ishlaydi |
| **CLI buyruqlar** | `/add`, `/addfolder`, `/sources`, `/reset`, `/exit` |

```bash
pip install chromadb sentence-transformers pypdf python-dotenv
python company_knowledge_bot.py
```

**Ishlatish namunasi:**
```
Kompaniya nomi (Enter = 'Kompaniya'): TechNova
❓ Savol (yoki buyruq): /add notes.txt
❓ Savol (yoki buyruq): /sources
❓ Savol (yoki buyruq): Ta'til siyosati qanday?
💬 Javob: ... (Manba: notes.txt)
📚 Manbalar: [{'source': 'notes.txt', 'score': 0.427}, ...]
```

### LangChain — Nima Uchun Ishlatilmadi

Solishtiruv uchun LangChain ekvivalenti ko'rib chiqildi (`RecursiveCharacterTextSplitter`, `Chroma.from_documents` va h.k.). Internship davomida **qo'lda yozish tanlandi**, chunki bu har bir RAG bosqichini (chunking, embedding, retrieval, generation) chuqur tushunishga yordam beradi. Production loyihada LangChain/LlamaIndex'ga o'tish — endi "qora quti ichida nima borligini" bilgan holda — ancha oson bo'ladi.

---

## ❗ Tez-tez Uchraydigan Xatolar (Barcha Haftalar Bo'yicha)

| Xato | Sabab | Yechim |
|---|---|---|
| `ANTHROPIC_API_KEY o'rnatilmagan` | `.env` fayli topilmadi | `.env` faylini script bilan bir papkada yarating |
| `HTTP 400: invalid_request_error` | Noto'g'ri parametr | Xato matnini to'liq o'qing (`e.read().decode()`) |
| `HTTP 404: not_found_error` (model) | Model nomi eskirgan | `claude-sonnet-5` ishlating |
| `Your credit balance is too low` | Hisobda kredit yo'q | Billing sahifasidan kredit qo'shing |
| `ImportError: DLL load failed (_pydantic_core / _multiarray_umath)` | Windows Application Control siyosati | IT'dan exclusion so'rash / WSL / Google Colab |
| `KeyError: 'text'` | `content[0]` da text yo'q edi (boshqa block turi) | `type == "text"` bo'yicha filtrlash (barcha skriptlarda tuzatilgan) |
| `FileNotFoundError: "C:\...\notes.txt"` (qo'shtirnoq bilan) | Explorer'dan "Copy as path" qo'shtirnoq qo'shadi | `clean_path()` funksiyasi avtomatik tozalaydi |
| Score past, lekin javob to'g'ri | Chunk hajmi katta, similarity "suyultirilgan" | Kichikroq/recursive chunking ishlating (`company_knowledge_bot.py`dagidek) |

---

## 📌 Yakuniy Xulosa — Nima O'rganildi

- [x] **1-2 hafta:** LLM asoslari, token/embedding/context window, prompt turlari, Claude API bilan ishlash (`urllib`)
- [x] **3-4 hafta:** Role-based prompting, structured JSON output, hallucination guard, schema validation, self-correction
- [x] **5-6 hafta:** Embedding (local, sentence-transformers), similarity search, FAISS va Chroma bilan vector DB, oddiy RAG
- [x] **7-8 hafta:** Recursive chunking, score threshold, ko'p hujjatli RAG, source citation, production-ready Docs Assistant

## 🔭 Keyingi Qadamlar (Ixtiyoriy Kengaytirish)

- [ ] MMR (diversity-based retrieval) amalga oshirish
- [ ] Semantic chunking qo'shish
- [ ] `asyncio` bilan batch qayta ishlash (bir nechta savol/hujjatni parallel)
- [ ] LangChain/LlamaIndex'ga real o'tish (production loyihada)
- [ ] Web interfeys (Streamlit/FastAPI) orqali Company Knowledge Bot'ni deploy qilish
- [ ] Unit testlar (`pytest`) — `chunk_recursive()`, `validate_output()`, `extract_json()` uchun

## 📚 Foydali Havolalar

- Anthropic Console: https://console.anthropic.com
- API Docs: https://docs.claude.com
- Billing: https://console.anthropic.com/settings/billing
- Sentence Transformers: https://www.sbert.net
- Chroma Docs: https://docs.trychroma.com
- FAISS: https://github.com/facebookresearch/faiss

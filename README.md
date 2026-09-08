# LLM Project — Claude API bilan Ishlash

Ushbu loyiha Claude (Anthropic) API'dan foydalanib matn bilan ishlash uchun yozilgan Python skriptlar to'plamidir. Barcha skriptlar `urllib` (standart kutubxona) orqali ishlaydi — hech qanday `anthropic` yoki `pydantic` paketi talab qilinmaydi, shuning uchun Windows'da uchraydigan DLL/importa xatolaridan xoli.

## 📁 Loyiha Strukturasi

```
llm-project/
│
├── .env                    # API key shu yerda saqlanadi (git-ga qo'shilmaydi!)
├── .gitignore
├── README.md               # Shu fayl
├── hello_claude.py         # Eng sodda test skript
├── summarizer.py           # Matn qisqartiruvchi
├── qa_bot.py                # Interactive savol-javob bot
└── review_analyzer.py       # Sentiment/JSON tahlil qiluvchi
```

## ⚙️ O'rnatish

### 1. Python

Python 3.10+ kerak. Tekshirish:

```bash
python --version
```

### 2. Kerakli paketlar

```bash
pip install python-dotenv
```

> **Eslatma:** `anthropic` SDK ataylab ishlatilmayapti — o'rniga `urllib.request` (Python standart kutubxonasi) ishlatiladi. Bu ba'zi Windows kompyuterlarida chiqadigan `pydantic_core` DLL xatolarini butunlay chetlab o'tadi.

### 3. API Key olish

1. https://console.anthropic.com ga kiring
2. **API Keys** bo'limidan yangi key yarating (`sk-ant-...` bilan boshlanadi)
3. **Plans & Billing** bo'limida hisobingizga kredit qo'shing (minimal $5)

### 4. `.env` faylini yaratish

Loyiha papkasida `.env` nomli fayl yarating:

```
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxxx
```

**Muhim qoidalar:**
- Qo'shtirnoqsiz yozing (`ANTHROPIC_API_KEY="..."` emas)
- `=` atrofida bo'sh joy qoldirmang
- `.env` faylini hech qachon GitHub'ga yuklamang (`.gitignore`ga qo'shilgan bo'lishi kerak)

## 🚀 Ishlatish

### Test qilish

```bash
python hello_claude.py
```

Muvaffaqiyatli bo'lsa, Claude'dan javob konsolga chiqadi.

### Matn qisqartirish

```bash
python summarizer.py
```

`summarizer.py` ichidagi `text` o'zgaruvchisini o'z matningizga almashtiring.

### Interactive Q&A bot

```bash
python qa_bot.py
```

Savol yozing, Enter bosing. Chiqish uchun `exit` yozing. Bot suhbat tarixini (context) eslab qoladi.

### Review Analyzer (JSON output)

```bash
python review_analyzer.py
```

Har bir review uchun `sentiment`, `confidence`, `rating`, `key_words` maydonlari bilan JSON qaytaradi.

## 🧠 Ishlatilgan Model

Barcha skriptlarda `claude-sonnet-5` modeli ishlatiladi. Boshqa variantlar:

| Model | Tavsif |
|---|---|
| `claude-sonnet-5` | Balanslangan — tezlik + sifat (default) |
| `claude-haiku-4-5-20251001` | Eng tez va arzon |
| `claude-opus-5` | Eng kuchli, murakkab vazifalar uchun |

Modelni o'zgartirish uchun har bir faylda `call_claude()` funksiyasidagi `model=` parametrini tahrirlang.

## 🔧 Umumiy Funksiya: `call_claude()`

Barcha skriptlar bitta umumiy pattern'ga asoslangan:

```python
def call_claude(messages, system=None, max_tokens=500, model="claude-sonnet-5"):
    ...
```

**Xususiyatlari:**
- `urllib.request` orqali to'g'ridan-to'g'ri Anthropic API'ga so'rov yuboradi
- `HTTPError` va `URLError`larni alohida ushlab, aniq xato xabari beradi
- Javobdagi `content` massivini **`type == "text"`** bo'yicha filtrlaydi (index bo'yicha emas!), chunki ba'zida javob bir nechta turdagi bloklardan iborat bo'lishi mumkin (masalan `thinking` + `text`)

```python
content = data.get("content", [])
text_parts = [block["text"] for block in content if block.get("type") == "text"]
```

## ❗ Tez-tez Uchraydigan Xatolar

| Xato | Sabab | Yechim |
|---|---|---|
| `ANTHROPIC_API_KEY o'rnatilmagan` | `.env` fayli topilmadi yoki bo'sh | `.env` faylini script bilan bir papkada yarating |
| `HTTP 400: invalid_request_error` | Noto'g'ri parametr yoki key format | Xato matnini to'liq o'qing (`e.read().decode()`) |
| `HTTP 401: authentication_error` | API key noto'g'ri | Yangi key yarating |
| `HTTP 404: not_found_error` (model) | Model nomi eskirgan | `claude-sonnet-5` ishlating |
| `Your credit balance is too low` | Hisobda kredit yo'q | Billing sahifasidan kredit qo'shing |
| `ModuleNotFoundError: dotenv` | Paket o'rnatilmagan | `pip install python-dotenv` |
| `KeyError: 'text'` | `content[0]` da text yo'q edi | Yangilangan `extract_text` logikasidan foydalaning (type bo'yicha filtrlash) |


## 📚 Foydali Havolalar

- Anthropic Console: https://console.anthropic.com
- API Docs: https://docs.claude.com
- Billing: https://console.anthropic.com/settings/billing

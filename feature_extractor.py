# feature_extractor.py
# 3-4 hafta task: Structured JSON AI Feature
# Review matndan sentiment va category ajratish
#
# Xususiyatlar:
#   - Role-based prompting
#   - Structured JSON output (ikki qatlamli himoya)
#   - Hallucination guard (faqat matndagi ma'lumot, "unknown" ruxsat etiladi)
#   - Output validation (schema bo'yicha tekshirish)

import os
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-5"


# ---------------------------------------------------------------------------
# 1. LOW-LEVEL API CALL (urllib — pydantic muammolarisiz)
# ---------------------------------------------------------------------------

def call_claude(messages, system=None, max_tokens=600, model=MODEL):
    """Claude API-ga so'rov yuborish va xavfsiz text qaytarish."""
    if not API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY o'rnatilmagan! .env faylini tekshiring.")

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

    # content massivini type bo'yicha filtrlash (index emas!)
    content = data.get("content", [])
    text_parts = [b["text"] for b in content if b.get("type") == "text"]

    if text_parts:
        return "\n".join(text_parts)

    block_types = [b.get("type") for b in content]
    raise RuntimeError(f"'text' blok topilmadi. Mavjud turlar: {block_types}")


# ---------------------------------------------------------------------------
# 2. JSON EXTRACTION HELPER (model ba'zan izoh qo'shib yuborishi mumkin)
# ---------------------------------------------------------------------------

def extract_json(text):
    """Model javobidan JSON-ni xavfsiz ajratib olish."""
    cleaned = text.strip()

    # Agar markdown code block bo'lsa (```json ... ```), tozalash
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]

    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1

    if start == -1 or end == 0:
        raise ValueError(f"JSON topilmadi. Xom javob: {text[:200]}")

    return json.loads(cleaned[start:end])


# ---------------------------------------------------------------------------
# 3. VALIDATION (schema bo'yicha tekshirish)
# ---------------------------------------------------------------------------

SCHEMA = {
    "required": ["sentiment", "category", "confidence", "rating", "key_phrases"],
    "enums": {
        "sentiment": ["Positive", "Negative", "Neutral", "Mixed"],
        "category": [
            "Product Quality", "Price", "Customer Service",
            "Delivery/Shipping", "Usability", "Durability",
            "General", "unknown",
        ],
    },
    "ranges": {
        "confidence": (0, 100),
        "rating": (1, 5),
    },
}


def validate_output(data, schema=SCHEMA):
    """JSON strukturani schema bo'yicha tekshirish."""
    errors = []

    # Majburiy fieldlar
    for field in schema["required"]:
        if field not in data:
            errors.append(f"Yo'q field: '{field}'")

    # Enum (ruxsat etilgan qiymatlar) tekshiruvi
    for field, allowed in schema.get("enums", {}).items():
        if field in data and data[field] not in allowed:
            errors.append(
                f"'{field}' = '{data[field]}' ruxsat etilmagan. Kerak: {allowed}"
            )

    # Range (son diapazoni) tekshiruvi
    for field, (min_val, max_val) in schema.get("ranges", {}).items():
        if field in data:
            try:
                value = float(data[field])
                if not (min_val <= value <= max_val):
                    errors.append(
                        f"'{field}' = {value} diapazondan tashqari [{min_val}-{max_val}]"
                    )
            except (TypeError, ValueError):
                errors.append(f"'{field}' raqam emas: {data[field]!r}")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# 4. MAIN FEATURE: ReviewFeatureExtractor
# ---------------------------------------------------------------------------

class ReviewFeatureExtractor:
    """
    Review matndan strukturali ma'lumot ajratadi:
    sentiment, category, confidence, rating, key_phrases.

    Role-based prompting + hallucination guard + validation bilan.
    """

    SYSTEM_PROMPT = """Siz 10 yillik tajribaga ega mijozlar tajribasi (CX) tahlilchisisiz.
Vazifangiz — mijoz sharhlarini obyektiv va biznes uchun foydali formatda tasniflash.

QOIDALAR (hallucination'ni oldini olish uchun):
1. Faqat matnda AYNAN yozilgan yoki undan bevosita xulosa qilinadigan
   ma'lumotlardan foydalaning.
2. Agar kategoriya matndan aniq bo'lmasa, "General" yoki "unknown" deb belgilang —
   hech qachon o'zingiz mahsulot nomi yoki tafsilot o'ylab topmang.
3. key_phrases faqat matnda ishlatilgan so'z/iboralar bo'lishi kerak,
   sizning o'z talqiningiz emas.
4. Agar sharh juda qisqa yoki noaniq bo'lsa, confidence qiymatini pasaytiring."""

    CATEGORIES = [
        "Product Quality", "Price", "Customer Service",
        "Delivery/Shipping", "Usability", "Durability", "General",
    ]

    def _build_prompt(self, review_text):
        categories_str = ", ".join(self.CATEGORIES)
        return f"""Ushbu mijoz sharhini tahlil qiling:

REVIEW: "{review_text}"

Quyidagi kategoriyalardan bittasini tanlang: {categories_str}
(Agar hech biri mos kelmasa yoki aniq bo'lmasa: "unknown")

FAQAT quyidagi formatda JSON qaytaring. Hech qanday izoh, sarlavha
yoki markdown belgilar qo'shmang. Javobingiz to'g'ridan-to'g'ri {{ dan boshlansin:

{{
    "sentiment": "Positive/Negative/Neutral/Mixed",
    "category": "yuqoridagi ro'yxatdan biri",
    "confidence": 0-100,
    "rating": 1-5,
    "key_phrases": ["matndan olingan aniq so'z/iboralar"]
}}"""

    def analyze(self, review_text, retry_on_invalid=True):
        """
        Review-ni tahlil qiladi va validatsiyadan o'tgan JSON qaytaradi.
        Agar validatsiya muvaffaqiyatsiz bo'lsa, bir marta qayta urinadi.
        """
        messages = [{"role": "user", "content": self._build_prompt(review_text)}]
        raw_text = call_claude(messages, system=self.SYSTEM_PROMPT)

        try:
            data = extract_json(raw_text)
        except (ValueError, json.JSONDecodeError) as e:
            return {"success": False, "error": f"JSON parse xatosi: {e}", "raw": raw_text}

        is_valid, errors = validate_output(data)

        if not is_valid and retry_on_invalid:
            # Bitta qayta urinish: xatolarni ko'rsatib, tuzatishni so'raymiz
            fix_prompt = f"""Sizning oldingi javobingiz noto'g'ri edi:
{json.dumps(data, ensure_ascii=False)}

Xatolar: {'; '.join(errors)}

Iltimos, to'g'ri formatda qaytadan JSON qaytaring (faqat JSON, izohsiz):"""

            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": fix_prompt})

            raw_text = call_claude(messages, system=self.SYSTEM_PROMPT)
            try:
                data = extract_json(raw_text)
                is_valid, errors = validate_output(data)
            except (ValueError, json.JSONDecodeError) as e:
                return {"success": False, "error": f"JSON parse xatosi (retry): {e}", "raw": raw_text}

        return {
            "success": is_valid,
            "data": data,
            "validation_errors": errors if not is_valid else [],
        }


# ---------------------------------------------------------------------------
# 5. DEMO / TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    extractor = ReviewFeatureExtractor()

    test_reviews = [
        "Bu restoran ajoyib! Taomlar mazali, xizmat tez. Faqat narxi biroz qimmat.",
        "Notog'ri hizmat, kechiktirilgan buyurtma. Kuryer 2 soat kech keldi.",
        "Telefonning kamerasi super, lekin batareya tez tugatiladi. 3-4 soatga yetadi.",
        "Yaxshi.",  # Qisqa/noaniq — past confidence bo'lishi kutiladi
        "Mahsulot sinib keldi, qadoqlash yomon edi. Pulimni qaytarib berishlarini so'rayman.",
    ]

    print("=" * 60)
    print("REVIEW FEATURE EXTRACTOR — Sentiment & Category Analysis")
    print("=" * 60)

    results = []
    for i, review in enumerate(test_reviews, 1):
        print(f"\n[{i}] Review: {review}")
        result = extractor.analyze(review)

        if result["success"]:
            d = result["data"]
            print(f"    ✅ Sentiment: {d['sentiment']} | Category: {d['category']}")
            print(f"    📊 Confidence: {d['confidence']}% | Rating: {d['rating']}/5")
            print(f"    🔑 Key phrases: {', '.join(d['key_phrases'])}")
        else:
            print(f"    ❌ Xato: {result.get('error', result.get('validation_errors'))}")

        results.append(result)

    # Natijalarni faylga saqlash
    output_path = "review_analysis_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✅ Natijalar saqlandi: {output_path}")
    success_count = sum(1 for r in results if r["success"])
    print(f"📈 Muvaffaqiyat darajasi: {success_count}/{len(results)}")
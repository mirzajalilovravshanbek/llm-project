# review_analyzer.py
import os
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from dotenv import load_dotenv

# .env faylini yuklash
load_dotenv()

# API URL va kalitini olish
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
API_URL = "https://api.anthropic.com/v1/messages"

# Claude API-ga so'rov yuborish va xavfsiz javob qaytarish
def call_claude(messages, system=None, max_tokens=500, model="claude-sonnet-5"):
    if not API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY o'rnatilmagan!")

    # So'rov payloadini tayyorlash
    payload = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system:
        payload["system"] = system

    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
    }

    # So'rov yuborish
    request = Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST"
    )

    try:
        with urlopen(request) as response:
            # Javobni o'qish va JSON formatida parse qilish
            data = json.loads(response.read().decode())

            # Debug uchun to'liq javobni chiqarish
            print("\n📦 TO'LIQ JAVOB (debug uchun):")
            print("="*50)

            # JSON formatida chiroyli chiqarish
            print(json.dumps(data, indent=2, ensure_ascii=False))
            print("="*50)
    except HTTPError as e:
        # BU MUHIM QISM — haqiqiy xato matnni o'qish
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()}")
    except URLError as e:
        raise RuntimeError(f"Ulanish xatosi: {e.reason}")

    # Xavfsiz text ajratib olish
    content = data.get("content", [])

    if not content:
        raise RuntimeError(f"Bo'sh javob: {json.dumps(data, ensure_ascii=False)}")
    # Agar content bo'sh bo'lmasa, text bloklarini ajratib olish
    text_parts = [block["text"] for block in content if block.get("type") == "text"]

    if text_parts:
        return "\n".join(text_parts)
    
    # Agar text blok topilmasa, mavjud blok turlarini ko'rsatish
    block_types = [b.get("type") for b in content]
    
    raise RuntimeError(f"'text' blok topilmadi. Mavjud turlar: {block_types}")

# ReviewAnalyzer Classi foydalanuvchi review-ni tahlil qiladi va JSON formatida natija qaytaradi
class ReviewAnalyzer:
    # Bu class foydalanuvchi review-ni tahlil qiladi va sentiment, 
    # rating va kalit so'zlarni JSON formatida qaytaradi
    def __init__(self):
        self.system = (
            "Siz 10 yillik tajribaga ega mijozlar tajribasi (CX) tahlilchisisiz. \n" 
            "Sizning vazifangiz — mijoz sharhlarini obyektiv, biznes uchun foydali \n"
            "formatda tasniflash. Hech qachon shaxsiy fikr bildirmang, faqat matnda \n"
            "aniq ko'rsatilgan faktlarga tayaning. Foydalanuvchi \n"
            "tomonidan berilgan review-ni tahlil qiling va sentiment, \n"
            "rating va asosiy kalit so'zlarni JSON formatida qaytaring."
        )

    # Review-ni tahlil qilish va JSON formatida natija qaytarish
    def analyze(self, review_text):
        messages = [
            {
                "role": "user",
                "content": f"""Ushbu review-ni tahlil qil va JSON-da qaytarish:

Review: "{review_text}"

FAQAT JSON qaytar (boshqa hech narsa, izoh ham yozma):
{{
    "sentiment": "Positive/Negative/Neutral",
    "confidence": 0-100,
    "rating": 1-5,
    "key_words": []
}}""",
            }
        ]

        text = call_claude(messages)

        # JSON qismini xavfsiz ajratib olish
        try:
            # JSON qismi matn ichida joylashgan bo'lishi mumkin, 
            # shuning uchun uni ajratib olishga harakat qilamiz
            start = text.find("{")
            end = text.rfind("}") + 1
            json_str = text[start:end]
            return json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            return {"error": "JSON parse qilishda xato", "raw": text}

# Test qilish uchun main block
if __name__ == "__main__":
    # ReviewAnalyzer instance yaratish
    analyzer = ReviewAnalyzer()

    reviews = [
        "Bu restoran ajoyib! Taomlar mazali, xizmat tez. Faqat narxi qimmat.",
        "Notog'ri hizmat, kechiktirilgan buyurtma. Boshqa joy qilgandan yaxshi emas.",
        "Telefonning kamerasi super, lekin batareya tez tugatiladi.",
    ]

    print("=== REVIEW ANALYZER ===\n")

    # Har bir review-ni tahlil qilish
    for review in reviews:
        print(f"Review: {review}")
        try:
            result = analyzer.analyze(review)
            print("\n📦 TO'LIQ JAVOB (debug uchun):")
            print("="*50)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("="*50)
            # JSON formatida chiroyli chiqarish
            if "error" not in result:
                print(f"  Sentiment: {result['sentiment']} ({result['confidence']}%)")
                print(f"  Rating: {result['rating']}/5")
                print(f"  Key words: {', '.join(result['key_words'])}")
            else:
                print(f"  ❌ {result['error']}")
                print(f"  Raw: {result['raw']}")
        except RuntimeError as e:
            print(f"  ❌ Xato: {e}")
        print()
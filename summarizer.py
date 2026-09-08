# summarizer.py
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
    """Claude API-ga so'rov yuborish va xavfsiz javob qaytarish"""
    if not API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY o'rnatilmagan!")

    # So'rov payloadini tayyorlash
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
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
        method="POST",
    )

    try:
        with urlopen(request) as response:
            # Javobni o'qish va JSON formatida parse qilish
            data = json.loads(response.read().decode())
    except HTTPError as e:
        # BU MUHIM QISM — haqiqiy xato matnni o'qish
        error_body = e.read().decode()
        raise RuntimeError(f"HTTP {e.code}: {error_body}")
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

# Matnni qisqartirish uchun TextSummarizer Classi
class TextSummarizer:
    def summarize(self, text):
        # Claude API-ga so'rov yuborish uchun kerakli system va messages tayyorlash
        system = (
            "Siz O'zbek tilida matnlarni qisqarta oladigan mutaxassissiz. "
            "Asosiy fikrlarni 2-3 jumlada ifodalang."
        )
        messages = [
            {"role": "user", "content": f"Ushbu matnni qisqartir:\n{text}"}
        ]
        return call_claude(messages, system=system)

# Test qilish uchun main block
if __name__ == "__main__":
    summarizer = TextSummarizer()

    text = """Artificial Intelligence (AI) hozirgi vaqtda dunyo iqtisodiyotini 
o'zgartirayapti. Mashina o'rganish va chuqur o'rganish texnologiyalari 
sohalarning ko'pchiligi uchun yangi imkoniyatlarni ochmoqda. LLM-lar, 
xususan, natural language processing-da inqilob keltirdi. Ular 
kasb-hunarlarni avtomatlashtirib, shaxslarga inson-kabi kommunikatsiya 
imkonini berayapti. Biroq, etika va xavfsizlik muammolari ham paydo bo'lmoqda."""

    print("Asl matn:")
    print(text)
    print("\n" + "=" * 50)
    print("Qisqartirma:")
    try:
        print(summarizer.summarize(text))
    except RuntimeError as e:
        print(f"❌ Xato: {e}")
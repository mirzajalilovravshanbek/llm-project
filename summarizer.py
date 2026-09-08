# summarizer.py
import os
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
API_URL = "https://api.anthropic.com/v1/messages"


def call_claude(messages, system=None, max_tokens=500, model="claude-sonnet-5"):
    """Claude API-ga so'rov yuborish va xavfsiz javob qaytarish"""
    if not API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY o'rnatilmagan!")

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

    request = Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )

    try:
        with urlopen(request) as response:
            data = json.loads(response.read().decode())
    except HTTPError as e:
        error_body = e.read().decode()
        raise RuntimeError(f"HTTP {e.code}: {error_body}")
    except URLError as e:
        raise RuntimeError(f"Ulanish xatosi: {e.reason}")

    # Xavfsiz text ajratib olish
    if "content" in data and len(data["content"]) > 0:
        block = data["content"][0]
        if "text" in block:
            return block["text"]
        raise RuntimeError(f"Kutilmagan content turi: {block.get('type')}")

    raise RuntimeError(f"Kutilmagan javob: {json.dumps(data, ensure_ascii=False)}")


class TextSummarizer:
    def summarize(self, text):
        system = (
            "Siz O'zbek tilida matnlarni qisqarta oladigan mutaxassissiz. "
            "Asosiy fikrlarni 2-3 jumlada ifodalang."
        )
        messages = [
            {"role": "user", "content": f"Ushbu matnni qisqartir:\n{text}"}
        ]
        return call_claude(messages, system=system)


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
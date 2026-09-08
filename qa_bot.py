# qa_bot.py
import os
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
API_URL = "https://api.anthropic.com/v1/messages"


def call_claude(messages, system=None, max_tokens=1000, model="claude-sonnet-5"):
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
            # Debug uchun to'liq javobni chiqarish
            print("\n✅ Claude javob berdi:")
            print("="*50)
            print("\n📦 TO'LIQ JAVOB (debug uchun):")
            print("="*50)
            # JSON formatida chiroyli chiqarish
            print(json.dumps(data, indent=2, ensure_ascii=False))
            print("="*50)
    except HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()}")
    except URLError as e:
        raise RuntimeError(f"Ulanish xatosi: {e.reason}")

    if "content" in data and len(data["content"]) > 0:
        block = data["content"][1]
        if "text" in block:
            return block["text"]
        raise RuntimeError(f"Kutilmagan content turi: {block.get('type')}")

    raise RuntimeError(f"Kutilmagan javob: {json.dumps(data, ensure_ascii=False)}")


class QABot:
    def __init__(self):
        self.conversation_history = []
        self.system = (
            "Siz qattiq bilimli Q&A boti. Aniq va sodda javoblar bering, O'zbek tilida."
        )

    def ask(self, question):
        self.conversation_history.append({"role": "user", "content": question})

        answer = call_claude(self.conversation_history, system=self.system)

        self.conversation_history.append({"role": "assistant", "content": answer})
        return answer


if __name__ == "__main__":
    bot = QABot()

    print("=== Q&A BOT ===")
    print("Chiqish uchun 'exit' yozing\n")

    while True:
        question = input("Savol: ").strip()

        if question.lower() == "exit":
            print("Xayr!")
            break

        if not question:
            continue

        print("\n⏳ Javob tayyorlanyapti...\n")
        try:
            answer = bot.ask(question)
            print(f"Bot: {answer}\n")
        except RuntimeError as e:
            print(f"❌ Xato: {e}\n")
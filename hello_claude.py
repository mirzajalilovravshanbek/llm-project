# hello_claude.py
import os
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from dotenv import load_dotenv

# .env faylini yuklash
load_dotenv()

# API kalitini olish
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    print("❌ ANTHROPIC_API_KEY o'rnatilmagan!")
    exit(1)

# API so'rovi (requests bilan)
url = "https://api.anthropic.com/v1/messages"

# So'rov yuborish uchun kerakli header va ma'lumotlar
headers = {
    "Content-Type": "application/json",
    "x-api-key": api_key,
    "anthropic-version": "2023-06-01"
}

payload = {
    "model": "claude-sonnet-5",
    "max_tokens": 500,
    "messages": [
        {"role": "user", "content": "Pythonda bu nima:  print('='*50)"}
    ]
}

try:
    print("🔄 Claude-ga so'rov yuborilmoqda...")
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    with urlopen(request) as response:
        body = response.read().decode()
        data = json.loads(body)

        # Debug uchun to'liq javobni chiqarish
        print("\n✅ Claude javob berdi:")
        print("="*50)
        print("\n📦 TO'LIQ JAVOB (debug uchun):")
        print("="*50)
        # JSON formatida chiroyli chiqarish
        print(json.dumps(data, indent=2, ensure_ascii=False))
        print("="*50)

        # Endi xavfsiz tarzda text-ni olishga harakat qilamiz
        if "content" in data and len(data["content"]) > 0:
            block = data["content"][0]
            if "text" in block:
                print("\n✅ Claude javobi:")
                print(block["text"])
            else:
                print(f"\n⚠️ 'text' kaliti yo'q. Bu blok turi: {block.get('type')}")
        else:
            print("\n⚠️ 'content' bo'sh yoki yo'q!")

except Exception as e:
    # BU MUHIM QISM — haqiqiy xato matnini o'qish
    error_body = e.read().decode()
    print(f"\n❌ HTTP Error {e.code}:")
    print("="*50)
    print(error_body)

except URLError as e:
    print(f"❌ Ulanish xatosi: {e.reason}")

except Exception as e:
    print(f"❌ Kutilmagan xato: {type(e).__name__}: {e}")
# hello_requests.py
import os
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    print("❌ ANTHROPIC_API_KEY o'rnatilmagan!")
    exit(1)

# API so'rovi (requests bilan - Pydantic yo'q!)
url = "https://api.anthropic.com/v1/messages"
headers = {
    "Content-Type": "application/json",
    "x-api-key": api_key,
    "anthropic-version": "2023-06-01"
}

payload = {
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 500,
    "messages": [
        {"role": "user", "content": "Salom! Men kimman?"}
    ]
}

try:
    print("🔄 Claude-ga so'rov yuborulmoqda...")
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    with urlopen(request) as response:
        body = response.read().decode()
        data = json.loads(body)
        print("\n✅ Claude javob berdi:")
        print("="*50)
        print(data['content'][0]['text'])
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
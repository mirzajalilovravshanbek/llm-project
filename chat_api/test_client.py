# test_client.py
# Chat API'ning streaming endpointini terminaldan sinash.
# Avval serverni ishga tushiring:  uvicorn chat_api:app --reload
# Keyin (boshqa terminalda):        python test_client.py

import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "http://127.0.0.1:8000"

try:  # Windows konsolida o'zbek harflari buzilmasligi uchun
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def stream_chat(message, session_id=None):
    """Xabarni yuboradi, javobni kelishi bilan chop etadi. session_id qaytaradi."""
    payload = {"message": message}
    if session_id:
        payload["session_id"] = session_id

    req = Request(
        f"{BASE_URL}/chat/stream",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue

                event = json.loads(line[5:].strip())
                kind = event.get("type")

                if kind == "session":
                    session_id = event["session_id"]
                elif kind == "text":
                    print(event["text"], end="", flush=True)  # Real vaqtda chiqadi
                elif kind == "error":
                    print(f"\n❌ Server xatosi: {event['message']}")
                elif kind == "done":
                    print()
    except HTTPError as e:
        print(f"❌ HTTP {e.code}: {e.read().decode('utf-8', 'replace')}")
    except URLError as e:
        print(f"❌ Serverga ulanib bo'lmadi: {e.reason}\n   Server ishlayaptimi? (uvicorn chat_api:app)")

    return session_id


def get_json(path):
    with urlopen(f"{BASE_URL}{path}", timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


if __name__ == "__main__":
    print("=== CHAT API TEST CLIENT ===")
    print("Buyruqlar: /new (yangi suhbat), /history, /sessions, /exit\n")

    session_id = None

    while True:
        text = input("Siz: ").strip()
        if not text:
            continue

        if text == "/exit":
            break
        elif text == "/new":
            session_id = None
            print("🆕 Yangi suhbat boshlandi\n")
            continue
        elif text == "/sessions":
            for s in get_json("/sessions")["sessions"]:
                print(f"  {s['id']} | {s['title'] or '(sarlavhasiz)'} | {s['message_count']} xabar")
            print()
            continue
        elif text == "/history":
            if not session_id:
                print("Hali suhbat yo'q\n")
            else:
                for m in get_json(f"/sessions/{session_id}/history")["messages"]:
                    print(f"  [{m['role']}] {m['content'][:100]}")
                print()
            continue

        print("Claude: ", end="", flush=True)
        session_id = stream_chat(text, session_id)
        print(f"(session: {session_id})\n")
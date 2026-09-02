import os
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

request = Request(
    "https://api.openai.com/v1/responses",
    data=json.dumps({"model": "gpt-5", "input": "Salom!"}).encode(),
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    method="POST",
)

try:
    with urlopen(request) as response:
        result = json.load(response)
except HTTPError as error:
    details = error.read().decode("utf-8", errors="replace")
    raise RuntimeError(f"OpenAI API request failed ({error.code}): {details}") from error
except URLError as error:
    raise RuntimeError(f"Could not connect to OpenAI API: {error.reason}") from error

output_text = "".join(
    content.get("text", "")
    for item in result.get("output", [])
    for content in item.get("content", [])
    if content.get("type") == "output_text"
)
print(output_text)
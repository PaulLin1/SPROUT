"""
openai_client.py
================
Minimal OpenAI vision client (Chat Completions) for the second frontier judge.
Key from env OPENAI_API_KEY else ~/.aesthetics_secrets/openai.key. Never written to outputs.
"""
from __future__ import annotations
import base64, json, os, time
import urllib.request, urllib.error

BASE = "https://api.openai.com/v1"


def load_key():
    k = os.environ.get("OPENAI_API_KEY")
    if k:
        return k.strip()
    p = os.path.expanduser("~/.aesthetics_secrets/openai.key")
    if os.path.exists(p):
        return open(p, encoding="utf-8").read().strip()
    raise RuntimeError("no OpenAI key in env or ~/.aesthetics_secrets/openai.key")


def _req(url, key, data=None, method="GET", timeout=90):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, method=method, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def list_models(key):
    j = _req(f"{BASE}/models", key)
    return [m["id"] for m in j.get("data", [])]


def generate(key, model, prompt, image_path=None, temperature=0.0,
             max_retries=5, response_json=True):
    content = [{"type": "text", "text": prompt}]
    if image_path:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(image_path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"}})
    payload = {"model": model, "messages": [{"role": "user", "content": content}]}
    if response_json:
        payload["response_format"] = {"type": "json_object"}
    if temperature is not None:
        payload["temperature"] = temperature
    last = None
    for attempt in range(max_retries):
        try:
            j = _req(f"{BASE}/chat/completions", key, data=payload, method="POST")
            return j["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            code = e.code
            detail = e.read()[:200]
            last = f"HTTP {code} {detail!r}"
            # some newer models reject temperature / response_format -> strip and retry once
            if code == 400 and "temperature" in str(detail) and "temperature" in payload:
                payload.pop("temperature"); continue
            if code == 400 and "response_format" in str(detail) and "response_format" in payload:
                payload.pop("response_format"); continue
            if code in (429, 500, 503):
                time.sleep(min(60.0, 2.0 * (2 ** attempt)))
            else:
                break
        except Exception as e:
            last = repr(e); time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"openai generate failed: {last}")

"""
anthropic_client.py
===================
Minimal Anthropic (Claude) vision client for the third frontier judge.
Key from env ANTHROPIC_API_KEY else ~/.aesthetics_secrets/anthropic.key. Never written out.
"""
from __future__ import annotations
import base64, json, os, time
import urllib.request, urllib.error

BASE = "https://api.anthropic.com/v1"
VERSION = "2023-06-01"


def load_key():
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k.strip()
    p = os.path.expanduser("~/.aesthetics_secrets/anthropic.key")
    if os.path.exists(p):
        return open(p, encoding="utf-8").read().strip()
    raise RuntimeError("no Anthropic key in env or ~/.aesthetics_secrets/anthropic.key")


def _headers(key):
    return {"x-api-key": key, "anthropic-version": VERSION, "content-type": "application/json"}


def _req(url, key, data=None, method="GET", timeout=90):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, method=method, headers=_headers(key))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def list_models(key):
    j = _req(f"{BASE}/models", key)
    return [m["id"] for m in j.get("data", [])]


def generate(key, model, prompt, image_path=None, max_tokens=300,
             temperature=0.0, max_retries=5):
    content = []
    if image_path:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(image_path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        content.append({"type": "image",
                        "source": {"type": "base64", "media_type": mime, "data": b64}})
    content.append({"type": "text", "text": prompt})
    payload = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
               "messages": [{"role": "user", "content": content}]}
    last = None
    for attempt in range(max_retries):
        try:
            j = _req(f"{BASE}/messages", key, data=payload, method="POST")
            parts = j.get("content", [])
            return "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        except urllib.error.HTTPError as e:
            code = e.code
            last = f"HTTP {code} {e.read()[:200]!r}"
            if code in (429, 500, 529, 503):
                time.sleep(min(60.0, 2.0 * (2 ** attempt)))
            else:
                break
        except Exception as e:
            last = repr(e); time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"anthropic generate failed: {last}")

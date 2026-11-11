"""
gemini_client.py
================
Minimal, dependency-light client for the Gemini REST API (generativelanguage.googleapis.com),
used as a frontier VLM "judge" for graded affect ratings of images.

Key handling: read from env GEMINI_API_KEY, else from ~/.aesthetics_secrets/gemini.key.
The key is NEVER written into outputs/repo. The provided key has an unusual prefix
(AQ....) so this client tries BOTH auth styles: ?key= query param and the
x-goog-api-key / Bearer headers, and reports which works.
"""
from __future__ import annotations
import base64, json, os, time
import urllib.request, urllib.error

BASE = "https://generativelanguage.googleapis.com/v1beta"


def load_key():
    k = os.environ.get("GEMINI_API_KEY")
    if k:
        return k.strip()
    p = os.path.expanduser("~/.aesthetics_secrets/gemini.key")
    if os.path.exists(p):
        return open(p, encoding="utf-8").read().strip()
    raise RuntimeError("no Gemini key in env or ~/.aesthetics_secrets/gemini.key")


def _req(url, data=None, headers=None, method="GET", timeout=60):
    body = json.dumps(data).encode() if data is not None else None
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=body, headers=h, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8", "replace"))


def list_models(key):
    """Return (auth_style, [model names]) trying both auth mechanisms."""
    attempts = [
        ("query", f"{BASE}/models?key={key}", {}),
        ("x-goog-api-key", f"{BASE}/models", {"x-goog-api-key": key}),
        ("bearer", f"{BASE}/models", {"Authorization": f"Bearer {key}"}),
    ]
    errs = []
    for style, url, hdr in attempts:
        try:
            st, j = _req(url, headers=hdr)
            names = [m.get("name", "") for m in j.get("models", [])]
            return style, names
        except urllib.error.HTTPError as e:
            errs.append(f"{style}: HTTP {e.code} {e.read()[:160]!r}")
        except Exception as e:
            errs.append(f"{style}: {e!r}")
    raise RuntimeError("list_models failed:\n  " + "\n  ".join(errs))


def _auth_url_hdr(key, model, auth_style):
    path = f"{BASE}/{model}:generateContent"
    if auth_style == "query":
        return f"{path}?key={key}", {}
    if auth_style == "x-goog-api-key":
        return path, {"x-goog-api-key": key}
    return path, {"Authorization": f"Bearer {key}"}


def generate(key, model, prompt, image_path=None, auth_style="query",
             temperature=0.0, max_retries=5, response_json=True):
    """Single generateContent call with optional inline image. Returns text string."""
    parts = [{"text": prompt}]
    if image_path:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(image_path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        parts.append({"inline_data": {"mime_type": mime, "data": b64}})
    gen_cfg = {"temperature": temperature}
    if response_json:
        gen_cfg["response_mime_type"] = "application/json"
    payload = {"contents": [{"parts": parts}], "generationConfig": gen_cfg}
    url, hdr = _auth_url_hdr(key, model, auth_style)
    last = None
    for attempt in range(max_retries):
        try:
            st, j = _req(url, data=payload, headers=hdr, method="POST", timeout=90)
            cands = j.get("candidates", [])
            if not cands:
                return ""  # safety block or empty
            return "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", []))
        except urllib.error.HTTPError as e:
            code = e.code
            last = f"HTTP {code} {e.read()[:200]!r}"
            if code in (429, 500, 503):
                time.sleep(min(60.0, 2.0 * (2 ** attempt)))
            else:
                break
        except Exception as e:
            last = repr(e)
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"generate failed: {last}")

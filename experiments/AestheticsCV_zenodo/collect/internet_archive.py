"""
internet_archive.py
===================
Keyless Internet Archive image adapter. Unlike Wikimedia, IA hosts user-uploaded
aesthetic imagery (vaporwave edits, liminal-space photo sets, etc.), so it can contain
the *post-processed* aesthetics, not just their subject matter.

Pipeline per query: advancedsearch -> item identifiers -> per-item metadata -> first
usable image file -> download URL. Per-item licence (licenseurl) is captured when present;
items without an explicit open licence are flagged 'unknown -- verify' so they can be
excluded from a redistributable set (kept only for internal/feasibility use).
"""
from __future__ import annotations
import requests

UA = {"User-Agent": "AestheticsCV-research/0.1 (academic stimulus collection; yl2884@cornell.edu)"}
TIMEOUT = 30
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
OPEN_LICENCE_HINTS = ("creativecommons.org", "publicdomain", "/cc0", "/by", "/by-sa")


def _search_ids(query, n):
    url = "https://archive.org/advancedsearch.php"
    params = [("q", f"{query} AND mediatype:image"), ("rows", str(n)),
              ("output", "json"), ("fl[]", "identifier"), ("fl[]", "licenseurl"),
              ("sort[]", "downloads desc")]
    try:
        r = requests.get(url, params=params, headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        docs = r.json().get("response", {}).get("docs", [])
    except (requests.RequestException, ValueError):
        return []
    return [(d.get("identifier"), d.get("licenseurl")) for d in docs if d.get("identifier")]


def _first_image(identifier):
    try:
        r = requests.get(f"https://archive.org/metadata/{identifier}", headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        meta = r.json()
    except (requests.RequestException, ValueError):
        return None, None
    files = meta.get("files", [])
    lic = (meta.get("metadata", {}) or {}).get("licenseurl", "")
    # prefer original (non-thumbnail) image files
    for f in files:
        name = f.get("name", "")
        if name.lower().endswith(IMG_EXT) and "_thumb" not in name.lower() \
           and not name.lower().startswith("__ia"):
            return f"https://archive.org/download/{identifier}/{requests.utils.quote(name)}", lic
    return None, lic


def internet_archive(query: str, n: int = 40) -> list[dict]:
    out = []
    for identifier, lic_search in _search_ids(query, n):
        url, lic_meta = _first_image(identifier)
        if not url:
            continue
        lic = (lic_meta or lic_search or "").strip()
        low = lic.lower()
        licence = lic if any(h in low for h in OPEN_LICENCE_HINTS) else "unknown -- verify"
        out.append({
            "url": url,
            "license": licence,
            "source_url": f"https://archive.org/details/{identifier}",
            "source": "internet_archive",
            "title": identifier[:200],
            "ext_id": identifier,
        })
    return out

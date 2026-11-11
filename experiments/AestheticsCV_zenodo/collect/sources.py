"""
sources.py
==========
Licensed-image source adapters. Each returns a list of candidate records:

    {"url": <direct image url>, "license": <short code>, "source_url": <landing page>,
     "source": <api name>, "title": <str>, "ext_id": <provider id>}

No image bytes are fetched here -- only metadata. Downloading + dedup + provenance
logging happen in build_corpus.py. Every record carries a license and a landing-page
URL so the published stimulus set documents IP per image.

Adapters:
  - openverse   : keyless, aggregates CC0/CC-BY/PDM across Flickr/Wikimedia/museums.
                  The primary workhorse.
  - wikimedia   : keyless, PD/CC from Wikimedia Commons (good for retro-tech, statues,
                  architecture, nature).
  - pexels/unsplash : OPTIONAL, only if PEXELS_API_KEY / UNSPLASH_ACCESS_KEY are set.
                  Free-license stock, strong for the photographic clusters.
"""

from __future__ import annotations
import os
import time
import requests

UA = {"User-Agent": "AestheticsCV-research/0.1 (academic stimulus collection; contact yl2884@cornell.edu)"}
TIMEOUT = 30

# Openverse license tokens we accept (open / attribution-only, no NC/ND restriction
# that would block a research stimulus set).
OPENVERSE_LICENSES = "cc0,pdm,by,by-sa"


def _get(url, **kw):
    for attempt in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=TIMEOUT, **kw)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# Openverse  (https://api.openverse.org/v1/images/)  -- no auth needed
# ---------------------------------------------------------------------------
def openverse(query: str, n: int = 60, page: int = 1) -> list[dict]:
    out = []
    url = "https://api.openverse.org/v1/images/"
    params = {
        "q": query,
        "license": OPENVERSE_LICENSES,
        "page_size": min(n, 200),
        "page": page,
        "mature": "false",
    }
    try:
        r = _get(url, params=params)
    except requests.RequestException:
        return out
    if r is None:
        return out
    for it in r.json().get("results", []):
        img_url = it.get("url")
        if not img_url:
            continue
        out.append({
            "url": img_url,
            "license": f'{it.get("license","")} {it.get("license_version","")}'.strip(),
            "source_url": it.get("foreign_landing_url") or it.get("url"),
            "source": f'openverse:{it.get("source","")}',
            "title": (it.get("title") or "")[:200],
            "ext_id": it.get("id"),
        })
    return out


# ---------------------------------------------------------------------------
# Wikimedia Commons  (generator=search + imageinfo with extmetadata) -- no auth
# ---------------------------------------------------------------------------
def wikimedia(query: str, n: int = 40) -> list[dict]:
    out = []
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"{query} filetype:bitmap",
        "gsrnamespace": 6,            # File: namespace
        "gsrlimit": min(n, 50),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size|mime",
        "iiurlwidth": 1280,            # thumb width; ensures long side clears the 512 floor
    }
    try:
        r = _get(url, params=params)
    except requests.RequestException:
        return out
    if r is None:
        return out
    pages = (r.json().get("query") or {}).get("pages", {})
    for _, pg in pages.items():
        ii = (pg.get("imageinfo") or [{}])[0]
        if not ii or ii.get("mime", "").split("/")[0] != "image":
            continue
        meta = ii.get("extmetadata", {}) or {}
        lic = (meta.get("LicenseShortName", {}) or {}).get("value", "")
        # keep only open licenses (skip "all rights reserved"/unknown)
        low = lic.lower()
        if not any(t in low for t in ("cc", "public domain", "pdm", "cc0")):
            continue
        out.append({
            "url": ii.get("thumburl") or ii.get("url"),
            "license": lic or "Wikimedia",
            "source_url": ii.get("descriptionurl") or ii.get("url"),
            "source": "wikimedia",
            "title": (pg.get("title") or "")[:200],
            "ext_id": pg.get("pageid"),
        })
    return out


# ---------------------------------------------------------------------------
# Pexels  (optional -- needs PEXELS_API_KEY)
# ---------------------------------------------------------------------------
def pexels(query: str, n: int = 40) -> list[dict]:
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        return []
    out = []
    url = "https://api.pexels.com/v1/search"
    try:
        r = requests.get(url, headers={**UA, "Authorization": key},
                         params={"query": query, "per_page": min(n, 80)}, timeout=TIMEOUT)
        r.raise_for_status()
    except requests.RequestException:
        return out
    for ph in r.json().get("photos", []):
        out.append({
            "url": (ph.get("src") or {}).get("large2x") or (ph.get("src") or {}).get("large"),
            "license": "Pexels (free)",
            "source_url": ph.get("url"),
            "source": "pexels",
            "title": (ph.get("alt") or "")[:200],
            "ext_id": ph.get("id"),
        })
    return out


# ---------------------------------------------------------------------------
# Unsplash  (optional -- needs UNSPLASH_ACCESS_KEY)
# ---------------------------------------------------------------------------
def unsplash(query: str, n: int = 30) -> list[dict]:
    key = os.environ.get("UNSPLASH_ACCESS_KEY")
    if not key:
        return []
    out = []
    url = "https://api.unsplash.com/search/photos"
    try:
        r = requests.get(url, headers={**UA, "Authorization": f"Client-ID {key}"},
                         params={"query": query, "per_page": min(n, 30)}, timeout=TIMEOUT)
        r.raise_for_status()
    except requests.RequestException:
        return out
    for ph in r.json().get("results", []):
        out.append({
            "url": (ph.get("urls") or {}).get("regular"),
            "license": "Unsplash (free)",
            "source_url": (ph.get("links") or {}).get("html"),
            "source": "unsplash",
            "title": (ph.get("description") or ph.get("alt_description") or "")[:200],
            "ext_id": ph.get("id"),
        })
    return out


def _load_optional():
    """Aesthetic-faithful re-sourcing adapters (separate modules)."""
    reg = {}
    try:
        from internet_archive import internet_archive
        reg["internet_archive"] = internet_archive
    except Exception:
        pass
    try:
        from flickr_cc import flickr_cc
        reg["flickr_cc"] = flickr_cc
    except Exception:
        pass
    return reg


ALL_ADAPTERS = {
    "openverse": openverse,
    "wikimedia": wikimedia,
    "pexels": pexels,
    "unsplash": unsplash,
    **_load_optional(),
}

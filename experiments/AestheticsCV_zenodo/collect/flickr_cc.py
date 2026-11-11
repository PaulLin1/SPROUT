"""
flickr_cc.py
============
Flickr Creative-Commons image adapter. This is the workhorse for *aesthetic-faithful*
re-sourcing: Flickr users tag and CC-license genuine post-processed aesthetic art
(#vaporwave, #frutigeraero, #weirdcore, #liminalspace), which is exactly what the
Wikimedia corpus lacks.

Requires a (free) Flickr API key in the FLICKR_API_KEY environment variable; returns []
with a notice if absent. License filter restricts to redistributable licences:
  4 = CC BY, 5 = CC BY-SA, 9 = CC0, 10 = Public Domain Mark, 7 = No known copyright.
"""
from __future__ import annotations
import os
import requests

UA = {"User-Agent": "AestheticsCV-research/0.1 (academic; yl2884@cornell.edu)"}
TIMEOUT = 30
LICENSE_NAME = {"4": "CC BY 2.0", "5": "CC BY-SA 2.0", "7": "No known copyright",
                "9": "CC0", "10": "Public Domain Mark"}
REDISTRIBUTABLE = "4,5,7,9,10"


def flickr_cc(query: str, n: int = 60) -> list[dict]:
    key = os.environ.get("FLICKR_API_KEY")
    if not key:
        return []
    url = "https://api.flickr.com/services/rest/"
    params = {
        "method": "flickr.photos.search", "api_key": key, "text": query,
        "license": REDISTRIBUTABLE, "content_type": 1, "media": "photos",
        "safe_search": 1, "sort": "relevance", "per_page": min(n, 200),
        "extras": "license,url_l,url_c,owner_name", "format": "json",
        "nojsoncallback": 1,
    }
    try:
        r = requests.get(url, params=params, headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError):
        return []
    out = []
    for p in data.get("photos", {}).get("photo", []):
        img = p.get("url_l") or p.get("url_c")
        if not img:
            continue
        out.append({
            "url": img,
            "license": LICENSE_NAME.get(str(p.get("license")), f"Flickr lic {p.get('license')}"),
            "source_url": f"https://www.flickr.com/photos/{p.get('owner')}/{p.get('id')}",
            "source": "flickr_cc",
            "title": (p.get("title") or "")[:200],
            "ext_id": p.get("id"),
        })
    return out

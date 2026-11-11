#!/usr/bin/env python3
"""
cover_collect.py -- build hand-picked boards (e.g. laptop twee) from web pages you choose.

Input: a CSV with columns  board,url,kind,note
  board   the board title, as a curator would write it ("laptop twee", "laptop twee ♡", "hyperpop")
  url     a Bandcamp artist page (all its releases are collected), a Bandcamp release page, any web page
          with a preview image (og:image), or a direct image link (.jpg/.png/.webp)
  kind    target (the aesthetic you want found) or decoy (its neighbours, for an honest test)
  note    free text: why it belongs, where you read it

Output: the same layout as arena_collect.py, so the two datasets merge. If --out already holds
photos.csv / links.csv / boards.csv (e.g. your Are.na dataset), new rows are ADDED to them.
  photos.csv  photo_id, image_path, image_url, block_title, source_url, kind, label
  links.csv   photo_id, board_id, board_title, connected_at
  boards.csv  board_id, slug, title, created_at, n_contents, found_by, role
  images/     <photo_id>.jpg (longest side <= --max-side)

Run
  pip install requests pillow pandas
  python cover_collect.py --sources laptop_twee_sources.csv --out arena_taste

Polite by default: one page request per second, every page cached under <out>/web_cache. Images are for
your own research; do not redistribute them.
"""
import argparse
import hashlib
import html
import io
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd

try:
    import requests
except ImportError:
    sys.exit("pip install requests pillow pandas")

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".gif")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


class Web:
    def __init__(self, cache, delay=1.0):
        self.cache, self.delay, self.last = Path(cache), delay, 0.0
        self.cache.mkdir(parents=True, exist_ok=True)
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "Mozilla/5.0 (research; personal moodboard dataset)"

    def page(self, url):
        f = self.cache / (hashlib.md5(url.encode()).hexdigest() + ".html")
        if f.exists():
            return f.read_text(encoding="utf-8", errors="replace")
        wait = self.delay - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()
        try:
            r = self.s.get(url, timeout=30)
        except requests.RequestException as e:
            log(f"  could not open {url}: {e}"); return None
        if r.status_code != 200:
            log(f"  {url}: HTTP {r.status_code}"); return None
        f.write_text(r.text, encoding="utf-8")
        return r.text

    def image(self, url):
        try:
            r = self.s.get(url, timeout=60)
            return r.content if r.status_code == 200 else None
        except requests.RequestException:
            return None


def meta(htm, prop):
    for pat in (rf'<meta[^>]+(?:property|name)=["\']{prop}["\'][^>]+content=["\']([^"\']+)["\']',
                rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']{prop}["\']'):
        m = re.search(pat, htm, re.I)
        if m:
            return html.unescape(m.group(1))
    return None


def is_bandcamp(url):
    return urlparse(url).netloc.endswith("bandcamp.com")


def is_release(url):
    return bool(re.search(r"/(album|track)/[^/?#]+", urlparse(url).path))


def releases_of_artist(web, url, cap):
    """All /album/ and /track/ links on a Bandcamp artist's root and /music pages."""
    root = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    found = []
    for u in (root, root + "/music"):
        htm = web.page(u)
        if not htm:
            continue
        for m in re.finditer(r'href=["\']([^"\']*/(?:album|track)/[^"\'?#]+)', htm):
            full = urljoin(root + "/", html.unescape(m.group(1)))
            if urlparse(full).netloc == urlparse(root).netloc and full not in found:
                found.append(full)
    return found[:cap]


def expand(web, url, cap):
    if url.lower().split("?")[0].endswith(IMG_EXT):
        return [("image", url)]
    if is_bandcamp(url) and not is_release(url):
        rel = releases_of_artist(web, url, cap)
        if not rel:
            log(f"  {url}: no releases found (is this the artist's Bandcamp address?)")
        return [("page", r) for r in rel]
    return [("page", url)]


def cover_of(web, page_url):
    htm = web.page(page_url)
    if not htm:
        return None, None
    img = meta(htm, "og:image")
    title = meta(htm, "og:title") or ""
    return (urljoin(page_url, img) if img else None), title


def save_image(data, dst, max_side):
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(data))
        im.seek(0)
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        tmp = dst.with_suffix(".part")
        im.save(tmp, "JPEG", quality=88)
        os.replace(tmp, dst)
        return True
    except Exception:
        return False


def merge(path, new, keys):
    if path.exists():
        old = pd.read_csv(path, dtype=str)
        new = pd.concat([old, new.astype(str)], ignore_index=True)
    return new.drop_duplicates(keys, keep="first")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", required=True, help="CSV with columns board,url,kind,note")
    ap.add_argument("--out", default="arena_taste")
    ap.add_argument("--label", default="laptop twee", help="label written for kind=target rows")
    ap.add_argument("--max-per-artist", type=int, default=30)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between page requests")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    web = Web(out / "web_cache", a.delay)
    src = pd.read_csv(a.sources, comment="#", skip_blank_lines=True).fillna("")
    for c in ("board", "url"):
        if c not in src:
            sys.exit(f"{a.sources} needs a '{c}' column")
    photos, links, boards = [], [], {}
    seen_img = set()
    for r in src.itertuples(index=False):
        board, url = str(r.board).strip(), str(r.url).strip()
        kind = (getattr(r, "kind", "") or "target").strip().lower()
        if not board or not url:
            continue
        bid = "manual:" + hashlib.md5(board.encode()).hexdigest()[:10]
        boards.setdefault(bid, dict(board_id=bid, slug=re.sub(r"\W+", "-", board.lower()).strip("-"), title=board,
                                    created_at="", n_contents=0, found_by="manual",
                                    role="aesthetic" if kind == "target" else "decoy"))
        got = 0
        for how, u in expand(web, url, a.max_per_artist):
            img, title = (u, "") if how == "image" else cover_of(web, u)
            if not img or img in seen_img:
                continue
            pid = "web_" + hashlib.md5(img.encode()).hexdigest()[:12]
            dst = out / "images" / f"{pid}.jpg"
            if not dst.exists():
                data = web.image(img)
                if not data or not save_image(data, dst, a.max_side):
                    log(f"  could not download {img}"); continue
            seen_img.add(img)
            photos.append(dict(photo_id=pid, image_path=f"images/{pid}.jpg", image_url=img, block_title=title,
                               source_url=u, kind="manual_" + kind, label=a.label if kind == "target" else board))
            links.append(dict(photo_id=pid, board_id=bid, board_title=board, connected_at=""))
            got += 1
        boards[bid]["n_contents"] += got
        log(f"  {board!r} <- {url}: {got} images")
    if not photos:
        sys.exit("nothing downloaded; check the URLs in the sources file")
    p = merge(out / "photos.csv", pd.DataFrame(photos), ["photo_id"])
    l = merge(out / "links.csv", pd.DataFrame(links), ["photo_id", "board_id"])
    b = merge(out / "boards.csv", pd.DataFrame(boards.values()), ["board_id"])
    p.to_csv(out / "photos.csv", index=False)
    l.to_csv(out / "links.csv", index=False)
    b.to_csv(out / "boards.csv", index=False)
    n_t = sum(1 for x in photos if x["kind"] == "manual_target")
    log(f"done: {len(photos)} images ({n_t} target, {len(photos) - n_t} decoy) on {len(boards)} boards -> {out}")
    per = pd.DataFrame(links).merge(pd.DataFrame(boards.values()), on="board_id").groupby(["title", "role"]).size()
    print(per.to_string())


if __name__ == "__main__":
    main()

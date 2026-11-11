#!/usr/bin/env python3
"""
arena_collect.py -- build a taste dataset from Are.na for the community-memory demo.

What it collects
  aesthetic channels   for each aesthetic name (from the Zenodo deposit's 1,115-aesthetic list, plus
                       any you add, e.g. "laptop twee"), public channels whose title matches the name
  literal channels     channels titled with plain words ("laptop", "office", "desk setup"...). These
                       give the search something literal to fall into, from the SAME platform, so the
                       demo has no domain shift.
  deposit images       the 1,978 labelled blocks in the deposit, with every channel each one is in
                       (arena_human_labels.csv + arena_block_channels.csv). No API calls needed.
  co-curation          optionally, for a sample of collected blocks, the other channels they appear in

How channels are found (in this order, automatically)
  1. Are.na search (v3). It needs a Premium account; without one it answers 403.
  2. The older v2 search, if Are.na still serves it.
  3. No search: a pool of known channels. It holds every channel named in the deposit (thousands, with
     titles), plus every channel found through co-curation as the run goes ("snowball"). Terms are
     matched against the pool's titles.
  4. Always: channels you name yourself with --seeds, for things you found on the Are.na website.

Outputs (in --out)
  photos.csv     photo_id, image_path, image_url, block_title, block_created_at, source_url, kind, label
  links.csv      photo_id, board_id, board_title, connected_at          (one row per block-in-channel)
  boards.csv     board_id, slug, title, created_at, n_contents, found_by, role
  images/        <photo_id>.jpg, re-encoded, longest side <= --max-side
  api_cache/     every API response, so re-runs are free and the run can be resumed after a stop

Run
  pip install requests pillow pandas
  export ARENA_TOKEN=...            # optional: personal access token from are.na/developers
                                    # (120 reads/min instead of 30 for guests)
  python arena_collect.py --probe frutiger-aero-ptg8tgmzbig      # 1) check the API's field names
  python arena_collect.py --deposit path/to/aesthetics --extra "laptop twee,office siren" \
      --n-aesthetics 60 --out arena_taste                         # 2) collect
  ... --seeds "laptop twee=some-channel-slug,another-slug;office siren=a-slug"   # channels you found yourself
  ... --seeds seeds.csv                                            # or a CSV with columns term,slug

Etiquette: requests are rate limited below Are.na's published limits and cached; images are for your
own research and must not be redistributed (the deposit marks them "copyright held by uploader;
research-only"). Check are.na/terms before sharing anything built from them.
"""
import argparse
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

try:
    import requests
except ImportError:
    sys.exit("pip install requests pillow pandas")

LITERAL = ["laptop", "laptops", "computer", "office", "office interior", "desk setup", "workspace", "keyboard",
           "notebook", "camera", "phone", "car", "chair", "kitchen", "bedroom", "living room", "architecture",
           "street", "portrait", "landscape", "flowers", "food", "fashion", "shoes", "typography"]


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(t).lower())).strip()


def trigrams(s):
    s = f" {s} "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def title_matches(term, title, exact=False):
    """Aesthetic terms: every word of the term is in the title (plural 's' ignored), or the titles are
    near-identical spellings ('laptop-twee', 'laptoptwee'). Literal words (exact=True): the title must BE
    the word, so 'laptop' never picks up 'laptop twee' channels."""
    t, c = norm(term), norm(title)
    if not t or not c:
        return False
    tw = {w.rstrip("s") for w in t.split()}
    cw = {w.rstrip("s") for w in c.split()}
    if exact:
        return tw == cw
    if tw <= cw:
        return True
    a, b = trigrams(t.replace(" ", "")), trigrams(c.replace(" ", ""))
    return len(a & b) / max(1, len(a | b)) >= 0.7


# ======================================================================================
# API client: rate limited, cached, tolerant of v3 and v2 response shapes
# ======================================================================================
class Arena:
    def __init__(self, cache, base="https://api.are.na", version="v3", token=None, rpm=None):
        self.cache, self.base, self.v = Path(cache), base.rstrip("/"), version
        self.cache.mkdir(parents=True, exist_ok=True)
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "taste-dataset-research/1.0"
        if token:
            self.s.headers["Authorization"] = f"Bearer {token}"
        self.rpm = rpm or (100 if token else 25)        # below the published 120 / 30 per minute
        self.stamps, self.lock, self.calls = deque(), threading.Lock(), 0
        self.last = None                                  # (status, url, short body) of the last network call
        self.search_mode = version                        # v3 -> v2 -> none, as each turns out unavailable

    def _wait(self):
        with self.lock:
            now = time.time()
            while self.stamps and now - self.stamps[0] > 60:
                self.stamps.popleft()
            if len(self.stamps) >= self.rpm:
                time.sleep(60 - (now - self.stamps[0]) + 0.05)
            self.stamps.append(time.time())

    def get(self, path, version=None, **params):
        ver = version or self.v
        params = {k: v for k, v in params.items() if v is not None}
        key = hashlib.md5((ver + path + json.dumps(params, sort_keys=True)).encode()).hexdigest()
        f = self.cache / f"{key}.json"
        if f.exists():
            return json.loads(f.read_text())
        url = f"{self.base}/{ver}/{path.lstrip('/')}"
        for attempt in range(8):
            self._wait()
            try:
                r = self.s.get(url, params=params, timeout=30)
            except requests.RequestException as e:
                log(f"  network error ({e}); retrying"); time.sleep(2 ** attempt); continue
            self.calls += 1
            self.last = (r.status_code, r.url, r.text[:600])
            if r.status_code == 429:
                reset = r.headers.get("Retry-After") or r.headers.get("X-RateLimit-Reset")
                try:
                    wait = float(reset)
                    wait = wait - time.time() if wait > 1e9 else wait          # epoch or seconds
                except (TypeError, ValueError):
                    wait = 2 ** attempt * 5
                log(f"  rate limited; waiting {max(5, wait):.0f}s"); time.sleep(max(5, wait)); continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt); continue
            if r.status_code >= 400:                       # never cached: a fixed token or URL must retry
                return {"_status": r.status_code, "_body": r.text[:600]}
            try:
                data = r.json()
            except ValueError:
                return {"_status": r.status_code, "_body": r.text[:600]}
            f.write_text(json.dumps(data))
            return data
        raise RuntimeError(f"gave up on {url} {params}")

    # ---------------- shape-tolerant helpers ----------------
    @staticmethod
    def items(resp, *keys):
        for k in ("data",) + keys:
            if isinstance(resp.get(k), list):
                return resp[k]
        return []

    @staticmethod
    def more(resp, page, n_items, per):
        m = resp.get("meta") or {}
        if "has_more_pages" in m:
            return bool(m["has_more_pages"])
        if "total_pages" in m:
            return page < int(m["total_pages"] or 0)
        if "total_pages" in resp:                                              # v2 style
            return page < int(resp["total_pages"] or 0)
        return n_items >= per

    def paged(self, path, list_keys, per=100, max_items=None, **params):
        page, out = 1, []
        while True:
            r = self.get(path, page=page, per=per, **params)
            if r.get("_status"):
                return out
            got = self.items(r, *list_keys)
            out += got
            if not got or (max_items and len(out) >= max_items) or not self.more(r, page, len(got), per):
                return out[:max_items] if max_items else out
            page += 1

    def search_raw(self, term, per=50):
        if self.search_mode == "v3":
            r = self.get("search", page=1, per=per, query=term, type="Channel")
            if r.get("_status") not in (401, 403):
                return r
            log(f"  v3 search is not available to this account (HTTP {r['_status']}: "
                f"{r.get('_body', '')[:140]}); trying the older v2 search")
            self.search_mode = "v2"
        if self.search_mode == "v2":
            r = self.get("search/channels", version="v2", page=1, per=per, q=term)
            if not r.get("_status"):
                return r
            log(f"  v2 search is not available either (HTTP {r['_status']}); finding channels through the "
                f"deposit and co-curation instead")
            self.search_mode = "none"
        return {"_status": "no-search"}

    def search_channels(self, term, per=50):
        return self.items(self.search_raw(term, per), "channels")

    def contents(self, slug, max_items):
        return self.paged(f"channels/{slug}/contents", ("contents",), per=100, max_items=max_items)

    def block_channels(self, block_id, max_items=100):
        if self.v == "v3":
            return self.paged(f"blocks/{block_id}/connections", ("channels",), per=100, max_items=max_items)
        return self.paged(f"blocks/{block_id}/channels", ("channels",), per=100, max_items=max_items)


def is_image(b):
    return (b.get("type") or b.get("class")) == "Image"


def image_url(b):
    im = b.get("image") or {}
    for path in (("large", "src"), ("large", "url"), ("display", "url"), ("display", "src"), ("medium", "src"),
                 ("original", "src"), ("original", "url"), ("src",), ("url",)):
        x = im
        for k in path:
            x = x.get(k) if isinstance(x, dict) else None
        if isinstance(x, str) and x.startswith("http"):
            return x
    stack = [im]
    while stack:                                                                 # any URL in the image object
        v = stack.pop()
        if isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, str) and v.startswith("http"):
            return v
    return None


def connected_at(b):
    c = b.get("connection") or {}
    return c.get("connected_at") or c.get("created_at") or b.get("connected_at") or b.get("created_at")


def source_url(b):
    s = b.get("source") or {}
    return s.get("url") if isinstance(s, dict) else None


def n_contents(ch):
    c = ch.get("counts") or {}
    return c.get("contents") or c.get("blocks") or ch.get("length") or 0


def is_public(ch):
    return (ch.get("visibility") or ch.get("status") or "public") != "private"


# ======================================================================================
# collection
# ======================================================================================
class Store:
    def __init__(self):
        self.photos, self.links, self.boards = {}, {}, {}
        self.collected = set()                       # slugs whose contents were fetched as aesthetic/literal

    def board(self, ch, found_by, role):
        bid = str(ch.get("id") or ch.get("slug"))
        if bid not in self.boards:
            self.boards[bid] = dict(board_id=bid, slug=ch.get("slug"), title=ch.get("title", ""),
                                    created_at=ch.get("created_at"), n_contents=n_contents(ch),
                                    found_by=found_by, role=role)
        elif self.boards[bid]["role"] == "cocuration" and role in ("aesthetic", "literal"):
            self.boards[bid].update(found_by=found_by, role=role)      # first seen via co-curation, now collected
        return bid

    def photo(self, b, kind, label=None):
        pid = str(b.get("id"))
        url = image_url(b)
        if not url:
            return None
        if pid not in self.photos:
            self.photos[pid] = dict(photo_id=pid, image_path=f"images/{pid}.jpg", image_url=url,
                                    block_title=b.get("title") or "", block_created_at=b.get("created_at"),
                                    source_url=source_url(b), kind=kind, label=label)
        elif label and not self.photos[pid].get("label"):
            self.photos[pid]["label"] = label
        return pid

    def link(self, pid, bid, title, when):
        k = (pid, bid)
        if k not in self.links:
            self.links[k] = dict(photo_id=pid, board_id=bid, board_title=title, connected_at=when)


def explain(api, resp, term):
    """Print what the API actually answered, so an empty result can be told apart from a failure."""
    log(f"  search for {term!r} returned no channels. What the API answered:")
    if resp.get("_status"):
        log(f"    HTTP {resp['_status']}: {resp.get('_body', '')[:400]}")
    else:
        log(f"    top-level keys {list(resp.keys())}; first 400 chars: {json.dumps(resp)[:400]}")
    if api.last:
        log(f"    last request: HTTP {api.last[0]} {api.last[1]}")


def has_size(c):
    return "counts" in c or "length" in c


def pool_candidates(api, pool, term, exact, a):
    """Channels in the pool whose title matches the term. Pool entries known only by slug and title are
    looked up (1 call each) to learn their size and visibility; at most 4x --channels-per-term lookups."""
    cands = [c for c in pool.values() if title_matches(term, c.get("title", ""), exact=exact)]
    cands.sort(key=lambda c: (has_size(c), n_contents(c)), reverse=True)
    out, lookups = [], 0
    for c in cands:
        if not has_size(c):
            if lookups >= 4 * a.channels_per_term:
                continue
            info = api.get(f"channels/{c['slug']}")
            lookups += 1
            if info.get("_status"):
                continue
            pool[c["slug"]] = c = info
        out.append(c)
    return out


def collect_channels(api, st, term, role, chans, a):
    """Fetch the contents of channels not collected before. Returns (channels collected, image links)."""
    n_img, n_ch = 0, 0
    for ch in chans:
        key = ch.get("slug") or str(ch.get("id"))
        if key in st.collected:
            continue
        st.collected.add(key)
        n_ch += 1
        bid = st.board(ch, term, role)
        for b in api.contents(ch.get("slug") or ch.get("id"), a.blocks_per_channel):
            if is_image(b):
                pid = st.photo(b, role)
                if pid:
                    st.link(pid, bid, ch.get("title", ""), connected_at(b)); n_img += 1
    return n_ch, n_img


def collect_term(api, st, term, role, a, pool):
    resp = api.search_raw(term)
    if resp.get("_status") == "no-search":
        found, via = pool_candidates(api, pool, term, role == "literal", a), "pool"
    else:
        found, via = api.items(resp, "channels"), "search"
        if not found:
            explain(api, resp, term)
    pub = [c for c in found if is_public(c)]
    tm = [c for c in pub if title_matches(term, c.get("title", ""), exact=(role == "literal"))]
    chans = [c for c in tm if n_contents(c) >= a.min_contents]
    if found and not chans and via == "search":
        ex = found[0]
        log(f"  {term!r}: search returned {len(found)} channels; {len(pub)} public, {len(tm)} with a matching title, "
            f"0 with >= {a.min_contents} items. First result: title={ex.get('title')!r}, counts={ex.get('counts')}, "
            f"keys={sorted(k for k in ex if not k.startswith('_'))}")
    chans = [c for c in chans if (c.get("slug") or str(c.get("id"))) not in st.collected]
    chans = sorted(chans, key=n_contents, reverse=True)[:a.channels_per_term]
    n_ch, n_img = collect_channels(api, st, term, role, chans, a)
    log(f"  {role:9s} {term!r}: {n_ch} new channels, {n_img} images (via {via})")
    return n_ch, n_img


def ingest_deposit(st, dep):
    lab = pd.read_csv(dep / "arena_human_labels.csv")
    for r in lab.itertuples(index=False):
        pid = str(r.block_id)
        if pid not in st.photos:
            st.photos[pid] = dict(photo_id=pid, image_path=f"images/{pid}.jpg", image_url=r.image_url,
                                  block_title=getattr(r, "title", "") or "", block_created_at=None,
                                  source_url=getattr(r, "source_url", None), kind="deposit", label=r.aesthetic)
        else:
            st.photos[pid]["label"] = st.photos[pid].get("label") or r.aesthetic
        bid = f"slug:{r.channel_slug}"
        st.boards.setdefault(bid, dict(board_id=bid, slug=r.channel_slug, title=r.channel_title, created_at=None,
                                       n_contents=getattr(r, "channel_length", None), found_by="deposit",
                                       role="deposit"))
        st.link(pid, bid, r.channel_title, getattr(r, "connected_at", None))
    bc = pd.read_csv(dep / "arena_block_channels.csv")
    for r in bc.itertuples(index=False):
        pid = str(r.block_id)
        if pid not in st.photos:
            continue
        bid = f"slug:{r.channel_slug}"
        st.boards.setdefault(bid, dict(board_id=bid, slug=r.channel_slug, title=r.channel_title, created_at=None,
                                       n_contents=None, found_by="deposit co-curation", role="cocuration"))
        st.link(pid, bid, r.channel_title, None)
    log(f"deposit: {len(lab)} labelled images, {len(bc)} block-channel rows")
    pool = {}
    for r in pd.concat([lab[["channel_slug", "channel_title"]], bc[["channel_slug", "channel_title"]]]).itertuples(index=False):
        if isinstance(r.channel_slug, str):
            pool.setdefault(r.channel_slug, {"slug": r.channel_slug, "title": str(r.channel_title)})
    return pool


def unify_slug_ids(st):
    """Deposit boards are keyed by slug, API boards by id. Merge them so one channel is one board."""
    by_slug = {b["slug"]: k for k, b in st.boards.items() if not k.startswith("slug:") and b.get("slug")}
    remap = {k: by_slug[b["slug"]] for k, b in st.boards.items() if k.startswith("slug:") and b["slug"] in by_slug}
    for old, new in remap.items():
        st.boards.pop(old, None)
    links = {}
    for (pid, bid), l in st.links.items():
        nb = remap.get(bid, bid)
        l = dict(l, board_id=nb)
        if (pid, nb) not in links or (l["connected_at"] and not links[(pid, nb)]["connected_at"]):
            links[(pid, nb)] = l
    st.links = links


def cocurate(api, st, a, pool, n, done):
    """For up to n not-yet-expanded blocks, every other channel they are in. New channels join the pool."""
    pids = [p for p, v in st.photos.items() if v["kind"] != "deposit" and p not in done]
    if not pids or n <= 0:
        return 0
    import random
    random.Random(len(done)).shuffle(pids)
    pids = pids[:n]
    before = len(pool)
    log(f"co-curation: other channels for {len(pids)} blocks (~{len(pids) / api.rpm:.0f} min at {api.rpm}/min)")
    for i, pid in enumerate(pids):
        done.add(pid)
        for ch in api.block_channels(pid, max_items=a.channels_per_block):
            if is_public(ch):
                bid = st.board(ch, "co-curation", "cocuration")
                st.link(pid, bid, ch.get("title", ""), None)
                if ch.get("slug"):
                    pool[ch["slug"]] = ch
        if i % 200 == 0:
            log(f"  {i}/{len(pids)}")
    log(f"co-curation: pool grew from {before} to {len(pool)} known channels")
    return len(pids)


def download(st, out, max_side, workers=8):
    from PIL import Image
    sess = requests.Session()
    sess.headers["User-Agent"] = "taste-dataset-research/1.0"
    (out / "images").mkdir(parents=True, exist_ok=True)
    todo = [p for p in st.photos.values() if not (out / p["image_path"]).exists()]
    log(f"images: {len(st.photos) - len(todo)} already on disk, downloading {len(todo)}")
    ok = [0]

    def one(p):
        dst = out / p["image_path"]
        for attempt in range(3):
            try:
                r = sess.get(p["image_url"], timeout=60)
                if r.status_code != 200:
                    time.sleep(2 ** attempt); continue
                im = Image.open(io.BytesIO(r.content))
                im.seek(0)
                im = im.convert("RGB")
                im.thumbnail((max_side, max_side))
                tmp = dst.with_suffix(".part")
                im.save(tmp, "JPEG", quality=88)
                os.replace(tmp, dst)
                ok[0] += 1
                return
            except Exception:
                time.sleep(2 ** attempt)

    with ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(one, todo)):
            if i % 500 == 0 and i:
                log(f"  {i}/{len(todo)}")
    log(f"images: {ok[0]} downloaded, {len(todo) - ok[0]} failed (kept out of photos.csv)")


def read_seeds(spec):
    """'term=slug,slug;term2=slug' or a CSV file with columns term,slug."""
    if not spec:
        return []
    if Path(spec).exists():
        df = pd.read_csv(spec)
        return [(str(t).strip(), str(s).strip()) for t, s in zip(df["term"], df["slug"])]
    out = []
    for part in spec.split(";"):
        if "=" in part:
            t, slugs = part.split("=", 1)
            out += [(t.strip(), s.strip()) for s in slugs.split(",") if s.strip()]
    return out


def probe(api, slug):
    """Print the real field names, so you can check the parser against the live API."""
    ch = api.get(f"channels/{slug}")
    print("channel keys:", sorted(k for k in ch if not k.startswith("_")) if isinstance(ch, dict) else ch)
    r = api.get(f"channels/{slug}/contents", page=1, per=10)
    print("contents top-level keys:", list(r.keys()))
    its = api.items(r, "contents")
    img = next((b for b in its if is_image(b)), None)
    if not img:
        print("no image block on the first page"); return
    print("image block keys:", sorted(img.keys()))
    print("image object keys:", json.dumps({k: (sorted(v) if isinstance(v, dict) else type(v).__name__)
                                            for k, v in (img.get("image") or {}).items()}))
    print("connection object:", json.dumps(img.get("connection"), default=str)[:400])
    print("-> parsed: id", img.get("id"), "| url", image_url(img), "| connected_at", connected_at(img))
    cons = api.block_channels(img.get("id"), max_items=3)
    print("block connections:", [(c.get("id"), c.get("slug"), c.get("title")) for c in cons])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deposit", default=None, help="folder with the unzipped Zenodo deposit (optional)")
    ap.add_argument("--n-aesthetics", type=int, default=60,
                    help="aesthetics to collect: the deposit's Are.na aesthetics first, then the most-linked names")
    ap.add_argument("--extra", default="laptop twee,office siren", help="comma-separated names to always include")
    ap.add_argument("--literal", default=",".join(LITERAL), help="plain-word channels (the literal readings)")
    ap.add_argument("--channels-per-term", type=int, default=8)
    ap.add_argument("--blocks-per-channel", type=int, default=200)
    ap.add_argument("--min-contents", type=int, default=10)
    ap.add_argument("--cocuration", type=int, default=2000, help="blocks to expand into all their channels (0 = off)")
    ap.add_argument("--channels-per-block", type=int, default=100)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--out", default="arena_taste")
    ap.add_argument("--api-version", default="v3", choices=["v3", "v2"])
    ap.add_argument("--base-url", default="https://api.are.na")
    ap.add_argument("--rpm", type=int, default=None, help="requests per minute (default 25 guest / 100 with token)")
    ap.add_argument("--probe", default=None, help="channel slug: print the API's field names and exit")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--seeds", default=None, help='"term=slug,slug;term=slug" or a CSV with columns term,slug')
    ap.add_argument("--snowball", type=int, default=2,
                    help="without search: rounds of co-curation that grow the channel pool (each --cocuration blocks)")
    a = ap.parse_args()
    out = Path(a.out)
    api = Arena(out / "api_cache", a.base_url, a.api_version, os.environ.get("ARENA_TOKEN"), a.rpm)
    if a.probe:
        probe(api, a.probe); return
    st = Store()
    terms, pool = [], {}
    if a.deposit:
        dep = Path(a.deposit)
        need = ["arena_human_labels.csv", "arena_block_channels.csv", "wiki_ecosystem_nodes.csv"]
        missing = [f for f in need if not (dep / f).exists()]
        if missing:
            hint = [str(p.parent) for p in Path(".").rglob("arena_human_labels.csv")][:3]
            sys.exit(f"--deposit {dep}: missing {', '.join(missing)}."
                     + (f" Found the deposit files in: {', '.join(hint)} (use --deposit with that folder)." if hint else ""))
        pool = ingest_deposit(st, dep)
        labelled = list(dict.fromkeys(pd.read_csv(dep / "arena_human_labels.csv")["aesthetic"].dropna()))
        nodes = pd.read_csv(dep / "wiki_ecosystem_nodes.csv").sort_values("degree", ascending=False)
        terms = labelled + [x for x in nodes["aesthetic"] if x not in labelled]
    extra = [x.strip() for x in a.extra.split(",") if x.strip()]
    terms = list(dict.fromkeys(extra + terms[:max(0, a.n_aesthetics - len(extra))]))
    literal = [x.strip() for x in a.literal.split(",") if x.strip()]
    est = (len(terms) + len(literal)) * (1 + a.channels_per_term * (1 + a.blocks_per_channel // 100)) + a.cocuration
    log(f"{len(terms)} aesthetics + {len(literal)} literal words; at most ~{est} API calls "
        f"(~{est / api.rpm:.0f} min at {api.rpm}/min; cached calls are free on re-runs)")
    seeds = read_seeds(a.seeds)
    seeded = {}
    for t, slug in seeds:
        info = api.get(f"channels/{slug}")
        if info.get("_status"):
            log(f"  seed {slug!r} for {t!r}: HTTP {info['_status']} (check the slug in the channel's URL)"); continue
        pool[slug] = info
        role = "literal" if t in literal else "aesthetic"
        n_ch, n = collect_channels(api, st, t, role, [info], a)
        c0, i0 = seeded.get(t, (0, 0))
        seeded[t] = (c0 + n_ch, i0 + n)
        log(f"  seed      {t!r}: channel {info.get('title')!r}, {n} images")
    summary = {}

    def run(words, role):
        for t in words:
            try:
                nc, ni = collect_term(api, st, t, role, a, pool)
            except Exception as e:
                log(f"  {t!r} failed: {e}"); nc, ni = 0, 0
            c0, i0 = seeded.get(t, (0, 0))
            prev = summary.get(t, dict(term=t, role=role, channels=c0, images=i0))
            summary[t] = dict(prev, channels=prev["channels"] + nc, images=prev["images"] + ni)
            if api.search_mode == "none" and not pool:
                sys.exit("Search is not available to this account and there is no channel pool to fall back on. "
                         "Pass --deposit (its channels seed the pool) and/or --seeds with channel slugs.")

    run(terms, "aesthetic")
    run(literal, "literal")
    done = set()
    if api.search_mode == "none":
        for rnd in range(a.snowball):
            if not cocurate(api, st, a, pool, a.cocuration, done):
                break
            empty = [t for t, v in summary.items() if v["channels"] < a.channels_per_term]
            log(f"snowball round {rnd + 1}: retrying {len(empty)} terms with fewer than {a.channels_per_term} channels")
            for t in empty:
                role = summary[t]["role"]
                cands = [c for c in pool_candidates(api, pool, t, role == "literal", a)
                         if (c.get("slug") or str(c.get("id"))) not in st.collected
                         and is_public(c) and n_contents(c) >= a.min_contents]
                cands = sorted(cands, key=n_contents, reverse=True)[:a.channels_per_term - summary[t]["channels"]]
                if cands:
                    nc, ni = collect_channels(api, st, t, role, cands, a)
                    summary[t]["channels"] += nc; summary[t]["images"] += ni
                    log(f"  {role:9s} {t!r}: +{nc} channels, +{ni} images (via snowball)")
    else:
        cocurate(api, st, a, pool, a.cocuration, done)
    summary = list(summary.values())
    unify_slug_ids(st)
    if not a.no_images:
        download(st, out, a.max_side)
    photos = pd.DataFrame(st.photos.values())
    if not a.no_images and len(photos):
        photos = photos[[(out / p).exists() for p in photos["image_path"]]]
    keep = set(photos["photo_id"]) if len(photos) else set()
    links = pd.DataFrame([l for l in st.links.values() if l["photo_id"] in keep])
    used = set(links["board_id"]) if len(links) else set()
    boards = pd.DataFrame([b for k, b in st.boards.items() if k in used])
    out.mkdir(parents=True, exist_ok=True)
    photos.to_csv(out / "photos.csv", index=False)
    links.to_csv(out / "links.csv", index=False)
    boards.to_csv(out / "boards.csv", index=False)
    sm = pd.DataFrame(summary)
    sm.to_csv(out / "collection_summary.csv", index=False)
    log(f"done: {len(photos)} images, {len(boards)} boards, {len(links)} image-board links, {api.calls} API calls")
    if len(sm):
        empty = sm[sm.channels == 0].term.tolist()
        if empty:
            log(f"no matching channels for: {', '.join(empty[:20])}{' ...' if len(empty) > 20 else ''}")
            if api.search_mode == "none":
                log("  without search, a term is only found if one of its channels is already known. Find channels "
                    "on are.na in a browser and pass them with --seeds \"term=slug,slug\" (the slug is the end of "
                    "the channel URL).")


if __name__ == "__main__":
    main()

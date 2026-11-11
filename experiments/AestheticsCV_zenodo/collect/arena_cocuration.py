"""
arena_cocuration.py
===================
Component 3 harvest: for each are.na block we already collected, fetch the FULL set of
channels users filed it into (GET /v2/blocks/:id/channels). This is an INDEPENDENT
behavioral signal (user co-filing) distinct from the Aesthetics-Wiki editors'
related-links -- the basis for a cross-source replication that defuses circularity.

Output: arena_block_channels.csv  (block_id, src_aesthetic, channel_title, channel_slug)
Resumable; polite pacing + 429 backoff (are.na throttles).
"""
from __future__ import annotations
import argparse, csv, os, sys, time, json, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.parse, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = "https://api.are.na/v2"
UA = {"User-Agent": "AestheticsCV-research/1.0 (academic; yl2884@cornell.edu)"}
PACE = 0.0          # concurrency replaces global pacing; per-request 429 backoff handles limits
_last = [0.0]


def _get(path, params=None):
    url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    last = None
    for attempt in range(7):
        w = PACE - (time.time() - _last[0])
        if w > 0:
            time.sleep(w)
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=25) as r:
                _last[0] = time.time()
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            _last[0] = time.time(); last = e
            time.sleep(min(60.0, 3.0 * (2 ** attempt)) if e.code == 429 else 1.5 * (attempt + 1))
        except Exception as e:
            _last[0] = time.time(); last = e; time.sleep(1.5 * (attempt + 1))
    print(f"  ! {path}: {last}", file=sys.stderr)
    return None


def block_channels(bid):
    out, page = [], 1
    while True:
        j = _get(f"/blocks/{urllib.parse.quote(str(bid))}/channels", {"per": 100, "page": page})
        if not j:
            break
        chans = j.get("channels") or []
        if not chans:
            break
        for c in chans:
            out.append((c.get("title") or "", c.get("slug") or ""))
        if page >= (j.get("total_pages") or page):
            break
        page += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    WORKERS = ap.parse_args().workers
    labels = list(csv.DictReader(open(os.path.join(ROOT, "arena_human_labels.csv"), encoding="utf-8")))
    src = {}
    for r in labels:
        src.setdefault(r["block_id"], r["aesthetic"])
    out_path = os.path.join(ROOT, "arena_block_channels.csv")

    done = set()
    if os.path.exists(out_path):
        for r in csv.DictReader(open(out_path, encoding="utf-8")):
            done.add(r["block_id"])
    todo = [b for b in src if b and b not in done]
    print(f"{len(src)} blocks, {len(done)} done, {len(todo)} to fetch", file=sys.stderr)

    new = not os.path.exists(out_path)
    f = open(out_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=["block_id", "src_aesthetic", "channel_title", "channel_slug"])
    if new:
        w.writeheader(); f.flush()
    lock = threading.Lock()
    counter = [0]

    def work(bid):
        chans = block_channels(bid)
        rows = ([{"block_id": bid, "src_aesthetic": src[bid], "channel_title": t,
                  "channel_slug": s} for t, s in chans]
                or [{"block_id": bid, "src_aesthetic": src[bid],
                     "channel_title": "", "channel_slug": ""}])
        with lock:
            w.writerows(rows); f.flush()
            counter[0] += 1
            if counter[0] % 50 == 0:
                print(f"  {counter[0]}/{len(todo)} blocks", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(as_completed([ex.submit(work, b) for b in todo]))
    f.close()
    print(f"done -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

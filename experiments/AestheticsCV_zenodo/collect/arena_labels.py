"""
arena_labels.py
===============
Harvest CROWD-CURATED image membership for each aesthetic from are.na via the keyless
public v2 API.

On are.na, users sort images into named channels. A channel called "vaporwave" full of
images is thousands of independent human decisions that "this image IS vaporwave". That
is exactly the image-level collective label the paper lacks: humans -- not our keyword
guesses -- define corpus membership. This directly addresses Limitation (i) (the corpus
must capture the post-processed *look*, not just subject matter).

Pipeline:
  1. search channels by aesthetic name        GET /v2/search/channels?q=
  2. keep the top channels by size (length)    -- the canonical, well-populated ones
  3. page through each channel's contents      GET /v2/channels/:slug/contents
  4. keep Image blocks; record image URL + original web source + the human label

Honest scope baked in:
  - are.na images are user-uploaded/linked and mostly COPYRIGHTED. We record URLs +
    source for FEATURE EXTRACTION (research analysis) only; they are NOT redistributed
    in the open deposit. The openly-licensed corpus remains the deposited one.
  - Channel membership is noisy (off-topic blocks happen) -> a CLIP relevance filter is
    applied downstream, same as for the IA pull. We DO NOT pre-filter here.
  - are.na curators are a design-literate subpopulation, not the general public. Noted.

Output (project root): arena_human_labels.csv
  columns: block_id, aesthetic, cluster, channel_slug, channel_title, channel_length,
           image_url, source_url, title, connected_at
"""
from __future__ import annotations
import argparse, csv, os, sys, time, json
import urllib.parse, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from aesthetic_map import AESTHETICS

BASE = "https://api.are.na/v2"
UA = {"User-Agent": "AestheticsCV-research/1.0 (academic; contact yl2884@cornell.edu)"}


PACE = 0.7          # min seconds between any two requests (be polite to the public API)
_last = [0.0]


def _get(path, params=None):
    url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    last = None
    for attempt in range(7):
        # global pacing
        wait = PACE - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=25) as r:
                _last[0] = time.time()
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            _last[0] = time.time()
            last = e
            if e.code == 429:                       # exponential backoff on throttle
                time.sleep(min(60.0, 3.0 * (2 ** attempt)))
            else:
                time.sleep(1.5 * (attempt + 1))
        except Exception as e:
            _last[0] = time.time()
            last = e
            time.sleep(1.5 * (attempt + 1))
    print(f"  ! {path} failed: {last}", file=sys.stderr)
    return None


def search_channels(query, want=3, min_len=25, max_len=4000):
    """Return up to `want` channel dicts for the aesthetic, biggest first."""
    j = _get("/search/channels", {"q": query, "per": 30, "page": 1})
    if not j:
        return []
    chans = j.get("channels", [])
    # title must plausibly match the aesthetic (avoid generic homonym channels)
    q = query.lower().replace(" ", "")
    def matches(c):
        t = (c.get("title") or "").lower().replace(" ", "")
        return q in t or t in q
    cand = [c for c in chans if matches(c) and min_len <= (c.get("length") or 0) <= max_len]
    cand.sort(key=lambda c: c.get("length") or 0, reverse=True)
    return cand[:want]


def channel_images(slug, cap=300):
    """Yield image blocks from a channel, paginating until cap or exhausted."""
    got, page = 0, 1
    while got < cap:
        j = _get(f"/channels/{urllib.parse.quote(slug)}/contents",
                 {"per": 50, "page": page, "direction": "desc"})
        if not j:
            break
        contents = j.get("contents") or []
        if not contents:
            break
        for b in contents:
            if (b.get("class") or "") != "Image":
                continue
            img = b.get("image") or {}
            url = ((img.get("large") or {}).get("url")
                   or (img.get("original") or {}).get("url")
                   or (img.get("display") or {}).get("url"))
            if not url:
                continue
            src = (b.get("source") or {}).get("url") or ""
            yield {
                "block_id": b.get("id"),
                "image_url": url,
                "source_url": src,
                "title": (b.get("title") or "").replace("\n", " ").strip(),
                "connected_at": b.get("connected_at") or "",
            }
            got += 1
            if got >= cap:
                break
        total_pages = j.get("total_pages") or page
        if page >= total_pages:
            break
        page += 1
        time.sleep(0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels-per", type=int, default=3, help="top-N channels per aesthetic")
    ap.add_argument("--cap-per-channel", type=int, default=250)
    ap.add_argument("--out", default=os.path.join(ROOT, "arena_human_labels.csv"))
    ap.add_argument("--resume", action="store_true",
                    help="keep aesthetics already present in --out; only fetch the gaps")
    ap.add_argument("--min-len", type=int, default=15)
    args = ap.parse_args()

    seen_blocks = set()
    rows = []
    done_aes = set()
    if args.resume and os.path.exists(args.out):
        for r in csv.DictReader(open(args.out, encoding="utf-8")):
            rows.append(r)
            seen_blocks.add(r["block_id"])
            done_aes.add(r["aesthetic"])
        print(f"resume: {len(rows)} existing rows, {len(done_aes)} aesthetics already done",
              file=sys.stderr)

    per_aes = {}
    for name, (cluster, query) in AESTHETICS.items():
        if name in done_aes:
            continue
        chans = search_channels(query, want=args.channels_per, min_len=args.min_len)
        if not chans:
            print(f"[{cluster:14s}] {name:18s} -> NO channels", file=sys.stderr)
            continue
        n_before = len(rows)
        for c in chans:
            slug, title, length = c.get("slug"), c.get("title"), c.get("length") or 0
            for img in channel_images(slug, cap=args.cap_per_channel):
                bid = img["block_id"]
                if bid in seen_blocks:          # dedup blocks across channels/aesthetics
                    continue
                seen_blocks.add(bid)
                rows.append({"aesthetic": name, "cluster": cluster,
                             "channel_slug": slug, "channel_title": title,
                             "channel_length": length, **img})
            time.sleep(0.3)
        per_aes[name] = len(rows) - n_before
        print(f"[{cluster:14s}] {name:18s} -> {len(chans)} chan(s), "
              f"{per_aes[name]} imgs", file=sys.stderr)

    cols = ["block_id", "aesthetic", "cluster", "channel_slug", "channel_title",
            "channel_length", "image_url", "source_url", "title", "connected_at"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    print(f"\n=== are.na harvest done ===")
    print(f"total image labels: {len(rows)}  (unique blocks: {len(seen_blocks)})")
    for cl in ["restorative", "hauntological", "inverted", "anemoia"]:
        n = sum(1 for r in rows if r["cluster"] == cl)
        print(f"  {cl:14s}: {n}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

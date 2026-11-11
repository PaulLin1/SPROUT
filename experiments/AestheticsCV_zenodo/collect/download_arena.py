"""
download_arena.py
=================
Download the are.na crowd-curated images (from arena_human_labels.csv) into a local
corpus for FEATURE EXTRACTION, and emit a manifest compatible with run_pipeline.py.

This is the re-sourced, aesthetic-FAITHFUL, HUMAN-VALIDATED corpus: membership was
decided by are.na curators, not our keyword guesses -> directly addresses Limitation (i)
(capturing the post-processed *look*, not just subject matter).

IMPORTANT (licensing): are.na blocks are user-uploaded/linked and mostly COPYRIGHTED.
These files are kept LOCAL for research feature-extraction ONLY and are NOT redistributed
in the open deposit. The manifest records this in the `license` column.

  - resolution floor (>= MIN_SIDE long side)
  - perceptual-hash dedup across the whole pull (catches the same image reused in
    multiple channels / aesthetics)
  - concurrent download with 429 backoff

Manifest columns: image_path, image_id, scene_category, cluster, license, source_url,
                  aesthetic, channel_slug, block_id
"""
from __future__ import annotations
import argparse, csv, io, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
UA = {"User-Agent": "AestheticsCV-research/1.0 (academic; yl2884@cornell.edu)"}
MIN_SIDE = 384            # are.na 'large' is capped ~1800px; many memes are small -> 384 floor
PHASH_HAMMING = 6
DL_TIMEOUT = 15


def fetch(rec):
    from PIL import Image
    import imagehash
    url = rec["image_url"]
    blob = None
    for attempt in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=DL_TIMEOUT)
            if r.status_code == 429:
                time.sleep(1.5 * (attempt + 1)); continue
            r.raise_for_status(); blob = r.content; break
        except Exception:
            time.sleep(0.8 * (attempt + 1))
    if blob is None:
        return None
    try:
        im = Image.open(io.BytesIO(blob)).convert("RGB")
    except Exception:
        return None
    if max(im.size) < MIN_SIDE:
        return None
    rec["_phash"] = imagehash.phash(im)
    rec["_image"] = im
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join(ROOT, "arena_human_labels.csv"))
    ap.add_argument("--imgroot", default=os.path.join(ROOT, "corpus_arena", "images"))
    ap.add_argument("--out", default=os.path.join(ROOT, "stimuli_manifest_arena.csv"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cand = list(csv.DictReader(open(args.labels, encoding="utf-8")))
    print(f"{len(cand)} candidate human-labelled images", file=sys.stderr)

    fetched = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, r): r for r in cand}
        done = 0
        for fut in as_completed(futs):
            done += 1
            r = fut.result()
            if r is not None:
                fetched.append(r)
            if done % 200 == 0:
                print(f"  {done}/{len(cand)}  kept {len(fetched)}", file=sys.stderr)
    print(f"decoded >= {MIN_SIDE}px: {len(fetched)}", file=sys.stderr)

    seen, rows, per = [], [], {}
    for r in fetched:
        ph = r["_phash"]
        if any((ph - p) <= PHASH_HAMMING for p in seen):
            continue
        seen.append(ph)
        cluster = r["cluster"]
        image_id = (r["block_id"] or "")[:16] or f"{len(rows):06d}"
        cdir = os.path.join(args.imgroot, cluster)
        os.makedirs(cdir, exist_ok=True)
        path = os.path.join(cdir, f"{image_id}.jpg")
        r["_image"].save(path, "JPEG", quality=92)
        rows.append({
            "image_path": path, "image_id": image_id,
            "scene_category": "generic", "cluster": cluster,
            "license": "arena-curated (copyright held by uploader; research-only, not redistributed)",
            "source_url": r.get("source_url", ""),
            "aesthetic": r["aesthetic"], "channel_slug": r["channel_slug"],
            "block_id": r["block_id"],
        })
        per[cluster] = per.get(cluster, 0) + 1

    cols = ["image_path", "image_id", "scene_category", "cluster", "license",
            "source_url", "aesthetic", "channel_slug", "block_id"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    print(f"\n=== arena corpus built: {len(rows)} unique images ===")
    for c in ("restorative", "hauntological", "inverted", "anemoia"):
        print(f"  {c:14s}: {per.get(c,0)}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

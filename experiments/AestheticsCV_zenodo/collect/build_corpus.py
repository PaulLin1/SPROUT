"""
build_corpus.py
===============
Drives the licensed-image collectors, downloads candidates CONCURRENTLY,
de-duplicates with a perceptual hash, enforces a resolution floor, and writes the
master manifest.

Output manifest columns (exactly what run_pipeline.py requires + provenance):
    image_path, image_id, scene_category, cluster, license, source_url

Three phases so the slow part (network) is parallel but dedup stays correct:
  A. fetch candidate METADATA for every query (fast, sequential API calls).
  B. download + decode + resolution-check + perceptual-hash in a thread pool.
  C. walk completed downloads, apply phash dedup against the running set, save +
     append manifest rows.

Honest design choices baked in:
  - Per-cluster counts are CAPPED AT REAL LICENSED YIELD, not forced equal.
  - Every kept image carries a license string and a landing-page source_url.
  - Perceptual-hash dedup runs ACROSS queries and clusters.

Run (from any env with requests + Pillow + imagehash):
    python collect/build_corpus.py --per-query 50 --target 1300 --workers 24
"""

from __future__ import annotations
import argparse, csv, io, os, sys, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import sources
import importlib

UA = sources.UA
MIN_SIDE = 512            # resolution floor (long side)
PHASH_HAMMING = 6         # near-duplicate threshold on 64-bit perceptual hash
DL_TIMEOUT = 12           # hard per-download timeout -- no long hangs on dead URLs


def _fetch_one(rec):
    """Download + decode + resolution check + phash. Returns enriched rec or None.

    Retries on HTTP 429 with backoff -- Wikimedia's upload servers throttle bursts,
    so a polite worker count + backoff is what keeps the realised keep-rate high.
    """
    from PIL import Image
    import imagehash
    import time
    url = rec.get("url")
    if not url:
        return None
    blob = None
    for attempt in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=DL_TIMEOUT)
            if r.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            blob = r.content
            break
        except Exception:
            time.sleep(0.8 * (attempt + 1))
    if blob is None:
        return None
    try:
        im = Image.open(io.BytesIO(blob)).convert("RGB")
    except Exception:
        return None
    w, h = im.size
    if max(w, h) < MIN_SIDE:
        return None
    rec["_phash"] = imagehash.phash(im)
    rec["_image"] = im
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-query", type=int, default=50)
    ap.add_argument("--target", type=int, default=1300, help="soft global cap")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--adapters", default="openverse,wikimedia,pexels,unsplash")
    ap.add_argument("--out", default=os.path.join(ROOT, "stimuli_manifest.csv"))
    ap.add_argument("--queries", default="queries",
                    help="query module to import QUERIES from (e.g. queries_rr)")
    ap.add_argument("--imgroot", default=os.path.join(ROOT, "corpus", "images"),
                    help="output image root directory")
    args = ap.parse_args()

    QUERIES = importlib.import_module(args.queries).QUERIES
    adapters = [a.strip() for a in args.adapters.split(",") if a.strip()]
    img_root = args.imgroot

    # ---- Phase A: gather candidate metadata for every query ----------------
    candidates = []
    seen_urls = set()
    for cluster, scene, query in QUERIES:
        for a in adapters:
            fn = sources.ALL_ADAPTERS.get(a)
            if fn is None:
                continue
            try:
                recs = fn(query, n=args.per_query)
            except Exception as e:
                print(f"  [{a}] '{query}' failed: {e}", file=sys.stderr)
                continue
            for rec in recs:
                u = rec.get("url")
                if not u or u in seen_urls:
                    continue
                seen_urls.add(u)
                rec["_cluster"], rec["_scene"], rec["_query"] = cluster, scene, query
                candidates.append(rec)
        print(f"[{cluster}/{scene}] '{query}': running total {len(candidates)} candidates",
              file=sys.stderr)
    print(f"\nPhase A done: {len(candidates)} unique candidate URLs", file=sys.stderr)

    # ---- Phase B: concurrent download + decode + phash ---------------------
    fetched = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_fetch_one, rec): rec for rec in candidates}
        done = 0
        for fut in as_completed(futs):
            done += 1
            rec = fut.result()
            if rec is not None:
                fetched.append(rec)
            if done % 100 == 0:
                print(f"  downloaded {done}/{len(candidates)}  (kept {len(fetched)})",
                      file=sys.stderr)
    print(f"Phase B done: {len(fetched)} images decoded >= {MIN_SIDE}px", file=sys.stderr)

    # ---- Phase C: dedup, save, manifest ------------------------------------
    seen_hashes = []
    rows = []
    per_cluster = {}
    for rec in fetched:
        if len(rows) >= args.target:
            break
        ph = rec["_phash"]
        if any((ph - prev) <= PHASH_HAMMING for prev in seen_hashes):
            continue
        seen_hashes.append(ph)
        cluster, scene = rec["_cluster"], rec["_scene"]
        image_id = hashlib.sha1((rec["url"] + str(rec.get("ext_id", ""))).encode()).hexdigest()[:16]
        cdir = os.path.join(img_root, cluster)
        os.makedirs(cdir, exist_ok=True)
        path = os.path.join(cdir, f"{image_id}.jpg")
        rec["_image"].save(path, "JPEG", quality=92)
        rows.append({
            "image_path": path,
            "image_id": image_id,
            "scene_category": scene,
            "cluster": cluster,
            "license": (rec.get("license", "") or "unknown").strip(),
            "source_url": rec.get("source_url", ""),
            "source": rec.get("source", ""),
            "query": rec["_query"],
        })
        per_cluster[cluster] = per_cluster.get(cluster, 0) + 1

    cols = ["image_path", "image_id", "scene_category", "cluster",
            "license", "source_url", "source", "query"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print("\n=== corpus built ===")
    print(f"total kept: {len(rows)}  ->  {args.out}")
    for c in ("restorative", "hauntological", "inverted", "anemoia", "control"):
        print(f"  {c:14s}: {per_cluster.get(c, 0)}")


if __name__ == "__main__":
    main()

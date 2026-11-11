"""
run_gemini_affect.py
====================
Component 1 (judge #1 = Gemini): rate every image's affect with Gemini, resumable and
rate-limit-tolerant. One row per image; incremental flush so a crash never loses work.

Usage:
  python vlm/run_gemini_affect.py --model models/gemini-2.5-flash \
      --manifests stimuli_manifest_arena.csv stimuli_manifest.csv --out vlm_gemini_affect.csv
"""
from __future__ import annotations
import argparse, csv, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemini_client as gc
import affect_prompts as ap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ["image_id", "source", "aesthetic", "cluster"] + ap.AXES + ["mode", "ok"]


def load_manifest(path):
    src = "arena" if "arena" in os.path.basename(path) else "corpus"
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        rows.append({"image_path": r["image_path"], "image_id": r["image_id"],
                     "source": src, "aesthetic": r.get("aesthetic", ""),
                     "cluster": r.get("cluster", "")})
    return rows


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--model", default="models/gemini-2.5-flash")
    ap_.add_argument("--manifests", nargs="+",
                     default=[os.path.join(ROOT, "stimuli_manifest_arena.csv")])
    ap_.add_argument("--out", default=os.path.join(ROOT, "vlm_gemini_affect.csv"))
    ap_.add_argument("--sleep", type=float, default=0.25)
    ap_.add_argument("--limit", type=int, default=0)
    args = ap_.parse_args()

    key = gc.load_key()
    items = []
    seen = set()
    for m in args.manifests:
        for r in load_manifest(m):
            if r["image_id"] in seen:
                continue
            seen.add(r["image_id"]); items.append(r)

    done = set()
    if os.path.exists(args.out):
        for r in csv.DictReader(open(args.out, encoding="utf-8")):
            done.add(r["image_id"])
    todo = [r for r in items if r["image_id"] not in done and os.path.exists(r["image_path"])]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(items)} images, {len(done)} done, {len(todo)} to do -> {args.out}", file=sys.stderr)

    new = not os.path.exists(args.out)
    f = open(args.out, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
    if new:
        w.writeheader(); f.flush()

    n_ok = 0
    for i, r in enumerate(todo):
        rec = {"image_id": r["image_id"], "source": r["source"],
               "aesthetic": r["aesthetic"], "cluster": r["cluster"], "ok": 0}
        try:
            txt = gc.generate(key, args.model, ap.PROMPT, image_path=r["image_path"],
                              auth_style="query")
            parsed = ap.parse(txt)
            if parsed:
                rec.update(parsed); rec["ok"] = 1; n_ok += 1
        except Exception as e:
            print(f"  err {r['image_id']}: {e}", file=sys.stderr)
        w.writerow(rec); f.flush()
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(todo)} (ok {n_ok})", file=sys.stderr)
        time.sleep(args.sleep)
    f.close()
    print(f"done: {n_ok}/{len(todo)} rated -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

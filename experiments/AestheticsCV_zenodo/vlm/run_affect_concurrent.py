"""
run_affect_concurrent.py
========================
Concurrent multi-provider affect rater (I/O-bound API calls -> thread pool gives ~10x).
One process per provider; resumable; single-writer (main thread) so no write races.

  python vlm/run_affect_concurrent.py --provider gemini   --model models/gemini-2.5-flash --out vlm_gemini_affect.csv --workers 12
  python vlm/run_affect_concurrent.py --provider openai    --model gpt-4o                  --out vlm_openai_affect.csv --workers 12
  python vlm/run_affect_concurrent.py --provider anthropic --model claude-sonnet-4-6       --out vlm_anthropic_affect.csv --workers 10
"""
from __future__ import annotations
import argparse, csv, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import affect_prompts as ap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ["image_id", "source", "aesthetic", "cluster"] + ap.AXES + ["mode", "ok"]


def make_caller(provider, model):
    if provider == "gemini":
        import gemini_client as cl
        key = cl.load_key()
        return lambda p: cl.generate(key, model, ap.PROMPT, image_path=p, auth_style="query")
    if provider == "openai":
        import openai_client as cl
        key = cl.load_key()
        return lambda p: cl.generate(key, model, ap.PROMPT, image_path=p)
    if provider == "anthropic":
        import anthropic_client as cl
        key = cl.load_key()
        return lambda p: cl.generate(key, model, ap.PROMPT, image_path=p)
    raise ValueError(provider)


def load_manifest(path):
    src = "arena" if "arena" in os.path.basename(path) else "corpus"
    return [{"image_path": r["image_path"], "image_id": r["image_id"], "source": src,
             "aesthetic": r.get("aesthetic", ""), "cluster": r.get("cluster", "")}
            for r in csv.DictReader(open(path, encoding="utf-8"))]


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--provider", required=True, choices=["gemini", "openai", "anthropic"])
    a.add_argument("--model", required=True)
    a.add_argument("--manifests", nargs="+",
                   default=[os.path.join(ROOT, "stimuli_manifest_arena.csv"),
                            os.path.join(ROOT, "stimuli_manifest.csv")])
    a.add_argument("--out", required=True)
    a.add_argument("--workers", type=int, default=10)
    a.add_argument("--limit", type=int, default=0)
    args = a.parse_args()
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)

    call = make_caller(args.provider, args.model)
    items, seen = [], set()
    for m in args.manifests:
        mp = m if os.path.isabs(m) else os.path.join(ROOT, m)
        for r in load_manifest(mp):
            if r["image_id"] not in seen:
                seen.add(r["image_id"]); items.append(r)
    done = set()        # only count SUCCESSFUL ratings as done -> failed rows get retried
    if os.path.exists(out):
        for r in csv.DictReader(open(out, encoding="utf-8")):
            if r.get("ok") == "1":
                done.add(r["image_id"])
    todo = [r for r in items if r["image_id"] not in done and os.path.exists(r["image_path"])]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[{args.provider}] {len(items)} imgs, {len(done)} done, {len(todo)} to do, "
          f"{args.workers} workers -> {out}", file=sys.stderr)

    def rate(item):
        rec = {k: item[k] for k in ("image_id", "source", "aesthetic", "cluster")}
        rec["ok"] = 0
        try:
            parsed = ap.parse(call(item["image_path"]))
            if parsed:
                rec.update(parsed); rec["ok"] = 1
        except Exception as e:
            rec["err"] = str(e)[:120]
        return rec

    new = not os.path.exists(out)
    f = open(out, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
    if new:
        w.writeheader(); f.flush()
    n_ok = n = 0
    errs = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(rate, it) for it in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            w.writerow(rec); f.flush()
            n += 1; n_ok += rec["ok"]
            if not rec["ok"]:
                errs += 1
            if n % 100 == 0:
                print(f"  {n}/{len(todo)} (ok {n_ok}, err {errs})", file=sys.stderr)
    f.close()
    print(f"[{args.provider}] done: {n_ok}/{len(todo)} rated -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

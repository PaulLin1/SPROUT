"""
run_prompt_ablation.py
======================
M3: is the unease over-attribution a low-level RESPONSE bias (fixable by prompting) or a deep
PERCEPTUAL/SEMANTIC prior? Re-rate the same human-warm-but-VLM-eerie are.na images under:
  baseline  - the original calibrated prompt (test-retest control)
  ground    - "identify the literal subject, rate the SCENE not its style"
  debias    - explicit "stylization (filters, grain, low-res, text, retro grading) is NOT
              evidence of threat/wrongness; rate the depicted scene"
If 'debias' removes the eerie, the bias is elicitation-level (a practical fix). If not, the
over-attribution is a learned association between these visual STYLES and unease.

Output: mechanism/mechanism_prompt.csv
"""
from __future__ import annotations
import argparse, csv, os, sys, random
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vlm"))
import affect_prompts as ap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_MODEL = {"gemini": "models/gemini-2.5-flash", "openai": "gpt-4o",
                 "anthropic": "claude-sonnet-4-6"}


def make_gen(provider, model):
    if provider == "gemini":
        import gemini_client as cl
        key = cl.load_key()
        return lambda prompt, p: cl.generate(key, model, prompt, image_path=p, auth_style="query")
    if provider == "openai":
        import openai_client as cl
        key = cl.load_key()
        return lambda prompt, p: cl.generate(key, model, prompt, image_path=p)
    if provider == "anthropic":
        import anthropic_client as cl
        key = cl.load_key()
        return lambda prompt, p: cl.generate(key, model, prompt, image_path=p)
    raise ValueError(provider)

GROUND = ("First identify the literal subject/scene of the image in your reasoning, then rate "
          "the affect a typical viewer would feel toward that SCENE (not its visual style). ")
DEBIAS = ("This is a curated 'aesthetic' image and may be stylized (filters, grain, low "
          "resolution, overlaid text, retro or cool colour grading). Stylization is NOT "
          "evidence of threat, horror, or wrongness. Rate the affect a typical viewer would "
          "feel toward the SCENE DEPICTED, not its editing. ")
PROMPTS = {"baseline": ap.PROMPT, "ground": GROUND + ap.PROMPT, "debias": DEBIAS + ap.PROMPT}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--provider", default="gemini", choices=["gemini", "openai", "anthropic"])
    a.add_argument("--model", default=None)
    a.add_argument("--n", type=int, default=120)
    args = a.parse_args()
    model = args.model or DEFAULT_MODEL[args.provider]
    gen = make_gen(args.provider, model)
    eerie = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "vlm_gemini_affect.csv"), encoding="utf-8")):
        if r.get("ok") == "1" and r.get("source") == "arena" and r.get("cluster") in ("restorative", "anemoia"):
            try:
                eerie[r["image_id"]] = (r["cluster"], float(r["eerie"]))
            except ValueError:
                pass
    man = {r["image_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "stimuli_manifest_arena.csv"), encoding="utf-8"))}
    cand = [i for i, (c, e) in eerie.items() if e >= 3 and i in man and os.path.exists(man[i]["image_path"])]
    rng = random.Random(20260614); rng.shuffle(cand)
    sample = cand[:args.n]
    print(f"{len(cand)} candidates; ablation on {len(sample)} x {len(PROMPTS)} prompts", file=sys.stderr)

    tasks = [(i, eerie[i][0], cond, man[i]["image_path"]) for i in sample for cond in PROMPTS]
    out = os.path.join(HERE, f"mechanism_prompt_{args.provider}.csv")
    done = set()                       # resume: skip (image_id, condition) already rated ok
    if os.path.exists(out):
        for r in csv.DictReader(open(out, encoding="utf-8")):
            if r.get("ok") == "1":
                done.add((r["image_id"], r["condition"]))
    tasks = [t for t in tasks if (t[0], t[2]) not in done]
    print(f"resume: {len(done)} already done, {len(tasks)} to rate", file=sys.stderr)
    new = not os.path.exists(out)
    f = open(out, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=["image_id", "cluster", "condition"] + ap.AXES + ["mode", "ok"],
                       extrasaction="ignore")
    if new:
        w.writeheader()

    def rate(t):
        i, cl, cond, p = t
        rec = {"image_id": i, "cluster": cl, "condition": cond, "ok": 0}
        try:
            parsed = ap.parse(gen(PROMPTS[cond], p))
            if parsed:
                rec.update(parsed); rec["ok"] = 1
        except Exception as e:
            rec["err"] = str(e)[:100]
        return rec

    n_ok = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for fut in as_completed([ex.submit(rate, t) for t in tasks]):
            r = fut.result(); w.writerow(r); f.flush(); n_ok += r["ok"]
    f.close()
    print(f"rated {n_ok}/{len(tasks)} -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

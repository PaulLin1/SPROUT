"""
run_intervention.py
===================
Causal test of WHY frontier VLMs over-attribute unease to internet aesthetics.

Hypothesis: the models read the aesthetics' POST-PROCESSING (degradation, low resolution,
desaturation, dark/cool grading) as 'wrong/uncanny', not the content. If so, NATURALIZING the
image (denoise + upscale + restore brightness/colour) should REDUCE the eerie rating, while
content is unchanged.

Design: sample are.na images that (a) humans filed as restorative/anemoia (should NOT be
eerie) but (b) Gemini rated eerie>=3. For each, create:
  control = re-encode original (test-retest baseline)
  T_quality = denoise + upscale + mild sharpen   (remove degradation)
  T_tone    = brightness + contrast + saturation  (remove cool/dark grading)
  T_natural = both
Re-rate all conditions with Gemini (same calibrated prompt). Paired deltas vs control isolate
which manipulation removes the unease attribution.

Output: mechanism/mechanism_intervention.csv (image_id, cluster, condition, eerie, valence, ...)
Edited images cached in mechanism/edited/.
"""
from __future__ import annotations
import argparse, csv, os, sys, random
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "vlm"))
import affect_prompts as ap

EDIT = os.path.join(HERE, "edited")
os.makedirs(EDIT, exist_ok=True)
DEFAULT_MODEL = {"gemini": "models/gemini-2.5-flash", "openai": "gpt-4o",
                 "anthropic": "claude-sonnet-4-6"}


def make_gen(provider, model):
    """Return gen(prompt, image_path) -> text, dispatching to the right client."""
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


def quality_restore(bgr):
    h, w = bgr.shape[:2]
    out = cv2.fastNlMeansDenoisingColored(bgr, None, 6, 6, 7, 21)   # remove grain/compression
    s = max(h, w)
    if s < 768:                                                     # upscale low-res
        f = 768.0 / s
        out = cv2.resize(out, (int(w*f), int(h*f)), interpolation=cv2.INTER_LANCZOS4)
    return out


def tone_restore(pil):
    pil = ImageEnhance.Brightness(pil).enhance(1.18)
    pil = ImageEnhance.Contrast(pil).enhance(1.12)
    pil = ImageEnhance.Color(pil).enhance(1.30)        # restore saturation
    return pil


def make_variants(path, iid):
    pil = Image.open(path).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    variants = {}
    # control: re-encode original at decent quality (test-retest baseline)
    p = os.path.join(EDIT, f"{iid}_control.jpg"); pil.save(p, "JPEG", quality=92); variants["control"] = p
    # quality
    q = cv2.cvtColor(quality_restore(bgr), cv2.COLOR_BGR2RGB)
    qp = Image.fromarray(q); qp = qp.filter(ImageFilter.UnsharpMask(radius=2, percent=80))
    p = os.path.join(EDIT, f"{iid}_quality.jpg"); qp.save(p, "JPEG", quality=92); variants["T_quality"] = p
    # tone
    tp = tone_restore(pil)
    p = os.path.join(EDIT, f"{iid}_tone.jpg"); tp.save(p, "JPEG", quality=92); variants["T_tone"] = p
    # natural = both
    np_ = tone_restore(qp)
    p = os.path.join(EDIT, f"{iid}_natural.jpg"); np_.save(p, "JPEG", quality=92); variants["T_natural"] = p
    return variants


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--provider", default="gemini", choices=["gemini", "openai", "anthropic"])
    a.add_argument("--model", default=None)
    a.add_argument("--n", type=int, default=120)
    args = a.parse_args()
    model = args.model or DEFAULT_MODEL[args.provider]
    gen = make_gen(args.provider, model)
    # candidate images: human restorative/anemoia, Gemini eerie>=3 (same set across models)
    eerie = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "vlm_gemini_affect.csv"), encoding="utf-8")):
        if r.get("ok") == "1" and r.get("source") == "arena":
            try:
                eerie[r["image_id"]] = (r["cluster"], float(r["eerie"]))
            except ValueError:
                pass
    man = {r["image_id"]: r for r in csv.DictReader(
        open(os.path.join(ROOT, "stimuli_manifest_arena.csv"), encoding="utf-8"))}
    cand = [iid for iid, (cl, ee) in eerie.items()
            if cl in ("restorative", "anemoia") and ee >= 3 and iid in man
            and os.path.exists(man[iid]["image_path"])]
    rng = random.Random(20260614); rng.shuffle(cand)
    sample = cand[:args.n]
    print(f"{len(cand)} candidates (restorative/anemoia rated eerie>=3); using {len(sample)}",
          file=sys.stderr)

    tasks = []
    for iid in sample:
        try:
            variants = make_variants(man[iid]["image_path"], iid)
        except Exception as e:
            print(f"  edit fail {iid}: {e}", file=sys.stderr); continue
        for cond, p in variants.items():
            tasks.append((iid, eerie[iid][0], cond, p))

    out_path = os.path.join(HERE, f"mechanism_intervention_{args.provider}.csv")
    done = set()                       # resume: skip (image_id, condition) already rated ok
    if os.path.exists(out_path):
        for r in csv.DictReader(open(out_path, encoding="utf-8")):
            if r.get("ok") == "1":
                done.add((r["image_id"], r["condition"]))
    tasks = [t for t in tasks if (t[0], t[2]) not in done]
    print(f"resume: {len(done)} already done, {len(tasks)} to rate", file=sys.stderr)
    new = not os.path.exists(out_path)
    f = open(out_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=["image_id", "cluster", "condition"] + ap.AXES + ["mode", "ok"],
                       extrasaction="ignore")
    if new:
        w.writeheader()

    def rate(t):
        iid, cl, cond, p = t
        rec = {"image_id": iid, "cluster": cl, "condition": cond, "ok": 0}
        try:
            parsed = ap.parse(gen(ap.PROMPT, p))
            if parsed:
                rec.update(parsed); rec["ok"] = 1
        except Exception as e:
            rec["err"] = str(e)[:100]
        return rec

    n_ok = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for fut in as_completed([ex.submit(rate, t) for t in tasks]):
            rec = fut.result(); w.writerow(rec); f.flush(); n_ok += rec["ok"]
    f.close()
    print(f"rated {n_ok}/{len(tasks)} variant-images -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

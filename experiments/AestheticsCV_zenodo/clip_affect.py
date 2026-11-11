"""
clip_affect.py
==============
Zero-shot affective probing with CLIP as a *computational observer*.

CLIP is trained on web-scale human image-text pairs, so its image-text alignment encodes
aggregated human associations. We use it to score each image on theory-relevant affective
axes via prompt-ensembled contrastive probes (P(positive pole) over a softmax of mean
text embeddings). This is NOT human affect data, but an independent, semantics-based
signal: a convergent test of whether the paper's structural findings also appear when the
construct is measured by a different mechanism (language alignment, not hand-built
features).

Key test: does CLIP's EERINESS score reproduce (a) the inverted>control contrast and,
crucially, (b) the scene x occupancy interaction (eerie strong where emptiness violates an
occupancy expectation, null where empty is ordinary)?

Output: clip_affect.csv (per-image axis scores) + printed analysis.
"""
from __future__ import annotations
import csv, math
import numpy as np

AXES = {
    "eerie": (
        ["an eerie, unsettling place", "a creepy liminal space", "an uncanny, wrong-feeling scene",
         "a disturbing empty place", "an unnerving abandoned interior"],
        ["an ordinary, comfortable place", "a normal everyday scene", "a pleasant familiar interior",
         "a reassuring lively place", "a calm ordinary room"],
    ),
    "nostalgia": (
        ["a nostalgic image evoking the past", "a wistful memory of earlier times",
         "a sentimental retro scene", "an image that makes you long for the past",
         "a nostalgic throwback aesthetic"],
        ["a neutral present-day image", "an ordinary contemporary scene", "a plain modern photo",
         "a forgettable everyday picture", "a present-moment snapshot"],
    ),
    "abandonment": (
        ["an empty abandoned place with no people", "a deserted space where people should be",
         "an unoccupied room that feels emptied", "a vacant place missing its occupants",
         "a place eerily devoid of people"],
        ["a lively place full of people", "a crowded busy scene", "a populated bustling space",
         "a place full of activity and people", "an occupied lively room"],
    ),
    "dreamlike": (
        ["a surreal dreamlike image", "a hazy oneiric scene", "an unreal dream-logic place",
         "a hypnagogic blurry vision", "a strange dreamcore image"],
        ["a realistic ordinary photograph", "a sharp documentary photo", "a plain literal image",
         "a normal real-world picture", "a mundane realistic scene"],
    ),
}


def num(x):
    try:
        v = float(x); return v if math.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def main():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def _text_feat(inp):
        out = model.text_model(**inp)
        return model.text_projection(out.pooler_output)

    def _image_feat(inp):
        out = model.vision_model(**inp)
        return model.visual_projection(out.pooler_output)

    def pole_embed(prompts):
        inp = proc(text=prompts, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            t = _text_feat(inp)
        t = t / t.norm(dim=-1, keepdim=True)
        return t.mean(0, keepdim=True)  # ensembled pole

    pole = {}
    for ax, (pos, neg) in AXES.items():
        pole[ax] = (pole_embed(pos), pole_embed(neg))

    rows = list(csv.DictReader(open("features.csv", encoding="utf-8")))
    logit_scale = model.logit_scale.exp().item()
    out = []
    for i, r in enumerate(rows):
        im = Image.open(r["image_path"]).convert("RGB")
        inp = proc(images=im, return_tensors="pt").to(dev)
        with torch.no_grad():
            v = _image_feat(inp)
        v = v / v.norm(dim=-1, keepdim=True)
        rec = {"image_id": r["image_id"], "cluster": r["cluster"],
               "scene_category": r["scene_category"]}
        for ax, (pp, npp) in pole.items():
            sp = (v @ pp.T).item() * logit_scale
            sn = (v @ npp.T).item() * logit_scale
            rec[f"clip_{ax}"] = float(np.exp(sp) / (np.exp(sp) + np.exp(sn)))  # P(positive pole)
        out.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    cols = ["image_id", "cluster", "scene_category"] + [f"clip_{a}" for a in AXES]
    with open("clip_affect.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(out)

    # ---- analysis ----
    CL = ["restorative", "hauntological", "inverted", "anemoia", "control"]
    print("\n=== CLIP zero-shot affect, per-cluster means ===")
    print("cluster".ljust(14) + "".join(a.rjust(12) for a in AXES))
    for c in CL:
        cells = []
        for a in AXES:
            v = [rec[f"clip_{a}"] for rec in out if rec["cluster"] == c]
            cells.append(f"{np.mean(v):.3f}".rjust(12))
        print(c.ljust(14) + "".join(cells))

    # scene x occupancy test on CLIP eerie (inverted vs control, within scene)
    PRIOR = ["classroom", "office", "swimming_pool", "shopping_mall", "corridor",
             "parking_lot", "playground"]
    def d(a, b):
        a, b = np.asarray(a), np.asarray(b)
        if len(a) < 2 or len(b) < 2: return np.nan
        sp = math.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
        return (a.mean()-b.mean())/(sp+1e-12)
    print("\n=== KEY TEST: CLIP eerie, inverted vs control, by scene (occupancy gradient) ===")
    print(f"{'scene':14s}{'inv_n':>6}{'ctl_n':>6}{'d_clip_eerie':>14}")
    ds, ws = [], []
    for s in PRIOR:
        iv = [rec["clip_eerie"] for rec in out if rec["cluster"]=="inverted" and rec["scene_category"]==s]
        ct = [rec["clip_eerie"] for rec in out if rec["cluster"]=="control" and rec["scene_category"]==s]
        dd = d(iv, ct)
        print(f"{s:14s}{len(iv):>6}{len(ct):>6}{dd:>14.2f}")
        if len(iv) >= 2 and len(ct) >= 2:
            ds.append(dd); ws.append(len(iv)+len(ct))
    if ds:
        print(f"sample-weighted within-category d (CLIP eerie) = {np.average(ds, weights=ws):.2f}")


if __name__ == "__main__":
    main()

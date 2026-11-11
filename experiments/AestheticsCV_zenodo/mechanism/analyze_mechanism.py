"""
analyze_mechanism.py
====================
M1 (correlational): within are.na images humans filed as restorative/anemoia (which should
NOT be eerie), which image properties predict Gemini's 'eerie' rating? Identifies the
candidate mechanism of the over-attribution.

M2 (causal): from mechanism_intervention.csv, paired change in eerie for each naturalization
manipulation vs the test-retest control. The manipulation that significantly lowers eerie is
the causal driver.
"""
from __future__ import annotations
import csv, os, sys, math
from collections import defaultdict
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

QFEATS = ["col_val_mean", "col_sat_mean", "col_colorfulness", "col_edge_density",
          "col_gloss_index", "col_warm_cool_ratio", "seg_emptiness_index", "seg_human_absence"]


def _f(x):
    try:
        v = float(x); return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def boot(d, n=5000, seed=0):
    rng = np.random.RandomState(seed); d = np.asarray(d, float)
    bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    out = []
    def P(s): print(s); out.append(s)

    # ---- M1 ----
    eerie = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "vlm_gemini_affect.csv"), encoding="utf-8")):
        if r.get("ok") == "1" and r.get("source") == "arena" and r.get("cluster") in ("restorative", "anemoia"):
            e = _f(r.get("eerie"))
            if e is not None:
                eerie[r["image_id"]] = e
    feats = {r["image_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "features_arena.csv"), encoding="utf-8"))}
    man = {r["image_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "stimuli_manifest_arena.csv"), encoding="utf-8"))}

    rows = []
    for iid, e in eerie.items():
        if iid not in feats or iid not in man:
            continue
        fr = feats[iid]
        rec = {"eerie": e}
        ok = True
        for q in QFEATS:
            v = _f(fr.get(q))
            if v is None:
                ok = False; break
            rec[q] = v
        if not ok:
            continue
        # add blur (Laplacian var) + resolution from the image
        p = man[iid]["image_path"]
        img = cv2.imread(p)
        if img is None:
            continue
        rec["blur_lapvar"] = float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        rec["resolution"] = float(max(img.shape[:2]))
        rows.append(rec)

    P("=" * 64)
    P("M1  MECHANISM (within human restorative/anemoia are.na images)")
    P(f"    n={len(rows)} images; what predicts Gemini 'eerie'?")
    P("=" * 64)
    cols = QFEATS + ["blur_lapvar", "resolution"]
    y = np.array([r["eerie"] for r in rows])
    P(f"\n  Pearson r of each property with VLM eerie (sorted):")
    cors = []
    for c in cols:
        x = np.array([r[c] for r in rows])
        r_ = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else 0.0
        cors.append((c, r_))
    for c, r_ in sorted(cors, key=lambda t: -abs(t[1])):
        arrow = "more eerie" if r_ > 0 else "less eerie"
        P(f"    {c:22s} r={r_:+.2f}   (higher -> {arrow})")

    # standardized multiple regression (numpy lstsq)
    X = np.array([[r[c] for c in cols] for r in rows])
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    ys = (y - y.mean()) / (y.std() + 1e-9)
    Xd = np.hstack([Xs, np.ones((len(Xs), 1))])
    beta, *_ = np.linalg.lstsq(Xd, ys, rcond=None)
    P(f"\n  standardized multiple-regression betas (unique contribution):")
    for c, b in sorted(zip(cols, beta[:-1]), key=lambda t: -abs(t[1])):
        P(f"    {c:22s} beta={b:+.2f}")
    pred = Xd @ beta
    P(f"    model R^2 = {1 - ((ys-pred)**2).sum()/((ys-ys.mean())**2).sum():.2f}")

    # ---- M2 ----
    PROV = ["gemini", "openai", "anthropic"]

    def paired(path, baseline, conds, title):
        if not os.path.exists(path):
            return None
        recs = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r.get("ok") == "1"]
        by = defaultdict(dict)
        for r in recs:
            e = _f(r.get("eerie"))
            if e is not None:
                by[r["image_id"]][r["condition"]] = e
        base = [by[i][baseline] for i in by if baseline in by[i]]
        if len(base) < 20:
            return None
        P(f"  [{title}] {baseline} mean eerie = {np.mean(base):.2f} (n={len(base)})")
        for cond in conds:
            d = [by[i][cond] - by[i][baseline] for i in by if cond in by[i] and baseline in by[i]]
            if len(d) >= 20:
                m, lo, hi = boot(d)
                sig = "REDUCES eerie" if hi < 0 else ("raises eerie" if lo > 0 else "n.s.")
                P(f"      {cond:11s} delta={m:+.2f} [{lo:+.2f},{hi:+.2f}] -> {sig} (n={len(d)})")
        return True

    P("\n" + "=" * 64)
    P("M2  CAUSAL INTERVENTION (image naturalization), paired eerie change vs control")
    P("=" * 64)
    for prov in PROV:
        paired(os.path.join(HERE, f"mechanism_intervention_{prov}.csv"),
               "control", ["T_quality", "T_tone", "T_natural"], prov)

    P("\n" + "=" * 64)
    P("M3  PROMPT ABLATION (reframe the question), paired eerie change vs baseline")
    P("=" * 64)
    for prov in PROV:
        paired(os.path.join(HERE, f"mechanism_prompt_{prov}.csv"),
               "baseline", ["ground", "debias"], prov)

    with open(os.path.join(HERE, "mechanism_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    P("\n-> mechanism/mechanism_report.txt")


if __name__ == "__main__":
    main()

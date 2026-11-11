"""
analyze_oasis.py
================
GENERALITY + HUMAN-ANCHOR test of the VLM affect bias, on OASIS (900 images with
gold-standard human valence/arousal norms, 1-7, N~100/image; Kurdi et al. 2017, BRM).

If the are.na "over-attribution of unease" is a real, general bias and not an internet-meme
quirk, it should reappear against HUMAN ground truth on a standard stimulus set:
  1. VALIDITY: VLM valence/arousal should correlate with human norms (they track affect).
  2. NEGATIVITY BIAS: on a common [0,1] scale, VLM valence should sit BELOW the human mean
     (paired difference < 0), i.e. models rate images more negative than people do.
  3. UNCANNY INFLATION: VLM 'eerie' should be non-trivial even for images humans rate as
     clearly POSITIVE -- unease attributed where humans feel none.
Output: oasis_report.txt
"""
from __future__ import annotations
import csv, os, math
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
JUDGES = {"gemini": "vlm_gemini_oasis.csv", "gpt4o": "vlm_openai_oasis.csv",
          "claude": "vlm_anthropic_oasis.csv"}


def load(path):
    if not os.path.exists(path):
        return {}
    d = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("ok") != "1":
            continue
        try:
            d[r["image_id"]] = {"valence": float(r["valence"]), "arousal": float(r["arousal"]),
                                "eerie": float(r["eerie"])}
        except (ValueError, KeyError):
            pass
    return d


def pear(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float(np.corrcoef(x, y)[0, 1])


def spear(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return pear(rx, ry)


def boot_ci(d, n=5000, seed=0):
    rng = np.random.RandomState(seed); d = np.asarray(d)
    bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.mean(d)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    judges = {k: load(os.path.join(ROOT, v)) for k, v in JUDGES.items()}
    judges = {k: v for k, v in judges.items() if v}
    norms = {}
    idkey = None
    for r in csv.DictReader(open(os.path.join(ROOT, "oasis", "OASIS.csv"), encoding="utf-8")):
        if idkey is None:
            idkey = [k for k in r if k.strip() == ""][0]
        norms[r[idkey]] = {"hv": float(r["Valence_mean"]), "ha": float(r["Arousal_mean"])}

    out = []
    def P(s): print(s); out.append(s)
    P("=" * 64); P(f"OASIS generality test (judges: {', '.join(judges)})"); P("=" * 64)

    # PER-MODEL (each model's own coverage) -- the cross-model boundary result
    P("\n[0] PER-MODEL vs human norms (valence Pearson r; negativity bias on 0-1; "
      "mean eerie on human-POSITIVE tercile):")
    for name, jd in judges.items():
        mids = [i for i in norms if i in jd]
        if len(mids) < 50:
            P(f"    {name:8s}: n={len(mids)} (too few)"); continue
        mvv = np.array([(jd[i]["valence"] - 1) / 4 for i in mids])
        mhv = np.array([(norms[i]["hv"] - 1) / 6 for i in mids])
        mve = np.array([jd[i]["eerie"] for i in mids])
        rr = pear(mvv, mhv); dm, dlo, dhi = boot_ci(mvv - mhv)
        oo = np.argsort(mhv); pos = oo[2 * len(mids) // 3:]
        P(f"    {name:8s} n={len(mids):4d}  valence r={rr:.2f}  "
          f"bias={dm:+.3f} [{dlo:+.3f},{dhi:+.3f}]  eerie|positive={mve[pos].mean():.2f}")

    ids = [i for i in norms if all(i in j for j in judges.values())]
    P(f"\nimages rated by all judges + with norms: {len(ids)}")
    if len(ids) < 50:
        P("not enough yet; rerun after batches finish."); _save(out); return

    # consensus VLM (mean across judges), normalized to [0,1]
    def cons(i, ax): return np.mean([judges[k][i][ax] for k in judges])
    vv = np.array([(cons(i, "valence") - 1) / 4 for i in ids])   # 1-5 -> 0-1
    va = np.array([(cons(i, "arousal") - 1) / 4 for i in ids])
    ve = np.array([cons(i, "eerie") for i in ids])               # 1-5
    hv = np.array([(norms[i]["hv"] - 1) / 6 for i in ids])        # 1-7 -> 0-1
    ha = np.array([(norms[i]["ha"] - 1) / 6 for i in ids])

    P("\n[1] VALIDITY (VLM tracks human norms):")
    P(f"    valence: Pearson r={pear(vv,hv):.2f}  Spearman={spear(vv,hv):.2f}")
    P(f"    arousal: Pearson r={pear(va,ha):.2f}  Spearman={spear(va,ha):.2f}")

    P("\n[2] NEGATIVITY BIAS (VLM minus human, common 0-1 scale; <0 = VLM more negative):")
    dv = vv - hv
    m, lo, hi = boot_ci(dv)
    P(f"    valence diff = {m:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  "
      f"({'VLM MORE NEGATIVE' if hi<0 else 'VLM more positive' if lo>0 else 'n.s.'})")
    da = va - ha
    m2, lo2, hi2 = boot_ci(da)
    P(f"    arousal diff = {m2:+.3f}  95% CI [{lo2:+.3f},{hi2:+.3f}]")

    P("\n[3] UNCANNY INFLATION (VLM 'eerie', 1-5, by human-valence tercile):")
    order = np.argsort(hv)
    t = len(ids) // 3
    neg, mid, pos = order[:t], order[t:2*t], order[2*t:]
    P(f"    human-NEGATIVE images: mean VLM eerie = {ve[neg].mean():.2f}")
    P(f"    human-NEUTRAL  images: mean VLM eerie = {ve[mid].mean():.2f}")
    P(f"    human-POSITIVE images: mean VLM eerie = {ve[pos].mean():.2f}")
    P(f"    -> eerie>=3 fraction among human-POSITIVE images: {(ve[pos]>=3).mean():.2f}")
    _save(out)


def _save(out):
    with open(os.path.join(ROOT, "oasis_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("\n-> oasis_report.txt")


if __name__ == "__main__":
    main()

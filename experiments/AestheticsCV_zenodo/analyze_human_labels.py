"""
analyze_human_labels.py
=======================
Use the COLLECTIVE HUMAN data (Aesthetics Wiki) as an external criterion for the paper's
affective-mechanism taxonomy -- no rating study required.

Two tests:

  A. AFFECT LEXICON.  Score each aesthetic's community-authored `key_values` on four
     affect dimensions (unease, nostalgia, comfort, refinement). Test whether the paper's
     R/H/I/A clusters differ as predicted:
        - inverted (uncanny) should top UNEASE
        - NOSTALGIA should appear ACROSS restorative/hauntological/inverted (shared
          substrate, the paper's central claim)
        - restorative should top COMFORT; anemoia should top REFINEMENT

  B. RELATED-AESTHETIC GRAPH.  The crowd-drawn `related_aesthetics` links form a similarity
     network. If humans link aesthetics by affective kinship, edges should fall WITHIN a
     mechanism cluster more than chance (assortativity). A permutation test gives p.

Outputs: printed report + human_affect_scores.csv
"""
from __future__ import annotations
import csv, os, re, math, random
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---- affect lexicon: term -> dimension (lowercased substring match) --------------
LEX = {
    "unease": ["uncann", "discomfort", "dread", "disorient", "alienat", "fear",
               "unsettl", "confusion", "surreal", "isolation", "loss of control",
               "anxiety", "eerie", "creep", "horror", "malfunction", "derealiz",
               "tension", "foreboding", "wrong"],
    "nostalgia": ["nostalg", "anemoia", "wistful", "retrofutur", "memory", "remember",
                  "childhood", "reclaim", "the past", "longing", "yearning",
                  "fabricated past", "throwback", "reminisc"],
    "comfort": ["comfort", "simplicity", "harmony", "tranquil", "optimism", "playful",
                "kindness", "positiv", "cozy", "innocence", "self-sufficien",
                "domesticity", "warmth", "peace", "calm", "wonder", "hope", "escapism"],
    "refinement": ["intellectual", "elegance", "opulence", "sophistic", "romantic",
                   "classic", "academ", "poetry", "literature", "tradition", "wealth",
                   "chivalry", "honor", "knowledge", "research", "mystery", "occult"],
}
CLUSTERS = ["restorative", "hauntological", "inverted", "anemoia"]


def score_text(text):
    t = (text or "").lower()
    sc = {}
    for dim, terms in LEX.items():
        sc[dim] = sum(1 for kw in terms if kw in t)
    return sc


def cohens_d(a, b):
    a, b = list(a), list(b)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = sum(a)/len(a), sum(b)/len(b)
    va = sum((x-ma)**2 for x in a)/(len(a)-1)
    vb = sum((x-mb)**2 for x in b)/(len(b)-1)
    sp = math.sqrt(((len(a)-1)*va + (len(b)-1)*vb)/(len(a)+len(b)-2))
    return (ma-mb)/sp if sp > 0 else float("nan")


def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "wiki_affect.csv"), encoding="utf-8")))
    have_kv = [r for r in rows if r["key_values"].strip()]

    # ---- A. affect lexicon ----------------------------------------------------
    scored = []
    for r in rows:
        sc = score_text(r["key_values"])
        tot = sum(sc.values()) or 1
        rec = {"aesthetic": r["aesthetic"], "cluster": r["cluster"], "has_kv": bool(r["key_values"].strip())}
        for dim in LEX:
            rec[f"n_{dim}"] = sc[dim]
            rec[f"p_{dim}"] = sc[dim] / tot      # within-aesthetic proportion
        scored.append(rec)

    by_cluster = defaultdict(list)
    for rec in scored:
        if rec["has_kv"]:
            by_cluster[rec["cluster"]].append(rec)

    print("=" * 70)
    print("A. COMMUNITY-AUTHORED AFFECT (Aesthetics Wiki key_values)")
    print(f"   {len(have_kv)}/{len(rows)} aesthetics have key_values")
    print("=" * 70)
    print(f"\nMean affect-term PROPORTION by mechanism cluster:")
    print(f"{'cluster':14s}" + "".join(d.rjust(12) for d in LEX) + "   n")
    cluster_means = {}
    for cl in CLUSTERS:
        recs = by_cluster[cl]
        cluster_means[cl] = {}
        cells = ""
        for dim in LEX:
            m = sum(r[f"p_{dim}"] for r in recs)/len(recs) if recs else float("nan")
            cluster_means[cl][dim] = m
            cells += f"{m:.2f}".rjust(12)
        print(f"{cl:14s}{cells}{len(recs):>4}")

    print("\nPredictions vs. data:")
    # P1: inverted tops unease
    top_unease = max(CLUSTERS, key=lambda c: cluster_means[c].get("unease", -1))
    print(f"  [P1] uncanny(inverted) tops UNEASE?      -> top = {top_unease}  "
          f"{'PASS' if top_unease=='inverted' else 'FAIL'}")
    inv = [r["p_unease"] for r in by_cluster["inverted"]]
    rest = [r["p_unease"] for c in CLUSTERS if c!="inverted" for r in by_cluster[c]]
    print(f"        inverted unease d vs rest = {cohens_d(inv, rest):.2f}")
    # P2: nostalgia present across R/H/I (shared substrate)
    print(f"  [P2] NOSTALGIA shared across restorative/hauntological/inverted?")
    for cl in ["restorative", "hauntological", "inverted", "anemoia"]:
        frac = sum(1 for r in by_cluster[cl] if r["n_nostalgia"]>0)/max(1,len(by_cluster[cl]))
        print(f"        {cl:14s}: {frac*100:4.0f}% of aesthetics mention nostalgia")
    # P3: restorative tops comfort; anemoia tops refinement
    top_comfort = max(CLUSTERS, key=lambda c: cluster_means[c].get("comfort", -1))
    top_refine = max(CLUSTERS, key=lambda c: cluster_means[c].get("refinement", -1))
    print(f"  [P3] restorative tops COMFORT?           -> top = {top_comfort}  "
          f"{'PASS' if top_comfort=='restorative' else 'FAIL'}")
    print(f"       anemoia tops REFINEMENT?            -> top = {top_refine}  "
          f"{'PASS' if top_refine=='anemoia' else 'FAIL'}")

    # ---- B. related-aesthetic graph assortativity -----------------------------
    print("\n" + "=" * 70)
    print("B. CROWD-DRAWN RELATED-AESTHETIC GRAPH -- does it recover the taxonomy?")
    print("=" * 70)
    edges = list(csv.DictReader(open(os.path.join(ROOT, "wiki_related_edges.csv"), encoding="utf-8")))
    # cluster of each aesthetic NAME (only those in our taxonomy)
    name2cluster = {r["aesthetic"]: r["cluster"] for r in rows}
    # normalise related target names to our taxonomy where possible
    def norm(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())
    norm2name = {norm(n): n for n in name2cluster}
    internal = []  # edges whose BOTH endpoints are in our taxonomy
    for e in edges:
        src = e["aesthetic"]
        tgt = norm2name.get(norm(e["related"]))
        if tgt and tgt != src:
            internal.append((src, tgt))
    same = sum(1 for s, t in internal if name2cluster[s] == name2cluster[t])
    n = len(internal)
    obs = same / n if n else float("nan")
    print(f"  edges within our taxonomy: {n}")
    print(f"  observed within-cluster edge fraction: {obs:.3f}")
    # permutation null: shuffle cluster labels over aesthetics
    names = list(name2cluster)
    labels = [name2cluster[x] for x in names]
    rng = random.Random(20260613)
    null = []
    for _ in range(5000):
        perm = labels[:]; rng.shuffle(perm)
        m = dict(zip(names, perm))
        s = sum(1 for a, b in internal if m[a] == m[b])
        null.append(s / n)
    p = (sum(1 for x in null if x >= obs) + 1) / (len(null) + 1)
    mu = sum(null)/len(null)
    print(f"  permutation null mean: {mu:.3f}   p(within >= observed) = {p:.4f}")
    print(f"  -> crowd links are {'MORE' if obs>mu else 'NOT more'} within-cluster than chance "
          f"({'significant' if p<0.05 else 'n.s.'})")

    # ---- write scores ---------------------------------------------------------
    cols = ["aesthetic", "cluster", "has_kv"] + [f"n_{d}" for d in LEX] + [f"p_{d}" for d in LEX]
    with open(os.path.join(ROOT, "human_affect_scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(scored)
    print("\n-> human_affect_scores.csv")


if __name__ == "__main__":
    main()

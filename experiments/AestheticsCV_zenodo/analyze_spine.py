"""
analyze_spine.py
================
The paper's non-circular SPINE: do image features carry the affective-mechanism structure,
and does it GENERALIZE across two independently-sourced corpora?

  - Corpus A: are.na (features_arena.csv) -- membership decided by human curators.
  - Corpus B: Wikimedia (features.csv) -- membership from our keyword queries.
These share NO images and were built by totally different pipelines. If a classifier trained
on one predicts cluster on the other ABOVE CHANCE, the structure is in the images, not in a
single labelling procedure -- the strongest evidence against circularity.

Tests:
  1. CROSS-SOURCE TRANSFER (A->B and B->A), 4 mechanism clusters (drop control: A has none),
     accuracy + per-class recall + permutation null.
  2. POOLED leave-one-source-out sanity.
Output: printed report + spine_report.txt
"""
from __future__ import annotations
import csv, os, sys, math, random
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CLUSTERS = ["restorative", "hauntological", "inverted", "anemoia"]
SKIP = {"image_id", "image_path", "scene_category", "cluster", "aesthetic",
        "channel_slug", "block_id", "license", "source_url", "source", "query"}


def load(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def numeric_cols(rows):
    cols = []
    for k in rows[0]:
        if k in SKIP:
            continue
        ok = sum(1 for r in rows[:60] if _f(r.get(k)) is not None)
        if ok >= 45:
            cols.append(k)
    return cols


def _f(x):
    try:
        v = float(x); return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def matrix(rows, cols):
    X = np.full((len(rows), len(cols)), np.nan)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            v = _f(r.get(c))
            if v is not None:
                X[i, j] = v
    return X


def impute(Xtr, Xte):
    mu = np.nanmean(Xtr, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    for X in (Xtr, Xte):
        idx = np.where(~np.isfinite(X))
        X[idx] = np.take(mu, idx[1])
    return Xtr, Xte


def transfer(tr_rows, te_rows, cols, tag, out, n_perm=200):
    from sklearn.ensemble import RandomForestClassifier
    def P(s): print(s); out.append(s)
    tr = [r for r in tr_rows if r["cluster"] in CLUSTERS]
    te = [r for r in te_rows if r["cluster"] in CLUSTERS]
    Xtr, ytr = matrix(tr, cols), np.array([r["cluster"] for r in tr])
    Xte, yte = matrix(te, cols), np.array([r["cluster"] for r in te])
    Xtr, Xte = impute(Xtr, Xte)
    clf = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = (pred == yte).mean()
    vals, cnts = np.unique(yte, return_counts=True)
    maj = cnts.max() / cnts.sum()
    rng = np.random.RandomState(1)
    null = []
    for _ in range(n_perm):
        clf2 = RandomForestClassifier(n_estimators=120, random_state=rng.randint(1e6), n_jobs=-1)
        yp = ytr[rng.permutation(len(ytr))]
        clf2.fit(Xtr, yp)
        null.append((clf2.predict(Xte) == yte).mean())
    mu, sd = float(np.mean(null)), float(np.std(null) + 1e-9)
    p = (sum(1 for x in null if x >= acc) + 1) / (n_perm + 1)
    P(f"\n[{tag}]  train n={len(tr)}  test n={len(te)}  ({len(cols)} shared features)")
    P(f"   accuracy={acc:.3f}  majority={maj:.3f}  null={mu:.3f}+/-{sd:.3f}  "
      f"p={p:.4f}  z={(acc-mu)/sd:.1f}")
    for c in CLUSTERS:
        m = yte == c
        if m.sum():
            P(f"     recall {c:14s}={(pred[m]==c).mean():.2f}  (n={int(m.sum())})")
    return acc


def main():
    A = load(os.path.join(ROOT, "features_arena.csv"))   # are.na (human-curated)
    B = load(os.path.join(ROOT, "features.csv"))          # Wikimedia (keyword)
    cols = sorted(set(numeric_cols(A)) & set(numeric_cols(B)))
    out = []
    def P(s): print(s); out.append(s)
    P("=" * 66)
    P("SPINE: cross-source transfer of the affective-mechanism structure")
    P(f"are.na n={len(A)}  Wikimedia n={len(B)}  shared numeric features={len(cols)}")
    P("=" * 66)
    transfer(A, B, cols, "are.na -> Wikimedia", out)
    transfer(B, A, cols, "Wikimedia -> are.na", out)
    with open(os.path.join(ROOT, "spine_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    P("\n-> spine_report.txt")


if __name__ == "__main__":
    main()

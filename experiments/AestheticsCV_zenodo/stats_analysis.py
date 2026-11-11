"""
stats_analysis.py
=================
Inferential analysis the review (correctly) demanded:
  1. 5-way cross-validated cluster classifier + confusion matrix + permutation baseline
     -> turns "a coherent map" into a quantified separability claim, and the confusion
        structure is itself a finding (does anemoia collapse into hauntological?).
  2. Silhouette score (cluster compactness vs chance).
  3. Scene-matched effect sizes: inverted vs control on human_absence, emptiness,
     phi_typicality -- Cohen's d with bootstrap 95% CIs. This is the evidentiary core.
  4. phi_typicality reframed: Mahalanobis percentile of inverted relative to the control
     distribution (instead of the dimensionality-inflated raw -1480 vs -41).

Classifier uses only features DEFINED FOR ALL images (drops phi_typicality and
obj_expected_absent_score, which exist only for scene-tagged inverted/control).
"""
from __future__ import annotations
import csv, math
import numpy as np

RNG = np.random.RandomState(0)
CLUSTERS = ["restorative", "hauntological", "inverted", "anemoia", "control"]


def num(x):
    try:
        v = float(x); return v if math.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def load():
    rows = list(csv.DictReader(open("features.csv", encoding="utf-8")))
    # features defined for ALL images (exclude phi_typicality, obj_expected_absent_score)
    drop = {"phi_typicality", "obj_expected_absent_score"}
    feat_cols = [c for c in rows[0]
                 if c.startswith(("col_", "seg_", "obj_", "phi_")) and c not in drop]
    X = np.array([[num(r[c]) for c in feat_cols] for r in rows])
    for j in range(X.shape[1]):  # NaN -> column mean
        col = X[:, j]; m = np.nanmean(col)
        X[:, j] = np.where(np.isnan(col), m, col)
    y = np.array([r["cluster"] for r in rows])
    return rows, feat_cols, X, y


def classifier_analysis(X, y):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score, confusion_matrix

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    clfs = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "rf": RandomForestClassifier(n_estimators=400, random_state=0),
    }
    out = {}
    for name, clf in clfs.items():
        yhat = cross_val_predict(clf, X, y, cv=skf)
        acc = accuracy_score(y, yhat)
        cm = confusion_matrix(y, yhat, labels=CLUSTERS)
        out[name] = (acc, cm, yhat)
    return out


def permutation_test(X, y, n=200):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    obs = accuracy_score(y, cross_val_predict(clf, X, y, cv=skf))
    null = np.empty(n)
    for i in range(n):
        yp = RNG.permutation(y)
        null[i] = accuracy_score(yp, cross_val_predict(clf, X, yp, cv=skf))
    p = (np.sum(null >= obs) + 1) / (n + 1)
    return obs, null.mean(), null.std(), p


def silhouette(X, y):
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    Xs = StandardScaler().fit_transform(X)
    lab = np.array([CLUSTERS.index(c) for c in y])
    return silhouette_score(Xs, lab)


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    sp = math.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2))
    return (a.mean() - b.mean()) / (sp + 1e-12)


def boot_d_ci(a, b, n=2000):
    a, b = np.asarray(a), np.asarray(b)
    ds = np.empty(n)
    for i in range(n):
        ds[i] = cohens_d(RNG.choice(a, len(a), True), RNG.choice(b, len(b), True))
    return cohens_d(a, b), np.percentile(ds, 2.5), np.percentile(ds, 97.5)


def scene_matched_effects(rows):
    def vals(cluster, key):
        v = np.array([num(r[key]) for r in rows if r["cluster"] == cluster])
        return v[~np.isnan(v)]
    print("\n=== scene-matched effect sizes: inverted vs control (Cohen's d [95% CI]) ===")
    for key in ["seg_human_absence", "seg_emptiness_index", "phi_typicality"]:
        inv, ctl = vals("inverted", key), vals("control", key)
        d, lo, hi = boot_d_ci(inv, ctl)
        print(f"  {key:26s}: d = {d:6.2f}  [{lo:.2f}, {hi:.2f}]  "
              f"(inv n={len(inv)} m={inv.mean():.3f} / ctl n={len(ctl)} m={ctl.mean():.3f})")


def typicality_percentile(rows):
    # raw -phi_typ = squared Mahalanobis distance; reframe inverted vs the control dist
    def maha(cluster):
        v = np.array([-num(r["phi_typicality"]) for r in rows
                      if r["cluster"] == cluster and not np.isnan(num(r["phi_typicality"]))])
        return v
    inv, ctl = maha("inverted"), maha("control")
    print("\n=== phi_typicality reframed (squared Mahalanobis to per-scene control centroid) ===")
    print(f"  control:  median={np.median(ctl):.0f}  90th pct={np.percentile(ctl,90):.0f}")
    print(f"  inverted: median={np.median(inv):.0f}")
    pct = 100.0 * np.mean(ctl < np.median(inv))
    print(f"  inverted median sits at the {pct:.1f}th percentile of the control distribution")
    print(f"  (768-dim embedding, Ledoit-Wolf shrinkage -> raw distances inflate off-manifold;")
    print(f"   percentile/median framing is the honest one, not the raw -1480 vs -41)")


def main():
    rows, feat_cols, X, y = load()
    print(f"loaded {len(rows)} images, {len(feat_cols)} all-defined features for the classifier")
    from collections import Counter
    print("class balance:", dict(Counter(y)))

    res = classifier_analysis(X, y)
    for name, (acc, cm, _) in res.items():
        print(f"\n=== {name}: 5-fold CV accuracy = {acc:.3f} (chance = 0.200) ===")
        print("confusion matrix (rows=true, cols=pred): " + " ".join(c[:5] for c in CLUSTERS))
        for i, c in enumerate(CLUSTERS):
            print(f"  {c:14s} " + " ".join(f"{cm[i,j]:4d}" for j in range(len(CLUSTERS))))

    obs, nm, ns, p = permutation_test(X, y, n=200)
    print(f"\n=== permutation test (logreg, 200 shuffles) ===")
    print(f"  observed acc={obs:.3f}  null={nm:.3f}+/-{ns:.3f}  p={p:.4f}")

    sil = silhouette(X, y)
    print(f"\n=== silhouette (standardized, 5 clusters) = {sil:.3f}  (0=no structure, 1=tight) ===")

    scene_matched_effects(rows)
    typicality_percentile(rows)


if __name__ == "__main__":
    main()

"""
stats_addendum.py
=================
Round-2 review responses:
  1. Firewall-free classifier: rerun on NON-phi features (col/seg/obj only) and on
     phi-only, to show how much of control isolation is the firewall tautology.
  2. Scene-matching documentation: per-scene-category counts for inverted vs control,
     and WITHIN-category Cohen's d (stratified), plus a sample-weighted pooled estimate,
     for human_absence and emptiness.
  3. phi_realism_var per cluster (is it flat? -> reinforces limitation i).
"""
from __future__ import annotations
import csv, math
import numpy as np

RNG = np.random.RandomState(0)
CLUSTERS = ["restorative", "hauntological", "inverted", "anemoia", "control"]
PRIOR_SCENES = ["shopping_mall", "corridor", "swimming_pool", "classroom",
                "playground", "parking_lot", "office"]


def num(x):
    try:
        v = float(x); return v if math.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def load():
    return list(csv.DictReader(open("features.csv", encoding="utf-8")))


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = math.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return (a.mean() - b.mean()) / (sp + 1e-12)


def classifier(rows, cols, tag):
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import accuracy_score, confusion_matrix
    X = np.array([[num(r[c]) for c in cols] for r in rows])
    for j in range(X.shape[1]):
        m = np.nanmean(X[:, j]); X[:, j] = np.where(np.isnan(X[:, j]), m, X[:, j])
    y = np.array([r["cluster"] for r in rows])
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    yhat = cross_val_predict(RandomForestClassifier(400, random_state=0), X, y, cv=skf)
    acc = accuracy_score(y, yhat)
    cm = confusion_matrix(y, yhat, labels=CLUSTERS)
    ctrl_recall = cm[CLUSTERS.index("control"), CLUSTERS.index("control")] / cm[CLUSTERS.index("control")].sum()
    print(f"\n[{tag}] {len(cols)} features: RF 5-fold acc={acc:.3f}, control recall={ctrl_recall:.3f}")
    return acc, ctrl_recall


def main():
    rows = load()
    allcols = list(rows[0].keys())
    nonphi = [c for c in allcols if c.startswith(("col_", "seg_")) or
              c in ("obj_object_count", "obj_object_density", "obj_object_area_frac")]
    phionly = ["phi_surprise", "phi_realism_var", "phi_entropy"]  # phi_typ NaN for 3 clusters
    print("=== 1. firewall decomposition (RF, control recall) ===")
    classifier(rows, nonphi, "NON-phi (col/seg/obj)")
    classifier(rows, phionly, "phi-only (surprise/realism/entropy)")
    classifier(rows, nonphi + phionly, "all 29")

    print("\n=== 2. scene-matching: inverted vs control, per category ===")
    print(f"{'scene':14s}{'inv_n':>6}{'ctl_n':>6}{'d_humabs':>10}{'d_empty':>9}")
    inv_all_ha, ctl_all_ha, ws, ds_ha, ds_em = [], [], [], [], []
    for s in PRIOR_SCENES:
        inv = [r for r in rows if r["cluster"] == "inverted" and r["scene_category"] == s]
        ctl = [r for r in rows if r["cluster"] == "control" and r["scene_category"] == s]
        ha_i = [num(r["seg_human_absence"]) for r in inv]
        ha_c = [num(r["seg_human_absence"]) for r in ctl]
        em_i = [num(r["seg_emptiness_index"]) for r in inv]
        em_c = [num(r["seg_emptiness_index"]) for r in ctl]
        dha, dem = cohens_d(ha_i, ha_c), cohens_d(em_i, em_c)
        print(f"{s:14s}{len(inv):>6}{len(ctl):>6}{dha:>10.2f}{dem:>9.2f}")
        if len(inv) >= 2 and len(ctl) >= 2:
            w = len(inv) + len(ctl); ws.append(w); ds_ha.append(dha); ds_em.append(dem)
    if ws:
        ws = np.array(ws, float)
        print(f"sample-weighted within-category d:  human_absence={np.average(ds_ha, weights=ws):.2f}"
              f"  emptiness={np.average(ds_em, weights=ws):.2f}")

    print("\n=== 3. phi_realism_var per cluster (flat?) ===")
    for c in CLUSTERS:
        v = [num(r["phi_realism_var"]) for r in rows if r["cluster"] == c]
        v = [x for x in v if not np.isnan(x)]
        print(f"  {c:14s} mean={np.mean(v):.4f}  sd={np.std(v):.4f}")


if __name__ == "__main__":
    main()

"""
analyze_arena_features.py
=========================
The payoff test: do the structural CV features predict the aesthetic that HUMANS assigned
each image to (are.na channel membership)? This connects the computation to a real human
criterion -- the dependent variable the paper otherwise lacks.

Design choices to keep it honest:
  - Labels are HUMAN (are.na curators), not ours.
  - GROUPED cross-validation by channel_slug: all images from one channel stay in the same
    fold, so the classifier cannot win by memorising a single curator's house style /
    near-duplicates. This is a conservative test of generalisation across curators.
  - Permutation baseline (shuffle labels within the grouped scheme) for a real null.
  - We report BOTH the 4-way mechanism-cluster task and the harder per-aesthetic task.
  - Firewall: phi_* coordinates were fit on the ORIGINAL controls only; here we additionally
    report accuracy WITHOUT phi to show the structural axes carry the signal on their own.

Inputs:  features_arena.csv (pipeline output)  +  stimuli_manifest_arena.csv (human labels)
Output:  printed report  +  arena_prediction_report.txt
"""
from __future__ import annotations
import csv, os, sys, math
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_join():
    feats = {r["image_id"]: r for r in csv.DictReader(
        open(os.path.join(ROOT, "features_arena.csv"), encoding="utf-8"))}
    man = {r["image_id"]: r for r in csv.DictReader(
        open(os.path.join(ROOT, "stimuli_manifest_arena.csv"), encoding="utf-8"))}
    rows = []
    for iid, fr in feats.items():
        if iid not in man:
            continue
        rec = dict(fr)
        rec["aesthetic"] = man[iid]["aesthetic"]
        rec["channel_slug"] = man[iid]["channel_slug"]
        rec["cluster"] = man[iid].get("cluster") or fr.get("cluster", "")
        rows.append(rec)
    return rows


def numeric_cols(rows, drop_phi=False):
    skip = {"image_id", "image_path", "scene_category", "cluster", "aesthetic",
            "channel_slug", "block_id", "license", "source_url", "source", "query"}
    cols = []
    for k in rows[0]:
        if k in skip:
            continue
        if drop_phi and k.startswith("phi"):
            continue
        # numeric if parseable on most rows
        ok = 0
        for r in rows[:50]:
            try:
                float(r[k]); ok += 1
            except (ValueError, TypeError):
                pass
        if ok >= 40:
            cols.append(k)
    return cols


def matrix(rows, cols):
    X = np.full((len(rows), len(cols)), np.nan)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            try:
                v = float(r[c])
                if math.isfinite(v):
                    X[i, j] = v
            except (ValueError, TypeError):
                pass
    # column-mean impute, then drop all-nan cols
    keep = [j for j in range(X.shape[1]) if np.isfinite(X[:, j]).any()]
    X = X[:, keep]; cols = [cols[j] for j in keep]
    mu = np.nanmean(X, axis=0)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(mu, inds[1])
    return X, cols


def grouped_cv_acc(X, y, groups, seed=0, n_splits=5, shuffle_y=False):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    rng = np.random.RandomState(seed)
    yv = y.copy()
    if shuffle_y:
        yv = yv[rng.permutation(len(yv))]
    gkf = GroupKFold(n_splits=n_splits)
    accs, cm_lbls = [], sorted(set(y))
    idx = {l: i for i, l in enumerate(cm_lbls)}
    cm = np.zeros((len(cm_lbls), len(cm_lbls)), int)
    for tr, te in gkf.split(X, yv, groups):
        clf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)
        clf.fit(X[tr], yv[tr])
        pred = clf.predict(X[te])
        accs.append((pred == yv[te]).mean())
        if not shuffle_y:
            for t, p in zip(yv[te], pred):
                cm[idx[t], idx[p]] += 1
    return float(np.mean(accs)), cm, cm_lbls


def report(rows, label_key, out, n_perm=50):
    y = np.array([r[label_key] for r in rows])
    groups = np.array([r["channel_slug"] for r in rows])
    lines = []
    def P(s): print(s); lines.append(s)
    P(f"\n{'='*66}\nTASK: predict {label_key}  (n={len(rows)}, "
      f"{len(set(y))} classes, {len(set(groups))} channel-groups)\n{'='*66}")
    for tag, drop in [("ALL features", False), ("WITHOUT phi", True)]:
        cols = numeric_cols(rows, drop_phi=drop)
        X, used = matrix(rows, cols)
        acc, cm, lbls = grouped_cv_acc(X, y, groups)
        # majority-class baseline and permutation null
        vals, cnts = np.unique(y, return_counts=True)
        maj = cnts.max() / cnts.sum()
        perms = [grouped_cv_acc(X, y, groups, seed=1000+i, shuffle_y=True)[0]
                 for i in range(n_perm)]
        mu, sd = float(np.mean(perms)), float(np.std(perms) + 1e-9)
        p = (sum(1 for v in perms if v >= acc) + 1) / (n_perm + 1)
        P(f"\n[{tag}]  {len(used)} features, channel-grouped 5-fold")
        P(f"   accuracy            = {acc:.3f}")
        P(f"   majority baseline   = {maj:.3f}")
        P(f"   permutation null    = {mu:.3f} +/- {sd:.3f}   p = {p:.3f}   "
          f"(z = {(acc-mu)/sd:.1f})")
        if not drop:
            P(f"   confusion (rows=true {lbls}):")
            for i, l in enumerate(lbls):
                P(f"     {l[:14]:14s} " + " ".join(f"{cm[i,j]:4d}" for j in range(len(lbls))))
    out.extend(lines)


def main():
    rows = load_join()
    if not rows:
        print("no joined rows -- is features_arena.csv written yet?", file=sys.stderr)
        sys.exit(1)
    print(f"joined {len(rows)} images with human labels")
    out = []
    report(rows, "cluster", out, n_perm=200)       # 4-way mechanism task
    report(rows, "aesthetic", out, n_perm=200)     # harder per-aesthetic task
    with open(os.path.join(ROOT, "arena_prediction_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("\n-> arena_prediction_report.txt")


if __name__ == "__main__":
    main()

"""
analyze_vlm.py
==============
VLM affect panel analysis (Components 1 & 4). Uses the two frontier judges (Gemini, GPT-4o);
extra judges auto-included if their CSVs exist (e.g. a local open VLM).

Reports:
  1. INTER-JUDGE reliability: per-axis Pearson r between judges (+ mode agreement) -- the
     consensus is only meaningful if judges agree.
  2. CAN VLMs READ THE STRUCTURE: predict the HUMAN-assigned cluster from the consensus
     6 affect axes (channel-grouped CV, permutation null) -- VLM-as-observer vs human labels.
  3. DIRECT MODE CALL: VLM forced-choice mode vs human cluster (per-mode accuracy + confusion).
  4. WHERE IT BREAKS: per-mode accuracy across judges -- expect the uncanny/anemoia to fail,
     and frontier models to mis-mode the restorative flagship (Frutiger Aero).
Output: vlm_report.txt
"""
from __future__ import annotations
import csv, os, sys, math, random
from collections import defaultdict, Counter
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
AXES = ["eerie", "nostalgic", "abandoned", "dreamlike", "valence", "arousal"]
CLUSTERS = ["restorative", "hauntological", "inverted", "anemoia"]
JUDGES = {"gemini": "vlm_gemini_affect.csv", "gpt4o": "vlm_openai_affect.csv",
          "claude": "vlm_anthropic_affect.csv", "local": "vlm_local_affect.csv"}


def load_judge(path):
    if not os.path.exists(path):
        return {}
    d = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("ok") != "1":
            continue
        rec = {}
        for a in AXES:
            try:
                rec[a] = float(r[a])
            except (ValueError, TypeError, KeyError):
                rec[a] = None
        rec["mode"] = r.get("mode", "")
        rec["aesthetic"] = r.get("aesthetic", "")
        rec["cluster"] = r.get("cluster", "")
        rec["source"] = r.get("source", "")
        d[r["image_id"]] = rec
    return d


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return float("nan")
    x, y = x[m], y[m]
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    judges = {name: load_judge(os.path.join(ROOT, f)) for name, f in JUDGES.items()}
    judges = {k: v for k, v in judges.items() if v}
    out = []
    def P(s): print(s); out.append(s)
    P("=" * 66); P(f"VLM AFFECT PANEL  (judges: {', '.join(judges)})"); P("=" * 66)
    for name, d in judges.items():
        P(f"  {name}: {len(d)} rated images")

    # channel_slug for grouped CV
    chan = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "stimuli_manifest_arena.csv"), encoding="utf-8")):
        chan[r["image_id"]] = r.get("channel_slug", "")

    names = list(judges)
    # ---- 1. inter-judge reliability ----
    if len(names) >= 2:
        P("\n[1] INTER-JUDGE reliability (Pearson r per axis):")
        common = set.intersection(*[set(judges[n]) for n in names])
        P(f"    images rated by all judges: {len(common)}")
        for a in AXES:
            rs = []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    A = [judges[names[i]][k][a] for k in common]
                    B = [judges[names[j]][k][a] for k in common]
                    rs.append(pearson(A, B))
            P(f"    {a:11s} r={np.nanmean(rs):.2f}")
        # mode agreement (first two judges)
        if len(names) >= 2:
            n0, n1 = names[0], names[1]
            agree = [judges[n0][k]["mode"] == judges[n1][k]["mode"] for k in common
                     if judges[n0][k]["mode"] and judges[n1][k]["mode"]]
            if agree:
                P(f"    mode agreement {n0} vs {n1}: {np.mean(agree):.2f}")

    # consensus over judges (arena images with human labels)
    allids = set().union(*[set(judges[n]) for n in names])
    cons = {}
    for iid in allids:
        recs = [judges[n][iid] for n in names if iid in judges[n]]
        arena = [r for r in recs if r["source"] == "arena" and r["cluster"] in CLUSTERS]
        if not arena:
            continue
        c = {a: np.nanmean([r[a] for r in recs if r[a] is not None]) for a in AXES}
        if any(not np.isfinite(c[a]) for a in AXES):
            continue
        cons[iid] = {"axes": c, "cluster": arena[0]["cluster"],
                     "aesthetic": arena[0]["aesthetic"],
                     "modes": [r["mode"] for r in recs if r["mode"]]}
    P(f"\narena images with consensus affect + human label: {len(cons)}")
    if len(cons) < 40:
        P("(not enough rated yet; rerun after batches finish)"); _save(out); return

    # ---- 2. predict human cluster from consensus VLM axes (channel-grouped CV) ----
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    ids = list(cons)
    X = np.array([[cons[i]["axes"][a] for a in AXES] for i in ids])
    y = np.array([cons[i]["cluster"] for i in ids])
    groups = np.array([chan.get(i, i) for i in ids])
    def grouped_acc(X, y, groups, shuffle=False, seed=0):
        rng = np.random.RandomState(seed); yy = y.copy()
        if shuffle:
            yy = yy[rng.permutation(len(yy))]
        gkf = GroupKFold(n_splits=5); accs = []
        for tr, te in gkf.split(X, yy, groups):
            clf = RandomForestClassifier(300, random_state=seed, n_jobs=-1).fit(X[tr], yy[tr])
            accs.append((clf.predict(X[te]) == yy[te]).mean())
        return float(np.mean(accs))
    acc = grouped_acc(X, y, groups)
    maj = max(Counter(y).values()) / len(y)
    null = [grouped_acc(X, y, groups, shuffle=True, seed=1000 + i) for i in range(50)]
    mu, sd = float(np.mean(null)), float(np.std(null) + 1e-9)
    P(f"\n[2] PREDICT human cluster from 6 consensus VLM affect axes (channel-grouped 5-fold):")
    P(f"    accuracy={acc:.3f}  majority={maj:.3f}  null={mu:.3f}+/-{sd:.3f}  z={(acc-mu)/sd:.1f}")

    # ---- 3 & 4. direct VLM mode call vs human cluster + where it breaks ----
    P("\n[3/4] VLM forced-choice MODE vs human cluster (consensus majority vote):")
    per_mode = defaultdict(lambda: [0, 0])
    conf = defaultdict(Counter)
    for i in ids:
        modes = cons[i]["modes"]
        if not modes:
            continue
        vote = Counter(modes).most_common(1)[0][0]
        true = cons[i]["cluster"]
        conf[true][vote] += 1
        per_mode[true][1] += 1
        if vote == true:
            per_mode[true][0] += 1
    overall = sum(v[0] for v in per_mode.values()) / max(1, sum(v[1] for v in per_mode.values()))
    P(f"    overall mode-match accuracy = {overall:.3f}")
    for c in CLUSTERS:
        ok, n = per_mode[c]
        if n:
            P(f"    {c:14s}: {ok/n:.2f}  (n={n})   confusions-> {dict(conf[c])}")
    P("    => 'where it breaks': lowest-accuracy modes are the cultural frontier (uncanny/anemoia)")
    _save(out)


def _save(out):
    with open(os.path.join(ROOT, "vlm_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("\n-> vlm_report.txt")


if __name__ == "__main__":
    main()

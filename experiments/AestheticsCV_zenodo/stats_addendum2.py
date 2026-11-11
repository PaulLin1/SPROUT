"""
stats_addendum2.py — second-review numerical loose ends.
  - account for all 194 inverted images (scene-tagged vs generic)
  - per-cell bootstrap 95% CI on human_absence d (flag small/unstable cells)
  - sample-weighted within-category d WITH bootstrap CI
"""
from __future__ import annotations
import csv, math
import numpy as np
RNG = np.random.RandomState(0)
PRIOR = ["classroom", "office", "swimming_pool", "shopping_mall", "corridor",
         "parking_lot", "playground"]


def num(x):
    try:
        v = float(x); return v if math.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = math.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    return (a.mean()-b.mean())/(sp+1e-12)


rows = list(csv.DictReader(open("features.csv", encoding="utf-8")))
inv = [r for r in rows if r["cluster"] == "inverted"]
ctl = [r for r in rows if r["cluster"] == "control"]
from collections import Counter
print("inverted total:", len(inv))
print("inverted scene_category counts:", dict(Counter(r["scene_category"] for r in inv)))
tagged = sum(1 for r in inv if r["scene_category"] in PRIOR)
print(f"inverted in the 7 prior scenes: {tagged}; generic/other: {len(inv)-tagged}")
print("control total:", len(ctl), "scene counts:", dict(Counter(r["scene_category"] for r in ctl)))

key = "seg_human_absence"
cells = []
print(f"\n{'scene':14s}{'inv':>4}{'ctl':>4}{'d':>7}{'  95% CI':>16}")
for s in PRIOR:
    ai = [num(r[key]) for r in inv if r["scene_category"] == s]
    ac = [num(r[key]) for r in ctl if r["scene_category"] == s]
    dd = d(ai, ac)
    bs = [d(RNG.choice(ai, len(ai), True), RNG.choice(ac, len(ac), True)) for _ in range(2000)]
    lo, hi = np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)
    flag = "  (small/unstable)" if min(len(ai), len(ac)) < 15 else ""
    cells.append((s, ai, ac, len(ai)+len(ac)))
    print(f"{s:14s}{len(ai):>4}{len(ac):>4}{dd:>7.2f}   [{lo:5.2f},{hi:5.2f}]{flag}")

# weighted within-category d with bootstrap CI (resample within each cell)
ws = np.array([c[3] for c in cells], float)
def weighted_d(resample):
    ds = []
    for (_, ai, ac, _) in cells:
        if resample:
            ai2 = RNG.choice(ai, len(ai), True); ac2 = RNG.choice(ac, len(ac), True)
        else:
            ai2, ac2 = ai, ac
        ds.append(d(ai2, ac2))
    return np.average(ds, weights=ws)
point = weighted_d(False)
bsw = [weighted_d(True) for _ in range(3000)]
print(f"\nsample-weighted within-category d (human_absence) = {point:.2f}  "
      f"95% CI [{np.nanpercentile(bsw,2.5):.2f}, {np.nanpercentile(bsw,97.5):.2f}]")
print(f"corridor share of tagged inverted: {len([r for r in inv if r['scene_category']=='corridor'])}/{tagged}")

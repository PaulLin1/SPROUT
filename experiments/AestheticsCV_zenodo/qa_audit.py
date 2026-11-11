"""
qa_audit.py
===========
Post-collection / post-run audit. Produces:
  - manifest summary: per-cluster counts, license breakdown, per-scene counts,
    how many images carry a priors-matched scene_category.
  - feature-table sanity: row count vs manifest, all-NaN columns, non-finite phi.
  - colour cluster-separation check on REAL images: per-cluster means of the
    discriminative colour features (hue_mean_deg, colorfulness, gloss_index,
    edge_density) -- the replication of the synthetic-stimulus separation.

    python qa_audit.py --manifest stimuli_manifest.csv --features features.csv
"""

from __future__ import annotations
import argparse, csv, math, os
from collections import Counter, defaultdict

PRIOR_SCENES = {"shopping_mall", "corridor", "swimming_pool", "classroom",
                "playground", "parking_lot", "office"}
CLUSTERS = ["restorative", "hauntological", "inverted", "anemoia", "control"]


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def audit_manifest(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"\n=== MANIFEST AUDIT ({len(rows)} images) ===")
    cl = Counter(r["cluster"] for r in rows)
    print("per-cluster:")
    for c in CLUSTERS:
        print(f"  {c:14s}: {cl.get(c,0)}")
    lic = Counter((r.get("license","") or "unknown").split()[0] if r.get("license") else "unknown"
                  for r in rows)
    print("license (first token):")
    for k, v in lic.most_common():
        print(f"  {k:18s}: {v}")
    matched = sum(1 for r in rows if r["scene_category"] in PRIOR_SCENES)
    print(f"priors-matched scene_category: {matched}/{len(rows)}")
    src = Counter(r.get("source","") for r in rows)
    print("source:")
    for k, v in src.most_common():
        print(f"  {k:20s}: {v}")
    return rows


def audit_features(path, manifest_rows):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"\n=== FEATURE-TABLE AUDIT ({len(rows)} rows) ===")
    if manifest_rows is not None:
        print(f"row count vs manifest: {len(rows)} / {len(manifest_rows)}",
              "OK" if len(rows) == len(manifest_rows) else "MISMATCH")
    cols = rows[0].keys() if rows else []
    # all-NaN / all-empty columns
    bad = []
    for c in cols:
        vals = [r.get(c, "") for r in rows]
        if all(_num(v) is None for v in vals) and c.startswith(("col_", "seg_", "obj_", "phi_")):
            bad.append(c)
    print(f"all-non-numeric feature columns: {bad if bad else 'none'}")
    # non-finite phi
    for c in [k for k in cols if k.startswith("phi_")]:
        n_bad = sum(1 for r in rows if _num(r.get(c)) is None)
        print(f"  {c:18s}: {n_bad} non-finite / {len(rows)}")
    return rows


def colour_separation(feature_rows):
    keys = ["col_hue_mean_deg", "col_colorfulness", "col_gloss_index",
            "col_edge_density", "col_sat_mean", "col_val_std"]
    by = defaultdict(lambda: defaultdict(list))
    for r in feature_rows:
        c = r.get("cluster", "?")
        for k in keys:
            v = _num(r.get(k))
            if v is not None:
                by[c][k].append(v)
    print("\n=== COLOUR CLUSTER-SEPARATION (real images) ===")
    header = "cluster".ljust(14) + "".join(k.replace("col_", "").rjust(13) for k in keys)
    print(header)
    for c in CLUSTERS:
        if c not in by:
            continue
        cells = []
        for k in keys:
            vals = by[c][k]
            cells.append(f"{(sum(vals)/len(vals)):.2f}".rjust(13) if vals else "-".rjust(13))
        print(c.ljust(14) + "".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="stimuli_manifest.csv")
    ap.add_argument("--features", default=None)
    args = ap.parse_args()

    mrows = audit_manifest(args.manifest) if os.path.exists(args.manifest) else None
    if args.features and os.path.exists(args.features):
        frows = audit_features(args.features, mrows)
        colour_separation(frows)


if __name__ == "__main__":
    main()

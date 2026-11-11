"""
run_pipeline.py
===============
Orchestrator. Reads a manifest CSV (image_path, image_id, scene_category, cluster,
license, source_url) and writes one design-matrix row per image.

Provenance columns (license, source_url) are MANDATORY and carried through to the
output, so the published stimulus set documents IP per image -- handle this at
ingestion, not after.

phi_surprise: computed in two passes. During the per-image loop we cache the raw
embedding-space reconstruction error (SurprisePCA, fit on control embeddings) and the
three low-level-complexity covariates. After the loop we residualise raw surprise on
complexity using the CONTROL rows as the baseline, so phi_surprise is *semantic*
surprise, not busyness. This is the pragmatic stand-in for the design's VQ-VAE.

Usage:
    python run_pipeline.py --manifest stimuli_manifest.csv --out features.csv \
        --reference control_embeddings.npz      # fit reference + SurprisePCA on controls

For a dry run of the cheap colour axis only, pass --color-only.
On the RTX 5070 Ti the full run handles ~1000-1400 images comfortably.
"""

from __future__ import annotations
import argparse, csv, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import color_gradient


REQUIRED_COLS = {"image_path", "image_id", "scene_category", "cluster"}
PROVENANCE_COLS = {"license", "source_url"}


def read_manifest(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("empty manifest")
    missing = REQUIRED_COLS - set(rows[0].keys())
    if missing:
        raise SystemExit(f"manifest missing columns: {missing}")
    if PROVENANCE_COLS - set(rows[0].keys()):
        print("WARNING: provenance columns (license, source_url) absent. "
              "A published stimulus set needs per-image licensing -- add them now.",
              file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="features.csv")
    ap.add_argument("--reference", help="npz with control embeddings+categories")
    ap.add_argument("--color-only", action="store_true",
                    help="run only the cheap colour/gradient axis (no model weights)")
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    from PIL import Image

    seg = obj_det = expect = embedder = ref = surprise = None
    phi_realism_variance = None
    if not args.color_only:
        from segmentation import SemanticSegmenter
        from object_analysis import OpenVocabDetector, SceneExpectation
        from ml_features import (Embedder, CanonicalReference, SurprisePCA,
                                 SurpriseModel, residualize_surprise,
                                 phi_realism_variance)
        seg = SemanticSegmenter()
        obj_det = OpenVocabDetector()
        expect = SceneExpectation()
        embedder = Embedder()
        if args.reference:
            data = np.load(args.reference, allow_pickle=True)
            emb = data["embeddings"]
            ref = CanonicalReference().fit(emb, list(data["categories"]))
            surprise = SurprisePCA().fit(emb)

    out_rows = []
    raw_surprise = []          # cached for the post-loop residualisation
    complexity = []
    is_control = []

    for i, r in enumerate(rows):
        pil = Image.open(r["image_path"]).convert("RGB")
        rec = {k: r.get(k, "") for k in (REQUIRED_COLS | PROVENANCE_COLS) if k in r}

        # axis 1 -- colour & gradient (always)
        cfeats = color_gradient.analyze(np.asarray(pil) / 255.0)
        rec.update({f"col_{k}": cfeats[k] for k in color_gradient.SCALAR_KEYS})

        if not args.color_only:
            import segmentation as S, object_analysis as O
            # axis 2 -- segmentation
            sfeats = S.analyze(pil, seg)
            rec.update({f"seg_{k}": sfeats[k] for k in S.SCALAR_KEYS})
            # axis 3 -- objects
            ofeats = O.analyze(pil, r["scene_category"], obj_det, expect)
            rec.update({f"obj_{k}": ofeats[k] for k in O.SCALAR_KEYS})
            # phi mechanism-proxies
            emb_i = embedder.embed(pil)
            z = emb_i.get("dino", emb_i.get("clip"))
            if ref is not None:
                rec["phi_typicality"] = (ref.phi_typicality(z, r["scene_category"])
                                         if r["scene_category"] in ref.mu
                                         else float("nan"))
                rec["phi_entropy"] = ref.phi_entropy(z)
            rec["phi_realism_var"] = phi_realism_variance(pil, embedder)
            # cache surprise inputs for the two-pass residualisation
            if surprise is not None:
                raw_surprise.append(surprise.raw_surprise(z))
                complexity.append(SurpriseModel.low_level_complexity(pil))
                is_control.append(r["cluster"] == "control")

        out_rows.append(rec)
        if (i + 1) % 25 == 0:
            print(f"  processed {i+1}/{len(rows)}", file=sys.stderr)

    # post-loop: residualise raw surprise on low-level complexity (controls = baseline)
    if not args.color_only and surprise is not None and raw_surprise:
        phi_s = residualize_surprise(np.array(raw_surprise),
                                     np.vstack(complexity),
                                     np.array(is_control, bool))
        for rec, v in zip(out_rows, phi_s):
            rec["phi_surprise"] = float(v)

    # write
    keys = sorted({k for row in out_rows for k in row})
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows x {len(keys)} cols -> {args.out}")


if __name__ == "__main__":
    main()

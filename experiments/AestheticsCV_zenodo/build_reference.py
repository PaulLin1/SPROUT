"""
build_reference.py
==================
Fit the canonical-scene reference from the POPULATED CONTROL images only.

Embeds every control image (cluster == "control") with DINOv2 (CLS token), groups by
scene_category, and saves control_embeddings.npz with arrays:
    embeddings : (N, D) float
    categories : (N,) str   -- scene_category per row

run_pipeline.py loads this to fit CanonicalReference (phi_typicality, phi_entropy)
and SurprisePCA (raw surprise). It must be controls-only: this is the "normal,
populated, everyday-scene" reference distribution, kept strictly separate from the
aesthetic stimuli and from any human ratings.

    python build_reference.py --manifest stimuli_manifest.csv --out control_embeddings.npz
"""

from __future__ import annotations
import argparse, csv, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pipeline"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(HERE, "stimuli_manifest.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "control_embeddings.npz"))
    ap.add_argument("--cluster", default="control")
    args = ap.parse_args()

    from PIL import Image
    from ml_features import Embedder

    with open(args.manifest, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["cluster"] == args.cluster]
    if not rows:
        raise SystemExit(f"no rows with cluster=={args.cluster} in manifest")

    embedder = Embedder(use_dino=True, use_clip=False)
    embs, cats = [], []
    for i, r in enumerate(rows):
        try:
            pil = Image.open(r["image_path"]).convert("RGB")
        except Exception as e:
            print(f"  skip {r['image_id']}: {e}", file=sys.stderr)
            continue
        z = embedder.embed(pil)["dino"]
        embs.append(z)
        cats.append(r["scene_category"])
        if (i + 1) % 25 == 0:
            print(f"  embedded {i+1}/{len(rows)}", file=sys.stderr)

    embs = np.vstack(embs)
    cats = np.array(cats)
    np.savez(args.out, embeddings=embs, categories=cats)
    print(f"saved {embs.shape[0]} control embeddings (dim {embs.shape[1]}) -> {args.out}")
    uniq, counts = np.unique(cats, return_counts=True)
    for u, c in zip(uniq, counts):
        print(f"  {u:16s}: {c}")


if __name__ == "__main__":
    main()

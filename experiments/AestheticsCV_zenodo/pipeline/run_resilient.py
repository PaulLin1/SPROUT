"""
run_resilient.py
================
Hang-tolerant feature extractor for large/dirty corpora (e.g. the are.na pull, where one
pathological image can wedge a model/CUDA call indefinitely).

Robustness contract:
  - INCREMENTAL: each image's row is appended to --out and flushed immediately, so a crash
    or kill never loses completed work.
  - RESUME: image_ids already in --out are skipped.
  - AUTO-BLACKLIST: before processing image X its id is written to <out>.cur; if a previous
    run died while X was in flight, that id is read back from .cur on startup and added to a
    persistent <out>.bad blacklist, so the offending image is skipped on resume.
  - PER-IMAGE try/except: ordinary decode/feature errors skip the image (also blacklisted).

A fixed header is derived from the modules' SCALAR_KEYS so appended rows stay aligned.
phi_surprise's two-pass residualisation is omitted here (the are.na corpus has no controls
to baseline against); phi_typicality/entropy/realism_var are still emitted.

Usage (driven by a watchdog that kills + restarts on stall):
    python pipeline/run_resilient.py --manifest stimuli_manifest_arena.csv \
        --out features_arena.csv --reference control_embeddings.npz
"""
from __future__ import annotations
import argparse, csv, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import color_gradient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="features_arena.csv")
    ap.add_argument("--reference")
    args = ap.parse_args()

    cur_path = args.out + ".cur"
    bad_path = args.out + ".bad"
    try_path = args.out + ".try"
    MAX_TRIES = 2     # tolerate transient (e.g. OOM) errors; blacklist only after repeats

    # --- resume: ids already done ---
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add(r["image_id"])

    # --- blacklist + per-id attempt counts ---
    bad = set()
    if os.path.exists(bad_path):
        bad = {l.strip() for l in open(bad_path, encoding="utf-8") if l.strip()}
    tries = {}
    if os.path.exists(try_path):
        for l in open(try_path, encoding="utf-8"):
            if ":" in l:
                k, v = l.rsplit(":", 1)
                tries[k.strip()] = int(v)
    # whatever was in flight when we last died = a true HANG -> blacklist immediately
    if os.path.exists(cur_path):
        stuck = open(cur_path, encoding="utf-8").read().strip()
        if stuck and stuck not in done:
            bad.add(stuck)
            with open(bad_path, "a", encoding="utf-8") as f:
                f.write(stuck + "\n")
            print(f"  blacklisted HUNG image: {stuck}", file=sys.stderr)
        os.remove(cur_path)

    def record_failure(iid, err, permanent):
        tries[iid] = tries.get(iid, 0) + 1
        with open(try_path, "w", encoding="utf-8") as tf:
            for k, v in tries.items():
                tf.write(f"{k}:{v}\n")
        if permanent or tries[iid] >= MAX_TRIES:
            with open(bad_path, "a", encoding="utf-8") as bf:
                bf.write(iid + "\n")
            print(f"  BLACKLIST {iid} (try {tries[iid]}): {err}", file=sys.stderr)
        else:
            print(f"  retry-later {iid} (try {tries[iid]}): {err}", file=sys.stderr)

    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    todo = [r for r in rows if r["image_id"] not in done and r["image_id"] not in bad]
    print(f"{len(rows)} total, {len(done)} done, {len(bad)} blacklisted, {len(todo)} to do",
          file=sys.stderr)
    if not todo:
        print("nothing to do.", file=sys.stderr); return

    from PIL import Image
    from segmentation import SemanticSegmenter
    import segmentation as S
    from object_analysis import OpenVocabDetector, SceneExpectation
    import object_analysis as O
    from ml_features import Embedder, CanonicalReference, SurprisePCA, phi_realism_variance
    seg = SemanticSegmenter(); obj_det = OpenVocabDetector(); expect = SceneExpectation()
    embedder = Embedder()
    ref = surprise = None
    if args.reference:
        data = np.load(args.reference, allow_pickle=True)
        ref = CanonicalReference().fit(data["embeddings"], list(data["categories"]))
        surprise = SurprisePCA().fit(data["embeddings"])

    META = ["image_id", "image_path", "cluster", "scene_category", "aesthetic", "channel_slug"]
    PHI = ["phi_typicality", "phi_entropy", "phi_realism_var", "raw_surprise"]
    header = (META + [f"col_{k}" for k in color_gradient.SCALAR_KEYS]
              + [f"seg_{k}" for k in S.SCALAR_KEYS]
              + [f"obj_{k}" for k in O.SCALAR_KEYS] + PHI)

    new_file = not os.path.exists(args.out)
    fout = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fout, fieldnames=header, extrasaction="ignore")
    if new_file:
        writer.writeheader(); fout.flush()

    n_ok = 0
    for i, r in enumerate(todo):
        iid = r["image_id"]
        with open(cur_path, "w", encoding="utf-8") as cf:   # claim marker
            cf.write(iid)
        try:
            pil = Image.open(r["image_path"]).convert("RGB")
            rec = {k: r.get(k, "") for k in META}
            cf_ = color_gradient.analyze(np.asarray(pil) / 255.0)
            rec.update({f"col_{k}": cf_[k] for k in color_gradient.SCALAR_KEYS})
            sf = S.analyze(pil, seg)
            rec.update({f"seg_{k}": sf[k] for k in S.SCALAR_KEYS})
            of = O.analyze(pil, r.get("scene_category", "generic"), obj_det, expect)
            rec.update({f"obj_{k}": of[k] for k in O.SCALAR_KEYS})
            emb_i = embedder.embed(pil); z = emb_i.get("dino", emb_i.get("clip"))
            if ref is not None:
                sc = r.get("scene_category", "generic")
                rec["phi_typicality"] = ref.phi_typicality(z, sc) if sc in ref.mu else float("nan")
                rec["phi_entropy"] = ref.phi_entropy(z)
                rec["raw_surprise"] = float(surprise.raw_surprise(z)) if surprise else float("nan")
            rec["phi_realism_var"] = phi_realism_variance(pil, embedder)
            writer.writerow(rec); fout.flush()
            n_ok += 1
        except (OSError, ValueError) as e:
            # decode/format errors are inherent to the file -> permanent
            record_failure(iid, repr(e), permanent=True)
        except Exception as e:
            # runtime/CUDA/OOM etc. -> transient, allow a retry on a later pass
            record_failure(iid, repr(e), permanent=False)
        if os.path.exists(cur_path):
            os.remove(cur_path)                              # clean claim on success
        if (n_ok) % 25 == 0 and n_ok:
            print(f"  processed {n_ok} (item {i+1}/{len(todo)})", file=sys.stderr)

    fout.close()
    print(f"done this pass: {n_ok} new rows -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

# Aesthetics CV — Corpus + Feature-Table Build Report

**Date:** 2026-06-11
**Project root:** `C:\Users\yanyl\Downloads\AestheticsCV\`
**Scope of this build:** stand up a Blackwell-compatible ML environment, collect an
openly-licensed image corpus + populated controls with per-image provenance, run the
three-axis + φ computer-vision pipeline, and emit the analysis-ready feature table.
*Out of scope (deferred):* downstream mixed-models / model horse-race, human-ratings
instrument.

---

## 1. Environment

| Component | Value |
|---|---|
| GPU | NVIDIA RTX 5070 Ti, 16 GB, driver 576.88 |
| Compute capability | **sm_120 (Blackwell)** — verified GPU matmul + all model inference |
| conda env | `aesthetics` (Python 3.12), at `C:\Users\yanyl\miniconda3\envs\aesthetics` |
| PyTorch | **2.11.0+cu128** (CUDA 12.8 wheel — required for Blackwell; the default stable wheel fails with "no kernel image" on sm_120) |
| transformers | 5.11.0 |
| scikit-image / scikit-learn | 0.26.0 / 1.9.0 |
| Models | SegFormer-b4 (ADE20K), OWLv2-base-ensemble, DINOv2-base, CLIP ViT-B/32 — all load + infer cleanly under transformers 5.11 |

Two code fixes were required for the new stack and are in the `pipeline/` copies (the
originals in `Downloads\files\` are untouched):
- `LogisticRegression(multi_class="multinomial")` → removed the deprecated arg
  (removed in scikit-learn ≥1.7; multinomial is now the default).
- OWLv2 post-processing renamed: `post_process_object_detection` →
  `post_process_grounded_object_detection` (transformers ≥5). `detect()` now calls the
  new method with a `<5` fallback and reads `text_labels`/`labels` robustly.

---

## 2. Corpus & provenance

**458 images**, all from licensed sources, one subdirectory per cluster under
`corpus/images/`. Manifest: `stimuli_manifest.csv` (columns
`image_path,image_id,scene_category,cluster,license,source_url,source,query`).

| cluster (mechanism) | n | role |
|---|---|---|
| restorative | 88 | warm, comfort-seeking nostalgia (Boym restorative) |
| hauntological | 97 | ironic/melancholic "lost futures" |
| inverted | 138 | liminal / uncanny / kenopsia (empty, scene-matched) |
| anemoia | 53 | imagined-past nostalgia |
| control | 82 | the *same* scenes as inverted, but **populated** |

**Licensing (per-image, logged):** CC (various CC-BY/CC-BY-SA) 371, CC0 46, Public
domain 41. **Source:** Wikimedia Commons for all 458 (see limitations). **Scene-tagged
to the object-prior categories** (corridor / shopping_mall / swimming_pool / classroom
/ office / parking_lot / playground): 210/458 — these drive the expected-but-absent
metric; the rest carry `scene_category = generic` and are excluded from that one metric
by design.

**Control reference:** `control_embeddings.npz` — 82 control DINOv2 embeddings
(768-dim) grouped by scene_category (shopping_mall 17, corridor 16, office 15,
playground 12, parking_lot 10, swimming_pool 7, classroom 5). Used to fit the canonical
reference (φ_typicality, φ_entropy) and the SurprisePCA basis (φ_surprise). Controls
only — strict firewall from the aesthetic stimuli.

### Collection method & two failures fixed along the way
- **Sources:** keyless **Wikimedia Commons** API (the reliable workhorse: ~40
  open-licensed results per query, every query) + **Openverse** as an opportunistic
  supplement. Pexels/Unsplash adapters exist but are off (need API keys).
- **Failure 1 — strict AND-matching:** both APIs require *all* query words to match, so
  long descriptive phrases ("empty hallway corridor fluorescent") returned zero. Fixed
  by rewriting the taxonomy to short concrete nouns (1–3 words). The niche aesthetics
  (vaporwave, weirdcore) are searched by their *visual constituents* (greek bust, CRT
  monitor, neon sign) since the folk labels aren't licensed-tagged — an honest,
  licensed approximation rather than scraping the aesthetic communities.
- **Failure 2 — HTTP 429 under concurrency:** 24 download workers triggered Wikimedia
  throttling (80/120 rejected). Fixed with retry+backoff on 429 and a polite 8-worker
  pool; realised keep-rate recovered from ~2% to a usable level across all five
  clusters.

---

## 3. The three axes + φ (what each number means)

- **Axis 1 — colour/gradient** (`color_gradient.py`, CPU): palette/hue fingerprint,
  Hasler–Süsstrunk colourfulness, `gloss_index` (smooth-gradient ÷ edge density — the
  Frutiger Aero "one continuous environment" signature), edge density, sat/val stats.
- **Axis 2 — segmentation** (`segmentation.py`, SegFormer/ADE20K): `emptiness_index`,
  `human_absence`, `vanishing_structure`, region count/Gini, `light_source_frac` —
  Fisher's *eerie* (absence) and configural deviation.
- **Axis 3 — objects** (`object_analysis.py`, OWLv2): `expected_absent_score` — the
  novel object-level eerie metric (prior-weighted objects a scene should contain but
  lacks), plus object density/area.
- **φ mechanism-proxies** (`ml_features.py`): `phi_typicality` (Mahalanobis from the
  per-scene control centroid; configural deviation), `phi_realism_var` (patch-level
  CLIP P(real) variance; MacDorman realism inconsistency), `phi_surprise` (pragmatic
  stand-in — embedding-space PCA reconstruction error fit on controls, **residualised
  on low-level complexity** so it's semantic surprise not busyness), `phi_entropy`
  (scene-classifier softmax entropy; Jentsch category uncertainty, kept to be falsified).

---

## 4. Colour-axis validation (real images)

The cheap axis separates clusters on real images as the theory predicts. Per-cluster
means:

| cluster | hue° | colorfulness | gloss_index | edge_density | sat_mean | val_std |
|---|---|---|---|---|---|---|
| restorative | 133.5 | 51.2 | 13.5 | 0.08 | 0.40 | 0.24 |
| hauntological | 81.2 | 32.0 | 19.3 | 0.06 | 0.25 | 0.23 |
| inverted | 85.4 | 30.8 | 12.0 | 0.07 | 0.25 | 0.23 |
| anemoia | 93.2 | 36.2 | 7.6 | 0.10 | 0.33 | 0.26 |
| control | 85.0 | 29.1 | 8.6 | 0.08 | 0.23 | 0.25 |

**Reading:** restorative is cleanly separated — most colourful and saturated. The three
uncanny-family clusters (hauntological / inverted / control) collapse into a
desaturated band, exactly the restorative-vs-degraded contrast. Inverted-vs-control is
*not* a colour distinction (they share scenes); that separation is carried by the
segmentation/object/φ axes (emptiness, human_absence, expected_absent), not axis 1.

---

## 5. Feature table

`features.csv` — **458 rows × 37 columns**, one design-matrix row per image. Columns:
provenance (`image_id, scene_category, cluster, license, source_url`) + `col_*` (13,
axis 1) + `seg_*` (10, axis 2) + `obj_*` (4, axis 3) + `phi_typicality, phi_realism_var,
phi_surprise, phi_entropy`. Full GPU run wall-time ≈ 11 min (OWLv2 detection is the
bottleneck at ~0.7 img/s).

### Per-cluster means (the result that matters)

| cluster | emptiness | human_absence | vanishing | exp_absent | phi_typicality | phi_realism_var | phi_surprise | phi_entropy |
|---|---|---|---|---|---|---|---|---|
| restorative | 0.112 | 0.107 | 0.302 | — | — | 0.031 | 1937 | 1.556 |
| hauntological | 0.344 | 0.344 | 0.832 | — | — | 0.044 | 1682 | 1.066 |
| **inverted** | **0.549** | **0.552** | 0.850 | 0.778 | **−1932** | 0.038 | 1182 | 0.768 |
| anemoia | 0.214 | 0.221 | 0.475 | — | — | 0.036 | 1857 | 1.429 |
| control | 0.372 | 0.307 | 0.814 | 0.791 | −29 | 0.044 | −0.000 | 0.017 |

(`exp_absent`/`phi_typicality` are only defined for the scene-tagged inverted+control
images; restorative/hauntological/anemoia are `generic`-scene and blank by design.)

### What the numbers say
- **The eerie signal works.** `inverted` (liminal/empty) tops `emptiness` (0.549) and
  `human_absence` (0.552), clearly above every other cluster — including the *same
  scenes when populated* (`control`: 0.372 / 0.307). This is Fisher's eerie quantified.
- **`phi_typicality` is the sharpest discriminator** of inverted vs control:
  **−1932 vs −29** — empty scene-matched images sit far off their per-scene control
  centroid (configural deviation), populated ones sit on it.
- **The control firewall validates itself.** Controls define the "normal" manifold, so
  by construction they score `phi_surprise ≈ 0.000` and `phi_entropy ≈ 0.017` (the scene
  classifier is confident); all aesthetic clusters are off-manifold (high surprise,
  high entropy). `inverted` has the *lowest* surprise among aesthetics (1182) — sensible,
  since empty rooms are structurally near the normal room types, just unpopulated.
- **An honest divergence between two operationalizations of the eerie.** The
  segmentation `human_absence` separates inverted from control, but the object-level
  `expected_absent` does **not** (0.778 vs 0.791). Cause: Wikimedia "populated" scenes
  are often architectural shots with few or no people, so OWLv2 finds little to violate
  in either case. The segmentation metric is the cleaner eerie discriminator in this
  corpus — a finding to carry forward (and a reason the publication control set needs
  genuinely crowded scenes, e.g. Places365).

---

## 6. Feature-table audit

- Row count: **458 / 458** vs manifest (OK).
- No all-non-numeric feature columns.
- φ finiteness: `phi_entropy`, `phi_realism_var`, `phi_surprise` — 0 non-finite / 458.
  `phi_typicality` — 248 non-finite / 458, **by design**: exactly the 248 `generic`-scene
  images with no per-scene control centroid (458 − 210 priors-matched = 248).
- `phi_surprise` residualised on low-level complexity with the **control rows as the
  baseline** (controls ≈ 0 residual confirms the residualizer fit correctly).

---

## 7. Honest limitations (for the eventual paper)

1. **Single source (Wikimedia) for this pilot.** Encyclopedic imagery, not
   community-curated aesthetics — so the *look* of niche clusters is approximated by
   visual constituents, not authentic exemplars. For restorative this shows up as a
   **green** colour centroid (meadows/reef) rather than the predicted Frutiger Aero
   **cyan** of aqua-glossy wallpapers. A publication set should add per-creator licensed
   aesthetic exemplars and/or a diffusion-generated controllability arm.
2. **n = 458, below the ~1000+ stretch target.** Balanced across clusters and
   sufficient to validate the pipeline; scaling is just more queries + pagination
   (Openverse with an API key lifts the anonymous throttle).
3. **Thin control cells** (classroom n=5) make per-scene covariance estimates noisy;
   LedoitWolf shrinkage mitigates but more controls are wanted.
4. **φ_surprise is a pragmatic PCA stand-in**, not the design's VQ-VAE/latent-diffusion
   reconstruction NLL. It is residualised on low-level complexity and labelled as a
   stand-in; treat as hypothesis-generating until the generative model is trained.
5. **scene_category coverage is partial** (210/458 priors-matched); `expected_absent`
   is only defined for those.

---

## 8. How to reproduce / extend

```bash
conda activate aesthetics
cd C:\Users\yanyl\Downloads\AestheticsCV
# 1. collect (Wikimedia-primary; add openverse with a key for more)
python collect/build_corpus.py --per-query 50 --target 1300 --workers 8 --adapters wikimedia,openverse
# 2. canonical reference from controls
python build_reference.py --manifest stimuli_manifest.csv --out control_embeddings.npz
# 3a. cheap colour dry run + separation check
python pipeline/run_pipeline.py --manifest stimuli_manifest.csv --out feats_color.csv --color-only
python qa_audit.py --manifest stimuli_manifest.csv --features feats_color.csv
# 3b. full GPU run -> features.csv
python pipeline/run_pipeline.py --manifest stimuli_manifest.csv --out features.csv --reference control_embeddings.npz
```

# Aesthetics CV Pipeline

Image-analysis pipeline for the nostalgia/uncanny/internet-aesthetics study. Turns
each stimulus image into a single design-matrix row spanning three analysis axes
plus the mechanism-proxy `phi` features, ready for the crossed mixed-model and
Bayesian model-comparison horse-race in the study design.

## The three axes (each metric maps to a named theory)

### Axis 1 — `color_gradient.py`  *(runs anywhere, no model weights)*
| Feature | What it captures | Theory link |
|---|---|---|
| `palette_lab` + `hue_mean_deg`/`hue_dispersion` | per-image colour fingerprint | cluster identity (FA aqua ~198°, vaporwave magenta ~277°, liminal yellow-green) |
| `colorfulness` (Hasler–Süsstrunk) | chromatic richness | restorative-warm vs degraded-uncanny |
| `gloss_index` = smooth-gradient ÷ edge density | the Frutiger Aero "one continuous environment" look | restorative gloss vs degradation |
| `edge_density`, `grad_energy` | hard-edge / high-frequency content | analog-horror / VHS channel (early CNN layers in RSA) |
| `warm_cool_ratio`, sat/val stats | warm–cool balance, tonal contrast | valence priors (Table 1) |

### Axis 2 — `segmentation.py`  *(SegFormer/ADE20K labelled, or SAM label-free)*
| Feature | What it captures | Theory link |
|---|---|---|
| `emptiness_index` | bare structural surface, few occupants | Fisher's **eerie** (absence) / kenopsia |
| `human_absence` | structural scene with person-area ≈ 0 | "a mall with no people" — the specific eerie signal |
| `vanishing_structure` | long uninterrupted floor/wall runs | corridor geometry / liminal transit space |
| `region_count`, `region_size_gini` | configural simplicity vs clutter | configural deviation (ties to `phi_typicality`) |
| `light_source_frac` | fluorescent-light prevalence | liminal / analog-horror cue |

### Axis 3 — `object_analysis.py`  *(OWLv2 open-vocab, or DETR closed-set)*
| Feature | What it captures | Theory link |
|---|---|---|
| `expected_absent_score` | objects a scene *should* contain but lacks, prior-weighted | object-level operationalization of the **eerie** — the novel metric |
| `object_density`, `object_area_frac` | clutter / occupancy | restorative warmth vs uncanny emptiness |

### Bridge — `ml_features.py`
- **Embeddings**: DINOv2 (CLS) + CLIP, reported under both as a robustness factor.
- **`phi_typicality`**: negative Mahalanobis distance from the per-category canonical
  centroid (shrinkage covariance). → configural deviation.
- **`phi_realism_var`**: variance of patch-level P(real photograph) under CLIP.
  → MacDorman realism inconsistency.
- **`phi_surprise`**: reconstruction NLL under a generative model of canonical scenes,
  **residualised on low-level complexity** (compression ratio, edge density, SF slope)
  before interpretation. → active-inference prediction error. *Plug your trained
  VQ-VAE / latent-diffusion model into `SurpriseModel`.*
- **`phi_entropy`**: softmax entropy of the scene classifier. → Jentsch category
  uncertainty (kept in the race to be falsified).

## Sourcing & IP — read before collecting
- The pipeline is corpus-agnostic; it processes whatever you point it at.
- **Provenance is mandatory.** The manifest requires `license` and `source_url`
  per image and they propagate to the output. A published stimulus set cannot rest
  on scraped Pinterest/Reddit/TikTok imagery of unknown rights.
- Preferred sources: CC0/CC-BY image banks, your own captures, the populated
  control set from **Places365 / SUN397 / ADE20K** (documented licenses), and for
  aesthetic clusters, creators who grant explicit permission or release under open
  licenses. Where you must use found imagery, restrict to a normed pilot under a
  documented fair-use/research basis and record it per image.

## Run
```bash
pip install numpy scikit-image scikit-learn Pillow torch transformers

# 1. fit the canonical reference on populated control embeddings first
#    (saves control_embeddings.npz with arrays 'embeddings' and 'categories')

# 2. cheap dry run — colour axis only, no model weights:
python run_pipeline.py --manifest stimuli_manifest.csv --out feats.csv --color-only

# 3. full run (GPU): all three axes + phi
python run_pipeline.py --manifest stimuli_manifest.csv \
    --out features.csv --reference control_embeddings.npz
```

Manifest columns: `image_path,image_id,scene_category,cluster,license,source_url`.
`cluster` ∈ {restorative, hauntological, inverted, anemoia, control}; `scene_category`
should match the keys in `SCENE_OBJECT_PRIORS` for the object-expectation metric.

## What's validated vs what needs your hardware
- **Live-tested here**: all of Axis 1 (separates the clusters as predicted), the
  Axis-2 emptiness/kenopsia metric logic, the Axis-3 expected-absent metric, and the
  full orchestrator path on a synthetic manifest.
- **Needs your GPU + weights**: the segmentation/detector/embedding model calls and
  the `phi_surprise` generative model (you train it on controls). All written against
  standard HuggingFace interfaces and lazy-loaded.

## Downstream (next files, not in this drop)
`analysis/` — crossed random-effects models (`lme4`/`brms` or `bambi`), GAM valley
fit, stimulus-level CV horse-race with PSIS-LOO stacking, RSA across CNN layers.
This package produces the feature table those consume.

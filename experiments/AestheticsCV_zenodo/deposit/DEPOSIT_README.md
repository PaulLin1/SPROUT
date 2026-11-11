# Internet Aesthetics Stimulus Set — Deposit

681 openly licensed images + computed features, companion to the manuscript
"The Many Pasts of Internet Aesthetics: Computationally Mapping Nostalgia and Its
Inversions" (Y. J. Lin, Cornell University).

## Contents
- `corpus/images/<cluster>/*.jpg` — the 681 stimulus images
- `stimuli_manifest.csv` — per-image: id, scene_category, cluster, license, source_url
- `features.csv` — the 681x37 design matrix (colour/segmentation/object + phi)
- `control_embeddings.npz` — canonical-reference DINOv2 embeddings (controls only)
- `AESTHETICS_CATALOG.md` — the aesthetics taxonomy
- code: `pipeline/`, `collect/`, `build_reference.py`, `qa_audit.py`,
  `make_figures.py`, `make_examples.py`, `stats_analysis.py`, `stats_addendum*.py`
- `CREDITS.md` — per-image attribution (CC-BY/SA compliance)
- `LICENSES.md` — licence breakdown + compilation licence
- `SHA256SUMS.txt` — integrity hashes

## Per-cluster counts
- restorative: 160
- hauntological: 123
- inverted: 194
- anemoia: 84
- control: 120

## Reproducibility
The corpus is reproducible from the collectors in `collect/` plus the query taxonomy; the manifest records the exact source URL and licence of each kept image. ~3300 candidates were retrieved per round and filtered to these by perceptual-hash de-duplication and a 512px resolution floor; discarded candidates are not retained.

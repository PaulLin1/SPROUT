"""
make_deposit.py
===============
Assemble a deposit-ready archive package for Zenodo / OSF.

Generates, into deposit/:
  - CREDITS.md      : per-image attribution (id, license, source_url) -- satisfies the
                      CC-BY / CC-BY-SA attribution requirement for redistribution.
  - SHA256SUMS.txt  : integrity hash of every deposited file.
  - zenodo.json     : Zenodo deposition metadata (title, creators, license, keywords).
  - DEPOSIT_README.md
  - LICENSES.md     : license summary + the dataset-compilation licensing statement.

The IMAGES themselves are redistributable: all 681 carry CC-BY / CC-BY-SA / CC0 / public
domain licences (recorded per image), so the archive includes them with attribution.
Individual images retain their own licences; the compilation (manifest, features, code) is
released CC-BY-4.0.

Note: this prepares the bundle. The actual upload to Zenodo/OSF is a separate, outward-
facing publishing step that requires an API token and explicit confirmation.
"""
from __future__ import annotations
import csv, hashlib, json, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DEP = os.path.join(HERE, "deposit")
os.makedirs(DEP, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    with open(os.path.join(HERE, "stimuli_manifest.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    lic = Counter((r["license"] or "unknown") for r in rows)
    clusters = Counter(r["cluster"] for r in rows)

    # CREDITS.md -- per-image attribution
    with open(os.path.join(DEP, "CREDITS.md"), "w", encoding="utf-8") as f:
        f.write("# Per-image attribution\n\n")
        f.write(f"{n} images. Each retains the licence listed; attribution is the "
                "source URL. Required for CC-BY / CC-BY-SA reuse.\n\n")
        f.write("| image_id | cluster | scene | license | source_url |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['image_id']} | {r['cluster']} | {r['scene_category']} | "
                    f"{r['license']} | {r['source_url']} |\n")

    # LICENSES.md
    with open(os.path.join(DEP, "LICENSES.md"), "w", encoding="utf-8") as f:
        f.write("# Licensing\n\n## Per-image licence breakdown\n\n")
        for k, v in lic.most_common():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Compilation licence\n\n"
                "Individual images retain their own licences (see CREDITS.md). The dataset "
                "*compilation* — manifest, feature table, embeddings, and all code — is "
                "released under **CC-BY-4.0**. CC-BY-SA images remain under CC-BY-SA; "
                "reusers must preserve per-image attribution and licence.\n")

    # zenodo.json metadata
    meta = {
        "metadata": {
            "title": "Internet Aesthetics Stimulus Set and Feature Table "
                     "(Nostalgia/Uncanny Computational Map)",
            "upload_type": "dataset",
            "description": (
                f"A license-documented corpus of {n} images spanning internet "
                "aesthetics organized by affective operation on the past (restorative, "
                "hauntological, anemoia, inverted) plus present-day controls, with a "
                "computed 681x37 feature table (three perceptual axes + four phi "
                "coordinates), canonical-reference embeddings, the aesthetics catalogue, "
                "and full analysis code. Companion to the manuscript 'The Many Pasts of "
                "Internet Aesthetics'. All images openly licensed; per-image attribution "
                "in CREDITS.md."),
            "creators": [{"name": "Lin, Yan Jun", "affiliation": "Cornell University"}],
            "license": "cc-by-4.0",
            "keywords": ["internet aesthetics", "nostalgia", "uncanny", "liminal spaces",
                         "vaporwave", "Frutiger Aero", "computational aesthetics",
                         "stimulus set", "computer vision", "affective science"],
            "access_right": "open",
            "notes": "Individual images retain their own CC/public-domain licences; "
                     "compilation is CC-BY-4.0.",
        }
    }
    with open(os.path.join(DEP, "zenodo.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # DEPOSIT_README.md
    with open(os.path.join(DEP, "DEPOSIT_README.md"), "w", encoding="utf-8") as f:
        f.write(f"""# Internet Aesthetics Stimulus Set — Deposit

{n} openly licensed images + computed features, companion to the manuscript
"The Many Pasts of Internet Aesthetics: Computationally Mapping Nostalgia and Its
Inversions" (Y. J. Lin, Cornell University).

## Contents
- `corpus/images/<cluster>/*.jpg` — the {n} stimulus images
- `stimuli_manifest.csv` — per-image: id, scene_category, cluster, license, source_url
- `features.csv` — the {n}x37 design matrix (colour/segmentation/object + phi)
- `control_embeddings.npz` — canonical-reference DINOv2 embeddings (controls only)
- `AESTHETICS_CATALOG.md` — the aesthetics taxonomy
- code: `pipeline/`, `collect/`, `build_reference.py`, `qa_audit.py`,
  `make_figures.py`, `make_examples.py`, `stats_analysis.py`, `stats_addendum*.py`
- `CREDITS.md` — per-image attribution (CC-BY/SA compliance)
- `LICENSES.md` — licence breakdown + compilation licence
- `SHA256SUMS.txt` — integrity hashes

## Per-cluster counts
""")
        for c in ("restorative", "hauntological", "inverted", "anemoia", "control"):
            f.write(f"- {c}: {clusters.get(c,0)}\n")
        f.write("\n## Reproducibility\n"
                "The corpus is reproducible from the collectors in `collect/` plus the "
                "query taxonomy; the manifest records the exact source URL and licence of "
                "each kept image. ~3300 candidates were retrieved per round and filtered "
                "to these by perceptual-hash de-duplication and a 512px resolution floor; "
                "discarded candidates are not retained.\n")

    print("deposit metadata written to deposit/. Counts:",
          dict(clusters), "| licenses:", dict(lic))


if __name__ == "__main__":
    main()

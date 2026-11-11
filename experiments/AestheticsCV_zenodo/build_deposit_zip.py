"""
build_deposit_zip.py
====================
Assemble the Zenodo deposit archive from the repo, EXCLUDING anything that cannot be
redistributed or is junk. Hard exclusions:
  - corpus_arena/  : 1,886 are.na images (COPYRIGHT-HELD; never redistributed)
  - corpus_rr/     : re-sourcing probe images (mixed / unknown licences)
  - caches, logs, tex intermediates, the old bundle, the token/state, backups
Includes: the 681 openly-licensed Wikimedia images (corpus/), all derived data (features,
ecosystem, VLM ratings, OASIS, mechanism, manifests), code, manuscript, figures, the
interactive explorer, the catalogue, and the deposit/ metadata. Prints a manifest summary
and a hard leak-check for the copyright folders.
"""
import os, zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "AestheticsCV_zenodo.zip")
EXCLUDE_DIRS = {"corpus_arena", "corpus_rr", "oasis", "__pycache__", ".git", ".ipynb_checkpoints"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
EXCLUDE_EXT = {".log", ".aux", ".out", ".tok", ".zip", ".synctex", ".fls", ".fdb_latexmk"}
EXCLUDE_FILES = {"manuscript_backup_predeco.tex", "deposit_state.json", "deposit_meta.json",
                 "AestheticsCV_deposit_bundle.zip", "_check.js", "_c.js",
                 "manuscript.log", "manuscript.aux", "manuscript.out",
                 # internal / submission working docs -- not for the public archive
                 "cover_letter_BRM.md", "review_package.md", "review_package_natcomm.md",
                 "review_package_human_labels.md"}


def main():
    if os.path.exists(OUT):
        os.remove(OUT)
    count = 0; skipped = 0; bytes_in = 0
    topdirs = {}
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, dns, fns in os.walk(ROOT):
            dns[:] = [d for d in dns if d not in EXCLUDE_DIRS]
            for fn in fns:
                ext = os.path.splitext(fn)[1].lower()
                if ext in EXCLUDE_EXT or fn in EXCLUDE_FILES:
                    skipped += 1; continue
                full = os.path.join(dp, fn)
                arc = os.path.relpath(full, ROOT)
                parts = arc.replace("\\", "/").split("/")
                if parts[0] in EXCLUDE_DIRS:
                    skipped += 1; continue
                # mechanism/ holds edited are.na images (copyright derivatives): keep CSV/code only
                if parts[0] == "mechanism" and ext in IMG_EXT:
                    skipped += 1; continue
                z.write(full, arc)
                count += 1
                bytes_in += os.path.getsize(full)
                top = parts[0] if len(parts) > 1 else "(root)"
                topdirs[top] = topdirs.get(top, 0) + 1

    # hard leak check
    with zipfile.ZipFile(OUT) as z:
        names = z.namelist()
    def _under(n, d):
        nn = n.replace("\\", "/")
        return nn == d or nn.startswith(d + "/")
    leak = [n for n in names
            if any(_under(n, d) for d in ("corpus_arena", "corpus_rr", "oasis"))
            or (_under(n, "mechanism") and os.path.splitext(n)[1].lower() in IMG_EXT)]
    secrets = [n for n in names if n.endswith((".key", ".pem", ".tok")) or "secret" in n.lower()]

    print(f"zip: {OUT}")
    print(f"files: {count}  skipped: {skipped}  size MB: {round(os.path.getsize(OUT)/1e6,1)}")
    print("top-level entries:")
    for k in sorted(topdirs):
        print(f"  {k:24s} {topdirs[k]}")
    print(f"LEAK CHECK -> corpus_arena/corpus_rr in zip: {len(leak)}  (must be 0)")
    print(f"SECRET CHECK -> key/token/secret files in zip: {len(secrets)}  (must be 0)")
    if leak or secrets:
        raise SystemExit("ABORT: copyright or secret material in zip")


if __name__ == "__main__":
    main()

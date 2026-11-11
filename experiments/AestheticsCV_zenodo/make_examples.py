"""
make_examples.py
================
Fig 1 -- representative stimuli per affective mode (the N images nearest each cluster centroid
in standardised feature space; genuine exemplars, not hand-picked). Clean GridSpec montage with
Times mode labels and a colour accent per mode (figstyle palette).
"""
import csv, math, os
import numpy as np
import figstyle
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image

figstyle.use(11)
HERE = os.path.dirname(os.path.abspath(__file__))
CLUSTERS = ["restorative", "hauntological", "inverted", "anemoia", "control"]
NICE = {"restorative": "Restorative", "hauntological": "Hauntological",
        "inverted": "Inverted (uncanny)", "anemoia": "Anemoia", "control": "Control"}
N_PER = 5


def num(x):
    try:
        v = float(x); return v if math.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "features.csv"), encoding="utf-8")))
    feat_cols = [c for c in rows[0] if c.startswith(("col_", "seg_", "obj_", "phi_"))]
    X = np.array([[num(r[c]) for c in feat_cols] for r in rows])
    for j in range(X.shape[1]):
        col = X[:, j]; m = np.nanmean(col); s = np.nanstd(col)
        X[:, j] = np.where(np.isnan(col), 0.0, (col - m) / (s + 1e-9))
    clusters = np.array([r["cluster"] for r in rows]); paths = np.array([r["image_path"] for r in rows])
    picks = {}
    for c in CLUSTERS:
        idx = np.where(clusters == c)[0]
        d = np.linalg.norm(X[idx] - X[idx].mean(0), axis=1)
        picks[c] = idx[np.argsort(d)[:N_PER]]

    fig, axes = plt.subplots(len(CLUSTERS), N_PER, figsize=(N_PER * 1.75, len(CLUSTERS) * 1.62))
    for i, c in enumerate(CLUSTERS):
        for j in range(N_PER):
            ax = axes[i, j]; ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("white"); sp.set_linewidth(1.2)
            if j < len(picks[c]):
                im = Image.open(paths[picks[c][j]]).convert("RGB")
                w, h = im.size; s = min(w, h)
                im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((300, 300), Image.LANCZOS)
                ax.imshow(np.asarray(im))
            if j == 0:
                col = figstyle.MODE_COLORS[c]
                # colour accent bar + mode label
                ax.add_patch(patches.Rectangle((-0.16, 0.0), 0.06, 1.0, transform=ax.transAxes,
                             clip_on=False, facecolor=col, edgecolor="none"))
                ax.text(-0.22, 0.5, NICE[c], rotation=90, va="center", ha="right",
                        transform=ax.transAxes, fontsize=11, color=figstyle.INK)
    fig.subplots_adjust(left=0.10, right=0.995, top=0.99, bottom=0.005, wspace=0.05, hspace=0.05)
    figstyle.save(fig, "fig0_examples")


if __name__ == "__main__":
    main()

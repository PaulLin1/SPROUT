"""
make_figures.py
===============
Generate the manuscript figures from features.csv. Writes PNG+PDF to figures/.

  fig1_pca.pdf       : 2D PCA of the standardised design matrix, coloured by cluster
                       (visual evidence that the feature space separates clusters)
  fig2_eerie.pdf     : per-cluster bar charts of the load-bearing eerie metrics
                       (emptiness, human_absence, phi_typicality, phi_entropy)
  fig3_heatmap.pdf   : per-cluster z-scored feature means heatmap (the full fingerprint)
"""
from __future__ import annotations
import csv, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
figstyle.use(10)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

CLUSTERS = ["restorative", "hauntological", "inverted", "anemoia", "control"]
COLORS = figstyle.MODE_COLORS


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def load():
    with open(os.path.join(HERE, "features.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    feat_cols = [c for c in rows[0] if c.startswith(("col_", "seg_", "obj_", "phi_"))]
    X = np.array([[num(r[c]) for c in feat_cols] for r in rows])
    clusters = np.array([r["cluster"] for r in rows])
    return rows, feat_cols, X, clusters


def standardize(X):
    # column-wise z-score; NaN -> column mean (phi_typicality is NaN for generic scenes)
    Xz = X.copy()
    for j in range(Xz.shape[1]):
        col = Xz[:, j]
        m = np.nanmean(col)
        s = np.nanstd(col)
        col = np.where(np.isnan(col), m, col)
        Xz[:, j] = (col - m) / (s + 1e-9)
    return Xz


def fig_pca(Xz, clusters):
    from sklearn.decomposition import PCA
    p = PCA(n_components=2).fit(Xz)
    Z = p.transform(Xz)
    ev = p.explained_variance_ratio_ * 100
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for c in CLUSTERS:
        m = clusters == c
        ax.scatter(Z[m, 0], Z[m, 1], s=22, alpha=0.7, label=c,
                   color=COLORS[c], edgecolors="none")
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}% var)")
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig1_pca.{ext}"), dpi=200)
    plt.close(fig)
    return ev


def fig_eerie(rows):
    # a spectrum spanning warm-nostalgia (colorfulness) -> off-manifold deviation
    # (surprise, entropy) -> eerie absence (human_absence), not an all-uncanny panel
    metrics = [("col_colorfulness", "colorfulness"),
               ("phi_surprise", r"$\phi_{\mathrm{surprise}}$"),
               ("phi_entropy", r"$\phi_{\mathrm{entropy}}$"),
               ("seg_human_absence", "human absence")]
    means = {c: {} for c in CLUSTERS}
    sems = {c: {} for c in CLUSTERS}
    for key, _ in metrics:
        for c in CLUSTERS:
            vals = np.array([num(r[key]) for r in rows if r["cluster"] == c])
            vals = vals[~np.isnan(vals)]
            means[c][key] = vals.mean() if len(vals) else np.nan
            sems[c][key] = vals.std() / math.sqrt(len(vals)) if len(vals) else 0.0
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.0))
    for ax, (key, label) in zip(axes, metrics):
        xs = range(len(CLUSTERS))
        ys = [means[c][key] for c in CLUSTERS]
        es = [sems[c][key] for c in CLUSTERS]
        ax.bar(xs, ys, yerr=es, color=[COLORS[c] for c in CLUSTERS], capsize=3)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([c[:5] for c in CLUSTERS], rotation=45, ha="right", fontsize=8)
        ax.set_title(label, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig2_eerie.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_heatmap(feat_cols, X, clusters):
    # per-cluster mean of z-scored features
    Xz = standardize(X)
    M = np.array([[np.nanmean(Xz[clusters == c, j]) for c in CLUSTERS]
                  for j in range(len(feat_cols))])  # (features, clusters)
    fig, ax = plt.subplots(figsize=(6.5, 9.5))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(len(CLUSTERS)))
    ax.set_xticklabels(CLUSTERS, rotation=45, ha="right")
    ax.set_yticks(range(len(feat_cols)))
    ax.set_yticklabels([c.replace("col_", "C:").replace("seg_", "S:")
                        .replace("obj_", "O:").replace("phi_", "P:") for c in feat_cols],
                       fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="z-score")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig3_heatmap.{ext}"), dpi=200)
    plt.close(fig)


def main():
    rows, feat_cols, X, clusters = load()
    Xz = standardize(X)
    ev = fig_pca(Xz, clusters)
    fig_eerie(rows)
    fig_heatmap(feat_cols, X, clusters)
    print(f"figures written to {FIGDIR}")
    print(f"PCA explained variance PC1/PC2 = {ev[0]:.1f}% / {ev[1]:.1f}%")


if __name__ == "__main__":
    main()

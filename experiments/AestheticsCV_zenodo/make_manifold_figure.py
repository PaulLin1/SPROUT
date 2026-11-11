"""
make_manifold_figure.py
=======================
Fig 6 -- "the leak of the uncanny." An affective manifold of the are.na images: each image is
embedded (t-SNE) by the frontier panel's NON-eerie affect ratings (valence, arousal, nostalgic,
abandoned, dreamlike), coloured by its HUMAN-assigned mode. The panel's eerie rating, withheld
from the embedding, is overlaid as a smooth shadow field (Nadaraya-Watson kernel smoothing,
masked to the data region). The eerie shadow spreads across the warm (restorative / anemoia)
territories: the over-attribution, made spatial. Everything is algorithmic (no hand-placed
coordinates): t-SNE positions, kernel-smoothed field, mode labels at computed medians via
adjustText.
"""
import csv, os
import numpy as np
import figstyle
from adjustText import adjust_text
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["gemini", "openai", "anthropic"]
AXES = ["nostalgic", "abandoned", "dreamlike", "valence", "arousal"]   # non-eerie -> embedding
MODES = ["restorative", "hauntological", "inverted", "anemoia"]
NICE = {"restorative": "Restorative", "hauntological": "Hauntological",
        "inverted": "Inverted (uncanny)", "anemoia": "Anemoia"}
EERIE_CMAP = LinearSegmentedColormap.from_list("eerie", ["#ffffff", "#cfd0d6", "#6b6c78", "#26262b"])


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load():
    # per model: image_id -> {eerie, axes..., cluster}
    per = {}
    for m in MODELS:
        p = os.path.join(HERE, f"vlm_{m}_affect.csv")
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r.get("ok") != "1":
                continue
            e = _f(r.get("eerie")); ax = [_f(r.get(a)) for a in AXES]
            if e is None or any(v is None for v in ax):
                continue
            per.setdefault(r["image_id"], []).append((r.get("cluster"), e, ax))
    ids, X, eerie, mode = [], [], [], []
    for iid, recs in per.items():
        if len(recs) < 2:                          # need >=2 models for a stable consensus
            continue
        cl = recs[0][0]
        if cl not in MODES:
            continue
        ids.append(iid)
        eerie.append(np.mean([r[1] for r in recs]))
        X.append(np.mean([r[2] for r in recs], axis=0))
        mode.append(cl)
    return np.array(X), np.array(eerie), np.array(mode)


def smooth_field(xy, vals, gx, gy, h):
    """Nadaraya-Watson: value + density on a grid; returns (field, density)."""
    GX, GY = np.meshgrid(gx, gy)
    flat = np.column_stack([GX.ravel(), GY.ravel()])
    field = np.empty(len(flat)); dens = np.empty(len(flat))
    inv = 1.0 / (2 * h * h)
    for i, p in enumerate(flat):
        d2 = ((xy - p) ** 2).sum(1)
        w = np.exp(-d2 * inv)
        sw = w.sum()
        dens[i] = sw
        field[i] = (w @ vals) / sw if sw > 1e-9 else np.nan
    return field.reshape(GX.shape), dens.reshape(GX.shape)


def main():
    figstyle.use(12)
    X, eerie, mode = load()
    print(f"images: {len(X)}  |  eerie range {eerie.min():.2f}-{eerie.max():.2f}")

    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    xy = TSNE(n_components=2, perplexity=40, init="pca", learning_rate="auto",
              random_state=7).fit_transform(Xz)
    xy = (xy - xy.mean(0)) / xy.std(0)             # normalise the plane

    # smooth eerie field on a fine grid, masked to where images actually are
    pad = 0.6
    gx = np.linspace(xy[:, 0].min() - pad, xy[:, 0].max() + pad, 320)
    gy = np.linspace(xy[:, 1].min() - pad, xy[:, 1].max() + pad, 320)
    field, dens = smooth_field(xy, eerie, gx, gy, h=0.16)
    field = np.ma.masked_where(dens < 0.06 * dens.max(), field)

    fig, ax = plt.subplots(figsize=(9.6, 8.4))
    lo, hi = np.percentile(eerie, 5), np.percentile(eerie, 95)
    cf = ax.contourf(gx, gy, field, levels=np.linspace(lo, hi, 12), cmap=EERIE_CMAP,
                     alpha=0.55, zorder=1, extend="both")
    ax.contour(gx, gy, field, levels=np.linspace(lo, hi, 12), colors="#7a7b85",
               linewidths=0.4, alpha=0.5, zorder=2)

    for md in MODES:
        sel = mode == md
        ax.scatter(xy[sel, 0], xy[sel, 1], s=16, c=figstyle.MODE_COLORS[md],
                   edgecolors="white", linewidths=0.25, alpha=0.85, zorder=3, label=NICE[md])

    texts = []
    for md in MODES:                               # label each territory at its median (computed)
        sel = mode == md
        mx, my = np.median(xy[sel, 0]), np.median(xy[sel, 1])
        texts.append(ax.text(mx, my, NICE[md], fontsize=14, fontweight="bold", zorder=6,
                     color=figstyle.INK,
                     path_effects=[pe.withStroke(linewidth=3.2, foreground="white")]))
    adjust_text(texts, ax=ax, expand=(1.3, 1.5))

    cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("model eerie rating (panel consensus, 1-5)")
    cb.outline.set_visible(False)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3, borderpad=0.2,
              markerscale=1.4, fontsize=12)
    ax.set_axis_off()
    fig.tight_layout()
    figstyle.save(fig, "manifold_uncanny")


if __name__ == "__main__":
    main()

"""
make_oasis_figure.py
====================
Fig 5 (NEW) -- the bias is bounded: on OASIS (images with human valence norms) all three models
track human valence (r=0.84-0.87) and reserve eerie for genuinely negative scenes. Small multiples,
one per model: VLM valence vs human-norm valence, points coloured by the model's eerie rating.
"""
import csv, os
import numpy as np
import figstyle
import matplotlib.pyplot as plt

figstyle.use(11)
ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = [("gemini", "vlm_gemini_oasis.csv", "Gemini 2.5 Flash"),
          ("openai", "vlm_openai_oasis.csv", "GPT-4o"),
          ("anthropic", "vlm_anthropic_oasis.csv", "Claude Sonnet 4.6")]

norms, idkey = {}, None
for r in csv.DictReader(open(os.path.join(ROOT, "oasis", "OASIS.csv"), encoding="utf-8")):
    if idkey is None:
        idkey = [k for k in r if k.strip() == ""][0]
    norms[r[idkey]] = float(r["Valence_mean"])


def load(f):
    d = {}
    for r in csv.DictReader(open(os.path.join(ROOT, f), encoding="utf-8")):
        if r.get("ok") == "1":
            try:
                d[r["image_id"]] = (float(r["valence"]), float(r["eerie"]))
            except (ValueError, KeyError):
                pass
    return d


fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharex=True, sharey=True)
sc = None
for ax, (mk, f, name) in zip(axes, MODELS):
    d = load(f)
    ids = [i for i in d if i in norms]
    hv = np.array([norms[i] for i in ids])           # human valence 1-7
    vv = np.array([d[i][0] for i in ids])            # VLM valence 1-5
    ee = np.array([d[i][1] for i in ids])            # VLM eerie 1-5
    r = float(np.corrcoef(hv, vv)[0, 1])
    sc = ax.scatter(hv, vv, c=ee, cmap=figstyle.SEQ, vmin=1, vmax=5, s=13, alpha=0.7,
                    edgecolors="none")
    b1, b0 = np.polyfit(hv, vv, 1)
    xx = np.array([hv.min(), hv.max()])
    ax.plot(xx, b1 * xx + b0, color=figstyle.ACCENT, lw=1.8, zorder=3)
    ax.text(0.05, 0.93, f"{name}\n$r={r:.2f}$  ($n={len(ids)}$)", transform=ax.transAxes,
            va="top", ha="left", fontsize=10.5)
    ax.set_xlabel("human valence norm (OASIS, 1–7)")
    ax.set_box_aspect(1)
axes[0].set_ylabel("VLM valence rating (1–5)")
cb = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02); cb.outline.set_visible(False)
cb.set_label("VLM eerie rating")
figstyle.save(fig, "oasis_calibration")

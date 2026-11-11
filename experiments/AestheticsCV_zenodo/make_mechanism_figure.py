"""
make_mechanism_figure.py
========================
Fig 4 (NEW) -- the over-attribution is interpretive, not perceptual. Forest plot of the paired
change in VLM 'eerie' for each manipulation, per model: editing the IMAGE (denoise/upscale/recolor)
leaves eerie unchanged (clustered at 0); reframing the QUESTION (ground / debias) lowers it
(left of 0). Replicated across Gemini, GPT-4o and Claude.
"""
import csv, os
from collections import defaultdict
import numpy as np
import figstyle
import matplotlib.pyplot as plt

figstyle.use(11)
HERE = os.path.dirname(os.path.abspath(__file__))
MECH = os.path.join(HERE, "mechanism")
MODELS = [("gemini", "Gemini 2.5 Flash", "#3b5b7a"),
          ("openai", "GPT-4o", "#b0822f"),
          ("anthropic", "Claude Sonnet 4.6", "#5f8a7a")]
# (file kind, baseline, condition, display label, group)
ROWS = [
    ("intervention", "control", "T_quality", "edit image: quality", "image"),
    ("intervention", "control", "T_tone", "edit image: tone", "image"),
    ("intervention", "control", "T_natural", "edit image: both", "image"),
    ("prompt", "baseline", "ground", "reframe: rate the scene", "prompt"),
    ("prompt", "baseline", "debias", "reframe: stylization≠threat", "prompt"),
]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def boot(d, n=5000, seed=0):
    rng = np.random.RandomState(seed); d = np.asarray(d, float)
    bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def deltas(kind, baseline, cond, model):
    p = os.path.join(MECH, f"mechanism_{kind}_{model}.csv")
    if not os.path.exists(p):
        return None
    by = defaultdict(dict)
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r.get("ok") == "1":
            e = _f(r.get("eerie"))
            if e is not None:
                by[r["image_id"]][r["condition"]] = e
    d = [by[i][cond] - by[i][baseline] for i in by if cond in by[i] and baseline in by[i]]
    return boot(d) if len(d) >= 20 else None


fig, ax = plt.subplots(figsize=(8.2, 5.2))
yticks, ylabels = [], []
y = 0
for kind, base, cond, label, group in ROWS:
    yticks.append(y); ylabels.append(label)
    for k, (mk, mname, mc) in enumerate(MODELS):
        res = deltas(kind, base, cond, mk)
        if res is None:
            continue
        m, lo, hi = res
        yy = y + (k - 1) * 0.22
        ax.plot([lo, hi], [yy, yy], color=mc, lw=1.6, solid_capstyle="round", zorder=2)
        ax.scatter([m], [yy], s=34, color=mc, edgecolors="white", linewidths=0.6, zorder=3,
                   label=mname if y == 0 else None)
    y += 1

ax.axvline(0, color=figstyle.INK, lw=0.9, zorder=1)
ax.axhline(2.5, color=figstyle.LIGHT, lw=0.8, zorder=0)     # divider: image vs prompt groups
ax.set_yticks(yticks); ax.set_yticklabels(ylabels)
ax.invert_yaxis()
ax.set_xlabel("change in VLM eerie rating ($\\Delta$, paired vs. baseline; 95% CI)")
ax.text(0.40, 1.0, "edit the image", va="center", ha="left", fontsize=9.5,
        color=figstyle.NEUTRAL, style="italic")
ax.text(0.40, 3.5, "reframe the question", va="center", ha="left", fontsize=9.5,
        color=figstyle.NEUTRAL, style="italic")
ax.annotate("$\\leftarrow$ reduces unease", xy=(-0.5, 4.5), fontsize=9,
            color=figstyle.NEUTRAL, ha="center")
ax.legend(title="model", loc="upper left", frameon=False)
ax.set_xlim(-0.95, 0.95)
fig.tight_layout()
figstyle.save(fig, "mechanism_forest")

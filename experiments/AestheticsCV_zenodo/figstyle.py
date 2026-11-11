"""
figstyle.py
===========
Shared, reliable figure style for the whole manuscript: registers Times New Roman (so
matplotlib does not silently fall back to DejaVu), sets a consistent Okabe-Ito colourblind-safe
palette, clean spines, and vector output. Every figure script calls figstyle.use().

No element is placed by hand-computed coordinates anywhere downstream; figures are produced by
algorithms (force-directed layout, grids, errorbars, scatter, imshow) + automatic label
de-confliction. This module just fixes the look.
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Restrained, desaturated editorial palette (colourblind-safe hues, muted) for the four modes.
# Used sparingly: most elements are ink/grey; colour only where it encodes data.
MODE_COLORS = {
    "restorative":   "#C7A35A",   # soft ochre (warmth)
    "hauntological": "#5E7A9B",   # slate blue (lost futures)
    "anemoia":       "#9E7BA6",   # muted plum (refined longing)
    "inverted":      "#6FA292",   # sage/teal (the uncanny)
    "control":       "#AEB4BA",   # cool grey
}
INK = "#1f1f1f"          # near-black for text/axes
NEUTRAL = "#9aa0a6"      # mid grey for de-emphasised elements
LIGHT = "#d7dadd"        # light grey fills
ACCENT = "#B4451F"       # single muted-vermillion accent for the one thing to highlight
SEQ = "Blues"            # single-hue sequential colormap (clean, not rainbow)


def _register_times():
    reg = []
    for fn in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
        p = os.path.join("C:/Windows/Fonts", fn)
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p); reg.append(fn)
            except Exception:
                pass
    return reg


def use(base=11):
    _register_times()
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",          # Times-compatible math
        "font.size": base + 1,
        "axes.titlesize": base + 2,
        "axes.labelsize": base + 1,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.fontsize": base,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "legend.frameon": False,
    })
    fam = plt.rcParams["font.family"]
    resolved = font_manager.FontProperties(family=fam).get_name()
    print(f"[figstyle] font.family={fam} -> resolved '{resolved}'")
    return resolved


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"))
    print(f"  -> figures/{name}.pdf/.png")

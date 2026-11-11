"""
make_vlm_flow.py
================
Fig 3 (new) -- the asymmetric leak into the uncanny. (A) an alluvial/Sankey flow from
human-assigned mode (left) to VLM consensus mode (right): warm genres each pour a third of
their mass into "inverted", while inverted itself stays put. (B) per-mode mode-match recall.
All ribbon geometry is computed from the confusion counts (no hand-placed coordinates).
The confusion heatmap moves to the supplement.
"""
import csv, os
from collections import Counter, defaultdict
import numpy as np
import figstyle
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle

figstyle.use(11)
ROOT = os.path.dirname(os.path.abspath(__file__))
CL = ["restorative", "hauntological", "anemoia", "inverted"]      # top -> bottom (warm -> uncanny)
NICE = {"restorative": "Restorative", "hauntological": "Hauntological",
        "anemoia": "Anemoia", "inverted": "Inverted (uncanny)"}
JUDGES = ["vlm_gemini_affect.csv", "vlm_openai_affect.csv", "vlm_anthropic_affect.csv"]


def confusion():
    modes = defaultdict(list); human = {}
    for jf in JUDGES:
        p = os.path.join(ROOT, jf)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r.get("ok") == "1" and r.get("source") == "arena" and r.get("cluster") in CL and r.get("mode") in CL:
                modes[r["image_id"]].append(r["mode"]); human[r["image_id"]] = r["cluster"]
    idx = {c: i for i, c in enumerate(CL)}
    C = np.zeros((len(CL), len(CL)))
    for iid, ms in modes.items():
        vote = Counter(ms).most_common(1)[0][0]
        C[idx[human[iid]], idx[vote]] += 1
    return C


def ribbon(ax, x0, x1, ya0, yb0, ya1, yb1, color, alpha):
    xm = (x0 + x1) / 2.0
    verts = [(x0, ya0), (xm, ya0), (xm, ya1), (x1, ya1),
             (x1, yb1), (xm, yb1), (xm, yb0), (x0, yb0), (x0, ya0)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none",
                           alpha=alpha, zorder=2))


def main():
    C = confusion()
    N = C.sum()
    Lt = C.sum(1); Rt = C.sum(0)
    GAP = 0.035
    avail = 1.0 - (len(CL) - 1) * GAP

    # node vertical extents (top=1 downward)
    def extents(tot):
        ys = {}; y = 1.0
        for i in range(len(CL)):
            h = tot[i] / N * avail
            ys[i] = (y, y - h); y = y - h - GAP
        return ys
    L = extents(Lt); R = extents(Rt)
    xL0, xL1, xR0, xR1 = 0.0, 0.045, 0.955, 1.0

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 5.6),
                                   gridspec_kw={"width_ratios": [2.5, 1], "wspace": 0.32})

    # node bars + labels
    for i, c in enumerate(CL):
        ya, yb = L[i]
        axA.add_patch(Rectangle((xL0, yb), xL1 - xL0, ya - yb, facecolor=figstyle.MODE_COLORS[c],
                      edgecolor="white", linewidth=0.5, zorder=3))
        axA.text(xL0 - 0.012, (ya + yb) / 2, f"{NICE[c]}", ha="right", va="center",
                 fontsize=12.5, fontweight="bold", color=figstyle.INK, zorder=4)
        ya, yb = R[i]
        axA.add_patch(Rectangle((xR0, yb), xR1 - xR0, ya - yb, facecolor=figstyle.MODE_COLORS[c],
                      edgecolor="white", linewidth=0.5, zorder=3))
        axA.text(xR1 + 0.012, (ya + yb) / 2, f"{NICE[c]}", ha="left", va="center",
                 fontsize=12.5, fontweight="bold", color=figstyle.INK, zorder=4)

    # ribbons: left node i sub-stacked by j; right node j sub-stacked by i
    lcur = {i: L[i][0] for i in range(len(CL))}
    rcur = {j: R[j][0] for j in range(len(CL))}
    order = sorted(((i, j) for i in range(len(CL)) for j in range(len(CL))),
                   key=lambda t: (t[0], t[1]))
    # pre-advance right cursors in i-order for each j so incoming stacks by source
    for (i, j) in order:
        c = C[i, j]
        if c <= 0:
            continue
        hL = c / N * avail; hR = c / N * avail
        ya0, yb0 = lcur[i], lcur[i] - hL
        ya1, yb1 = rcur[j], rcur[j] - hR
        emph = 0.62 if (j == CL.index("inverted") and i != j) else 0.34   # highlight the leak
        ribbon(axA, xL1, xR0, ya0, yb0, ya1, yb1, figstyle.MODE_COLORS[CL[i]], emph)
        lcur[i] = yb0; rcur[j] = yb1

    axA.text(xL1, 1.06, "human curation", ha="center", va="bottom", fontsize=12,
             color=figstyle.NEUTRAL, style="italic")
    axA.text(xR0, 1.06, "model reading", ha="center", va="bottom", fontsize=12,
             color=figstyle.NEUTRAL, style="italic")
    axA.set_xlim(-0.26, 1.26); axA.set_ylim(-0.02, 1.12); axA.set_axis_off()
    axA.set_title("a", loc="left", fontweight="bold")

    # panel B: recall
    rown = C / C.sum(1, keepdims=True).clip(min=1)
    rec = [rown[i, i] for i in range(len(CL))]
    colors = [figstyle.ACCENT if c == "inverted" else figstyle.NEUTRAL for c in CL]
    axB.bar(range(len(CL)), rec, color=colors, width=0.68)
    axB.axhline(0.25, ls=(0, (4, 3)), c=figstyle.NEUTRAL, lw=0.9)
    axB.text(len(CL) - 0.5, 0.27, "chance", ha="right", va="bottom", fontsize=9, color=figstyle.NEUTRAL)
    axB.set_xticks(range(len(CL)))
    axB.set_xticklabels([NICE[c] for c in CL], fontsize=9, rotation=20, ha="right")
    axB.set_ylabel("VLM mode-match recall"); axB.set_ylim(0, 1)
    for i, v in enumerate(rec):
        axB.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10, color=figstyle.INK)
    axB.spines[["top", "right"]].set_visible(False)
    axB.set_title("b", loc="left", fontweight="bold")

    figstyle.save(fig, "vlm_flow")
    inv = CL.index("inverted")
    for i, c in enumerate(CL):
        if i != inv:
            print(f"{c:14s} -> inverted: {rown[i, inv]:.2f}")
    print("inverted -> inverted:", round(rown[inv, inv], 2))


if __name__ == "__main__":
    main()

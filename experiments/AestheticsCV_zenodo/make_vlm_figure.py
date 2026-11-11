"""
make_vlm_figure.py
==================
Fig 3 -- frontier VLMs over-attribute unease. (A) human-assigned mode vs VLM consensus mode
(row-normalised confusion); (B) per-mode mode-match recall. Algorithmic (imshow + bars),
styled via figstyle (Times, Okabe-Ito).
"""
import csv, os
from collections import Counter, defaultdict
import numpy as np
import figstyle
import matplotlib.pyplot as plt

figstyle.use(11)
ROOT = os.path.dirname(os.path.abspath(__file__))
CL = ["restorative", "hauntological", "anemoia", "inverted"]
NICE = {"restorative": "restorative", "hauntological": "hauntological",
        "anemoia": "anemoia", "inverted": "inverted\n(uncanny)"}
JUDGES = ["vlm_gemini_affect.csv", "vlm_openai_affect.csv", "vlm_anthropic_affect.csv"]

modes = defaultdict(list); human = {}
for jf in JUDGES:
    p = os.path.join(ROOT, jf)
    if not os.path.exists(p):
        continue
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r.get("ok") == "1" and r.get("source") == "arena" and r.get("cluster") in CL and r.get("mode") in CL:
            modes[r["image_id"]].append(r["mode"]); human[r["image_id"]] = r["cluster"]

conf = np.zeros((len(CL), len(CL))); idx = {c: i for i, c in enumerate(CL)}
for iid, ms in modes.items():
    vote = Counter(ms).most_common(1)[0][0]
    conf[idx[human[iid]], idx[vote]] += 1
rown = conf / conf.sum(1, keepdims=True).clip(min=1)
n = len(modes)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"wspace": 0.35})

im = axA.imshow(rown, cmap=figstyle.SEQ, vmin=0, vmax=1, aspect="equal")
axA.set_xticks(range(len(CL))); axA.set_xticklabels([c[:5] for c in CL])
axA.set_yticks(range(len(CL))); axA.set_yticklabels([NICE[c].replace("\n", " ") for c in CL])
axA.set_xlabel("VLM consensus mode"); axA.set_ylabel("human-assigned mode")
for i in range(len(CL)):
    for j in range(len(CL)):
        axA.text(j, i, f"{rown[i,j]:.2f}", ha="center", va="center", fontsize=10,
                 color="white" if rown[i, j] > 0.55 else figstyle.INK)
for sp in axA.spines.values():
    sp.set_visible(False)
axA.set_title("a", loc="left", fontweight="bold")
cb = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04); cb.outline.set_visible(False)
cb.set_label("row fraction")

rec = [rown[idx[c], idx[c]] for c in CL]
colors = [figstyle.ACCENT if c == "inverted" else figstyle.NEUTRAL for c in CL]
axB.bar(range(len(CL)), rec, color=colors, width=0.68)
axB.axhline(0.25, ls=(0, (4, 3)), c=figstyle.NEUTRAL, lw=0.9)
axB.text(len(CL) - 0.5, 0.27, "chance", ha="right", va="bottom", fontsize=9, color=figstyle.NEUTRAL)
axB.set_xticks(range(len(CL))); axB.set_xticklabels([NICE[c] for c in CL])
axB.set_ylabel("VLM mode-match recall"); axB.set_ylim(0, 1)
for i, v in enumerate(rec):
    axB.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10, color=figstyle.INK)
axB.set_title("b", loc="left", fontweight="bold")

figstyle.save(fig, "vlm_where_it_breaks")
print("per-mode recall:", {c: round(r, 2) for c, r in zip(CL, rec)})

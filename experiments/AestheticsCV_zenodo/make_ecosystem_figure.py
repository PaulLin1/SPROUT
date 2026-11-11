"""
make_ecosystem_figure.py
========================
Fig 2 -- the internet-aesthetics curation network, drawn by a real force-directed algorithm
(ForceAtlas2). ALL 1,115 aesthetics in the giant component are plotted; nodes are coloured by
their community's dominant community-authored affect and sized by curatorial degree. Up to
N_LABEL_TOTAL aesthetics are labelled: the 26 focal aesthetics (the R/H/I/A taxonomy, bold)
plus the leading hubs, with font size scaled by degree. Large vector canvas (zoomable).
"""
import csv, os, re, sys
import numpy as np
import networkx as nx
import figstyle
from adjustText import adjust_text
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "collect"))
from aesthetic_map import AESTHETICS          # focal seed aesthetics -> (cluster, query)

figstyle.use(12)
plt.rcParams["savefig.dpi"] = 150             # keep the (very large) PNG preview manageable
ROOT = os.path.dirname(os.path.abspath(__file__))
AFF = ["unease", "nostalgia", "comfort", "refinement"]
AFF_COLOR = {"unease": "#6FA292", "nostalgia": "#5E7A9B",
             "comfort": "#C7A35A", "refinement": "#9E7BA6"}
AFF_LABEL = {"unease": "uncanny / unease", "nostalgia": "nostalgia",
             "comfort": "comfort / warmth", "refinement": "refinement / longing"}
N_LABEL_TOTAL = 100                           # total aesthetics labelled (focal seeds + hubs)
# obscure / disliked aesthetics to keep OUT of the labels (normalised names)
EXCLUDE = {"animecore", "mcbling", "groovival", "webcore", "urbancore", "vectordelia",
           "memphisdesign", "memphislite", "wackypomo", "silicondreams", "dollette", "chav"}


def norm(s): return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def main():
    nodes = list(csv.DictReader(open(os.path.join(ROOT, "wiki_ecosystem_nodes.csv"), encoding="utf-8")))
    edges = list(csv.DictReader(open(os.path.join(ROOT, "wiki_ecosystem_edges.csv"), encoding="utf-8")))
    info = {r["aesthetic"]: r for r in nodes}
    nbn = {norm(r["aesthetic"]): r["aesthetic"] for r in nodes}

    G = nx.Graph(); G.add_nodes_from(info)
    for e in edges:
        a = e["aesthetic"]; b = nbn.get(norm(e["related"]))
        if b and b != a and a in info:
            G.add_edge(a, b)
    G.remove_nodes_from(list(nx.isolates(G)))
    giant = max(nx.connected_components(G), key=len)
    G = G.subgraph(giant).copy()

    # community -> dominant affect
    comm_nodes = {}
    for n in G:
        comm_nodes.setdefault(int(info[n]["community"]), []).append(n)
    comm_aff = {}
    for c, ns in comm_nodes.items():
        agg = {a: np.mean([float(info[n].get(f"p_{a}", 0) or 0) for n in ns]) for a in AFF}
        comm_aff[c] = max(agg, key=agg.get)
    node_color = [AFF_COLOR[comm_aff[int(info[n]["community"])]] for n in G]
    deg = dict(G.degree())
    sizes = [4 + 2.0 * np.sqrt(deg[n]) for n in G]

    # ---- labels: focal seeds (matched) first, then leading hubs, up to N_LABEL_TOTAL ----
    nodes_norm = {norm(n): n for n in G}

    def match(seed):
        s = norm(seed)
        if s in nodes_norm:
            return nodes_norm[s]
        for k, v in nodes_norm.items():               # tolerant: "Y2K" -> "Y2K Futurism"
            if len(s) >= 5 and (k.startswith(s) or s.startswith(k)):
                return v
        return None

    seed_nodes = []
    for name in AESTHETICS:
        m = match(name)
        if m and m not in seed_nodes:
            seed_nodes.append(m)
    seed_set = set(seed_nodes)
    ranked = [n for n in sorted(G, key=lambda n: deg[n], reverse=True) if norm(n) not in EXCLUDE]
    label_nodes = list(dict.fromkeys(seed_nodes + ranked))[:N_LABEL_TOTAL]

    print(f"layout (ForceAtlas2) on {G.number_of_nodes()} nodes / {G.number_of_edges()} edges ...")
    print(f"  focal seeds matched: {len(seed_nodes)}/{len(AESTHETICS)}; total labelled: {len(label_nodes)}")
    pos = nx.forceatlas2_layout(G, max_iter=800, scaling_ratio=4.5, gravity=0.55,
                                linlog=True, seed=7)

    # font size scaled by degree (hubs large, tail small); focal seeds at least 11pt, bold
    ld = np.array([deg[n] for n in label_nodes], float)
    dmin, dmax = float(ld.min()), float(ld.max())

    def fsize(n):
        t = (deg[n] - dmin) / (dmax - dmin) if dmax > dmin else 0.5
        return 9.5 + 11.0 * np.sqrt(t)

    fig, ax = plt.subplots(figsize=(16, 11.6))           # landscape, fills a rotated full page
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#9aa0a6", width=0.2, alpha=0.10)
    xs = [pos[n][0] for n in G]; ys = [pos[n][1] for n in G]
    ax.scatter(xs, ys, s=sizes, c=node_color, edgecolors="white", linewidths=0.25, zorder=2)

    texts = []
    for n in label_nodes:
        seed = n in seed_set
        fs = 16.0 if seed else fsize(n)
        texts.append(ax.text(pos[n][0], pos[n][1], n, ha="center", va="center",
                     fontsize=fs, fontweight="bold" if seed else "normal",
                     color=figstyle.INK if seed else "#4a4a4a",
                     zorder=200 + int(fs) if seed else 50 + int(fs),
                     path_effects=[pe.withStroke(linewidth=3.4 if seed else 2.6, foreground="white")]))
    adjust_text(texts, ax=ax, expand=(1.1, 1.25), force_text=(0.3, 0.4),
                arrowprops=dict(arrowstyle="-", color="#b0b0b0", lw=0.4))

    handles = [plt.Line2D([0], [0], marker="o", ls="", markersize=13, markerfacecolor=AFF_COLOR[a],
               markeredgecolor="white", label=AFF_LABEL[a]) for a in AFF]
    ax.legend(handles=handles, title="community dominant affect", loc="upper left",
              frameon=False, handletextpad=0.3, borderpad=0.2, fontsize=15, title_fontsize=15)
    ax.set_axis_off()
    fig.tight_layout()
    figstyle.save(fig, "ecosystem_network")


if __name__ == "__main__":
    main()

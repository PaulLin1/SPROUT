"""
analyze_ecosystem.py
====================
The novel core: does the AFFECTIVE-TEMPORAL organization of internet visual culture EMERGE
from collective human curation, with no researcher labels?

We build the similarity network of the whole Aesthetics-Wiki ecosystem (nodes = aesthetics,
edges = community-authored `related_aesthetics` links), detect communities with Louvain
(unsupervised), and ask:

  1. STRUCTURE: is the network modular (communities far above a degree-preserving null)?
  2. AFFECT EMERGES: are the emergent communities affectively coherent -- i.e. do aesthetics
     in the same community share `key_values` affect terms more than chance?
  3. TAXONOMY RECOVERED: do our four a-priori mechanism classes (R/H/I/A), assigned to 26
     seed aesthetics on theoretical grounds, concentrate within the emergent communities
     (NMI vs a label-permutation null)?
  4. TIME: do communities separate along decade_of_origin (the temporal axis)?

Inputs:  wiki_ecosystem.csv, wiki_ecosystem_edges.csv  (+ collect/aesthetic_map for seeds)
Output:  ecosystem_report.txt, wiki_ecosystem_nodes.csv
"""
from __future__ import annotations
import csv, os, re, sys, random, math
from collections import defaultdict, Counter
import networkx as nx

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "collect"))
from aesthetic_map import AESTHETICS               # seed name -> (cluster, query)
from analyze_human_labels import LEX, score_text   # affect lexicon


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def first_year(s):
    m = re.search(r"(1[89]\d0|20[0-4]0|19\d0|20\d0|\b(18|19|20)\d{2}\b)", s or "")
    if not m:
        return None
    y = re.search(r"\d{4}", m.group(0))
    return int(y.group(0)) if y else None


def main():
    nodes = list(csv.DictReader(open(os.path.join(ROOT, "wiki_ecosystem.csv"), encoding="utf-8")))
    edges = list(csv.DictReader(open(os.path.join(ROOT, "wiki_ecosystem_edges.csv"), encoding="utf-8")))
    name_by_norm = {norm(r["aesthetic"]): r["aesthetic"] for r in nodes}
    info = {r["aesthetic"]: r for r in nodes}
    R = []
    def P(s): print(s); R.append(s)

    # ---- build undirected graph on aesthetic nodes ----
    G = nx.Graph()
    G.add_nodes_from(info)
    kept = 0
    for e in edges:
        a = e["aesthetic"]; b = name_by_norm.get(norm(e["related"]))
        if b and b != a:
            G.add_edge(a, b); kept += 1
    G.remove_nodes_from(list(nx.isolates(G)))
    P("=" * 70)
    P("ECOSYSTEM NETWORK (collective human curation of internet aesthetics)")
    P("=" * 70)
    P(f"nodes (aesthetics): {G.number_of_nodes()}   edges (related-links): {G.number_of_edges()}")
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    P(f"connected components: {len(comps)}   giant component: {len(comps[0])} "
      f"({100*len(comps[0])/G.number_of_nodes():.0f}%)")

    # ---- 1. communities (Louvain, deterministic seed) ----
    comms = nx.community.louvain_communities(G, seed=20260613, resolution=1.0)
    comms = sorted(comms, key=len, reverse=True)
    node2comm = {n: i for i, c in enumerate(comms) for n in c}
    Q = nx.community.modularity(G, comms)
    # degree-preserving null modularity
    rng = random.Random(7)
    null_Q = []
    for _ in range(50):
        try:
            Gr = nx.double_edge_swap(G.copy(), nswap=2*G.number_of_edges(), max_tries=20*G.number_of_edges())
        except Exception:
            Gr = G
        null_Q.append(nx.community.modularity(Gr, comms))
    muQ = sum(null_Q)/len(null_Q)
    P(f"\n[1] STRUCTURE: {len(comms)} communities, modularity Q = {Q:.3f} "
      f"(rewired null {muQ:.3f}) -> {'modular' if Q > muQ + 0.1 else 'weak'}")
    P(f"    community sizes (top 10): {[len(c) for c in comms[:10]]}")

    # ---- 2. affect emerges: per-community lexicon profile ----
    P(f"\n[2] AFFECT EMERGES: dominant affect dimension per large community")
    P(f"    {'comm':>4} {'size':>5}  " + "".join(d[:6].rjust(8) for d in LEX) + "   exemplars")
    comm_profile = {}
    for i, c in enumerate(comms):
        if len(c) < 8:
            continue
        agg = Counter()
        for n in c:
            for d, v in score_text(info[n].get("key_values", "")).items():
                agg[d] += v
        tot = sum(agg.values()) or 1
        prof = {d: agg[d]/tot for d in LEX}
        comm_profile[i] = prof
        dom = max(LEX, key=lambda d: prof[d])
        ex = sorted(c, key=lambda n: G.degree(n), reverse=True)[:4]
        P(f"    {i:>4} {len(c):>5}  " + "".join(f"{prof[d]:.2f}".rjust(8) for d in LEX)
          + f"   {dom:>10} | " + ", ".join(ex))

    # community affect homogeneity vs permutation null (assign each node its argmax dim)
    def node_dim(n):
        sc = score_text(info[n].get("key_values", ""))
        return max(sc, key=sc.get) if sum(sc.values()) else None
    labeled = [n for n in G if node_dim(n)]
    obs = sum(1 for u, v in G.edges() if u in labeled and v in labeled
              and node_dim(u) == node_dim(v))
    base = [(u, v) for u, v in G.edges() if u in labeled and v in labeled]
    if base:
        dims = {n: node_dim(n) for n in labeled}
        keys = list(dims); vals = [dims[k] for k in keys]
        null = []
        for _ in range(2000):
            random.shuffle(vals)
            m = dict(zip(keys, vals))
            null.append(sum(1 for u, v in base if m[u] == m[v]))
        p = (sum(1 for x in null if x >= obs) + 1)/(len(null)+1)
        P(f"    affect-homophily: {obs}/{len(base)} edges link same-affect aesthetics; "
          f"null mean {sum(null)/len(null):.0f}; p={p:.4f}")

    # ---- 3. a-priori taxonomy recovered among seeds ----
    seeds = {}
    for name, (cl, _) in AESTHETICS.items():
        nn = name_by_norm.get(norm(name))
        if nn and nn in node2comm:
            seeds[nn] = cl
    P(f"\n[3] TAXONOMY RECOVERED: {len(seeds)} seed aesthetics located in the network")
    if len(seeds) >= 8:
        from sklearn.metrics import normalized_mutual_info_score as nmi
        y_true = [seeds[n] for n in seeds]
        y_comm = [node2comm[n] for n in seeds]
        obs_nmi = nmi(y_true, y_comm)
        rng2 = random.Random(11)
        perm = []
        for _ in range(2000):
            yt = y_true[:]; rng2.shuffle(yt); perm.append(nmi(yt, y_comm))
        pn = (sum(1 for x in perm if x >= obs_nmi)+1)/(len(perm)+1)
        P(f"    NMI(a-priori cluster, emergent community) = {obs_nmi:.3f}  "
          f"(perm null {sum(perm)/len(perm):.3f}, p={pn:.4f})")
        # cross-tab
        ct = defaultdict(Counter)
        for n in seeds:
            ct[seeds[n]][node2comm[n]] += 1
        for cl in ["restorative", "hauntological", "inverted", "anemoia"]:
            P(f"      {cl:14s}-> communities { dict(ct[cl]) }")

    # ---- 4. temporal axis ----
    P(f"\n[4] TIME: median decade_of_origin per large community")
    for i, c in enumerate(comms):
        if len(c) < 8:
            continue
        ys = [first_year(info[n].get("decade_of_origin", "")) for n in c]
        ys = [y for y in ys if y]
        if ys:
            ys.sort()
            P(f"    comm {i:>2} (n={len(c):>3}): median year {ys[len(ys)//2]}  "
              f"[{ys[0]}--{ys[-1]}], {len(ys)} dated")

    # ---- node table ----
    with open(os.path.join(ROOT, "wiki_ecosystem_nodes.csv"), "w", newline="", encoding="utf-8") as f:
        cols = ["aesthetic", "community", "degree", "year"] + [f"p_{d}" for d in LEX]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for n in G:
            sc = score_text(info[n].get("key_values", "")); tot = sum(sc.values()) or 1
            w.writerow({"aesthetic": n, "community": node2comm[n], "degree": G.degree(n),
                        "year": first_year(info[n].get("decade_of_origin", "")) or "",
                        **{f"p_{d}": round(sc[d]/tot, 3) for d in LEX}})

    with open(os.path.join(ROOT, "ecosystem_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(R))
    P("\n-> ecosystem_report.txt, wiki_ecosystem_nodes.csv")


if __name__ == "__main__":
    main()

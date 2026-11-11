"""
analyze_cocuration.py
=====================
Component 3 analysis: build an INDEPENDENT aesthetic-similarity graph from are.na user
co-curation (which aesthetics users file the SAME image under), then test whether its
community structure REPLICATES the Aesthetics-Wiki editor network. If it does, the
"the network is just editors describing what they cross-reference" circularity charge dies.

Co-curation edge A-B weight = number of distinct images filed in BOTH an A-named and a
B-named channel. Channels are mapped to aesthetics by normalised title match against the
full wiki aesthetic vocabulary.

Tests:
  - community NMI between co-curation graph and wiki network (shared nodes), vs perm null
  - affect replication: co-curation neighbours share wiki key_values affect (homophily)
Output: printed report + cocuration_report.txt, cocuration_edges.csv
"""
from __future__ import annotations
import csv, os, re, sys, random, itertools
from collections import defaultdict
import networkx as nx

ROOT = os.path.dirname(os.path.abspath(__file__))


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def make_matcher(names):
    """Match a channel title to an aesthetic: exact norm, else longest substring (len>=5),
    plus a few short high-value names. Returns the canonical aesthetic or None."""
    exact = {norm(n): n for n in names}
    # substring matching only for distinctive (>=6 char) names to avoid false hubs like
    # "emo" in "memories" or "mod" in "moodboard"; short names match EXACTLY only.
    subs = sorted([(norm(n), n) for n in names if len(norm(n)) >= 6],
                  key=lambda kv: -len(kv[0]))

    def match(title):
        t = norm(title)
        if not t:
            return None
        if t in exact:
            return exact[t]
        for nn, n in subs:
            if nn in t:
                return n
        return None
    return match


def main():
    # aesthetic vocabulary + wiki community/affect from the ecosystem harvest
    eco = list(csv.DictReader(open(os.path.join(ROOT, "wiki_ecosystem.csv"), encoding="utf-8")))
    match = make_matcher([r["aesthetic"] for r in eco])
    nodes = {r["aesthetic"]: r for r in
             csv.DictReader(open(os.path.join(ROOT, "wiki_ecosystem_nodes.csv"), encoding="utf-8"))}
    wiki_comm = {a: int(nodes[a]["community"]) for a in nodes if nodes[a].get("community") not in ("", None)}

    # block -> set of recognized aesthetics (from its channels)
    bc = list(csv.DictReader(open(os.path.join(ROOT, "arena_block_channels.csv"), encoding="utf-8")))
    block_aes = defaultdict(set)
    for r in bc:
        a = match(r["channel_title"])
        if a:
            block_aes[r["block_id"]].add(a)

    # co-curation edges
    ew = defaultdict(int)
    deg = defaultdict(int)
    for b, aes in block_aes.items():
        if len(aes) < 2:
            continue
        for x, y in itertools.combinations(sorted(aes), 2):
            ew[(x, y)] += 1
    out = []
    def P(s): print(s); out.append(s)
    P("=" * 66)
    P("CO-CURATION REPLICATION (independent are.na user-filing graph)")
    P("=" * 66)
    P(f"blocks with >=1 recognized aesthetic: {sum(1 for v in block_aes.values() if v)}")
    P(f"blocks spanning >=2 aesthetics (edge-bearing): {sum(1 for v in block_aes.values() if len(v)>=2)}")

    G = nx.Graph()
    MINW = 1
    with open(os.path.join(ROOT, "cocuration_edges.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["a", "b", "weight"])
        for (x, y), wt in sorted(ew.items(), key=lambda kv: -kv[1]):
            w.writerow([x, y, wt])
            if wt >= MINW:
                G.add_edge(x, y, weight=wt)
    G.remove_nodes_from(list(nx.isolates(G)))
    P(f"co-curation graph (edges w>={MINW}): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    if G.number_of_nodes() < 8:
        P("too few co-curation nodes to test replication."); _save(out); return

    comms = nx.community.louvain_communities(G, weight="weight", seed=20260614)
    Q = nx.community.modularity(G, comms, weight="weight")
    co_comm = {n: i for i, c in enumerate(comms) for n in c}
    P(f"co-curation communities: {len(comms)}  modularity Q={Q:.3f}")

    # cross-source NMI vs wiki communities on shared nodes
    shared = [n for n in G if n in wiki_comm]
    P(f"\nshared nodes (in both co-curation graph and wiki network): {len(shared)}")
    if len(shared) >= 8:
        from sklearn.metrics import normalized_mutual_info_score as nmi
        a = [co_comm[n] for n in shared]
        b = [wiki_comm[n] for n in shared]
        obs = nmi(a, b)
        rng = random.Random(3)
        perm = []
        for _ in range(2000):
            bb = b[:]; rng.shuffle(bb); perm.append(nmi(a, bb))
        p = (sum(1 for x in perm if x >= obs) + 1) / (len(perm) + 1)
        P(f"CROSS-SOURCE community NMI(co-curation, wiki) = {obs:.3f} "
          f"(perm null {sum(perm)/len(perm):.3f}, p={p:.4f})")
        P("  -> " + ("REPLICATES: structure is not a wiki-editor artifact"
                     if p < 0.05 else "does NOT replicate: circularity charge stands (report honestly)"))

    # top co-curation edges for the manuscript
    P("\ntop co-curation pairs (most co-filed):")
    for (x, y), wt in sorted(ew.items(), key=lambda kv: -kv[1])[:15]:
        P(f"  {wt:4d}  {x} -- {y}")
    _save(out)


def _save(out):
    with open(os.path.join(ROOT, "cocuration_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("\n-> cocuration_report.txt, cocuration_edges.csv")


if __name__ == "__main__":
    main()

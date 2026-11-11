"""
wiki_ecosystem.py
=================
SCALE the collective-human backbone from 26 hand-picked aesthetics to the ENTIRE
Aesthetics-Wiki ecosystem (~1,200 community-authored articles). This turns the validation
from "our taxonomy is recovered" into a study object in its own right: the affective-
temporal organization of internet visual culture, as encoded by collective human authorship.

For every main-namespace article carrying an {{Aesthetic}} infobox we capture:
    key_values (affect terms), key_motifs, key_colours, decade_of_origin,
    related_aesthetics (-> the ecosystem similarity NETWORK), primary_platform
Enumeration: MediaWiki allpages (namespace 0) with continuation; keyless.

Outputs (project root):
  wiki_ecosystem.csv         one row per aesthetic with infobox fields
  wiki_ecosystem_edges.csv   directed related-aesthetic edges (the full network)
"""
from __future__ import annotations
import csv, os, sys, time, json
import urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import wiki_affect as W   # reuse parse_infobox, _clean, split_terms, FIELDS, _get

API = W.API
UA = W.UA


def all_article_titles():
    titles, apcont = [], None
    while True:
        params = {"action": "query", "list": "allpages", "apnamespace": 0,
                  "aplimit": 500, "apfilterredir": "nonredirects", "format": "json"}
        if apcont:
            params["apcontinue"] = apcont
        j = W._get(params)
        if not j:
            break
        titles += [p["title"] for p in j.get("query", {}).get("allpages", [])]
        if "continue" in j:
            apcont = j["continue"]["apcontinue"]
        else:
            break
        time.sleep(0.2)
    return titles


def main():
    titles = all_article_titles()
    print(f"{len(titles)} main-namespace articles enumerated", file=sys.stderr)

    rows, edges = [], []
    n_aes = 0
    for i, title in enumerate(titles):
        res = W.get_wikitext(title)
        if res is None:
            continue
        ctitle, wt = res
        if "{{Aesthetic" not in wt and "{{ Aesthetic" not in wt:
            continue                      # not an aesthetic article (lists, hubs, etc.)
        ib = W.parse_infobox(wt)
        # require at least one substantive infobox field to count it
        if not any(ib.get(f) for f in ("key_values", "key_motifs", "related_aesthetics")):
            continue
        n_aes += 1
        rows.append({"aesthetic": ctitle, **{f: ib.get(f, "") for f in W.FIELDS}})
        for rel in W.split_terms(ib.get("related_aesthetics", "")):
            edges.append({"aesthetic": ctitle, "related": rel})
        if n_aes % 100 == 0:
            print(f"  {n_aes} aesthetics ({i+1}/{len(titles)} pages scanned)", file=sys.stderr)
        time.sleep(0.15)

    with open(os.path.join(ROOT, "wiki_ecosystem.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["aesthetic"] + W.FIELDS)
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(ROOT, "wiki_ecosystem_edges.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["aesthetic", "related"])
        w.writeheader(); w.writerows(edges)

    kv = sum(1 for r in rows if r["key_values"])
    print(f"\n=== ecosystem harvest ===")
    print(f"aesthetics with an infobox: {len(rows)}")
    print(f"  with key_values: {kv}")
    print(f"related-aesthetic edges: {len(edges)}")
    print("-> wiki_ecosystem.csv, wiki_ecosystem_edges.csv")


if __name__ == "__main__":
    main()

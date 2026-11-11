"""
wiki_affect.py
==============
Harvest COMMUNITY-AUTHORED affect labels for each aesthetic from the Aesthetics Wiki
(aesthetics.fandom.com) via the keyless MediaWiki API.

Each aesthetic page carries an {{Aesthetic}} infobox with structured fields:
    |key_motifs= |key_colours= |key_values= |related_aesthetics= |other_names= ...
`key_values` is the collective human characterisation of the aesthetic's *affect*
(e.g. Liminal Space -> "Uncanniness, surreality, nostalgia, familiarity, discomfort,
comfort"). This is the category-level human dependent variable: a criterion the paper's
computational structure can be tested against WITHOUT running a rating study.

Honest scope: the wiki is editor-authored, not a large-n rating distribution -- it is a
collective *taxonomy*, not crowd ratings. Treated as such.

Outputs (to project root):
  wiki_affect.csv        one row per aesthetic: cluster, resolved title, raw infobox fields
  wiki_related_edges.csv crowd-drawn similarity graph: aesthetic -> related_aesthetic
"""
from __future__ import annotations
import csv, os, re, sys, time, json
import urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from aesthetic_map import AESTHETICS

API = "https://aesthetics.fandom.com/api.php"
UA = {"User-Agent": "AestheticsCV-research/1.0 (academic; contact yl2884@cornell.edu)"}
FIELDS = ["other_names", "decade_of_origin", "key_motifs", "key_colours",
          "key_values", "related_aesthetics"]


def _get(params):
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:
            time.sleep(1.0 * (attempt + 1))
            last = e
    print(f"  ! request failed: {last}", file=sys.stderr)
    return None


def resolve_title(query):
    """Return the real page title (handles redirects / near-misses) or None."""
    j = _get({"action": "query", "list": "search", "srsearch": query,
              "srlimit": 1, "format": "json"})
    if not j:
        return None
    hits = j.get("query", {}).get("search", [])
    return hits[0]["title"] if hits else None


def get_wikitext(title):
    """Return (canonical_title, wikitext) following redirects, or None if no such page."""
    j = _get({"action": "parse", "page": title, "prop": "wikitext",
              "redirects": 1, "format": "json"})
    if not j or "parse" not in j:
        return None
    return j["parse"].get("title", title), j["parse"]["wikitext"]["*"]


def _clean(val):
    """Strip wiki markup from an infobox field value -> plain comma list."""
    if val is None:
        return ""
    v = val
    v = re.sub(r"<\s*br\s*/?\s*>", ", ", v, flags=re.I)     # <br> -> comma FIRST
    v = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", v)  # [[Link|label]] -> label
    v = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", v)     # [url label] -> label
    v = re.sub(r"\[https?://\S+\]", " ", v)                  # bare [url] -> drop
    v = re.sub(r"\{\{[^}]*\}\}", " ", v)                     # drop templates
    v = re.sub(r"<[^>]+>", " ", v)                           # drop remaining html tags
    v = v.replace("'''", "").replace("''", "")               # bold/italic
    v = re.sub(r"[\*\n]+", ", ", v)                          # bullets / stray newlines -> comma
    v = re.sub(r"(\s*,\s*)+", ", ", v)                       # collapse repeated commas
    v = re.sub(r"\s+", " ", v).strip(" ,;")
    return v


def parse_infobox(wikitext):
    """Extract the {{Aesthetic}} infobox params we care about."""
    out = {f: "" for f in FIELDS}
    # grab the infobox block (first {{Aesthetic ... }} balanced-ish region)
    m = re.search(r"\{\{\s*Aesthetic\b", wikitext, flags=re.I)
    region = wikitext[m.start():] if m else wikitext
    # a value ends at the next infobox param (on a new line OR inline) or the close
    stop = (r"(?=\n\s*\||\|\s*(?:other_names|decade_of_origin|creator_s|key_motifs|"
            r"key_colours|key_values|related_aesthetics|primary_platform|related_media|"
            r"subgenres|title\d|image\d|caption\d)\s*=|\n\s*\}\}|\Z)")
    for f in FIELDS:
        mm = re.search(r"\|\s*" + re.escape(f) + r"\s*=\s*(.*?)" + stop,
                       region, flags=re.S | re.I)
        if mm:
            out[f] = _clean(mm.group(1))
    return out


def split_terms(s):
    return [t.strip() for t in re.split(r"[,;/]", s) if t.strip()]


def main():
    rows, edges = [], []
    for name, (cluster, query) in AESTHETICS.items():
        # try the EXACT page title first (following redirects); only search if absent
        res = get_wikitext(query)
        if res is None:
            alt = resolve_title(query)
            res = get_wikitext(alt) if alt else None
        if res is None:
            print(f"[{cluster:14s}] {name:18s} -> NO PAGE", file=sys.stderr)
            rows.append({"aesthetic": name, "cluster": cluster, "wiki_title": "",
                         **{f: "" for f in FIELDS}})
            continue
        title, wt = res
        ib = parse_infobox(wt)
        rows.append({"aesthetic": name, "cluster": cluster, "wiki_title": title, **ib})
        for rel in split_terms(ib.get("related_aesthetics", "")):
            edges.append({"aesthetic": name, "cluster": cluster, "related": rel})
        kv = ib.get("key_values", "")
        print(f"[{cluster:14s}] {name:18s} -> {title:22s} | key_values: {kv[:70]}",
              file=sys.stderr)
        time.sleep(0.3)

    cols = ["aesthetic", "cluster", "wiki_title"] + FIELDS
    with open(os.path.join(ROOT, "wiki_affect.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    with open(os.path.join(ROOT, "wiki_related_edges.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["aesthetic", "cluster", "related"])
        w.writeheader(); w.writerows(edges)

    got = sum(1 for r in rows if r["key_values"])
    print(f"\n=== wiki harvest done ===")
    print(f"aesthetics with key_values: {got}/{len(rows)}")
    print(f"related-aesthetic edges: {len(edges)}")
    print(f"-> wiki_affect.csv, wiki_related_edges.csv")


if __name__ == "__main__":
    main()

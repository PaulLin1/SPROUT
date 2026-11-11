"""
aesthetic_map.py
================
Central registry mapping each internet aesthetic to (a) its affective-mechanism
cluster (the paper's R/H/I/A taxonomy) and (b) the search term used to resolve its
Aesthetics-Wiki page and are.na channels.

This is the bridge between the paper's *theoretical* grouping and the *collective
human* labelling we harvest:
  - Aesthetics Wiki  -> community-AUTHORED affect descriptors per aesthetic (category level)
  - are.na channels  -> crowd-CURATED image membership per aesthetic (image level)

Clusters: restorative / hauntological / inverted (uncanny) / anemoia.
(Controls are populated present-day scenes; they have no folk-aesthetic label, so they
are absent here -- collective human data only exists for the *named* aesthetics.)
"""

# name -> (cluster, wiki_query)   wiki_query is a best-guess page title; the harvester
# resolves redirects / near-misses via the MediaWiki search fallback.
AESTHETICS = {
    # ---- restorative ----------------------------------------------------------
    "Frutiger Aero":   ("restorative", "Frutiger Aero"),
    "Y2K":             ("restorative", "Y2K"),
    "Cottagecore":     ("restorative", "Cottagecore"),
    "Kidcore":         ("restorative", "Kidcore"),
    "Synthwave":       ("restorative", "Synthwave"),
    "Googie":          ("restorative", "Googie"),
    "Atompunk":        ("restorative", "Atompunk"),
    "Raygun Gothic":   ("restorative", "Raygun Gothic"),

    # ---- hauntological --------------------------------------------------------
    "Vaporwave":       ("hauntological", "Vaporwave"),
    "Mallsoft":        ("hauntological", "Mallsoft"),
    "Cassette Futurism": ("hauntological", "Cassette Futurism"),
    "Dieselpunk":      ("hauntological", "Dieselpunk"),
    "Retrofuturism":   ("hauntological", "Retrofuturism"),
    "Faded":           ("hauntological", "Faded"),

    # ---- inverted / uncanny ---------------------------------------------------
    "Liminal Space":   ("inverted", "Liminal Space"),
    "The Backrooms":   ("inverted", "The Backrooms"),
    "Poolcore":        ("inverted", "Poolcore"),
    "Weirdcore":       ("inverted", "Weirdcore"),
    "Dreamcore":       ("inverted", "Dreamcore"),
    "Analog Horror":   ("inverted", "Analog Horror"),

    # ---- anemoia --------------------------------------------------------------
    "Dark Academia":   ("anemoia", "Dark Academia"),
    "Light Academia":  ("anemoia", "Light Academia"),
    "Fairycore":       ("anemoia", "Fairycore"),
    "Goblincore":      ("anemoia", "Goblincore"),
    "Royalcore":       ("anemoia", "Royalcore"),
    "Old Money":       ("anemoia", "Old Money"),
}

CLUSTERS = ["restorative", "hauntological", "inverted", "anemoia"]

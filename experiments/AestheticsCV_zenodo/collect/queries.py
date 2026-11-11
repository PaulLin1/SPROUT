"""
queries.py
==========
The collection taxonomy. Each entry is one search query mapped to:

  cluster        : the affective-mechanism cluster used in the study design
                   {restorative, hauntological, inverted, anemoia, control}
  scene_category : matched to object_analysis.SCENE_OBJECT_PRIORS keys where it makes
                   sense (drives the expected-but-absent metric). "generic" otherwise.
  terms          : the query string sent to the licensed image APIs.

Organisation is by *what affective operation the aesthetic performs on the past*
(Boym's restorative vs. hauntological, the inverted/uncanny family, pure anemoia),
NOT by folk aesthetic name -- so a reviewer sees a principled stimulus taxonomy.

Queries are SHORT (1-3 words). Both back-end APIs do strict AND matching, so long
descriptive phrases ("empty hallway corridor fluorescent") return zero; short
concrete nouns are what actually retrieve licensed images. The visual constituents of
a niche aesthetic (greek bust, neon sign, CRT monitor for vaporwave) are searchable
even when the folk label is not -- this is the honest, licensed way to approximate the
look without scraping the aesthetic communities.

The inverted (liminal) and control clusters deliberately share scene_category labels
(corridor, shopping_mall, swimming_pool, classroom, office, parking_lot, playground)
so the expected-but-absent score contrasts the SAME scene populated vs. empty.
"""

PRIOR_SCENES = ["shopping_mall", "corridor", "swimming_pool", "classroom",
                "playground", "parking_lot", "office"]

# (cluster, scene_category, query terms)  -- short, concrete, AND-match friendly
QUERIES = [
    # ---- restorative nostalgia: warm, comfort-seeking, techno-optimist ----------
    ("restorative", "generic", "water bubbles"),
    ("restorative", "generic", "blue sky clouds"),
    ("restorative", "generic", "green meadow"),
    ("restorative", "generic", "coral reef"),
    ("restorative", "generic", "tropical fish"),
    ("restorative", "generic", "dewdrop leaf"),
    ("restorative", "generic", "aquarium"),
    ("restorative", "generic", "cottage garden"),
    ("restorative", "generic", "sunset beach"),
    ("restorative", "generic", "waterfall nature"),
    ("restorative", "generic", "rainbow sky"),
    ("restorative", "generic", "flower field"),

    # ---- hauntological: ironic / melancholic, "lost futures" -------------------
    ("hauntological", "generic", "greek bust"),
    ("hauntological", "generic", "marble statue"),
    ("hauntological", "generic", "plaster sculpture"),
    ("hauntological", "generic", "crt monitor"),
    ("hauntological", "generic", "retro computer"),
    ("hauntological", "generic", "vintage television"),
    ("hauntological", "generic", "neon sign"),
    ("hauntological", "generic", "arcade machine"),
    ("hauntological", "generic", "cassette tape"),
    ("hauntological", "generic", "1980s interior"),
    ("hauntological", "generic", "shopping arcade"),
    ("hauntological", "generic", "palm trees sunset"),

    # ---- inverted / uncanny: liminal spaces, kenopsia (EMPTY, scene-matched) ----
    ("inverted", "corridor",       "empty corridor"),
    ("inverted", "corridor",       "empty hallway"),
    ("inverted", "corridor",       "hospital corridor"),
    ("inverted", "corridor",       "underground passage"),
    ("inverted", "shopping_mall",  "empty mall"),
    ("inverted", "shopping_mall",  "deserted shopping centre"),
    ("inverted", "swimming_pool",  "empty swimming pool"),
    ("inverted", "swimming_pool",  "abandoned pool"),
    ("inverted", "classroom",      "empty classroom"),
    ("inverted", "office",         "empty office"),
    ("inverted", "parking_lot",    "parking garage"),
    ("inverted", "parking_lot",    "underground car park"),
    ("inverted", "playground",     "empty playground"),
    ("inverted", "corridor",       "stairwell"),
    ("inverted", "generic",        "waiting room"),
    ("inverted", "generic",        "hotel corridor"),

    # ---- anemoia: pure imagined-past nostalgia, pasts never lived ---------------
    ("anemoia", "generic", "fairytale castle"),
    ("anemoia", "generic", "castle interior"),
    ("anemoia", "generic", "enchanted forest"),
    ("anemoia", "generic", "misty forest"),
    ("anemoia", "generic", "old library"),
    ("anemoia", "generic", "palace interior"),
    ("anemoia", "generic", "victorian interior"),
    ("anemoia", "generic", "stately home"),
    ("anemoia", "generic", "mushroom forest"),
    ("anemoia", "generic", "medieval hall"),

    # ---- retro-futurism: optimistic strand -> RESTORATIVE -----------------------
    ("restorative", "generic", "googie architecture"),
    ("restorative", "generic", "space age design"),
    ("restorative", "generic", "raygun gothic"),
    ("restorative", "generic", "atomic age interior"),
    ("restorative", "generic", "world fair pavilion"),
    ("restorative", "generic", "synthwave neon grid"),
    ("restorative", "generic", "retro futuristic city"),

    # ---- retro-futurism: decayed / lost-futures strand -> HAUNTOLOGICAL ----------
    ("hauntological", "generic", "abandoned space age building"),
    ("hauntological", "generic", "cassette futurism"),
    ("hauntological", "generic", "dieselpunk machine"),
    ("hauntological", "generic", "retro control room"),
    ("hauntological", "generic", "vintage spaceport"),

    # ---- additional named nostalgia aesthetics (round 2 coverage) ---------------
    ("anemoia", "generic", "fairycore mushroom"),
    ("anemoia", "generic", "dark academia library"),
    ("anemoia", "generic", "royal palace gold"),
    ("inverted", "swimming_pool", "tiled pool room"),
    ("inverted", "generic", "vhs analog horror"),

    # ---- control: the SAME scenes as inverted, but POPULATED --------------------
    ("control", "shopping_mall",  "shopping mall"),
    ("control", "shopping_mall",  "mall food court"),
    ("control", "corridor",       "school hallway"),
    ("control", "corridor",       "hospital hallway"),
    ("control", "swimming_pool",  "public swimming pool"),
    ("control", "classroom",      "classroom students"),
    ("control", "classroom",      "lecture hall"),
    ("control", "office",         "open plan office"),
    ("control", "office",         "office workers"),
    ("control", "parking_lot",    "parking lot"),
    ("control", "playground",     "playground children"),
    ("control", "playground",     "public park"),
]

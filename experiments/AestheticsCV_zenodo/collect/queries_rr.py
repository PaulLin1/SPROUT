"""
queries_rr.py
=============
Re-sourcing query taxonomy for the aesthetic-FAITHFUL corpus (Registered-Report phase).

Unlike the Wikimedia query set (which had to search visual *constituents* because the
folk labels aren't encyclopedically tagged), Flickr CC and the Internet Archive ARE tagged
by aesthetic name. So here we query the aesthetics DIRECTLY---this is the whole point of
re-sourcing: to capture the post-processed look (grading, compositing, degradation,
typography) that carries the affective grammar.

The decayed-retro-futurism strand is over-sampled on purpose: it is the n=1 gap that
currently blocks the within-domain restorative-vs-hauntological test.
"""

# (cluster, scene_category, query)  -- scene_category mostly 'generic'; the inverted/
# control scene-matched pairs keep prior-scene tags for the eerie contrast.
QUERIES = [
    # ---- restorative (incl. optimistic retro-futurism) -------------------------
    ("restorative", "generic", "frutiger aero"),
    ("restorative", "generic", "frutiger aero wallpaper"),
    ("restorative", "generic", "y2k aesthetic"),
    ("restorative", "generic", "cottagecore"),
    ("restorative", "generic", "kidcore"),
    ("restorative", "generic", "synthwave"),
    ("restorative", "generic", "googie architecture"),
    ("restorative", "generic", "atompunk"),
    ("restorative", "generic", "raygun gothic"),

    # ---- hauntological (incl. DECAYED retro-futurism -- oversampled) -----------
    ("hauntological", "generic", "vaporwave"),
    ("hauntological", "generic", "vaporwave aesthetic"),
    ("hauntological", "generic", "mallsoft"),
    ("hauntological", "generic", "dead mall"),
    ("hauntological", "generic", "cassette futurism"),
    ("hauntological", "generic", "dieselpunk"),
    ("hauntological", "generic", "abandoned space age"),
    ("hauntological", "generic", "retro futurism decay"),
    ("hauntological", "generic", "atomic age ruins"),

    # ---- inverted / uncanny ----------------------------------------------------
    ("inverted", "generic",       "liminal space"),
    ("inverted", "generic",       "liminal spaces aesthetic"),
    ("inverted", "generic",       "the backrooms"),
    ("inverted", "swimming_pool", "poolrooms"),
    ("inverted", "generic",       "weirdcore"),
    ("inverted", "generic",       "dreamcore"),
    ("inverted", "generic",       "analog horror"),
    ("inverted", "corridor",      "empty liminal hallway"),
    ("inverted", "shopping_mall", "empty liminal mall"),

    # ---- anemoia ---------------------------------------------------------------
    ("anemoia", "generic", "dark academia"),
    ("anemoia", "generic", "light academia"),
    ("anemoia", "generic", "fairycore"),
    ("anemoia", "generic", "goblincore"),
    ("anemoia", "generic", "royaltycore"),
    ("anemoia", "generic", "old money aesthetic"),

    # ---- control (populated present-day, scene-matched) ------------------------
    ("control", "shopping_mall",  "crowded shopping mall"),
    ("control", "corridor",       "busy school hallway"),
    ("control", "swimming_pool",  "public pool people"),
    ("control", "classroom",      "classroom students"),
    ("control", "office",         "busy office people"),
    ("control", "parking_lot",    "parking lot cars people"),
    ("control", "playground",     "playground children playing"),
]

"""
affect_prompts.py
=================
Shared, CALIBRATED affect-rating prompt for the VLM judge panel. Rubric anchors and the
"reserve 5 for unambiguous cases; most images are 2-3" instruction are deliberate, to
counter the documented systematic INFLATION of VLM/LLM affect ratings (multi-judge
consensus + calibrated scales are the literature-prescribed mitigations).
"""
import json, re

AXES = ["eerie", "nostalgic", "abandoned", "dreamlike", "valence", "arousal"]
MODES = ["restorative", "hauntological", "anemoia", "inverted"]

PROMPT = (
    "You are a careful visual-affect annotator rating ONE image. "
    "Use the FULL 1-5 scale and reserve 5 for unambiguous, intense cases; most images are 2-3. "
    "Rate 1 (not at all) to 5 (extremely):\n"
    "- eerie: unsettling, uncanny, wrong-feeling\n"
    "- nostalgic: evokes longing for the past\n"
    "- abandoned: empty/deserted, missing people who should be there\n"
    "- dreamlike: surreal, hazy, dream-logic\n"
    "Then valence (1 very negative to 5 very positive) and arousal (1 calm to 5 intense). "
    "Then pick the single best MODE for the image's relation to the past:\n"
    "- restorative: warm, optimistic remembered/idealized past (e.g. cosy, bright retro)\n"
    "- hauntological: melancholic 'lost futures', decayed retro, ironic mall/corporate nostalgia\n"
    "- anemoia: longing for a past one never lived (refined, scholarly, old-world)\n"
    "- inverted: familiar-but-wrong, eerie/uncanny, liminal, dreamcore/weirdcore\n"
    'Return ONLY compact JSON: {"eerie":int,"nostalgic":int,"abandoned":int,'
    '"dreamlike":int,"valence":int,"arousal":int,"mode":str}'
)


def parse(txt):
    """Robustly parse the JSON object from a model reply. Returns dict or None."""
    if not txt:
        return None
    try:
        d = json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            return None
        try:
            d = json.loads(m.group(0))
        except Exception:
            return None
    out = {}
    for a in AXES:
        try:
            v = int(round(float(d.get(a))))
            out[a] = v if 1 <= v <= 5 else None
        except (TypeError, ValueError):
            out[a] = None
    mode = str(d.get("mode", "")).strip().lower()
    out["mode"] = mode if mode in MODES else ""
    return out

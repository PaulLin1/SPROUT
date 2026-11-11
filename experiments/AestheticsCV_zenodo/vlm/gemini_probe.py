"""
gemini_probe.py
===============
Component 0: validate the Gemini key + endpoint BEFORE any batch.
Lists available models (trying all auth styles), picks a vision-capable Flash model,
and runs ONE graded-affect rating on a real are.na image. Prints everything; writes nothing.
"""
import csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemini_client as gc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROMPT = (
    "You are rating the AFFECT a single image evokes, as a careful annotator. "
    "Use the FULL 1-5 scale and reserve 5 for unambiguous, intense cases; most images are 2-3. "
    "Rate each dimension 1 (not at all) to 5 (extremely):\n"
    "- eerie: unsettling, uncanny, wrong-feeling\n"
    "- nostalgic: evokes longing for the past\n"
    "- abandoned: empty, deserted, missing people who should be there\n"
    "- dreamlike: surreal, hazy, dream-logic\n"
    "Also give valence (1 very negative to 5 very positive) and arousal (1 calm to 5 intense), "
    "and pick the single best mode: restorative, hauntological, anemoia, or inverted.\n"
    'Return ONLY JSON: {"eerie":int,"nostalgic":int,"abandoned":int,"dreamlike":int,'
    '"valence":int,"arousal":int,"mode":str}'
)


def main():
    key = gc.load_key()
    print(f"key loaded ({len(key)} chars, prefix {key[:3]}...)")
    style, names = gc.list_models(key)
    print(f"AUTH STYLE THAT WORKS: {style}")
    gem = [n for n in names if "gemini" in n.lower()]
    print(f"{len(names)} models; gemini models:")
    for n in gem:
        print("  ", n)

    # choose a vision Flash model
    pref = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    chosen = None
    for p in pref:
        for n in gem:
            if p in n and "thinking" not in n and "lite" not in n and "image" not in n and "tts" not in n:
                chosen = n if n.startswith("models/") else f"models/{n}"; break
        if chosen:
            break
    if not chosen and gem:
        chosen = gem[0]
    print(f"\nCHOSEN MODEL: {chosen}")

    # one real are.na image
    rows = list(csv.DictReader(open(os.path.join(ROOT, "stimuli_manifest_arena.csv"), encoding="utf-8")))
    sample = next(r for r in rows if os.path.exists(r["image_path"]))
    print(f"test image: {sample['aesthetic']} | {os.path.basename(sample['image_path'])}")
    txt = gc.generate(key, chosen, PROMPT, image_path=sample["image_path"], auth_style=style)
    print("\nRAW RESPONSE:\n", txt)
    try:
        print("\nPARSED:", json.loads(txt))
        print("\n*** PROBE OK: key works, model rates affect, JSON parses ***")
    except Exception as e:
        print(f"\n(JSON parse note: {e})")


if __name__ == "__main__":
    main()


from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Paths
IMAGE_DIR = Path("images")
OUTPUT_CSV = Path("images.csv")

# Model
MODEL_NAME = "Salesforce/blip-image-captioning-base"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}...")

processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

# Image formats to process
extensions = {".jpg", ".jpeg", ".png", ".webp"}

# Load existing CSV so reruns don't discard previous captions
if OUTPUT_CSV.exists():
    df = pd.read_csv(OUTPUT_CSV)
    captions = {
        row["image_path"]: row["caption"]
        for _, row in df.iterrows()
    }
else:
    captions = {}

image_paths = sorted(
    p for p in IMAGE_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
)

if not image_paths:
    raise RuntimeError(f"No images found in {IMAGE_DIR.resolve()}")

for path in image_paths:
    relative_path = path.as_posix()

    # Skip images that already have captions
    if relative_path in captions and pd.notna(captions[relative_path]):
        print(f"Skipping {relative_path}")
        continue

    try:
        image = Image.open(path).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt"
        ).to(device)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=50
            )

        caption = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        captions[relative_path] = caption
        print(f"{relative_path}: {caption}")

        # Save after each image in case the script is interrupted
        rows = [
            {
                "image_id": f"img{i:03d}",
                "image_path": image_path,
                "caption": caption_text,
            }
            for i, (image_path, caption_text)
            in enumerate(sorted(captions.items()), start=1)
        ]

        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    except Exception as exc:
        print(f"Error processing {relative_path}: {exc}")

print(f"\nFinished. Captions saved to {OUTPUT_CSV.resolve()}")

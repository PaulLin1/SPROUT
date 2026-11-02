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

# Class of each image, by image number (files are named <number>.<ext>)
CLASSES = {
    "laptop twee": range(1, 11),
    "laptop": range(11, 21),
    "random": range(21, 31),
    "office siren": range(31, 41),
    "office": range(41, 51),
    "siren": range(51, 61),
}


def class_of(number: int) -> str:
    for name, numbers in CLASSES.items():
        if number in numbers:
            return name
    raise ValueError(f"Image {number} is not in any class")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}...")

processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

# Image formats to process
extensions = {".jpg", ".jpeg", ".png", ".webp"}

image_paths = sorted(
    (p for p in IMAGE_DIR.iterdir()
     if p.is_file() and p.suffix.lower() in extensions),
    key=lambda p: int(p.stem),
)

if not image_paths:
    raise RuntimeError(f"No images found in {IMAGE_DIR.resolve()}")

rows = []

for path in image_paths:
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

    number = int(path.stem)
    print(f"{path.as_posix()}: {caption}")

    rows.append({
        "image_id": f"img{number:03d}",
        "image_path": path.as_posix(),
        "caption": caption,
        "class": class_of(number),
    })

pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

print(f"\nFinished. Captions saved to {OUTPUT_CSV.resolve()}")

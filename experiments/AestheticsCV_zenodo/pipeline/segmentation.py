"""
segmentation.py
===============
Axis 2 of 3: image segmentation and scene-region structure.

Two backends, swappable:

  - SemanticSegmenter  : ADE20K-trained model (SegFormer / Mask2Former). Gives
                         *labelled* regions (wall, floor, ceiling, person, sky,
                         water, light, door...). This is what powers the
                         "emptiness" / kenopsia metrics.
  - PromptlessSegmenter: SAM (Segment Anything) for class-agnostic masks when you
                         want region-count / region-size statistics without labels.

Metrics computed map to the design:

  - emptiness_index      -> Fisher's *eerie* (absence): proportion of the frame that
                            is structural surface (wall/floor/ceiling) with no
                            occupant objects. Liminal spaces maximise this.
  - human_absence        -> the specific "a mall with no people" signal: structural
                            scene class present, person area ~ 0.
  - region_count / size  -> configural simplicity vs clutter (ties to typicality phi)
  - vanishing_structure  -> long uninterrupted floor/wall runs (corridor geometry)
  - light_source_frac    -> fluorescent-light prevalence (liminal/analog-horror cue)

Heavy model weights are NOT downloaded in this file's import; they load lazily on
first call so the module imports cheaply on machines without GPUs.
"""

from __future__ import annotations
import numpy as np

# ADE20K class groups we care about. Index sets are filled from the model's
# id2label at load time; these are the *semantic groups* the metrics use.
STRUCTURE_LABELS = {"wall", "floor", "ceiling", "road", "sidewalk", "pavement",
                    "stairs", "stairway", "runway", "path", "floor;flooring"}
OCCUPANT_LABELS  = {"person", "chair", "table", "sofa", "bed", "car", "plant",
                    "potted plant", "cabinet", "shelf", "book", "bottle", "box"}
LIGHT_LABELS     = {"light", "lamp", "chandelier", "sconce", "spotlight"}
OPENING_LABELS   = {"door", "doorway", "window", "windowpane", "screen"}
NATURE_LABELS    = {"sky", "water", "sea", "tree", "grass", "mountain", "plant"}


class SemanticSegmenter:
    """Labelled semantic segmentation via HuggingFace transformers (lazy load)."""

    def __init__(self, model_name: str = "nvidia/segformer-b4-finetuned-ade-512-512",
                 device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._proc = None
        self._model = None
        self._id2label = None

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from transformers import (AutoImageProcessor,
                                  SegformerForSemanticSegmentation)
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._proc = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = (SegformerForSemanticSegmentation
                       .from_pretrained(self.model_name).to(self.device).eval())
        self._id2label = {int(k): v.lower()
                          for k, v in self._model.config.id2label.items()}

    def label_map(self, pil_image) -> np.ndarray:
        """Return HxW int array of ADE20K class ids."""
        import torch
        self._ensure()
        inputs = self._proc(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self._model(**inputs).logits
        up = torch.nn.functional.interpolate(
            logits, size=pil_image.size[::-1], mode="bilinear", align_corners=False)
        return up.argmax(1)[0].cpu().numpy()

    def id2label(self):
        self._ensure()
        return self._id2label


def _group_mask(label_map, id2label, label_set):
    ids = {i for i, name in id2label.items()
           if any(tok in name for tok in label_set)}
    return np.isin(label_map, list(ids)) if ids else np.zeros_like(label_map, bool)


def _largest_run(binary_row_or_col):
    """Longest contiguous True run -- used for 'uninterrupted surface' geometry."""
    best = cur = 0
    for v in binary_row_or_col:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def scene_region_features(label_map: np.ndarray, id2label: dict) -> dict:
    H, W = label_map.shape
    total = H * W
    struct = _group_mask(label_map, id2label, STRUCTURE_LABELS)
    occ    = _group_mask(label_map, id2label, OCCUPANT_LABELS)
    person_ids = {i for i, n in id2label.items() if "person" in n}
    person = np.isin(label_map, list(person_ids)) if person_ids else np.zeros_like(label_map, bool)
    light  = _group_mask(label_map, id2label, LIGHT_LABELS)
    nature = _group_mask(label_map, id2label, NATURE_LABELS)

    struct_frac = struct.mean()
    occ_frac    = occ.mean()

    # region count / size from connected components of the label map
    from skimage.measure import label as cc_label
    comps = cc_label(label_map, connectivity=1)
    sizes = np.bincount(comps.ravel())[1:]
    region_count = len(sizes)
    region_size_gini = _gini(sizes) if region_count else 0.0

    # vanishing structure: longest uninterrupted structural run, row- and col-wise
    row_runs = max((_largest_run(struct[r]) for r in range(0, H, max(1, H // 64))), default=0)
    col_runs = max((_largest_run(struct[:, c]) for c in range(0, W, max(1, W // 64))), default=0)
    vanishing = max(row_runs / W, col_runs / H)

    return {
        "structure_frac": float(struct_frac),
        "occupant_frac": float(occ_frac),
        "person_frac": float(person.mean()),
        "light_source_frac": float(light.mean()),
        "nature_frac": float(nature.mean()),
        # emptiness: lots of bare structural surface, few occupants
        "emptiness_index": float(struct_frac * (1.0 - min(1.0, occ_frac * 3))),
        # the 'no people where people belong' signal: indoor-structure heavy, person ~0
        "human_absence": float(struct_frac * (1.0 - min(1.0, person.mean() * 50))),
        "region_count": int(region_count),
        "region_size_gini": float(region_size_gini),
        "vanishing_structure": float(vanishing),
    }


def _gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


class PromptlessSegmenter:
    """SAM class-agnostic masks for label-free region statistics (lazy load)."""

    def __init__(self, checkpoint: str = "facebook/sam-vit-base", device=None):
        self.checkpoint = checkpoint
        self.device = device
        self._gen = None

    def _ensure(self):
        if self._gen is not None:
            return
        import torch
        from transformers import pipeline
        self.device = self.device or (0 if torch.cuda.is_available() else -1)
        self._gen = pipeline("mask-generation", model=self.checkpoint, device=self.device)

    def region_stats(self, pil_image, points_per_side: int = 16) -> dict:
        self._ensure()
        out = self._gen(pil_image, points_per_side=points_per_side)
        masks = out["masks"]
        areas = np.array([m.sum() for m in masks], float)
        total = pil_image.size[0] * pil_image.size[1]
        return {
            "sam_region_count": int(len(masks)),
            "sam_area_gini": _gini(areas),
            "sam_largest_frac": float(areas.max() / total) if len(areas) else 0.0,
            "sam_coverage": float(min(1.0, areas.sum() / total)),
        }


def analyze(pil_image, segmenter: SemanticSegmenter) -> dict:
    """Full segmentation feature vector for one stimulus (labelled backend)."""
    lm = segmenter.label_map(pil_image)
    return scene_region_features(lm, segmenter.id2label())


SCALAR_KEYS = [
    "structure_frac", "occupant_frac", "person_frac", "light_source_frac",
    "nature_frac", "emptiness_index", "human_absence", "region_count",
    "region_size_gini", "vanishing_structure",
]

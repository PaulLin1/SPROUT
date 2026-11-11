"""
object_analysis.py
==================
Axis 3 of 3: object detection and object-level scene expectation.

The novel metric here is *expected-but-absent*: scenes carry priors about which
objects belong (a mall expects people and storefront clutter; a pool expects
people and water toys). The liminal/eerie signal is precisely the violation of
that prior -- the objects that *should* be present at high probability but are
detected at ~zero. This is the object-level twin of the segmentation human_absence
metric and a direct, quantitative handle on Fisher's eerie.

Backends:
  - Detector : open-vocabulary detector (OWLv2 / Grounding-DINO) so you can query
               arbitrary object prompts, OR a closed-set DETR for COCO objects.
  - SceneExpectation : a per-scene-category prior over object classes, estimated
               from the populated control corpus (Places + COCO co-occurrence).

Weights load lazily.
"""

from __future__ import annotations
import numpy as np


class OpenVocabDetector:
    """OWLv2 open-vocabulary detection (lazy load). Query any text labels."""

    def __init__(self, model_name: str = "google/owlv2-base-patch16-ensemble",
                 device=None, score_thresh: float = 0.10):
        self.model_name = model_name
        self.device = device
        self.score_thresh = score_thresh
        self._proc = None
        self._model = None

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._proc = Owlv2Processor.from_pretrained(self.model_name)
        self._model = (Owlv2ForObjectDetection
                       .from_pretrained(self.model_name).to(self.device).eval())

    def detect(self, pil_image, labels: list[str]) -> list[dict]:
        import torch
        self._ensure()
        inputs = self._proc(text=[labels], images=pil_image,
                            return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        target = torch.tensor([pil_image.size[::-1]]).to(self.device)
        # transformers >=5 renamed the OWLv2 post-processor; keep a fallback for <5.
        post = getattr(self._proc, "post_process_grounded_object_detection", None)
        if post is not None:
            res = post(outputs, target_sizes=target, threshold=self.score_thresh,
                       text_labels=[labels])[0]
        else:
            res = self._proc.post_process_object_detection(
                outputs, target_sizes=target, threshold=self.score_thresh)[0]
        # label may arrive as ready strings ("text_labels") or integer indices ("labels")
        text_labels = res.get("text_labels")
        idx_labels = res.get("labels")
        dets = []
        for i, (box, score) in enumerate(zip(res["boxes"], res["scores"])):
            if text_labels is not None:
                name = text_labels[i]
            elif idx_labels is not None:
                name = labels[int(idx_labels[i])]
            else:
                continue
            dets.append({"label": str(name), "score": float(score),
                         "box": [float(v) for v in box]})
        return dets


# Prior over objects that "belong" in each scene category. In practice estimate
# these frequencies from your populated control corpus; defaults encode strong
# priors for the categories most relevant to the liminal/eerie contrast.
SCENE_OBJECT_PRIORS = {
    "shopping_mall": {"person": 0.97, "shopping bag": 0.6, "bench": 0.5, "store sign": 0.7},
    "corridor":      {"person": 0.85, "door": 0.9, "sign": 0.5},
    "swimming_pool": {"person": 0.9, "pool float": 0.4, "towel": 0.4, "ladder": 0.6},
    "classroom":     {"person": 0.95, "chair": 0.9, "desk": 0.9, "whiteboard": 0.7},
    "playground":    {"person": 0.9, "swing": 0.7, "slide": 0.6},
    "parking_lot":   {"person": 0.7, "car": 0.95},
    "office":        {"person": 0.9, "chair": 0.9, "monitor": 0.8, "desk": 0.85},
}


class SceneExpectation:
    """Compute object density and the expected-but-absent violation score."""

    def __init__(self, priors: dict = None):
        self.priors = priors or SCENE_OBJECT_PRIORS

    def features(self, detections: list[dict], scene_category: str,
                 frame_area: float) -> dict:
        present = {}
        for d in detections:
            present[d["label"]] = max(present.get(d["label"], 0.0), d["score"])
        n_obj = len(detections)
        total_obj_area = 0.0
        for d in detections:
            x0, y0, x1, y1 = d["box"]
            total_obj_area += max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
        density = n_obj / (frame_area + 1e-9) * 1e5  # objects per 100k px

        prior = self.priors.get(scene_category, {})
        # expected-but-absent: sum over expected objects of P(expected)*(1 - detected)
        absent = 0.0
        detail = {}
        for obj, p in prior.items():
            d_conf = present.get(obj, 0.0)
            contrib = p * (1.0 - d_conf)
            absent += contrib
            detail[obj] = contrib
        # normalise by total prior mass so it's a fraction of expectation violated
        norm = sum(prior.values()) + 1e-9
        return {
            "object_count": int(n_obj),
            "object_density": float(density),
            "object_area_frac": float(min(1.0, total_obj_area / (frame_area + 1e-9))),
            "expected_absent_score": float(absent / norm),   # 0=all present, 1=all missing
            "expected_absent_detail": detail,
        }


def analyze(pil_image, scene_category: str, detector: OpenVocabDetector,
            expectation: SceneExpectation, extra_labels: list[str] = None) -> dict:
    """Full object feature vector. Queries the union of generic + scene-expected labels."""
    prior_labels = list(expectation.priors.get(scene_category, {}).keys())
    generic = ["person", "chair", "door", "window", "light", "plant", "sign",
               "car", "water", "stairs"]
    labels = sorted(set(prior_labels + generic + (extra_labels or [])))
    frame_area = pil_image.size[0] * pil_image.size[1]
    dets = detector.detect(pil_image, labels)
    feats = expectation.features(dets, scene_category, frame_area)
    feats["detections"] = dets
    return feats


SCALAR_KEYS = ["object_count", "object_density", "object_area_frac",
               "expected_absent_score"]

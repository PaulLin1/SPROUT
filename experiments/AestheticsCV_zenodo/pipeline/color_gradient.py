"""
color_gradient.py
=================
Axis 1 of 3: colour-and-gradient analysis.

Every metric below maps to a hypothesis in the study design:

  - palette_lab / hue circular stats  -> cluster fingerprints (Frutiger Aero aqua,
                                          vaporwave magenta/teal, liminal yellow-green)
  - colorfulness (Hasler-Susstrunk)   -> restorative-warm vs degraded-uncanny contrast
  - gradient_smoothness / edge_density -> the Frutiger Aero "everything in one
                                          environment" signature: smooth gradients,
                                          few hard edges. Also separates clean stimuli
                                          from VHS/weirdcore degradation.
  - warm_cool_ratio, sat/val stats    -> valence priors for Table 1.

All features are computed once per image and cached. No human data is touched here.
"""

from __future__ import annotations
import numpy as np
from PIL import Image
from skimage import color as skcolor
from skimage.feature import canny
from skimage.filters import sobel
from sklearn.cluster import KMeans


def _load_rgb(path_or_array, max_side: int = 512) -> np.ndarray:
    """Load -> RGB float [0,1], downscaled so the long side <= max_side."""
    if isinstance(path_or_array, np.ndarray):
        img = path_or_array
        if img.dtype != np.float64 and img.max() > 1.0:
            img = img / 255.0
    else:
        im = Image.open(path_or_array).convert("RGB")
        w, h = im.size
        scale = max_side / max(w, h)
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        img = np.asarray(im, dtype=np.float64) / 255.0
    return img


def dominant_palette(rgb: np.ndarray, k: int = 6, seed: int = 0):
    """K-means palette in CIELAB (perceptually ~uniform), returned with weights.

    Returns dict with lab centroids, rgb centroids, and proportion of pixels each
    cluster owns -- this is the per-image colour fingerprint.
    """
    lab = skcolor.rgb2lab(rgb).reshape(-1, 3)
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(lab)
    counts = np.bincount(km.labels_, minlength=k).astype(float)
    weights = counts / counts.sum()
    order = np.argsort(-weights)
    lab_centroids = km.cluster_centers_[order]
    weights = weights[order]
    rgb_centroids = skcolor.lab2rgb(lab_centroids.reshape(1, -1, 3)).reshape(-1, 3)
    return {
        "palette_lab": lab_centroids,          # (k,3)
        "palette_rgb": np.clip(rgb_centroids, 0, 1),
        "palette_weights": weights,            # (k,) sums to 1
    }


def colorfulness(rgb: np.ndarray) -> float:
    """Hasler & Susstrunk (2003) colourfulness metric. Higher = more colourful."""
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean_root = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std_root + 0.3 * mean_root) * 255.0  # scale to conventional range


def hue_circular_stats(rgb: np.ndarray, sat_floor: float = 0.10):
    """Circular mean hue and dispersion, weighted by saturation.

    Low-saturation pixels carry little hue information, so we weight by S and
    ignore near-grey pixels (the liminal/weirdcore desaturation case).
    """
    hsv = skcolor.rgb2hsv(rgb)
    H = hsv[..., 0].ravel() * 2 * np.pi      # hue -> radians
    S = hsv[..., 1].ravel()
    mask = S >= sat_floor
    if mask.sum() < 16:                       # essentially achromatic image
        return {"hue_mean_deg": np.nan, "hue_dispersion": 1.0,
                "chromatic_fraction": float(mask.mean())}
    w = S[mask]
    C = np.sum(w * np.cos(H[mask])) / w.sum()
    Sn = np.sum(w * np.sin(H[mask])) / w.sum()
    R = np.hypot(C, Sn)                        # mean resultant length in [0,1]
    mean_hue = (np.degrees(np.arctan2(Sn, C))) % 360
    return {
        "hue_mean_deg": float(mean_hue),
        "hue_dispersion": float(1.0 - R),     # 0 = single hue, 1 = hues cancel
        "chromatic_fraction": float(mask.mean()),
    }


def warm_cool_ratio(rgb: np.ndarray) -> float:
    """Fraction of chromatic energy in warm hues vs cool hues.

    Warm ~ [0-60] U [300-360] deg, Cool ~ [180-260] deg (the aqua/blue band that
    dominates Frutiger Aero and the teal of vaporwave). >1 warm-leaning, <1 cool.
    """
    hsv = skcolor.rgb2hsv(rgb)
    H = hsv[..., 0].ravel() * 360.0
    S = hsv[..., 1].ravel()
    warm = ((H < 60) | (H >= 300))
    cool = ((H >= 180) & (H < 260))
    warm_e = (S[warm]).sum()
    cool_e = (S[cool]).sum()
    return float(warm_e / (cool_e + 1e-6))


def gradient_structure(rgb: np.ndarray):
    """Separate *smooth gradient* coverage from *hard edge* density.

    Frutiger Aero is gradient-heavy and edge-sparse (glossy, continuous tone).
    Weirdcore/analog-horror degradation injects high-frequency noise -> high edge
    density and low smooth coverage. This is the structural channel that the RSA
    in the main design expects to live in *early* CNN layers.

    smooth_fraction : proportion of pixels whose local luminance gradient is small
                      but nonzero (i.e. a gentle ramp, not flat, not an edge)
    edge_density    : Canny edge pixel fraction
    grad_energy     : mean Sobel magnitude (overall busyness)
    """
    lum = skcolor.rgb2gray(rgb)
    g = sobel(lum)
    g_norm = g / (g.max() + 1e-9)
    edges = canny(lum, sigma=1.5)
    # "smooth gradient" band: gentle, non-flat, non-edge
    smooth = (g_norm > 0.01) & (g_norm < 0.10) & (~edges)
    return {
        "grad_energy": float(g.mean()),
        "edge_density": float(edges.mean()),
        "smooth_fraction": float(smooth.mean()),
        "gloss_index": float(smooth.mean() / (edges.mean() + 1e-3)),  # FA signature
    }


def saturation_value_stats(rgb: np.ndarray):
    hsv = skcolor.rgb2hsv(rgb)
    S, V = hsv[..., 1], hsv[..., 2]
    return {
        "sat_mean": float(S.mean()), "sat_std": float(S.std()),
        "val_mean": float(V.mean()), "val_std": float(V.std()),
    }


def analyze(path_or_array, k_palette: int = 6) -> dict:
    """Full colour-gradient feature vector for one stimulus."""
    rgb = _load_rgb(path_or_array)
    feats = {}
    pal = dominant_palette(rgb, k=k_palette)
    feats["colorfulness"] = colorfulness(rgb)
    feats.update(hue_circular_stats(rgb))
    feats["warm_cool_ratio"] = warm_cool_ratio(rgb)
    feats.update(gradient_structure(rgb))
    feats.update(saturation_value_stats(rgb))
    # flatten palette into scalar-friendly fields for the feature table
    feats["palette_rgb"] = pal["palette_rgb"]
    feats["palette_weights"] = pal["palette_weights"]
    feats["palette_lab"] = pal["palette_lab"]
    return feats


# scalar-only view for assembling the design matrix (drops the array fields)
SCALAR_KEYS = [
    "colorfulness", "hue_mean_deg", "hue_dispersion", "chromatic_fraction",
    "warm_cool_ratio", "grad_energy", "edge_density", "smooth_fraction",
    "gloss_index", "sat_mean", "sat_std", "val_mean", "val_std",
]


def scalar_vector(feats: dict) -> dict:
    return {k: feats[k] for k in SCALAR_KEYS}

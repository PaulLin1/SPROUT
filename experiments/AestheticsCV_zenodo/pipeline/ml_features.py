"""
ml_features.py
==============
The machine-learning bridge. Turns each image into:

  (1) a deep embedding z  (DINOv2 + CLIP)
  (2) the four perceptual mechanism-proxies phi from the study design:
        phi_typicality      (configural deviation; Diel & Lewis)
        phi_realism_var     (realism inconsistency; MacDorman)
        phi_surprise        (prediction error; active inference)  -- residualised
        phi_entropy         (category uncertainty; Jentsch -- included to falsify)
  (3) a flattened design-matrix row merging axes 1-3 (colour, segmentation, object)
      plus phi, ready for the crossed mixed-model / Bayesian horse-race.

The phi values are computed ONLY against a canonical-scene reference distribution
estimated from the populated control corpus -- never against the human ratings,
preserving the firewall between feature extraction and analysis.

Heavy models load lazily. Everything is cache-friendly (one row per image to disk).
"""

from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------------
# Embeddings
# ----------------------------------------------------------------------------
class Embedder:
    """DINOv2 and/or CLIP image embeddings (lazy load). Report under both."""

    def __init__(self, use_dino=True, use_clip=True, device=None):
        self.use_dino, self.use_clip = use_dino, use_clip
        self.device = device
        self._dino = self._dino_proc = None
        self._clip = self._clip_proc = None

    def _ensure(self):
        import torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.use_dino and self._dino is None:
            from transformers import AutoModel, AutoImageProcessor
            self._dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self._dino = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device).eval()
        if self.use_clip and self._clip is None:
            from transformers import CLIPModel, CLIPProcessor
            self._clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()

    def embed(self, pil_image) -> dict:
        import torch
        self._ensure()
        out = {}
        if self.use_dino:
            inp = self._dino_proc(images=pil_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                z = self._dino(**inp).last_hidden_state[:, 0]      # CLS token
            out["dino"] = z[0].cpu().numpy()
        if self.use_clip:
            inp = self._clip_proc(images=pil_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                z = self._clip.get_image_features(**inp)
            out["clip"] = z[0].cpu().numpy()
        return out


# ----------------------------------------------------------------------------
# Canonical-scene reference model (fit once on the populated control corpus)
# ----------------------------------------------------------------------------
class CanonicalReference:
    """Per-category Gaussian in embedding space + a softmax scene classifier.

    Fit on control (populated everyday) images grouped by scene category. Provides
    phi_typicality (Mahalanobis) and phi_entropy (classifier softmax entropy).
    """

    def __init__(self):
        self.mu = {}          # category -> mean vector
        self.prec = {}        # category -> shrinkage precision matrix
        self.clf = None       # optional logistic scene classifier

    def fit(self, embeddings: np.ndarray, categories: list[str]):
        from sklearn.covariance import LedoitWolf
        from sklearn.linear_model import LogisticRegression
        cats = np.array(categories)
        for c in np.unique(cats):
            X = embeddings[cats == c]
            self.mu[c] = X.mean(0)
            lw = LedoitWolf().fit(X)          # shrinkage: dim >> n_c
            self.prec[c] = lw.precision_
        self.clf = LogisticRegression(max_iter=2000)
        self.clf.fit(embeddings, cats)
        return self

    def phi_typicality(self, z: np.ndarray, category: str) -> float:
        d = z - self.mu[category]
        return float(-(d @ self.prec[category] @ d))   # negative Mahalanobis^2

    def phi_entropy(self, z: np.ndarray) -> float:
        p = self.clf.predict_proba(z.reshape(1, -1))[0]
        p = np.clip(p, 1e-12, 1)
        return float(-(p * np.log(p)).sum())


# ----------------------------------------------------------------------------
# Patch realism variance  (MacDorman realism inconsistency)
# ----------------------------------------------------------------------------
def phi_realism_variance(pil_image, clip_embedder: Embedder, n_tiles: int = 4) -> float:
    """Variance across patches of CLIP alignment to 'a real photograph'.

    High variance = within-image mixing of realism levels (the inconsistency
    MacDorman argues drives eeriness). Orthogonal to the global typicality term.
    """
    import torch
    clip_embedder._ensure()
    W, H = pil_image.size
    scores = []
    for i in range(n_tiles):
        for j in range(n_tiles):
            box = (W*j//n_tiles, H*i//n_tiles, W*(j+1)//n_tiles, H*(i+1)//n_tiles)
            patch = pil_image.crop(box)
            inp = clip_embedder._clip_proc(
                text=["a real photograph", "a synthetic rendering"],
                images=patch, return_tensors="pt", padding=True).to(clip_embedder.device)
            with torch.no_grad():
                logits = clip_embedder._clip(**inp).logits_per_image.softmax(-1)
            scores.append(float(logits[0, 0]))      # P(real) for this patch
    return float(np.var(scores))


# ----------------------------------------------------------------------------
# Surprise (prediction error) -- with the mandatory complexity residualisation
# ----------------------------------------------------------------------------
class SurpriseModel:
    """Reconstruction-NLL surprise under a generative model of canonical scenes.

    Train a VQ-VAE / small latent-diffusion model on the populated control corpus,
    then surprise(x) = reconstruction NLL. CRITICAL: residualise on low-level
    complexity before interpreting as *semantic* surprise (pre-registered), else
    you are measuring JPEG size. We expose both raw and residual forms.
    """

    def __init__(self, recon_model=None):
        self.recon_model = recon_model      # plug your trained model here
        self._resid_coefs = None            # fit complexity -> raw_surprise regression

    @staticmethod
    def low_level_complexity(pil_image) -> np.ndarray:
        """Cheap complexity covariates: compression ratio, edge density, SF slope."""
        import io
        from skimage import color as skcolor
        from skimage.feature import canny
        arr = np.asarray(pil_image.convert("RGB"))
        # JPEG compressibility
        buf = io.BytesIO(); pil_image.convert("RGB").save(buf, "JPEG", quality=85)
        comp_ratio = buf.tell() / (arr.size + 1e-9)
        lum = skcolor.rgb2gray(arr / 255.0)
        edge = canny(lum, sigma=1.5).mean()
        # radial spatial-frequency slope
        F = np.abs(np.fft.fftshift(np.fft.fft2(lum)))
        cy, cx = np.array(F.shape) // 2
        yy, xx = np.mgrid[0:F.shape[0], 0:F.shape[1]]
        r = np.hypot(yy - cy, xx - cx).astype(int)
        rad = np.bincount(r.ravel(), F.ravel()) / (np.bincount(r.ravel()) + 1e-9)
        rmask = (np.arange(len(rad)) > 2) & (rad > 0)
        slope = np.polyfit(np.log(np.arange(len(rad))[rmask] + 1e-9),
                           np.log(rad[rmask] + 1e-9), 1)[0]
        return np.array([comp_ratio, edge, slope])

    def raw_surprise(self, pil_image) -> float:
        if self.recon_model is None:
            raise RuntimeError("plug a trained recon_model (VQ-VAE / latent diffusion)")
        return float(self.recon_model.reconstruction_nll(pil_image))

    def fit_residualizer(self, raw_surprise: np.ndarray, complexity: np.ndarray):
        from sklearn.linear_model import LinearRegression
        self._resid_coefs = LinearRegression().fit(complexity, raw_surprise)
        return self

    def phi_surprise_residual(self, raw_s: float, complexity_row: np.ndarray) -> float:
        pred = self._resid_coefs.predict(complexity_row.reshape(1, -1))[0]
        return float(raw_s - pred)          # semantic surprise = residual


# ----------------------------------------------------------------------------
# Pragmatic phi_surprise: embedding-space reconstruction error (PCA on controls)
# ----------------------------------------------------------------------------
class SurprisePCA:
    """Pragmatic stand-in for the design's VQ-VAE phi_surprise.

    Raw surprise = squared reconstruction error of an image's DINOv2 embedding under
    a low-rank PCA basis learned from the populated *control* corpus only. An image
    whose embedding lies far off the manifold of normal scenes reconstructs poorly ->
    high raw surprise. Computed in embedding space (cheap, no generative model to
    train) and explicitly labelled as the stand-in for the eventual latent-diffusion
    / VQ-VAE reconstruction NLL.

    CRITICAL: raw_surprise must be residualised on low-level complexity (see
    residualize_surprise + SurpriseModel.low_level_complexity) before it is
    interpreted as *semantic* surprise -- otherwise it partly tracks busyness.
    """

    def __init__(self, n_components: int = 64):
        self.n_components = n_components
        self.pca = None

    def fit(self, control_embeddings: np.ndarray):
        from sklearn.decomposition import PCA
        n, d = control_embeddings.shape
        k = max(1, min(self.n_components, n - 1, d))
        self.pca = PCA(n_components=k).fit(control_embeddings)
        return self

    def raw_surprise(self, z: np.ndarray) -> float:
        z2 = z.reshape(1, -1)
        recon = self.pca.inverse_transform(self.pca.transform(z2))
        return float(((z2 - recon) ** 2).sum())


def residualize_surprise(raw: np.ndarray, complexity: np.ndarray,
                         fit_mask: np.ndarray) -> np.ndarray:
    """Residualise raw surprise on low-level complexity covariates.

    The complexity->surprise baseline is fit on `fit_mask` rows (the controls), so
    "expected surprise given complexity" is defined by normal scenes, then residuals
    are returned for ALL rows: phi_surprise = raw - predicted. Preserves the firewall
    (baseline learns the relationship from controls only).
    """
    from sklearn.linear_model import LinearRegression
    raw = np.asarray(raw, float)
    complexity = np.asarray(complexity, float)
    if int(fit_mask.sum()) < complexity.shape[1] + 2:
        fit_mask = np.ones(len(raw), bool)      # fall back to all rows if too few controls
    reg = LinearRegression().fit(complexity[fit_mask], raw[fit_mask])
    return raw - reg.predict(complexity)


# ----------------------------------------------------------------------------
# Assemble one design-matrix row across all three axes + phi
# ----------------------------------------------------------------------------
def assemble_row(image_id: str, scene_category: str, cluster: str,
                 color_feats: dict, seg_feats: dict, obj_feats: dict,
                 phi: dict) -> dict:
    """Merge axis-1/2/3 scalar features + phi into a single flat record."""
    import color_gradient, segmentation, object_analysis
    row = {"image_id": image_id, "scene_category": scene_category, "cluster": cluster}
    row.update({f"col_{k}": color_feats[k] for k in color_gradient.SCALAR_KEYS})
    row.update({f"seg_{k}": seg_feats[k]   for k in segmentation.SCALAR_KEYS})
    row.update({f"obj_{k}": obj_feats[k]   for k in object_analysis.SCALAR_KEYS})
    row.update({f"phi_{k}": v for k, v in phi.items()})
    return row

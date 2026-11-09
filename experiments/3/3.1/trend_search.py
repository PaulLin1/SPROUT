"""Trend-aware image retrieval (the "laptop twee" problem).

"laptop twee" is an aesthetic, not laptops, so plain SigLIP text->image returns
literal laptops. This module scores five channels separately and fuses them late:

  board    fuzzy board<->query relevance r_c (bge dense + rapidfuzz/char-3gram lexical,
           sigmoid-calibrated) x board specificity, aggregated over ALL of a photo's
           boards with noisy-OR (computed in log space through the sparse P x C matrix A).
  visual   query-by-example: trend seed images (from relevant boards and/or explicit
           example images) -> multi-query soft-max kNN + weighted centroid / k-means
           prototypes, with optional de-literalization of object words ("laptop").
  ppr      personalized PageRank over board edges (A row-normalized, both directions)
           + a visual kNN graph over SigLIP image embeddings.
  caption  BM25 + bge dense over captions.
  siglip   direct SigLIP text->image (fallback for non-trend queries such as "car").

Fusion: per-channel z-scores (PPR log-transformed first), weighted sum or RRF, with
weights interpolated by `trend_strength` (how strongly the query matches boards).

Usage
-----
    import trend_search as ts
    index = ts.build_index(df, collections_df, cache_dir="trend_cache")
    res = ts.search("laptop twee")                     # ranked DataFrame
    ts.explain(res.photo_id.iloc[0], "laptop twee")    # boards + per-channel scores
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from PIL import Image

SIGLIP_NAME = "google/siglip2-base-patch16-224"
BGE_NAME = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CHANNELS = ("board", "visual", "ppr", "caption", "siglip")

# Fusion weights at trend_strength = 1 (query clearly names a board trend) and
# trend_strength = 0 (no matching boards -> SigLIP + captions). Interpolated linearly.
TREND_WEIGHTS = {"board": 1.0, "visual": 1.0, "ppr": 0.8, "caption": 0.3, "siglip": 0.2}
FALLBACK_WEIGHTS = {"board": 0.3, "visual": 0.3, "ppr": 0.2, "caption": 0.6, "siglip": 1.0}

# Words dropped before *lexical* title matching only (dense matching sees the full title).
LEX_STOPWORDS = {
    "a", "an", "the", "and", "of", "my", "for", "in", "on", "with",
    "aesthetic", "aesthetics", "vibe", "vibes", "core", "inspo", "inspiration",
    "mood", "moodboard", "board", "ideas", "pics", "photos", "photography",
}

# Concrete object words that trigger de-literalization when they appear in a query.
# COCO-80 plus a few common photo nouns; override with `literal_terms=`.
LITERAL_VOCAB = set("""
person bicycle car motorcycle airplane bus train truck boat traffic light fire hydrant
stop sign parking meter bench bird cat dog horse sheep cow elephant bear zebra giraffe
backpack umbrella handbag tie suitcase frisbee skis snowboard ball kite bat glove
skateboard surfboard racket bottle glass cup fork knife spoon bowl banana apple sandwich
orange broccoli carrot pizza donut cake chair couch sofa plant bed table toilet tv
television laptop computer mouse remote keyboard phone smartphone microwave oven toaster
sink refrigerator fridge book clock vase scissors teddy bear toothbrush desk lamp camera
coffee mug window door flower flowers tree car cars shoe shoes dress hat
""".split())


# --------------------------------------------------------------------------------------
# Devices / models (lazy singletons)
# --------------------------------------------------------------------------------------

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_MODELS: dict = {}


def _siglip():
    if "siglip" not in _MODELS:
        from transformers import AutoModel, AutoProcessor

        device = get_device()
        model = AutoModel.from_pretrained(SIGLIP_NAME).to(device).eval()
        proc = AutoProcessor.from_pretrained(SIGLIP_NAME)
        _MODELS["siglip"] = (model, proc, device)
    return _MODELS["siglip"]


def set_siglip(model, processor, device: str | None = None):
    """Reuse a SigLIP model already loaded in the notebook instead of loading a second copy."""
    _MODELS["siglip"] = (model, processor, device or str(next(model.parameters()).device))


def _bge():
    if "bge" not in _MODELS:
        from sentence_transformers import SentenceTransformer

        _MODELS["bge"] = SentenceTransformer(BGE_NAME, device=get_device())
    return _MODELS["bge"]


def _features(out) -> torch.Tensor:
    """get_*_features returns a tensor in older transformers and a ModelOutput in newer ones
    (where `pooler_output` holds the projected embedding)."""
    if torch.is_tensor(out):
        return out
    for attr in ("pooler_output", "image_embeds", "text_embeds"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    return out[0]


def _load_rgb(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def embed_images_batch(image_paths, batch_size: int = 1024, num_workers: int = 8) -> torch.Tensor:
    """SigLIP image embeddings, L2-normalized, (N, D) float32 on the model device."""
    model, proc, device = _siglip()
    out = []
    with ThreadPoolExecutor(num_workers) as pool:
        for start in range(0, len(image_paths), batch_size):
            images = list(pool.map(_load_rgb, image_paths[start:start + batch_size]))
            inputs = proc(images=images, return_tensors="pt").to(device)
            with torch.inference_mode():
                emb = _features(model.get_image_features(**inputs))
            out.append(F.normalize(emb.float(), p=2, dim=-1))
    return torch.cat(out, dim=0)


def embed_text_batch(texts, batch_size: int = 1024) -> torch.Tensor:
    """SigLIP text embeddings, L2-normalized. SigLIP was trained with max_length=64 padding."""
    model, proc, device = _siglip()
    out = []
    for start in range(0, len(texts), batch_size):
        inputs = proc(
            text=list(texts[start:start + batch_size]),
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            emb = _features(model.get_text_features(**inputs))
        out.append(F.normalize(emb.float(), p=2, dim=-1))
    return torch.cat(out, dim=0)


def embed_bge(texts, query: bool = False, batch_size: int = 256) -> np.ndarray:
    texts = [BGE_QUERY_PREFIX + t for t in texts] if query else list(texts)
    return _bge().encode(
        texts, batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=len(texts) > 5000,
    ).astype(np.float32)


# --------------------------------------------------------------------------------------
# Text normalization
# --------------------------------------------------------------------------------------

def normalize_title(s) -> str:
    """Lowercase, NFKC, strip emoji/punctuation (-> space), collapse whitespace."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = "".join(ch if unicodedata.category(ch)[0] in "LN" else " " for ch in s)
    return re.sub(r"\s+", " ", s).strip()


def _lex_form(s: str) -> str:
    return " ".join(t for t in s.split() if t not in LEX_STOPWORDS)


def _tokenize(s: str) -> list[str]:
    return re.findall(r"\w+", str(s).lower())


# --------------------------------------------------------------------------------------
# Disk cache helpers (incremental: only new keys get embedded)
# --------------------------------------------------------------------------------------

def _cached_embed(cache_dir: Path, name: str, keys: list[str], embed_fn, verbose=True) -> np.ndarray:
    """Return embeddings for `keys` (in order), embedding only keys missing from the cache."""
    kpath, epath = cache_dir / f"{name}_keys.npy", cache_dir / f"{name}_emb.npy"
    if kpath.exists() and epath.exists():
        old_keys = np.load(kpath, allow_pickle=False).tolist()
        old_emb = np.load(epath, allow_pickle=False)
    else:
        old_keys, old_emb = [], None
    pos = {k: i for i, k in enumerate(old_keys)}
    uniq = list(dict.fromkeys(keys))
    missing = [k for k in uniq if k not in pos]
    if missing:
        if verbose:
            print(f"[{name}] embedding {len(missing)} new / {len(uniq)} total")
        new_emb = np.asarray(embed_fn(missing), dtype=np.float32)
        emb = new_emb if old_emb is None else np.concatenate([old_emb, new_emb])
        all_keys = old_keys + missing
        np.save(kpath, np.array(all_keys, dtype=str))
        np.save(epath, emb)
        pos = {k: i for i, k in enumerate(all_keys)}
    else:
        emb = old_emb
        if verbose:
            print(f"[{name}] cache hit ({len(uniq)})")
    return emb[[pos[k] for k in keys]]


def _fingerprint(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(repr(p).encode())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------------------
# Graph helpers
# --------------------------------------------------------------------------------------

def build_knn_graph(X: torch.Tensor, k: int = 10, chunk: int = 4096) -> sp.csr_matrix:
    """Cosine kNN over L2-normalized rows of X, computed in GPU chunks with topk.
    Returns raw similarities (clamped at 0), self excluded, as a P x P sparse matrix."""
    P = X.shape[0]
    k = min(k, P - 1)
    rows, cols, vals = [], [], []
    for start in range(0, P, chunk):
        end = min(start + chunk, P)
        sims = X[start:end] @ X.T
        ar = torch.arange(end - start, device=X.device)
        sims[ar, ar + start] = -float("inf")
        v, i = sims.topk(k, dim=1)
        rows.append(np.repeat(np.arange(start, end), k))
        cols.append(i.cpu().numpy().ravel())
        vals.append(v.clamp(min=0).float().cpu().numpy().ravel())
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(P, P)
    )


def row_normalize(M: sp.spmatrix) -> sp.csr_matrix:
    M = sp.csr_matrix(M, dtype=np.float32)
    s = np.asarray(M.sum(axis=1)).ravel()
    inv = np.divide(1.0, s, out=np.zeros_like(s), where=s > 0)
    return sp.diags(inv) @ M


def _z(x: np.ndarray, clip: float | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    if sd < 1e-12:
        return np.zeros_like(x)
    z = (x - x.mean()) / sd
    return np.clip(z, -clip, clip) if clip else z


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------------------

@dataclass
class BoardParams:
    w_dense: float = 0.6
    w_lex: float = 0.4
    t: float = 0.62        # sigmoid center on w_dense*dense + w_lex*lex
    tau: float = 0.05      # sigmoid temperature
    r_floor: float = 0.05  # r_c below this -> 0 (stops noisy-OR from accumulating noise)
    cov_floor: float = 0.5 # raw *= cov_floor + (1-cov_floor)*coverage; 1.0 disables the gate


@dataclass
class SearchParams:
    max_seeds: int = 256
    seed_min_frac: float = 0.25   # seeds need s_board >= frac * max s_board
    seed_min_abs: float = 0.05
    lse_temp: float = 0.05
    beta: float = 0.85
    gamma: float = 0.5
    ppr_iters: int = 20
    z_clip: float = 6.0
    rrf_k: int = 60
    ts_scale: float = 0.5         # trend_strength = clip(sum top-5 eff / ts_scale, 0, 1)


@dataclass
class Seeds:
    vecs: torch.Tensor           # (S, D) normalized
    w: np.ndarray                # (S,)
    nodes: np.ndarray            # (S,) corpus index or -1 for external images
    source: list = field(default_factory=list)  # "board" | "example"

    def __len__(self):
        return len(self.w)


class TrendIndex:
    def __init__(self, df, collections_df, cache_dir="trend_cache", knn_k=10,
                 batch_size=1024, verbose=True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.board_params = BoardParams()
        self.params = SearchParams()
        self.device = get_device()
        self._memo: dict = {}

        # ---- photos
        self.df = df.reset_index(drop=True).copy()
        self.df["photo_id"] = self.df["photo_id"].astype(str)
        if self.df["photo_id"].duplicated().any():
            raise ValueError("duplicate photo_id in df")
        self.photo_ids = self.df["photo_id"].tolist()
        self.photo_to_idx = {p: i for i, p in enumerate(self.photo_ids)}
        self.image_paths = self.df["image_path"].astype(str).tolist()
        self.path_to_idx = {p: i for i, p in enumerate(self.image_paths)}
        self.captions = self.df["caption"].fillna("").astype(str).tolist()
        P = len(self.df)

        # ---- photo x collection incidence A (P x C)
        m = collections_df[["photo_id", "collection_id", "collection_title"]].copy()
        m["photo_id"] = m["photo_id"].astype(str)
        m["collection_id"] = m["collection_id"].astype(str)
        m = m[m["photo_id"].isin(self.photo_to_idx)].drop_duplicates(["photo_id", "collection_id"])
        self.collections = (
            m.drop_duplicates("collection_id")[["collection_id", "collection_title"]]
            .reset_index(drop=True)
        )
        self.collections["collection_title"] = self.collections["collection_title"].fillna("").astype(str)
        self.collection_to_idx = {c: i for i, c in enumerate(self.collections["collection_id"])}
        C = len(self.collections)
        rows = m["photo_id"].map(self.photo_to_idx).to_numpy()
        cols = m["collection_id"].map(self.collection_to_idx).to_numpy()
        fp_A = _fingerprint(self.photo_ids, list(self.collections["collection_id"]), len(m))
        A_path = self.cache_dir / f"A_{fp_A}.npz"
        if A_path.exists():
            self.A = sp.load_npz(A_path).tocsr()
        else:
            self.A = sp.csr_matrix((np.ones(len(m), np.float32), (rows, cols)), shape=(P, C))
            sp.save_npz(A_path, self.A)
        self.board_size = np.asarray(self.A.sum(axis=0)).ravel()
        self.has_board = np.asarray(self.A.getnnz(axis=1)).ravel() > 0
        spec = 1.0 / np.log(2.0 + self.board_size)
        self.spec = spec / (1.0 / np.log(3.0))  # size 1 -> 1.0
        self.A_T = self.A.T.tocsr()
        self.Bp = row_normalize(self.A)          # photo -> its boards
        self.Bc = row_normalize(self.A_T)        # board -> its photos (divides by board size)

        # ---- board titles: normalize once per collection, embed each unique title once
        norm = [normalize_title(t) for t in self.collections["collection_title"]]
        self.collections["title_norm"] = norm
        self.titles_u = list(dict.fromkeys(norm))
        t_pos = {t: i for i, t in enumerate(self.titles_u)}
        self.title_idx = np.array([t_pos[t] for t in norm], dtype=np.int64)
        self.title_emb = _cached_embed(self.cache_dir, "bge_titles", self.titles_u,
                                       lambda xs: embed_bge(xs), verbose)
        self.titles_lex = [_lex_form(t) for t in self.titles_u]
        self.titles_nospace = [t.replace(" ", "") for t in self.titles_lex]
        vocab = {}
        r_, c_ = [], []
        for i, t in enumerate(self.titles_lex):
            for tok in set(t.split()):
                r_.append(i)
                c_.append(vocab.setdefault(tok, len(vocab)))
        self.title_vocab = list(vocab)
        self.title_vocab_M = sp.csr_matrix((np.ones(len(r_), np.float32), (r_, c_)),
                                           shape=(len(self.titles_lex), max(len(vocab), 1)))
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=1)
        self.title_tfidf = self.tfidf.fit_transform([t or " " for t in self.titles_lex])

        # ---- SigLIP image embeddings (cached per image_path)
        img = _cached_embed(self.cache_dir, "siglip_img", self.image_paths,
                            lambda xs: embed_images_batch(xs, batch_size).cpu().numpy(), verbose)
        self.img = img
        self.img_t = torch.from_numpy(img).to(self.device)
        self.img_mean = F.normalize(self.img_t.mean(0), dim=0)

        # ---- visual kNN graph
        fp_knn = _fingerprint(self.image_paths, knn_k)
        knn_path = self.cache_dir / f"knn_{fp_knn}.npz"
        if knn_path.exists():
            knn = sp.load_npz(knn_path).tocsr()
            if verbose:
                print(f"[knn] cache hit ({knn_path.name})")
        else:
            if verbose:
                print(f"[knn] building k={knn_k} over {P} images on {self.device}")
            knn = build_knn_graph(self.img_t, knn_k)
            sp.save_npz(knn_path, knn)
        self.knn_raw = knn
        self.K_T = row_normalize(knn).T.tocsr()

        # ---- captions: bge dense (cached) + BM25
        self.cap_emb = _cached_embed(self.cache_dir, "bge_captions", self.captions,
                                     lambda xs: embed_bge(xs), verbose)
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi([_tokenize(c) or ["_"] for c in self.captions])

        if verbose:
            print(f"index: P={P} photos, C={C} boards ({len(self.titles_u)} unique titles), "
                  f"A nnz={self.A.nnz}, kNN nnz={knn.nnz}, device={self.device}")

    # ---------------------------------------------------------------- board channel
    def board_relevance(self, query: str) -> pd.DataFrame:
        """Per-collection relevance r_c, specificity spec_c and eff_c = r_c * spec_c."""
        from rapidfuzz import fuzz, process

        bp = self.board_params
        qn = normalize_title(query)
        q_lex = _lex_form(qn) or qn
        dense_u = self.title_emb @ embed_bge([qn])[0]
        sort_r = process.cdist([q_lex], self.titles_lex, scorer=fuzz.token_sort_ratio, workers=-1)[0] / 100
        nosp_r = process.cdist([q_lex.replace(" ", "")], self.titles_nospace, scorer=fuzz.ratio, workers=-1)[0] / 100
        tfidf_r = np.asarray((self.title_tfidf @ self.tfidf.transform([q_lex]).T).todense()).ravel()
        lex_u = 0.5 * np.maximum(sort_r, nosp_r) + 0.5 * tfidf_r
        empty = np.array([len(t) == 0 for t in self.titles_lex])
        dense_u[empty] = 0
        lex_u[empty] = 0
        cov_u = self._coverage(q_lex)
        raw_u = (bp.w_dense * dense_u + bp.w_lex * lex_u) * (bp.cov_floor + (1 - bp.cov_floor) * cov_u)
        r_u = _sigmoid((raw_u - bp.t) / bp.tau)
        r_u[r_u < bp.r_floor] = 0.0

        ti = self.title_idx
        out = self.collections[["collection_id", "collection_title"]].copy()
        out["size"] = self.board_size.astype(int)
        out["dense"] = dense_u[ti]
        out["lex"] = lex_u[ti]
        out["cov"] = cov_u[ti]
        out["raw"] = raw_u[ti]
        out["r"] = r_u[ti]
        out["spec"] = self.spec
        out["eff"] = out["r"] * out["spec"]
        return out

    def _coverage(self, q_lex: str) -> np.ndarray:
        """IDF-weighted fraction of query tokens a title covers, per unique title.

        Dense similarity alone rates "Laptops" close to "laptop twee"; the rare token ("twee")
        is what makes the trend. Each query token is matched fuzzily against the title's tokens
        (fuzz.ratio) and as a substring of the de-spaced title (catches "laptoptwee"); tokens are
        weighted by IDF over board titles, so covering only the common word scores low."""
        from rapidfuzz import fuzz, process

        q_toks = q_lex.split()
        U = len(self.titles_u)
        if not q_toks or U == 0:
            return np.ones(U)
        tok_r = process.cdist(q_toks, self.title_vocab, scorer=fuzz.ratio, workers=-1) / 100  # (nq, V)
        match = np.zeros((len(q_toks), U))
        for j in range(len(q_toks)):
            match[j] = self.title_vocab_M.multiply(tok_r[j][None, :]).max(axis=1).toarray().ravel()
        long_q = [j for j, t in enumerate(q_toks) if len(t) >= 4]
        if long_q:
            part = process.cdist([q_toks[j] for j in long_q], self.titles_nospace,
                                 scorer=fuzz.partial_ratio, workers=-1) / 100
            match[long_q] = np.maximum(match[long_q], part)
        match = np.clip((match - 0.6) / 0.4, 0, 1)            # ratio 0.6 -> 0, 1.0 -> 1
        df_tok = (match >= 0.8).sum(axis=1)
        idf = np.log((U + 1) / (df_tok + 1)) + 1
        return (idf[:, None] * match).sum(0) / idf.sum()

    def _board_scores(self, eff: np.ndarray) -> np.ndarray:
        """Noisy-OR over all of a photo's boards: 1 - prod(1 - eff_c), in log space."""
        eff = np.clip(eff, 0.0, 1.0 - 1e-6)
        return 1.0 - np.exp(self.A @ np.log1p(-eff))

    def board_breakdown(self, query: str, photo_ids=None, min_eff: float = 1e-4) -> pd.DataFrame:
        """Long table: for each photo (default: every photo with s_board > 0), which boards
        contributed and by how much. `share` splits s_board proportionally to -log(1-eff)."""
        rel = self.board_relevance(query)
        eff = rel["eff"].to_numpy()
        s_board = self._board_scores(eff)
        if photo_ids is None:
            pidx = np.flatnonzero(s_board > 0)
        else:
            pidx = np.array([self.photo_to_idx[str(p)] for p in photo_ids])
        sub = self.A[pidx].tocoo()
        e = eff[sub.col]
        keep = e >= min_eff
        p_loc, c, e = sub.row[keep], sub.col[keep], e[keep]
        logc = -np.log1p(-np.clip(e, 0, 1 - 1e-6))
        out = pd.DataFrame({
            "photo_id": np.array(self.photo_ids, dtype=object)[pidx[p_loc]],
            "collection_id": rel["collection_id"].to_numpy()[c],
            "collection_title": rel["collection_title"].to_numpy()[c],
            "board_size": rel["size"].to_numpy()[c],
            "r": rel["r"].to_numpy()[c],
            "spec": rel["spec"].to_numpy()[c],
            "eff": e,
            "s_board": s_board[pidx[p_loc]],
        })
        tot = out.groupby("photo_id")["eff"].transform(lambda x: -np.log1p(-np.clip(x, 0, 1 - 1e-6)).sum())
        out["share"] = np.where(tot > 0, logc / tot.to_numpy() * out["s_board"], 0.0)
        return out.sort_values(["s_board", "photo_id", "eff"], ascending=[False, True, False]).reset_index(drop=True)

    def trend_groups(self, query: str, top_n: int = 200, distance: float = 0.15) -> pd.DataFrame:
        """Cluster near-duplicate titles among the top_n most relevant boards (average-linkage
        agglomerative). Distance = 1 - max(bge cosine, token_sort_ratio, de-spaced ratio), so
        "laptoptwee" / "twee laptop" join "laptop twee" even when bge alone says > 0.15.
        `in_query_group` marks the cluster of the most relevant title."""
        from rapidfuzz import fuzz, process
        from sklearn.cluster import AgglomerativeClustering

        rel = self.board_relevance(query)
        rel["title_norm"] = self.collections["title_norm"]
        rel["title_u"] = self.title_idx
        u = (rel.sort_values("raw", ascending=False)
             .drop_duplicates("title_u").head(top_n).reset_index(drop=True))
        if len(u) < 2:
            u["group"] = 0
        else:
            ti = u["title_u"].to_numpy()
            E = self.title_emb[ti]
            lex = [self.titles_lex[i] for i in ti]
            nosp = [self.titles_nospace[i] for i in ti]
            sim = np.maximum.reduce([
                E @ E.T,
                process.cdist(lex, lex, scorer=fuzz.token_sort_ratio, workers=-1) / 100,
                process.cdist(nosp, nosp, scorer=fuzz.ratio, workers=-1) / 100,
            ])
            dist = np.clip(1 - sim, 0, None)
            np.fill_diagonal(dist, 0)
            u["group"] = AgglomerativeClustering(
                n_clusters=None, metric="precomputed", linkage="average", distance_threshold=distance,
            ).fit_predict(dist)
        u["in_query_group"] = u["group"] == u["group"].iloc[0]
        agg = rel.groupby("title_u").agg(n_boards=("collection_id", "size"), n_photos=("size", "sum"))
        u = u.join(agg, on="title_u")
        return u[["title_norm", "group", "in_query_group", "r", "dense", "lex", "n_boards", "n_photos"]]

    # ---------------------------------------------------------------- seeds / visual
    def _seeds(self, s_board: np.ndarray, example_images=None) -> Seeds:
        sp_ = self.params
        nodes, w, src = [], [], []
        if s_board.max() > 0:
            thr = max(sp_.seed_min_abs, sp_.seed_min_frac * s_board.max())
            idx = np.argsort(-s_board)[: sp_.max_seeds]
            idx = idx[s_board[idx] >= thr]
            nodes += idx.tolist()
            w += s_board[idx].tolist()
            src += ["board"] * len(idx)
        ext_paths = []
        for ex in example_images or []:
            ex = str(ex)
            i = self.path_to_idx.get(ex, self.photo_to_idx.get(ex))
            if i is None:
                ext_paths.append(ex)
            elif i in nodes:
                w[nodes.index(i)] = 1.0
                src[nodes.index(i)] = "example"
            else:
                nodes.append(i)
                w.append(1.0)
                src.append("example")
        D = self.img_t.shape[1]
        vecs = self.img_t[torch.as_tensor(nodes, dtype=torch.long, device=self.device)] if nodes \
            else torch.empty((0, D), device=self.device)
        node_arr = np.array(nodes, dtype=np.int64)
        if ext_paths:
            ext = embed_images_batch(ext_paths).to(self.device)
            vecs = torch.cat([vecs, ext])
            w += [1.0] * len(ext_paths)
            node_arr = np.concatenate([node_arr, -np.ones(len(ext_paths), np.int64)])
            src += ["example"] * len(ext_paths)
        return Seeds(vecs=vecs, w=np.asarray(w, np.float32), nodes=node_arr, source=src)

    def literal_terms(self, query: str) -> list[str]:
        toks = normalize_title(query).split()
        return [t for t in toks if t in LITERAL_VOCAB or (t.endswith("s") and t[:-1] in LITERAL_VOCAB)]

    def literal_directions(self, terms: list[str], space: str = "text") -> torch.Tensor | None:
        """Unit directions to project out. 'text': SigLIP text("a photo of a <term>") as in
        the spec. 'image': mean of the corpus images that text retrieves minus the corpus mean,
        i.e. the same concept expressed in image space (sidesteps the modality gap)."""
        if not terms:
            return None
        L = embed_text_batch([f"a photo of a {t}" for t in terms]).to(self.device)
        if space == "image":
            k = int(np.clip(len(self.photo_ids) // 200, 5, 50))
            dirs = []
            for l in L:
                top = (self.img_t @ l).topk(k).indices
                dirs.append(F.normalize(self.img_t[top].mean(0) - self.img_t.mean(0), dim=0))
            L = torch.stack(dirs)
        Q, _ = torch.linalg.qr(L.T.float().cpu())     # orthonormal basis (D, m)
        return Q.T.to(self.device)

    def _visual(self, seeds: Seeds, literal: torch.Tensor | None = None) -> dict:
        P = self.img_t.shape[0]
        if len(seeds) == 0:
            z = np.zeros(P)
            return {"visual": z, "visual_knn": z, "visual_proto": z, "n_protos": 0}
        S = seeds.vecs
        w = torch.as_tensor(seeds.w, device=self.device)
        logw = torch.log(w / w.sum())
        T = self.params.lse_temp
        knn_lse, top3 = [], []
        for start in range(0, P, 8192):
            sims = self.img_t[start:start + 8192] @ S.T
            knn_lse.append(T * torch.logsumexp(sims / T + logw, dim=1))
            k3 = min(3, sims.shape[1])
            top3.append(sims.topk(k3, dim=1).values.mean(1))
        knn_lse = torch.cat(knn_lse).cpu().numpy()
        top3 = torch.cat(top3).cpu().numpy()

        # prototypes: weighted centroid + k-means centers (k grows with seed count)
        protos = [F.normalize((S * w[:, None]).sum(0), dim=0)]
        n = len(seeds)
        k = 1 if n < 4 else 2 if n < 12 else 3
        if k > 1:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(
                S.cpu().numpy(), sample_weight=seeds.w)
            protos += list(F.normalize(torch.as_tensor(km.cluster_centers_, device=self.device,
                                                       dtype=S.dtype), dim=1))
        Pm = torch.stack(protos)
        if literal is not None:
            Pm = F.normalize(Pm - (Pm @ literal.T) @ literal, dim=1)
        proto = (self.img_t @ Pm.T).max(dim=1).values.cpu().numpy()
        return {"visual": 0.5 * knn_lse + 0.5 * proto, "visual_knn": knn_lse,
                "visual_top3": top3, "visual_proto": proto, "n_protos": len(protos)}

    # ---------------------------------------------------------------- PPR
    def _ppr(self, seeds: Seeds) -> np.ndarray:
        P = len(self.photo_ids)
        v = np.zeros(P)
        if len(seeds) == 0:
            return v
        inside = seeds.nodes >= 0
        np.add.at(v, seeds.nodes[inside], seeds.w[inside])
        if (~inside).any():  # external examples -> spread over their nearest corpus images
            ext = seeds.vecs[torch.as_tensor(np.flatnonzero(~inside), device=self.device)]
            sims = ext @ self.img_t.T
            vals, idx = sims.topk(min(10, P), dim=1)
            vals = vals.clamp(min=0)
            vals = vals / vals.sum(1, keepdim=True).clamp(min=1e-12)
            for wi, vi, ii in zip(seeds.w[~inside], vals.cpu().numpy(), idx.cpu().numpy()):
                np.add.at(v, ii, wi * vi)
        v /= v.sum()
        p = self.params
        g = p.gamma * self.has_board  # photos without boards send all mass through kNN edges
        s = v.copy()
        for _ in range(p.ppr_iters):
            board_hop = self.Bc.T @ (self.Bp.T @ (g * s))
            knn_hop = self.K_T @ ((1 - g) * s)
            s = (1 - p.beta) * v + p.beta * (board_hop + knn_hop)
        return s

    # ---------------------------------------------------------------- caption / siglip
    def _caption(self, query: str) -> dict:
        bm = np.asarray(self.bm25.get_scores(_tokenize(query)), dtype=np.float64)
        dense = self.cap_emb @ embed_bge([query], query=True)[0]
        return {"caption": 0.5 * _z(bm) + 0.5 * _z(dense), "caption_bm25": bm, "caption_dense": dense}

    def _siglip(self, query: str, template: str = "{}", literal_text: torch.Tensor | None = None) -> np.ndarray:
        q = embed_text_batch([template.format(query)]).to(self.device)[0]
        if literal_text is not None:  # remove "a photo of a laptop" from the query vector
            q = F.normalize(q - literal_text.T @ (literal_text @ q), dim=0)
        return (self.img_t @ q).cpu().numpy()

    # ---------------------------------------------------------------- search
    def search(self, query: str, example_images=None, weights: dict | None = None,
               fusion: str = "z", deliteralize: str | bool = False, literal_terms=None,
               drop_channels=(), exclude_ids=None, siglip_template: str = "{}",
               top_k: int | None = None) -> pd.DataFrame:
        """Rank every photo for `query`.

        weights        None -> adaptive (interpolated by trend_strength); a dict -> used as-is,
                       missing channels get 0 (e.g. {"siglip": 1} = direct SigLIP only).
        fusion         "z" (weighted sum of z-scores) or "rrf" (weighted reciprocal rank fusion).
        deliteralize   False | "text" | "image" (True == "text"): project object-word directions
                       ("laptop") out of the visual prototypes ("text": SigLIP text direction as
                       specced; "image": the same concept estimated in image space). Also projects
                       the text direction out of the SigLIP query and drops the word from the
                       caption query, since those channels read the literal word too.
        drop_channels  zero these channels after adaptive weighting.
        exclude_ids    photo_ids removed from the returned ranking (e.g. seeds during eval).
        """
        key = (query, tuple(map(str, example_images or ())), repr(weights), fusion, str(deliteralize),
               repr(literal_terms), tuple(drop_channels), siglip_template)
        if key in self._memo:
            res = self._memo[key]
        else:
            res = self._search(query, example_images, weights, fusion, deliteralize,
                               literal_terms, drop_channels, siglip_template)
            if len(self._memo) > 64:
                self._memo.pop(next(iter(self._memo)))
            self._memo[key] = res
        out = res
        if exclude_ids is not None:
            out = out[~out["photo_id"].isin(set(map(str, exclude_ids)))].copy()
            out["rank"] = np.arange(1, len(out) + 1)
            out.attrs = res.attrs
        return out.head(top_k) if top_k else out

    def _search(self, query, example_images, weights, fusion, deliteralize, literal_terms,
                drop_channels, siglip_template) -> pd.DataFrame:
        P = len(self.photo_ids)
        rel = self.board_relevance(query)
        eff = rel["eff"].to_numpy()
        s_board = self._board_scores(eff)
        trend_strength = float(np.clip(np.sort(eff)[::-1][:5].sum() / self.params.ts_scale, 0, 1))

        seeds = self._seeds(s_board, example_images)
        if deliteralize is True:
            deliteralize = "text"
        terms = (literal_terms if literal_terms is not None else self.literal_terms(query)) if deliteralize else []
        literal = self.literal_directions(terms, deliteralize) if terms else None
        literal_text = (literal if deliteralize == "text" else self.literal_directions(terms, "text")) \
            if terms else None
        vis = self._visual(seeds, literal)
        ppr = self._ppr(seeds)
        # De-literalization also reaches the text channels: the literal word is dropped from the
        # caption query and its SigLIP text direction is projected out of the query vector.
        cap_query = query
        if terms:
            rest = [t for t in normalize_title(query).split() if t not in set(terms)]
            cap_query = " ".join(rest) or query
        cap = self._caption(cap_query)
        sig = self._siglip(query, siglip_template, literal_text)

        raw = {"board": s_board, "visual": vis["visual"], "ppr": ppr,
               "caption": cap["caption"], "siglip": sig}
        # Example-only queries carry trend evidence even with no matching boards.
        strength = max(trend_strength, 1.0 if example_images else 0.0)
        if weights is None:
            w = {c: strength * TREND_WEIGHTS[c] + (1 - strength) * FALLBACK_WEIGHTS[c] for c in CHANNELS}
        else:
            w = {c: float(weights.get(c, 0.0)) for c in CHANNELS}
        if len(seeds) == 0:
            w["visual"] = w["ppr"] = 0.0
        for c in drop_channels:
            w[c] = 0.0
        active = [c for c in CHANNELS if w[c] > 0]

        fused_in = dict(raw)
        fused_in["ppr"] = np.log(ppr + 1e-3 / P)
        contrib = {}
        if fusion == "rrf":
            for c in CHANNELS:
                x = fused_in[c]
                order = np.argsort(-x, kind="stable")
                rank = np.empty(P)
                rank[order] = np.arange(1, P + 1)
                valid = x > 0 if c == "board" else np.ones(P, bool)  # zero board score = no evidence
                contrib[c] = np.where(valid, w[c] / (self.params.rrf_k + rank), 0.0)
        else:
            clip = None if len(active) == 1 else self.params.z_clip
            for c in CHANNELS:
                contrib[c] = w[c] * _z(fused_in[c], clip)
        final = sum(contrib[c] for c in CHANNELS)

        out = pd.DataFrame({
            "photo_id": self.photo_ids,
            "image_path": self.image_paths,
            "caption": self.captions,
        })
        if "collection_title" in self.df:
            out["collection_title"] = self.df["collection_title"].to_numpy()
        for c in CHANNELS:
            out[f"s_{c}"] = raw[c]
        out["s_visual_knn"] = vis["visual_knn"]
        out["s_visual_proto"] = vis["visual_proto"]
        out["s_caption_bm25"] = cap["caption_bm25"]
        out["s_caption_dense"] = cap["caption_dense"]
        for c in CHANNELS:
            out[f"c_{c}"] = contrib[c]
        out["score"] = final
        cmat = np.stack([contrib[c] for c in CHANNELS], 1)
        out["top_channel"] = np.array(CHANNELS)[cmat.argmax(1)]
        is_seed = np.zeros(P, bool)
        is_seed[seeds.nodes[seeds.nodes >= 0]] = True
        out["is_seed"] = is_seed
        out = out.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
        out["rank"] = np.arange(1, P + 1)
        out.attrs = {
            "query": query, "trend_strength": trend_strength, "weights": w, "fusion": fusion,
            "n_seeds": len(seeds), "n_example_seeds": int(sum(s == "example" for s in seeds.source)),
            "seed_ids": [self.photo_ids[i] for i in seeds.nodes if i >= 0],
            "literal_terms": terms, "deliteralize": deliteralize, "n_protos": vis["n_protos"],
        }
        return out

    def explain(self, photo_id, query: str, **search_kwargs) -> dict:
        """Contributing boards and per-channel scores for one photo."""
        photo_id = str(photo_id)
        res = self.search(query, **search_kwargs)
        row = res[res["photo_id"] == photo_id].iloc[0]
        w = res.attrs["weights"]
        channels = pd.DataFrame({
            "raw": [row[f"s_{c}"] for c in CHANNELS],
            "weight": [w[c] for c in CHANNELS],
            "contribution": [row[f"c_{c}"] for c in CHANNELS],
            "channel_rank": [int((res[f"s_{c}"] > row[f"s_{c}"]).sum() + 1) for c in CHANNELS],
        }, index=list(CHANNELS))
        boards = self.board_breakdown(query, [photo_id], min_eff=0.0)
        boards = boards.drop(columns=["photo_id", "s_board"])
        return {
            "photo_id": photo_id, "rank": int(row["rank"]), "score": float(row["score"]),
            "top_channel": row["top_channel"], "caption": row["caption"],
            "trend_strength": res.attrs["trend_strength"], "is_seed": bool(row["is_seed"]),
            "channels": channels, "boards": boards,
        }


# --------------------------------------------------------------------------------------
# Module-level API
# --------------------------------------------------------------------------------------

_INDEX: TrendIndex | None = None


def build_index(df, collections_df, cache_dir="trend_cache", knn_k=10, batch_size=1024,
                verbose=True) -> TrendIndex:
    """Build (or load from `cache_dir`) every embedding, A, the kNN graph, board text
    embeddings and lexical indices. Also becomes the default index for search()/explain()."""
    global _INDEX
    _INDEX = TrendIndex(df, collections_df, cache_dir, knn_k, batch_size, verbose)
    return _INDEX


def _idx(index):
    if index is None:
        if _INDEX is None:
            raise RuntimeError("call build_index() first")
        return _INDEX
    return index


def search(query, example_images=None, weights=None, index=None, **kw) -> pd.DataFrame:
    return _idx(index).search(query, example_images=example_images, weights=weights, **kw)


def explain(photo_id, query, index=None, **kw) -> dict:
    return _idx(index).explain(photo_id, query, **kw)


# --------------------------------------------------------------------------------------
# Data prep for experiments
# --------------------------------------------------------------------------------------

def inject_trend(df, collections_df, inject_df, image_root: str, boards: dict,
                 n: int = 10, id_prefix: str = "laptop_twee_"):
    """Append the first `n` rows of inject_df to df (all at once) and add board membership.

    boards: {collection_id: (title, [inject row positions])}. Rows not listed in any board
    are added to df with NO board membership (held-out). Returns (df, collections_df) with
    df["collection_title"] recomputed from the new membership.
    """
    inj = inject_df.iloc[:n].reset_index(drop=True)
    new_rows = pd.DataFrame({
        "photo_id": [f"{id_prefix}{i}" for i in range(len(inj))],
        "image_path": [f"{image_root.rstrip('/')}/{p}" for p in inj["image_path"]],
        "caption": inj["caption"].astype(str).tolist(),
    })
    base = df.drop(columns=["collection_title"], errors="ignore")
    base = base[~base["photo_id"].astype(str).str.startswith(id_prefix)]
    df2 = pd.concat([base, new_rows], ignore_index=True)

    coll = collections_df[~collections_df["photo_id"].astype(str).str.startswith(id_prefix)].copy()
    coll["collection_id"] = coll["collection_id"].astype(str)
    add = [{"photo_id": f"{id_prefix}{i}", "collection_id": cid, "collection_title": title}
           for cid, (title, members) in boards.items() for i in members]
    coll = pd.concat([coll, pd.DataFrame(add)], ignore_index=True)

    titles = (coll.groupby("photo_id")["collection_title"]
              .apply(lambda x: ", ".join(x.dropna().astype(str).unique())).rename("collection_title"))
    df2 = df2.merge(titles, left_on="photo_id", right_index=True, how="left")
    return df2, coll


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------

def _ranks_of(res: pd.DataFrame, ids, score_col: str) -> list[float]:
    """Rank of each id within `res` by `score_col`. Ties are ranked pessimistically (last
    position of the tie), so an all-zero channel can't score recall hits by tying."""
    from scipy.stats import rankdata

    r = rankdata(-res[score_col].to_numpy(), method="max")
    pos = dict(zip(res["photo_id"], r))
    return [float(pos[i]) for i in ids]


HELDOUT_METHODS = {
    "siglip only": dict(weights={"siglip": 1}),
    "caption only": dict(weights={"caption": 1}),
    "board only": dict(weights={"board": 1}),
    "visual only": dict(weights={"visual": 1}),
    "visual only + delit(text)": dict(weights={"visual": 1}, deliteralize="text"),
    "visual only + delit(image)": dict(weights={"visual": 1}, deliteralize="image"),
    "ppr only": dict(weights={"ppr": 1}),
    "fused": dict(),
    "fused + delit(text)": dict(deliteralize="text"),
    "fused + delit(image)": dict(deliteralize="image"),
    "fused (rrf)": dict(fusion="rrf"),
}


def evaluate_heldout(index: TrendIndex, query: str, heldout_ids, board_ids,
                     ks=(25, 50, 100), methods=HELDOUT_METHODS) -> pd.DataFrame:
    """Rank of held-out trend images (in df, no boards) after removing seeds and the in-board
    trend images from every ranking, so every method ranks the same candidate set."""
    base = index.search(query)
    exclude = set(map(str, board_ids)) | set(base.attrs["seed_ids"])
    heldout_ids = [str(i) for i in heldout_ids]
    assert not exclude & set(heldout_ids), "held-out images leaked into seeds"
    rows = []
    for name, kw in methods.items():
        res = index.search(query, exclude_ids=exclude, **kw)
        ranks = _ranks_of(res, heldout_ids, "score")
        row = {"test": "held-out", "method": name, "n_candidates": len(res),
               "ranks": sorted(int(round(r)) for r in ranks), "median_rank": float(np.median(ranks))}
        for k in ks:
            row[f"recall@{k}"] = float(np.mean([r <= k for r in ranks]))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_fuzzy(index: TrendIndex, query: str, variant_ids: dict, injected_ids,
                   ks=(10, 25)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """variant_ids: {collection_id: title}. Returns (per-variant board table, retrieval rows)."""
    rel = index.board_relevance(query)
    rel["r_rank"] = rel["raw"].rank(ascending=False, method="min").astype(int)
    per = rel[rel["collection_id"].isin(variant_ids)][
        ["collection_id", "collection_title", "size", "dense", "lex", "cov", "raw", "r", "spec", "eff", "r_rank"]]
    groups = index.trend_groups(query)
    grp_titles = set(groups.loc[groups["in_query_group"], "title_norm"])
    per = per.assign(in_trend_group=[normalize_title(t) in grp_titles for t in per["collection_title"]],
                     match=lambda d: d["r"] >= 0.5)
    injected_ids = [str(i) for i in injected_ids]
    rows = []
    for name, kw in {"board only": dict(weights={"board": 1}), "fused": dict()}.items():
        res = index.search(query, **kw)
        ranks = _ranks_of(res, injected_ids, "score")
        row = {"test": "fuzzy-title", "method": name, "n_candidates": len(res),
               "ranks": sorted(int(round(r)) for r in ranks), "median_rank": float(np.median(ranks))}
        for k in ks:
            row[f"recall@{k}"] = float(np.mean([r <= k for r in ranks]))
        row["variants_matched"] = f"{int(per['match'].sum())}/{len(per)}"
        rows.append(row)
    return per.reset_index(drop=True), pd.DataFrame(rows)


def evaluate_regression(index: TrendIndex, query: str = "car", pattern: str = r"\bcars?\b",
                        k: int = 25) -> pd.DataFrame:
    """precision@k against a weak caption label. Note: the caption channel reads the same
    captions as the label, so 'fused - caption' is the leakage-free comparison."""
    rx = re.compile(pattern, re.IGNORECASE)
    methods = {"siglip only": dict(weights={"siglip": 1}), "fused": dict(),
               "fused - caption": dict(drop_channels=("caption",)), "fused (rrf)": dict(fusion="rrf")}
    rows = []
    base = None
    for name, kw in methods.items():
        res = index.search(query, top_k=k, **kw)
        p = float(np.mean([bool(rx.search(c)) for c in res["caption"]]))
        base = p if base is None else base
        rows.append({"test": f"regression '{query}'", "method": name, f"precision@{k}": p,
                     "trend_strength": res.attrs["trend_strength"],
                     "not_worse_than_siglip": p >= base})
    return pd.DataFrame(rows)


def print_board_calibration(index: TrendIndex, queries, top: int = 50):
    """Print the top boards per query so t / tau can be calibrated by eye."""
    with pd.option_context("display.max_rows", top + 5, "display.width", 200,
                           "display.max_colwidth", 40):
        for q in queries:
            rel = index.board_relevance(q).sort_values("raw", ascending=False).head(top)
            print(f"\n=== {q!r}  (t={index.board_params.t}, tau={index.board_params.tau})")
            print(rel[["collection_title", "size", "dense", "lex", "cov", "raw", "r", "spec", "eff"]]
                  .round(3).to_string(index=False))


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------

def plot_grid(res: pd.DataFrame, title: str | None = None, n: int = 25, ncols: int = 5,
              highlight_ids=None, thumb: int = 256):
    """n images in a grid, labelled with rank, top channel and score. Highlighted ids in green."""
    import matplotlib.pyplot as plt

    top = res.head(n)
    nrows = math.ceil(len(top) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.9 * nrows))
    hl = set(map(str, highlight_ids or []))
    for ax, (_, row) in zip(np.ravel(axes), top.iterrows()):
        try:
            im = _load_rgb(row["image_path"])
            im.thumbnail((thumb, thumb))
            ax.imshow(im)
        except Exception as e:  # missing file shouldn't kill the grid
            ax.text(0.5, 0.5, type(e).__name__, ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        color = "green" if row["photo_id"] in hl else "black"
        ax.set_xlabel(f"#{row['rank']} {row['top_channel']} ({row['score']:.2f})",
                      fontsize=8, color=color)
        if row["photo_id"] in hl:
            for s in ax.spines.values():
                s.set_edgecolor("green")
                s.set_linewidth(3)
    for ax in np.ravel(axes)[len(top):]:
        ax.axis("off")
    a = res.attrs
    sub = f"trend_strength={a.get('trend_strength', 0):.2f}  seeds={a.get('n_seeds', 0)}"
    if a.get("literal_terms"):
        sub += f"  delit={a['deliteralize']}:{a['literal_terms']}"
    fig.suptitle(f"{title or a.get('query', '')}\n{sub}", fontsize=11)
    fig.tight_layout()
    return fig

"""Text → embedding (a list of floating-point numbers).

Two implementations live here:

  - `BGEEmbedder`     — real semantic embedder using sentence-transformers
                        with BAAI/bge-small-en-v1.5 by default. ~384-dim
                        output. Lazy-loaded; the model weights download on
                        first use (~130MB) and need internet access.

  - `HashingEmbedder` — deterministic, offline stub. Hashes tokens into a
                        fixed-dim bag-of-words vector and L2-normalizes.
                        NOT semantic — it just captures word overlap, like
                        a poor-man's TF without the IDF. Used in tests and
                        in environments where the BGE model can't be
                        downloaded.

Both implement the `Embedder` protocol so the rest of the system doesn't
care which is in use.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable

from regrag.retrieval.bm25 import tokenize


@runtime_checkable
class Embedder(Protocol):
    """The retrieval layer's embedder contract.

    `embed_texts` and `embed_query` are separate methods because some
    embedders (older BGE, E5) use different prompts for indexing vs.
    querying. BGE-v1.5 doesn't, but the API leaves the door open.
    """

    @property
    def dim(self) -> int:
        """Dimensionality of the output vectors."""

    @property
    def name(self) -> str:
        """Stable identifier for this embedder. Recorded in the audit log
        so a future replay knows which embedder produced the index."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents (chunks)."""

    def embed_query(self, query: str) -> list[float]:
        """Embed a single user query."""


# ---- HashingEmbedder ----


class HashingEmbedder:
    """Deterministic offline-only stub embedder.

    Hashes each token to a fixed position in a `dim`-sized vector,
    increments that position, then L2-normalizes the result. Cosine
    similarity between two such vectors approximates token-overlap —
    NOT meaning. Use only for tests and offline development.
    """

    def __init__(self, *, dim: int = 256) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"hashing-{self._dim}"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed_one(query)

    def _embed_one(self, text: str) -> list[float]:
        tokens = tokenize(text)
        vec = [0.0] * self._dim
        if not tokens:
            return vec
        for token in tokens:
            position = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self._dim
            vec[position] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


# ---- BGEEmbedder ----


class BGEEmbedder:
    """Real semantic embedder via sentence-transformers.

    Default model: BAAI/bge-small-en-v1.5 (384-dim, ~130MB download on first
    use). Lazy-loads — importing this module never triggers the download;
    only the first `embed_*` call does.

    For data-residency-restricted environments, point `model_name` at a
    locally-hosted mirror or a model already cached on disk.
    """

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: object | None = None
        self._dim_cache: int | None = None

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        assert self._dim_cache is not None
        return self._dim_cache

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        assert self._model is not None
        # BGE-v1.5 docs: no prefix needed for documents.
        embeddings = self._model.encode(  # type: ignore[attr-defined]
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, query: str) -> list[float]:
        self._ensure_loaded()
        assert self._model is not None
        # BGE-v1.5 docs: queries don't need a prefix either; older BGE did.
        embedding = self._model.encode(  # type: ignore[attr-defined]
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        return list(map(float, embedding))

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "BGEEmbedder requires sentence-transformers. Install it via "
                "`uv sync` (it's in the default deps), or use HashingEmbedder "
                "for offline-only environments."
            ) from exc
        self._model = SentenceTransformer(self._model_name)
        # sentence-transformers isn't fully typed; mypy sees `_model` as
        # `object | Any` and can't resolve the method. Blanket ignore is
        # fine here — it's the canonical SentenceTransformer API surface.
        self._dim_cache = int(
            self._model.get_sentence_embedding_dimension()  # type: ignore
        )

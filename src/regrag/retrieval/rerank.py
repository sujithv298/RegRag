"""Rerankers — final relevance pass over the fused top-N.

Three implementations:

  - `NoOpReranker`            — returns input unchanged. Used when reranking
                                is disabled or in tests where we want the
                                fusion output untouched.
  - `LexicalOverlapReranker`  — offline-only stub. Scores each chunk by
                                token overlap between query and the chunk's
                                source_text (the rule body, no prefix).
                                Crude but useful for testing the
                                architecture without huggingface.co access.
  - `CrossEncoderReranker`    — real cross-encoder via sentence-transformers
                                (`BAAI/bge-reranker-base` by default).
                                Lazy-loaded; weights download on first use.

Rerankers operate on the small (<=20-ish) post-fusion candidate set, so
they can afford to be expensive. The cross-encoder reads each (query,
chunk) pair through a transformer — much more accurate than embedder-based
similarity, way too slow for the full corpus.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from regrag.retrieval.bm25 import tokenize, tokenize_query
from regrag.retrieval.types import ScoredChunk


@runtime_checkable
class Reranker(Protocol):
    """Re-score a list of `ScoredChunk` for a given query."""

    @property
    def name(self) -> str:
        """Stable identifier; recorded in the audit log."""

    def rerank(
        self, query: str, scored_chunks: list[ScoredChunk], *, top_k: int = 5
    ) -> list[ScoredChunk]:
        """Return up to `top_k` chunks re-scored by this reranker, sorted desc."""


# ---- NoOpReranker ----


class NoOpReranker:
    """Identity reranker. Returns the first `top_k` of input unchanged."""

    @property
    def name(self) -> str:
        return "noop"

    def rerank(
        self, query: str, scored_chunks: list[ScoredChunk], *, top_k: int = 5
    ) -> list[ScoredChunk]:
        del query  # unused
        return scored_chunks[:top_k]


# ---- LexicalOverlapReranker ----


class LexicalOverlapReranker:
    """Offline stub: re-score by query-token overlap with source_text.

    Scores in [0, 1]: fraction of *query* tokens that appear in the chunk
    body. Uses `source_text`, not `text`, so the citation/heading prefix
    we glued on for retrieval doesn't inflate scores.

    NOT a real cross-encoder. Use `CrossEncoderReranker` in production.
    """

    @property
    def name(self) -> str:
        return "lexical-overlap"

    def rerank(
        self, query: str, scored_chunks: list[ScoredChunk], *, top_k: int = 5
    ) -> list[ScoredChunk]:
        if not scored_chunks:
            return []
        query_tokens = set(tokenize_query(query))
        if not query_tokens:
            return [
                ScoredChunk(chunk=sc.chunk, score=0.0, retriever="rerank")
                for sc in scored_chunks[:top_k]
            ]
        rescored: list[ScoredChunk] = []
        for sc in scored_chunks:
            body_tokens = set(tokenize(sc.chunk.source_text))
            overlap = len(query_tokens & body_tokens)
            score = overlap / len(query_tokens)
            rescored.append(
                ScoredChunk(chunk=sc.chunk, score=score, retriever="rerank")
            )
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:top_k]


# ---- CrossEncoderReranker ----


class CrossEncoderReranker:
    """Real cross-encoder via sentence-transformers `CrossEncoder`.

    Lazy-loaded. Default model: `BAAI/bge-reranker-base` (~280MB on first
    use, downloaded from huggingface.co). For data-residency-restricted
    environments, point `model_name` at a locally-hosted path.

    Scores are the model's raw relevance logits — higher = more relevant.
    Don't compare them directly to RRF or BM25 scores; they live on
    yet another scale.
    """

    DEFAULT_MODEL = "BAAI/bge-reranker-base"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: object | None = None

    @property
    def name(self) -> str:
        return self._model_name

    def rerank(
        self, query: str, scored_chunks: list[ScoredChunk], *, top_k: int = 5
    ) -> list[ScoredChunk]:
        if not scored_chunks:
            return []
        self._ensure_loaded()
        assert self._model is not None
        # Cross-encoders read (query, document) pairs together. We pass the
        # full chunk.text (with prefix) — the model is robust to small
        # amounts of formatting noise and the prefix carries signal.
        pairs = [[query, sc.chunk.text] for sc in scored_chunks]
        scores = self._model.predict(  # type: ignore[attr-defined]
            pairs, show_progress_bar=False
        )
        rescored = [
            ScoredChunk(chunk=sc.chunk, score=float(s), retriever="rerank")
            for sc, s in zip(scored_chunks, scores, strict=True)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:top_k]

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "CrossEncoderReranker requires sentence-transformers. Install "
                "via `uv sync`, or use LexicalOverlapReranker / NoOpReranker "
                "for offline-only environments."
            ) from exc
        self._model = CrossEncoder(self._model_name)

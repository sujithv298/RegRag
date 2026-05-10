"""Hybrid retriever — BM25 + dense fused via RRF, then reranked.

Pipeline per `search(query, k)`:

    1. BM25Index.search(query, k=candidate_k)         → list[ScoredChunk] (bm25)
    2. DenseRetriever.search(query, k=candidate_k)    → list[ScoredChunk] (dense)
    3. rrf_fuse([bm25, dense], top_k=candidate_k)     → list[ScoredChunk] (hybrid)
    4. Reranker.rerank(query, fused, top_k=k)         → list[ScoredChunk] (rerank)

Defaults are tuned for the v1 single-Part scale (~1k chunks):
  - candidate_k=20 — pull 20 from each retriever; the fused set is up to 40
    unique chunks before rerank.
  - rrf_k=60      — standard.
  - top_k passed to .search() is the user-facing final size, default 5.

The reranker is pluggable (defaults to NoOpReranker so the pipeline runs
identically to RRF-only when no reranker is configured). For production,
inject a CrossEncoderReranker.
"""

from __future__ import annotations

from regrag.chunking import Chunk
from regrag.retrieval.bm25 import BM25Index
from regrag.retrieval.dense import DenseRetriever
from regrag.retrieval.fusion import DEFAULT_K_RRF, rrf_fuse
from regrag.retrieval.rerank import NoOpReranker, Reranker
from regrag.retrieval.types import ScoredChunk

DEFAULT_CANDIDATE_K = 20


class HybridRetriever:
    """Compose BM25 + dense + (optional) reranker into a single search API."""

    def __init__(
        self,
        *,
        bm25: BM25Index,
        dense: DenseRetriever,
        reranker: Reranker | None = None,
        rrf_k: int = DEFAULT_K_RRF,
        candidate_k: int = DEFAULT_CANDIDATE_K,
    ) -> None:
        self._bm25 = bm25
        self._dense = dense
        self._reranker: Reranker = reranker or NoOpReranker()
        self._rrf_k = rrf_k
        self._candidate_k = candidate_k

    @property
    def reranker(self) -> Reranker:
        return self._reranker

    def add(self, chunks: list[Chunk]) -> None:
        """Add to both underlying retrievers in lockstep."""
        self._bm25.add(chunks)
        self._dense.add(chunks)

    def search(self, query: str, *, k: int = 5) -> list[ScoredChunk]:
        """Run BM25 + dense, fuse with RRF, rerank, return top-k."""
        bm25_results = self._bm25.search(query, k=self._candidate_k)
        dense_results = self._dense.search(query, k=self._candidate_k)
        fused = rrf_fuse(
            [bm25_results, dense_results],
            k_rrf=self._rrf_k,
            top_k=self._candidate_k,
        )
        return self._reranker.rerank(query, fused, top_k=k)

    def __len__(self) -> int:
        # Both retrievers should be in lockstep; report the dense store's count.
        return len(self._dense)

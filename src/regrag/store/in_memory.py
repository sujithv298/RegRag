"""In-memory VectorStore — the default for v1.

A list of `(chunk, embedding)` pairs plus a brute-force cosine-similarity
loop at search time. O(N * dim) per query, which is fine for the v1
single-Part scale (~1k chunks, 384-dim → ~400k multiply-adds per query,
sub-millisecond on any laptop).

When the corpus grows past ~50k chunks or latency starts to matter, swap
in `ChromaStore` (already implemented) without changing any other code.
"""

from __future__ import annotations

import math

from regrag.chunking import Chunk
from regrag.retrieval.types import ScoredChunk


class InMemoryVectorStore:
    """Pure-Python brute-force vector store. No external deps."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._embeddings: list[list[float]] = []

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings length mismatch: "
                f"{len(chunks)} chunks vs {len(embeddings)} embeddings"
            )
        self._chunks.extend(chunks)
        self._embeddings.extend(embeddings)

    def search(
        self, query_embedding: list[float], *, k: int = 10
    ) -> list[ScoredChunk]:
        if not self._embeddings:
            return []
        scored = [
            (
                _cosine_similarity(query_embedding, emb),
                chunk,
            )
            for emb, chunk in zip(self._embeddings, self._chunks, strict=True)
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            ScoredChunk(chunk=chunk, score=float(score), retriever="dense")
            for score, chunk in scored[:k]
        ]

    def __len__(self) -> int:
        return len(self._chunks)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    For L2-normalized inputs (both our default embedders produce them),
    this reduces to the dot product. We compute the norms anyway for
    correctness when callers pass un-normalized vectors.
    """
    if len(a) != len(b):
        raise ValueError(f"vector dim mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)

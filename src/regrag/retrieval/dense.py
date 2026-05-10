"""Dense retriever: embed the query, ask the store for nearest neighbors.

Thin wrapper that ties `Embedder` and `VectorStore` together. Most of the
real work is delegated:

    DenseRetriever.add(chunks)
        ↓
    embedder.embed_texts(chunk.text for chunk in chunks)
        ↓
    store.add(chunks, embeddings)

    DenseRetriever.search(query)
        ↓
    embedder.embed_query(query)
        ↓
    store.search(query_embedding, k=k)
        ↓
    list[ScoredChunk]  (retriever="dense")
"""

from __future__ import annotations

from regrag.chunking import Chunk
from regrag.retrieval.embeddings import Embedder
from regrag.retrieval.types import ScoredChunk
from regrag.store.base import VectorStore


class DenseRetriever:
    """Compose an Embedder + VectorStore into a query-able retriever."""

    def __init__(self, *, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def store(self) -> VectorStore:
        return self._store

    def add(self, chunks: list[Chunk]) -> None:
        """Embed `chunks` and persist them in the underlying store."""
        if not chunks:
            return
        embeddings = self._embedder.embed_texts([c.text for c in chunks])
        self._store.add(chunks, embeddings)

    def search(self, query: str, *, k: int = 10) -> list[ScoredChunk]:
        """Embed the query, return the top-k nearest chunks from the store."""
        if not query.strip():
            return []
        query_embedding = self._embedder.embed_query(query)
        return self._store.search(query_embedding, k=k)

    def __len__(self) -> int:
        return len(self._store)

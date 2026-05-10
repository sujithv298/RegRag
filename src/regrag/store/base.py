"""VectorStore protocol — the contract every store implementation honors.

A vector store is a database that holds (chunk, embedding) pairs and
answers nearest-neighbor queries: given a query embedding, return the K
chunks whose embeddings are closest, with a similarity score for each.

Why a protocol rather than a base class:
- Forkers integrating their own infra (pgvector, Pinecone, Weaviate,
  Qdrant, an internal bank-hosted FAISS service) implement this protocol
  without inheriting from anything in this codebase.
- The retrieval layer depends only on this protocol — no cycles, no
  Chroma-specific imports leaking into the rest of the system.

Score semantics:
    Higher = more similar. Cosine similarity, range [-1, 1] in theory but
    typically [0, 1] for L2-normalized embeddings (which both our default
    embedders produce).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from regrag.chunking import Chunk
from regrag.retrieval.types import ScoredChunk


@runtime_checkable
class VectorStore(Protocol):
    """Add chunks with their embeddings; query by embedding."""

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert chunks alongside their pre-computed embeddings.

        Raises:
            ValueError: if `len(chunks) != len(embeddings)`.
        """

    def search(self, query_embedding: list[float], *, k: int = 10) -> list[ScoredChunk]:
        """Return the top-k chunks most similar to `query_embedding`.

        Each ScoredChunk carries `retriever="dense"`.
        """

    def __len__(self) -> int:
        """Number of chunks currently in the store."""

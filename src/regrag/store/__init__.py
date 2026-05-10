"""Vector + metadata persistence.

Default in v1: `InMemoryVectorStore` — pure Python, zero infra. Adequate
for any single-Part corpus (~1k chunks).

Production swap-in: `ChromaStore` — real vector database, persistent or
ephemeral, scales to ~millions of chunks.

Both implement the `VectorStore` protocol in `base.py`. Forkers needing
pgvector / Pinecone / Weaviate / Qdrant / an internal bank service
implement that protocol once and the rest of the codebase doesn't change.
"""

from __future__ import annotations

from regrag.store.base import VectorStore
from regrag.store.chroma import ChromaStore
from regrag.store.in_memory import InMemoryVectorStore

__all__ = ["ChromaStore", "InMemoryVectorStore", "VectorStore"]

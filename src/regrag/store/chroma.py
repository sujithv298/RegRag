"""ChromaDB-backed VectorStore.

Real-vector-database swap-in for `InMemoryVectorStore`. Use this when:
  - the corpus grows past what brute-force search comfortably handles, or
  - you want persistent storage that survives process restarts, or
  - you need the metadata-filter capabilities Chroma exposes.

Persistence:
    `ChromaStore(persist_path="./.chroma")` writes to disk. Omitting
    `persist_path` opens an ephemeral in-memory client (still backed by
    Chroma's HNSW index, just not persisted).

Distance vs. similarity:
    Chroma returns *distances* (lower = closer). We convert to a
    similarity score (higher = better) so `ScoredChunk.score` follows
    the same convention as `InMemoryVectorStore`.
"""

from __future__ import annotations

from typing import Any

from regrag.chunking import Chunk
from regrag.retrieval.types import ScoredChunk

DEFAULT_COLLECTION = "regrag_chunks"


class ChromaStore:
    """ChromaDB-backed implementation of the VectorStore protocol."""

    def __init__(
        self,
        *,
        persist_path: str | None = None,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "ChromaStore requires chromadb. Install via `uv sync` (it's in the default deps)."
            ) from exc

        if persist_path:
            self._client: Any = chromadb.PersistentClient(path=persist_path)
        else:
            self._client = chromadb.EphemeralClient()
        self._collection: Any = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        # Chroma roundtrips Chunk metadata as a flat primitives dict, which
        # would lose `paragraph_path` (a tuple) and `is_interpretation`
        # (interacts with comment_id). Easier to keep a Python-side map
        # of chunk_id -> Chunk and re-hydrate from there on search.
        self._chunks_by_id: dict[str, Chunk] = {}

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings length mismatch: "
                f"{len(chunks)} chunks vs {len(embeddings)} embeddings"
            )
        if not chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {"citation_path": c.citation_path, "section": c.section or ""} for c in chunks
            ],
        )
        for c in chunks:
            self._chunks_by_id[c.chunk_id] = c

    def search(self, query_embedding: list[float], *, k: int = 10) -> list[ScoredChunk]:
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )
        ids_lists = result.get("ids") or []
        distances_lists = result.get("distances") or []
        if not ids_lists or not ids_lists[0]:
            return []
        scored: list[ScoredChunk] = []
        for chunk_id, distance in zip(ids_lists[0], distances_lists[0], strict=True):
            chunk = self._chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            # Chroma cosine "distance" is 1 - cosine_similarity, so invert.
            similarity = 1.0 - float(distance)
            scored.append(ScoredChunk(chunk=chunk, score=similarity, retriever="dense"))
        return scored

    def __len__(self) -> int:
        return int(self._collection.count())

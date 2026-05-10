"""Shared types for the retrieval layer.

A `ScoredChunk` is what every retriever returns: a chunk plus a score plus
a label saying which retriever produced the score. The label matters because
in Phase 4c we'll fuse multiple retrievers' results, and we want to be able
to inspect "which retriever surfaced this chunk?" in the audit log.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from regrag.chunking import Chunk

RetrieverLabel = Literal["bm25", "dense", "hybrid", "rerank"]


class ScoredChunk(BaseModel):
    """One chunk with the score a retriever assigned to it."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float = Field(description="Retriever-specific score. Higher = more relevant.")
    retriever: RetrieverLabel = Field(
        description="Which retriever produced this score. Tracked through the audit log."
    )

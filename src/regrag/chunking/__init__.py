"""Hierarchy-preserving chunking.

A naive RAG over CFR text uses a recursive character splitter and loses
paragraph addressing. We do the opposite: each leaf paragraph becomes one
chunk, with the citation address and section heading prepended so retrieval
matches on either, and splitting only happens *within* a leaf when it
exceeds the embedder's context window.

Two outputs from this module:

    HierarchyNode  — the typed leaf model produced by ingestion (lives in
                     hierarchy.py because chunking is its primary consumer).
    Chunk          — the searchable unit produced by chunking, indexed by
                     retrieval, cited by the verifier.

See `docs/chunking.md` for the chunking strategy and edge cases.
"""

from __future__ import annotations

from regrag.chunking.chunk import Chunk
from regrag.chunking.chunker import DEFAULT_MAX_TOKENS, chunk_nodes
from regrag.chunking.hierarchy import HierarchyNode

__all__ = ["DEFAULT_MAX_TOKENS", "Chunk", "HierarchyNode", "chunk_nodes"]

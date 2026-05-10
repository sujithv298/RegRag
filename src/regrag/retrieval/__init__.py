"""Retrieval: turn a user query into a ranked list of chunks.

Phase 4a (done): BM25 keyword retrieval.
Phase 4b (done): dense retrieval via embeddings + a vector store.
Phase 4c (current): hybrid (BM25 + dense fused via RRF) + cross-encoder rerank.

Public surface:

  - `BM25Index`            — keyword retriever (Phase 4a)
  - `DenseRetriever`       — semantic retriever (Phase 4b)
  - `HybridRetriever`      — fused + reranked composition (Phase 4c)
  - Embedders, vector-store types, and rerankers reachable via submodules
"""

from __future__ import annotations

from regrag.retrieval.bm25 import BM25Index, tokenize, tokenize_query
from regrag.retrieval.dense import DenseRetriever
from regrag.retrieval.embeddings import BGEEmbedder, Embedder, HashingEmbedder
from regrag.retrieval.fusion import rrf_fuse
from regrag.retrieval.hybrid import HybridRetriever
from regrag.retrieval.rerank import (
    CrossEncoderReranker,
    LexicalOverlapReranker,
    NoOpReranker,
    Reranker,
)
from regrag.retrieval.types import ScoredChunk

__all__ = [
    "BGEEmbedder",
    "BM25Index",
    "CrossEncoderReranker",
    "DenseRetriever",
    "Embedder",
    "HashingEmbedder",
    "HybridRetriever",
    "LexicalOverlapReranker",
    "NoOpReranker",
    "Reranker",
    "ScoredChunk",
    "rrf_fuse",
    "tokenize",
    "tokenize_query",
]

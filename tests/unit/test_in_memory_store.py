"""Tests for InMemoryVectorStore — the default v1 vector store."""

from __future__ import annotations

import math

import pytest

from regrag.chunking.hierarchy import HierarchyNode
from regrag.chunking import chunk_nodes
from regrag.store import InMemoryVectorStore, VectorStore


def _make_chunks() -> list:
    nodes = [
        HierarchyNode(
            title=12, part=1005, section="1005.1", paragraph_path=("a",),
            section_heading="§ 1005.1 Authority and purpose.", text="Authority text.",
        ),
        HierarchyNode(
            title=12, part=1005, section="1005.1", paragraph_path=("b",),
            section_heading="§ 1005.1 Authority and purpose.", text="Purpose text.",
        ),
    ]
    return chunk_nodes(nodes)


def test_in_memory_store_implements_protocol() -> None:
    assert isinstance(InMemoryVectorStore(), VectorStore)


def test_empty_store_returns_empty_search() -> None:
    store = InMemoryVectorStore()
    assert store.search([0.5, 0.5], k=5) == []


def test_add_then_len_reflects_count() -> None:
    chunks = _make_chunks()
    store = InMemoryVectorStore()
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])
    assert len(store) == 2


def test_add_mismatched_lengths_raises() -> None:
    chunks = _make_chunks()
    store = InMemoryVectorStore()
    with pytest.raises(ValueError):
        store.add(chunks, [[1.0, 0.0]])  # 2 chunks, 1 embedding


def test_search_returns_chunks_in_similarity_order() -> None:
    chunks = _make_chunks()
    store = InMemoryVectorStore()
    # First chunk's embedding is "north"; second is "east".
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])
    # A query pointing slightly north-of-east should rank chunk 0 first
    # if it's closer in cosine, chunk 1 first if closer to east.
    results_north = store.search([1.0, 0.1], k=2)
    results_east = store.search([0.1, 1.0], k=2)
    assert results_north[0].chunk.citation_path == chunks[0].citation_path
    assert results_east[0].chunk.citation_path == chunks[1].citation_path


def test_search_scores_use_cosine_similarity() -> None:
    """Identical query and chunk embedding → similarity 1.0 (modulo float)."""
    chunks = _make_chunks()
    store = InMemoryVectorStore()
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])
    results = store.search([1.0, 0.0], k=1)
    assert results
    assert math.isclose(results[0].score, 1.0, abs_tol=1e-9)


def test_search_respects_k() -> None:
    chunks = _make_chunks()
    store = InMemoryVectorStore()
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])
    results = store.search([0.5, 0.5], k=1)
    assert len(results) == 1


def test_scored_chunks_carry_dense_label() -> None:
    chunks = _make_chunks()
    store = InMemoryVectorStore()
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])
    results = store.search([1.0, 0.0], k=1)
    assert all(r.retriever == "dense" for r in results)

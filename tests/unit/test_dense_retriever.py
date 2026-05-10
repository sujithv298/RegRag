"""Tests for DenseRetriever against the real Reg E excerpt fixture.

Uses HashingEmbedder (offline) + InMemoryVectorStore. The point of these
tests is the *architecture* — that DenseRetriever wires Embedder and
VectorStore correctly. Semantic-quality tests against the real BGE model
belong in tests/integration/ once the sandbox / CI has network access.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from regrag.chunking import chunk_nodes
from regrag.ingest import parse_part_xml
from regrag.retrieval import DenseRetriever, HashingEmbedder, ScoredChunk
from regrag.store import InMemoryVectorStore

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "reg_e_part_1005_excerpt.xml"


@pytest.fixture(scope="module")
def retriever() -> DenseRetriever:
    nodes = parse_part_xml(FIXTURE.read_bytes())
    chunks = chunk_nodes(nodes)
    r = DenseRetriever(embedder=HashingEmbedder(), store=InMemoryVectorStore())
    r.add(chunks)
    return r


def test_retriever_holds_all_chunks(retriever: DenseRetriever) -> None:
    assert len(retriever) == 10


def test_search_returns_scored_chunks(retriever: DenseRetriever) -> None:
    results = retriever.search("consumer liability for unauthorized transfers", k=5)
    assert results
    assert all(isinstance(r, ScoredChunk) for r in results)
    assert all(r.retriever == "dense" for r in results)


def test_search_results_are_sorted_by_score(retriever: DenseRetriever) -> None:
    results = retriever.search("consumer liability", k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_returns_no_results(retriever: DenseRetriever) -> None:
    assert retriever.search("", k=5) == []
    assert retriever.search("   ", k=5) == []


def test_search_finds_relevant_section(retriever: DenseRetriever) -> None:
    """With the lexical hashing stub, queries that share tokens with the
    target section's text/heading should rank that section's chunks high."""
    results = retriever.search("liability for unauthorized transfers", k=3)
    assert results
    citations = [r.chunk.citation_path for r in results]
    # All top results should be from §1005.6 (the section about liability).
    assert all("1005.6" in c or "Comment 6" in c for c in citations)


def test_add_with_no_chunks_is_a_noop() -> None:
    r = DenseRetriever(embedder=HashingEmbedder(), store=InMemoryVectorStore())
    r.add([])
    assert len(r) == 0

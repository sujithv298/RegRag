"""Tests for `regrag.retrieval.bm25`.

Two clusters of tests:
  1. Tokenization — the part most likely to silently break legal-text search.
  2. End-to-end search behavior against the real Reg E excerpt fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from regrag.chunking import chunk_nodes
from regrag.ingest import parse_part_xml
from regrag.retrieval import BM25Index, ScoredChunk, tokenize

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "reg_e_part_1005_excerpt.xml"


# ---- Tokenization ----


def test_tokenize_lowercases() -> None:
    assert tokenize("Consumer Liability") == ["consumer", "liability"]


def test_tokenize_keeps_section_numbers_intact() -> None:
    """1005.6 must be one token, not three. Otherwise citation-style queries
    would have to spell "1005 dot 6" and we'd lose the most useful exact
    match in legal text."""
    assert tokenize("12 CFR 1005.6") == ["12", "cfr", "1005.6"]


def test_tokenize_splits_paragraph_designators() -> None:
    """(b)(1) tokenizes as separate b and 1. Parens become token boundaries.
    The chunker already lifts the paragraph_path out of the body so this
    duplication isn't a problem in practice."""
    assert tokenize("§ 1005.6(b)(1) Timely") == ["1005.6", "b", "1", "timely"]


def test_tokenize_drops_punctuation_and_section_symbol() -> None:
    assert "§" not in tokenize("§ 1005.1")


# ---- End-to-end search ----


@pytest.fixture(scope="module")
def index() -> BM25Index:
    nodes = parse_part_xml(FIXTURE.read_bytes())
    chunks = chunk_nodes(nodes)
    idx = BM25Index()
    idx.add(chunks)
    return idx


def test_empty_index_returns_empty() -> None:
    idx = BM25Index()
    assert idx.search("anything") == []


def test_search_returns_scored_chunks(index: BM25Index) -> None:
    results = index.search("liability unauthorized transfer", k=5)
    assert len(results) > 0
    assert all(isinstance(r, ScoredChunk) for r in results)
    assert all(r.retriever == "bm25" for r in results)
    # Scores monotonically non-increasing.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_citation_query_lands_in_correct_section_family(index: BM25Index) -> None:
    """A citation-style query should rank chunks from that section's family on top.

    Note: pure BM25 cannot reliably pick the *exact* paragraph for a
    citation-only query in a small corpus where many chunks share the
    citation-prefix tokens (everything in §1005.6 has '1005.6' in its
    citation line). Phase 4b's dense embeddings + Phase 4c's reranker
    are what tighten this from 'right family' to 'right paragraph'."""
    results = index.search("1005.6(b)(1)", k=3)
    assert results
    citations = [r.chunk.citation_path for r in results]
    # Top 3 should all be from the §1005.6 family (rule or interpretation).
    assert all("1005.6" in c or "Comment 6" in c for c in citations)


def test_content_query_finds_specific_paragraph(index: BM25Index) -> None:
    """A query rich in content tokens unique to one paragraph should rank that
    paragraph first. §1005.6(b)(1) is the only chunk that combines 'two
    business days', 'timely notice', and the $50 cap."""
    results = index.search("two business days timely notice $50 liability", k=3)
    assert results
    assert results[0].chunk.citation_path == "12 CFR 1005.6(b)(1)"


def test_search_for_topic_words_finds_section_via_heading(index: BM25Index) -> None:
    """The body of §1005.6(a) doesn't say 'unauthorized transfers' prominently;
    the section heading we glued on (Phase 3) does. This test exists to make
    the heading-prefix design pay off — if it stops working, we want to know."""
    results = index.search("liability for unauthorized transfers", k=3)
    assert results
    citations = [r.chunk.citation_path for r in results]
    # All top results should be from §1005.6 (the section about liability).
    assert all("1005.6" in c for c in citations)


def test_search_filters_zero_score_chunks(index: BM25Index) -> None:
    """A query with no token overlap with the corpus should return nothing,
    not low-score noise."""
    results = index.search("xylophone narwhal blockchain", k=10)
    assert results == []


def test_search_respects_k(index: BM25Index) -> None:
    results = index.search("consumer", k=2)
    assert len(results) <= 2


def test_index_len_reflects_added_chunks(index: BM25Index) -> None:
    # The fixture produces 10 chunks (Phase 3 confirmed).
    assert len(index) == 10

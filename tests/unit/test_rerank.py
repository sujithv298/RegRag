"""Tests for the offline rerankers.

We don't test CrossEncoderReranker here for the same reason we skip
BGEEmbedder — it requires a model download. An integration smoke test
belongs in tests/integration/.
"""

from __future__ import annotations

from regrag.chunking import chunk_nodes
from regrag.chunking.hierarchy import HierarchyNode
from regrag.retrieval import (
    LexicalOverlapReranker,
    NoOpReranker,
    Reranker,
    ScoredChunk,
)


def _make_scored(*texts_and_citations: tuple[str, str]) -> list[ScoredChunk]:
    """Build ScoredChunks from (body_text, fake_section) pairs."""
    nodes = [
        HierarchyNode(
            title=12,
            part=1005,
            section=section,
            paragraph_path=("a",),
            section_heading=f"§ {section} Test.",
            text=body,
        )
        for body, section in texts_and_citations
    ]
    chunks = chunk_nodes(nodes)
    return [
        ScoredChunk(chunk=c, score=1.0 - i * 0.1, retriever="hybrid") for i, c in enumerate(chunks)
    ]


# ---- Protocol conformance ----


def test_noop_reranker_implements_protocol() -> None:
    assert isinstance(NoOpReranker(), Reranker)


def test_lexical_overlap_reranker_implements_protocol() -> None:
    assert isinstance(LexicalOverlapReranker(), Reranker)


# ---- NoOpReranker ----


def test_noop_returns_input_unchanged() -> None:
    inputs = _make_scored(
        ("alpha body", "1005.1"),
        ("bravo body", "1005.2"),
        ("charlie body", "1005.3"),
    )
    out = NoOpReranker().rerank("anything", inputs, top_k=10)
    assert [r.chunk.chunk_id for r in out] == [r.chunk.chunk_id for r in inputs]


def test_noop_respects_top_k() -> None:
    inputs = _make_scored(
        ("a", "1005.1"),
        ("b", "1005.2"),
        ("c", "1005.3"),
    )
    out = NoOpReranker().rerank("q", inputs, top_k=2)
    assert len(out) == 2


# ---- LexicalOverlapReranker ----


def test_lexical_reranker_orders_by_query_overlap() -> None:
    """Chunk that contains more query words should rank higher after rerank,
    regardless of the input order."""
    inputs = _make_scored(
        ("This paragraph mentions zero query words.", "1005.1"),
        ("Consumer liability for unauthorized transfers and access devices.", "1005.6"),
        ("Some unrelated content about disclosures.", "1005.7"),
    )
    out = LexicalOverlapReranker().rerank("consumer liability unauthorized", inputs, top_k=3)
    # The §1005.6 chunk has all three query content tokens — should be #1.
    assert out[0].chunk.section == "1005.6"


def test_lexical_reranker_score_in_zero_to_one() -> None:
    inputs = _make_scored(
        ("Consumer liability text.", "1005.6"),
        ("Other content.", "1005.7"),
    )
    out = LexicalOverlapReranker().rerank("consumer liability", inputs, top_k=2)
    assert all(0.0 <= r.score <= 1.0 for r in out)


def test_lexical_reranker_labels_results_rerank() -> None:
    inputs = _make_scored(("Consumer.", "1005.1"))
    out = LexicalOverlapReranker().rerank("consumer", inputs, top_k=1)
    assert all(r.retriever == "rerank" for r in out)


def test_lexical_reranker_uses_source_text_not_prefix() -> None:
    """The reranker must score against `source_text` (rule body), not the
    `text` field which has the citation+heading prefix glued on. Otherwise
    a query that only matches the citation prefix would falsely score
    high — defeating the point of rerank as a precision pass."""
    inputs = _make_scored(("Body says nothing about consumer.", "1005.6"))
    out = LexicalOverlapReranker().rerank(
        # The chunk's text DOES have "1005.6" in the prefix; if the reranker
        # were scoring against text it would match. Source_text doesn't.
        "1005.6",
        inputs,
        top_k=1,
    )
    assert out[0].score == 0.0


def test_lexical_reranker_handles_empty_input() -> None:
    out = LexicalOverlapReranker().rerank("anything", [], top_k=5)
    assert out == []

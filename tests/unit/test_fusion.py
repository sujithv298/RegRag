"""Tests for `regrag.retrieval.fusion.rrf_fuse`.

The RRF formula is small and worth pinning down precisely so future
changes (different k_rrf, weighted variants) don't silently shift behavior.
"""

from __future__ import annotations

from regrag.chunking import chunk_nodes
from regrag.chunking.hierarchy import HierarchyNode
from regrag.retrieval import ScoredChunk, rrf_fuse


def _make_chunks(n: int) -> list:
    nodes = [
        HierarchyNode(
            title=12,
            part=1005,
            section=f"1005.{i}",
            paragraph_path=("a",),
            section_heading=f"§ 1005.{i} Test.",
            text=f"Body of section {i}.",
        )
        for i in range(1, n + 1)
    ]
    return chunk_nodes(nodes)


def _scored(chunks: list, retriever: str = "bm25") -> list[ScoredChunk]:
    """Wrap chunks into ScoredChunks with descending dummy scores."""
    return [
        ScoredChunk(chunk=c, score=float(len(chunks) - i), retriever=retriever)
        for i, c in enumerate(chunks)
    ]


# ---- Edge cases ----


def test_empty_input_returns_empty() -> None:
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_single_list_passes_through_in_rank_order() -> None:
    chunks = _make_chunks(3)
    fused = rrf_fuse([_scored(chunks)], top_k=3)
    assert [r.chunk.citation_path for r in fused] == [c.citation_path for c in chunks]


# ---- Math ----


def test_chunk_in_both_lists_outranks_chunk_in_one() -> None:
    """The whole point of RRF: appearing in two retrievers' top-N is more
    valuable than appearing in one. This is what makes hybrid > either alone."""
    chunks = _make_chunks(4)
    list_a = _scored([chunks[0], chunks[1]], retriever="bm25")
    list_b = _scored([chunks[1], chunks[2]], retriever="dense")
    # chunks[1] is in both lists at rank 2 and 1 → highest fused score.
    fused = rrf_fuse([list_a, list_b], top_k=4)
    assert fused[0].chunk.citation_path == chunks[1].citation_path


def test_rrf_score_uses_rank_not_raw_score() -> None:
    """RRF should ignore the absolute scores in input lists. Two lists with
    wildly different score scales should produce the same fusion as the same
    chunks with the same ranks."""
    chunks = _make_chunks(2)
    list_low = [
        ScoredChunk(chunk=chunks[0], score=0.001, retriever="bm25"),
        ScoredChunk(chunk=chunks[1], score=0.0005, retriever="bm25"),
    ]
    list_high = [
        ScoredChunk(chunk=chunks[0], score=1000.0, retriever="dense"),
        ScoredChunk(chunk=chunks[1], score=500.0, retriever="dense"),
    ]
    fused_low = rrf_fuse([list_low, list_high.copy()], top_k=2)
    # Same ranks but flipped scale on both sides:
    list_low_flipped = [
        ScoredChunk(chunk=chunks[0], score=999_999.0, retriever="bm25"),
        ScoredChunk(chunk=chunks[1], score=999_998.0, retriever="bm25"),
    ]
    fused_high = rrf_fuse([list_low_flipped, list_high], top_k=2)
    assert [r.chunk.citation_path for r in fused_low] == [r.chunk.citation_path for r in fused_high]


def test_fused_results_are_labeled_hybrid() -> None:
    chunks = _make_chunks(2)
    fused = rrf_fuse([_scored(chunks)], top_k=2)
    assert all(r.retriever == "hybrid" for r in fused)


def test_top_k_caps_output() -> None:
    chunks = _make_chunks(5)
    fused = rrf_fuse([_scored(chunks)], top_k=3)
    assert len(fused) == 3


def test_k_rrf_smoothing_constant_is_respected() -> None:
    """With k_rrf=0 the score for rank-1 is 1.0; with k_rrf=60 it's 1/61."""
    chunks = _make_chunks(1)
    fused_zero = rrf_fuse([_scored(chunks)], k_rrf=0, top_k=1)
    fused_default = rrf_fuse([_scored(chunks)], k_rrf=60, top_k=1)
    assert abs(fused_zero[0].score - 1.0) < 1e-9
    assert abs(fused_default[0].score - 1.0 / 61.0) < 1e-9

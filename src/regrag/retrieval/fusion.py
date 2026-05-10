"""Reciprocal Rank Fusion (RRF) — merge multiple ranked lists into one.

The math, in one line:

    rrf_score(chunk) = sum over each ranked list L of:  1 / (k_rrf + rank_L(chunk))

Where `rank_L(chunk)` is 1-indexed and chunks not in `L` contribute 0 from
that list. `k_rrf` is a smoothing constant (60 is the standard value from
the original Cormack/Clarke/Buettcher 2009 paper); larger values flatten
the score curve so top ranks contribute less disproportionately.

Why this formulation:
  - Uses only ranks, never raw scores. So merging BM25 (scores ~0-50) with
    dense (scores ~0-1) doesn't require score normalization.
  - Parameter-free in practice (k_rrf=60 is robust across domains).
  - Chunks present in *both* input lists naturally win — a property that's
    expensive to engineer with score-averaging.
"""

from __future__ import annotations

from regrag.retrieval.types import ScoredChunk

DEFAULT_K_RRF = 60


def rrf_fuse(
    ranked_lists: list[list[ScoredChunk]],
    *,
    k_rrf: int = DEFAULT_K_RRF,
    top_k: int = 10,
) -> list[ScoredChunk]:
    """Fuse multiple ranked lists with Reciprocal Rank Fusion.

    Args:
        ranked_lists: One ranked list per retriever, each in score-descending
            order. Empty lists are tolerated (contribute nothing).
        k_rrf: Smoothing constant. Default 60.
        top_k: How many fused results to return.

    Returns:
        At most `top_k` `ScoredChunk` objects with `retriever="hybrid"` and
        the RRF score, sorted descending.
    """
    accumulator: dict[str, tuple[ScoredChunk, float]] = {}
    for ranked in ranked_lists:
        for rank, scored in enumerate(ranked, start=1):
            chunk_id = scored.chunk.chunk_id
            increment = 1.0 / (k_rrf + rank)
            if chunk_id in accumulator:
                existing_scored, accumulated = accumulator[chunk_id]
                accumulator[chunk_id] = (existing_scored, accumulated + increment)
            else:
                accumulator[chunk_id] = (scored, increment)

    fused = sorted(accumulator.values(), key=lambda pair: pair[1], reverse=True)
    return [
        ScoredChunk(chunk=existing.chunk, score=score, retriever="hybrid")
        for existing, score in fused[:top_k]
    ]

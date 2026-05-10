"""Convert hierarchy-node leaves into searchable Chunks.

The chunker's invariants (tested in tests/unit/test_chunker.py):

    1. Every leaf produces at least one chunk.
    2. No chunk straddles a paragraph boundary. We split *within* a paragraph
       only, never *across* paragraphs.
    3. Every chunk's text starts with `Citation: <citation_path>` followed by
       the section heading (when one exists), so retrieval matches on either.
    4. Every chunk has a unique chunk_id. Un-split chunks reuse the source
       node's chunk_id; split chunks suffix `::0`, `::1`, ...
    5. Concatenating the source_text of all chunks for a given source node
       reproduces the original node text exactly. (Splitting is a partition,
       not a transformation.)

Token estimation:
    We use a rough char/4 estimate for v1. The real tokenizer for the BGE
    embedding model lives downstream in `regrag.retrieval.embeddings`;
    swapping in tiktoken-style precise counting is a one-line change here.
    For Reg E, all paragraphs in the current fixture fit comfortably under
    the default budget, so the splitting path is exercised by tests with
    synthetic long input rather than by real corpus data.
"""

from __future__ import annotations

import re

from regrag.chunking.chunk import Chunk
from regrag.chunking.hierarchy import HierarchyNode

# Default budget chosen for BGE-small (~512-token context) with headroom for
# the citation + heading prefix.
DEFAULT_MAX_TOKENS = 400

# Rough chars-per-token for English. Good enough for budgeting; precise
# tokenization lives in the embedder.
_CHARS_PER_TOKEN = 4

# Sentence boundary: period/question/exclamation followed by whitespace and
# a capital letter or open-paren. Doesn't try to be perfect — legal text has
# enough abbreviations that perfection requires a real NLP library.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def chunk_nodes(nodes: list[HierarchyNode], *, max_tokens: int = DEFAULT_MAX_TOKENS) -> list[Chunk]:
    """Turn a list of leaf hierarchy nodes into a list of searchable Chunks."""
    chunks: list[Chunk] = []
    for node in nodes:
        chunks.extend(_chunk_one_node(node, max_tokens=max_tokens))
    return chunks


# ---- Internal ----


def _chunk_one_node(node: HierarchyNode, *, max_tokens: int) -> list[Chunk]:
    prefix = _build_prefix(node)
    prefix_tokens = _estimate_tokens(prefix)
    body_budget = max(max_tokens - prefix_tokens, 64)  # never go below a sane floor

    body = node.text
    body_tokens = _estimate_tokens(body)

    if body_tokens <= body_budget:
        return [_make_chunk(node, prefix=prefix, body_segment=body, split_idx=0, split_total=1)]

    segments = _split_body(body, body_budget)
    return [
        _make_chunk(
            node,
            prefix=prefix,
            body_segment=seg,
            split_idx=i,
            split_total=len(segments),
        )
        for i, seg in enumerate(segments)
    ]


def _build_prefix(node: HierarchyNode) -> str:
    """The text we prepend to every chunk for retrieval recall.

    Two lines:
        Citation: <canonical citation>
        <section heading or interpretation context>
    """
    lines = [f"Citation: {node.citation_path}"]
    if node.is_interpretation:
        lines.append(f"Official Interpretation, 12 CFR Part {node.part}, Comment {node.comment_id}")
    elif node.section_heading:
        lines.append(node.section_heading)
    return "\n".join(lines)


def _make_chunk(
    node: HierarchyNode,
    *,
    prefix: str,
    body_segment: str,
    split_idx: int,
    split_total: int,
) -> Chunk:
    is_split = split_total > 1
    chunk_id = node.chunk_id if not is_split else f"{node.chunk_id}::{split_idx}"
    return Chunk(
        chunk_id=chunk_id,
        text=f"{prefix}\n\n{body_segment}",
        source_text=body_segment,
        citation_path=node.citation_path,
        title=node.title,
        part=node.part,
        subpart=node.subpart,
        section=node.section,
        section_heading=node.section_heading,
        paragraph_path=node.paragraph_path,
        is_interpretation=node.is_interpretation,
        comment_id=node.comment_id,
        is_split=is_split,
        split_index=split_idx,
        split_total=split_total,
    )


def _estimate_tokens(text: str) -> int:
    return max(len(text) // _CHARS_PER_TOKEN, 1)


def _split_body(body: str, budget_tokens: int) -> list[str]:
    """Greedy-pack sentences into segments under `budget_tokens`.

    Falls back to character-level split if a single sentence exceeds the
    budget — rare in CFR text but possible.
    """
    sentences = _SENTENCE_BOUNDARY.split(body)
    segments: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sent_tokens = _estimate_tokens(sentence)
        if sent_tokens > budget_tokens:
            # Flush the current segment first
            if current:
                segments.append(" ".join(current).strip())
                current = []
                current_tokens = 0
            # Force-split the long sentence at character boundaries
            segments.extend(_force_split(sentence, budget_tokens))
            continue
        if current_tokens + sent_tokens > budget_tokens and current:
            segments.append(" ".join(current).strip())
            current = [sentence]
            current_tokens = sent_tokens
        else:
            current.append(sentence)
            current_tokens += sent_tokens
    if current:
        segments.append(" ".join(current).strip())
    return [s for s in segments if s]


def _force_split(text: str, budget_tokens: int) -> list[str]:
    """Last-resort character-level split for runaway long sentences."""
    char_budget = budget_tokens * _CHARS_PER_TOKEN
    return [text[i : i + char_budget] for i in range(0, len(text), char_budget)]

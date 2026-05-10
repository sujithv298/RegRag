"""Tests for `regrag.chunking.chunker`.

The chunker has five invariants stated in chunker.py — each gets a test
here. We use the real Reg E excerpt fixture for the un-split path and a
synthetic long node for the split path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from regrag.chunking import Chunk, chunk_nodes
from regrag.chunking.hierarchy import HierarchyNode
from regrag.ingest import parse_part_xml

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "reg_e_part_1005_excerpt.xml"


@pytest.fixture(scope="module")
def chunks_from_fixture() -> list[Chunk]:
    nodes = parse_part_xml(FIXTURE.read_bytes())
    return chunk_nodes(nodes)


# ---- Invariant 1: every leaf produces at least one chunk ----


def test_every_leaf_produces_at_least_one_chunk(chunks_from_fixture: list[Chunk]) -> None:
    nodes = parse_part_xml(FIXTURE.read_bytes())
    citations_in = {n.citation_path for n in nodes}
    citations_out = {c.citation_path for c in chunks_from_fixture}
    assert citations_in == citations_out


def test_short_paragraphs_yield_one_chunk_each(chunks_from_fixture: list[Chunk]) -> None:
    # Reg E excerpt paragraphs are all short — none should split.
    assert len(chunks_from_fixture) == 10
    assert all(not c.is_split for c in chunks_from_fixture)
    assert all(c.split_total == 1 for c in chunks_from_fixture)


# ---- Invariant 2: no chunk straddles a paragraph boundary ----


def test_each_chunk_has_exactly_one_citation_path(chunks_from_fixture: list[Chunk]) -> None:
    # If a chunk straddled paragraphs, you couldn't assign one citation
    # to it. The frozen pydantic model already enforces "one string"; this
    # test verifies that string is non-empty and well-formed.
    for c in chunks_from_fixture:
        assert c.citation_path
        assert c.citation_path.startswith("12 CFR ")


# ---- Invariant 3: every chunk text starts with citation + heading prefix ----


def test_every_chunk_text_includes_citation_prefix(chunks_from_fixture: list[Chunk]) -> None:
    for c in chunks_from_fixture:
        first_line = c.text.split("\n", 1)[0]
        assert first_line == f"Citation: {c.citation_path}"


def test_rule_chunks_include_section_heading(chunks_from_fixture: list[Chunk]) -> None:
    rule_chunks = [c for c in chunks_from_fixture if not c.is_interpretation]
    for c in rule_chunks:
        assert c.section_heading is not None
        assert c.section_heading in c.text


def test_interpretation_chunks_include_interpretation_context(
    chunks_from_fixture: list[Chunk],
) -> None:
    interp_chunks = [c for c in chunks_from_fixture if c.is_interpretation]
    assert len(interp_chunks) == 2
    for c in interp_chunks:
        assert "Official Interpretation" in c.text
        assert c.comment_id is not None
        assert c.comment_id in c.text


# ---- Invariant 4: chunk_ids are unique ----


def test_chunk_ids_are_unique(chunks_from_fixture: list[Chunk]) -> None:
    ids = [c.chunk_id for c in chunks_from_fixture]
    assert len(ids) == len(set(ids))


# ---- Invariant 5: source_text reconstructs original node text ----


def test_source_text_matches_original_for_unsplit_chunks(
    chunks_from_fixture: list[Chunk],
) -> None:
    nodes = parse_part_xml(FIXTURE.read_bytes())
    nodes_by_citation = {n.citation_path: n for n in nodes}
    for c in chunks_from_fixture:
        assert c.source_text == nodes_by_citation[c.citation_path].text


# ---- Splitting path (synthetic long input) ----


def test_long_node_splits_into_multiple_chunks() -> None:
    # Build a synthetic node whose body greatly exceeds the default budget.
    long_body = " ".join(
        f"Sentence number {i} contains some words to take up some token budget." for i in range(200)
    )
    node = HierarchyNode(
        title=12,
        part=1005,
        subpart="A",
        section="1005.99",
        section_heading="§ 1005.99 Synthetic test section.",
        paragraph_path=("a",),
        text=long_body,
    )

    chunks = chunk_nodes([node])

    assert len(chunks) > 1, "long node should split"
    assert all(c.is_split for c in chunks)
    assert {c.split_total for c in chunks} == {len(chunks)}
    # Split indices are 0..N-1 in order
    assert [c.split_index for c in chunks] == list(range(len(chunks)))
    # chunk_ids are suffixed with ::N
    assert all(c.chunk_id.endswith(f"::{c.split_index}") for c in chunks)
    # All splits share the same citation_path (no straddling — all from one paragraph)
    assert {c.citation_path for c in chunks} == {"12 CFR 1005.99(a)"}


def test_split_chunks_partition_the_original_text() -> None:
    """source_text concatenated across splits == original body. Splitting
    is a partition, not a transformation — no text added, none lost."""
    long_body = " ".join(
        f"Sentence {i} has some content to fill space and force a split." for i in range(150)
    )
    node = HierarchyNode(
        title=12,
        part=1005,
        section="1005.99",
        section_heading="§ 1005.99 Test.",
        paragraph_path=("a",),
        text=long_body,
    )

    chunks = chunk_nodes([node])

    # Reconstruct: join split source_texts with a space (since the splitter
    # rejoins sentences with " "). Should match the original (modulo
    # whitespace normalization).
    reconstructed = " ".join(c.source_text for c in chunks)
    assert _normalize(reconstructed) == _normalize(long_body)


# ---- Helpers ----


def _normalize(s: str) -> str:
    """Collapse whitespace runs to a single space so we can compare
    reconstructed-from-splits text against the original."""
    return " ".join(s.split())

"""Tests for `regrag.ingest.parser` against the handcrafted Reg E excerpt.

These tests encode the chunker's invariants (see ARCHITECTURE.md): no chunk
straddles a section boundary, every leaf has a citation_path, etc.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from regrag.chunking.hierarchy import HierarchyNode
from regrag.ingest.parser import parse_part_xml

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "reg_e_part_1005_excerpt.xml"


@pytest.fixture(scope="module")
def parsed() -> list[HierarchyNode]:
    return parse_part_xml(FIXTURE.read_bytes())


# ---- Sanity ----


def test_parses_some_nodes(parsed: list[HierarchyNode]) -> None:
    assert len(parsed) > 0
    # Two paragraphs in §1005.1, six in §1005.6, two interpretation comments.
    assert len(parsed) == 10


def test_all_nodes_have_citation_path(parsed: list[HierarchyNode]) -> None:
    for node in parsed:
        assert node.citation_path  # non-empty string


def test_all_nodes_have_chunk_id(parsed: list[HierarchyNode]) -> None:
    chunk_ids = [n.chunk_id for n in parsed]
    assert len(chunk_ids) == len(set(chunk_ids)), "chunk_ids must be unique"


# ---- Hierarchy fields land correctly ----


def test_section_1005_1_paragraphs(parsed: list[HierarchyNode]) -> None:
    s1 = [n for n in parsed if n.section == "1005.1" and not n.is_interpretation]
    assert len(s1) == 2
    assert s1[0].paragraph_path == ("a",)
    assert s1[1].paragraph_path == ("b",)
    assert s1[0].subpart == "A"
    assert "Authority" in s1[0].text
    # Italic markup must be flattened, not preserved as structure.
    assert "<I>" not in s1[0].text
    assert "<i>" not in s1[0].text


def test_section_1005_6_nested_paragraphs(parsed: list[HierarchyNode]) -> None:
    s6 = [n for n in parsed if n.section == "1005.6" and not n.is_interpretation]
    paths = [n.paragraph_path for n in s6]
    assert paths == [
        ("a",),
        ("b",),
        ("b", "1"),
        ("b", "2"),
        ("b", "2", "i"),
        ("b", "2", "ii"),
    ]


def test_section_heading_propagated_to_every_leaf(parsed: list[HierarchyNode]) -> None:
    s6 = [n for n in parsed if n.section == "1005.6" and not n.is_interpretation]
    headings = {n.section_heading for n in s6}
    assert headings == {"§ 1005.6 Liability of consumer for unauthorized transfers."}


# ---- Citation-path formatting ----


def test_citation_path_for_simple_paragraph(parsed: list[HierarchyNode]) -> None:
    s1_a = next(n for n in parsed if n.section == "1005.1" and n.paragraph_path == ("a",))
    assert s1_a.citation_path == "12 CFR 1005.1(a)"


def test_citation_path_for_nested_paragraph(parsed: list[HierarchyNode]) -> None:
    s6_b1 = next(
        n for n in parsed if n.section == "1005.6" and n.paragraph_path == ("b", "1")
    )
    assert s6_b1.citation_path == "12 CFR 1005.6(b)(1)"


def test_citation_path_for_doubly_nested_paragraph(parsed: list[HierarchyNode]) -> None:
    s6_b2i = next(
        n for n in parsed if n.section == "1005.6" and n.paragraph_path == ("b", "2", "i")
    )
    assert s6_b2i.citation_path == "12 CFR 1005.6(b)(2)(i)"


# ---- Official Interpretations ----


def test_interpretations_parsed(parsed: list[HierarchyNode]) -> None:
    interps = [n for n in parsed if n.is_interpretation]
    assert len(interps) == 2
    assert interps[0].comment_id == "6(b)-1"
    assert interps[1].comment_id == "6(b)-2"


def test_interpretation_citation_path(parsed: list[HierarchyNode]) -> None:
    interps = [n for n in parsed if n.is_interpretation]
    assert interps[0].citation_path == "12 CFR 1005, Comment 6(b)-1"


def test_interpretation_text_starts_with_topic_word(parsed: list[HierarchyNode]) -> None:
    interps = [n for n in parsed if n.is_interpretation]
    # The italic intro word should remain in the text body.
    assert interps[0].text.startswith("Knowledge.")
    assert interps[1].text.startswith("Two business days.")


# ---- Invariants ----


def test_no_chunk_straddles_section(parsed: list[HierarchyNode]) -> None:
    """Each parsed leaf belongs to exactly one section (or is an interpretation)."""
    for node in parsed:
        if node.is_interpretation:
            assert node.section is None
        else:
            assert node.section is not None


def test_no_xml_designators_leak_into_body(parsed: list[HierarchyNode]) -> None:
    """The leading paragraph designator must be lifted to paragraph_path,
    not left in the body. Otherwise the chunker would re-include it as text
    and the citation extractor would see "(b)(1)" twice."""
    for node in parsed:
        if node.paragraph_path:
            assert not node.text.startswith(f"({node.paragraph_path[0]})"), (
                f"Designator leaked into body for {node.citation_path}: {node.text[:40]}"
            )


def test_part_and_title_set_on_all_nodes(parsed: list[HierarchyNode]) -> None:
    for node in parsed:
        assert node.title == 12
        assert node.part == 1005

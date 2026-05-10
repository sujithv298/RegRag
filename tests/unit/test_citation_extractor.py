"""Tests for `regrag.citations.extractor.CitationExtractor`."""

from __future__ import annotations

from regrag.citations import CitationExtractor


def test_extracts_tagged_citation() -> None:
    text = "The consumer's liability is $50 [CFR:1005.6(b)(1)]."
    out = CitationExtractor().extract(text)
    assert len(out) == 1
    assert out[0].citation_path == "12 CFR 1005.6(b)(1)"


def test_extracts_free_form_rule_citation() -> None:
    text = "Under 12 CFR 1005.6(b)(1), liability is capped."
    out = CitationExtractor().extract(text)
    assert len(out) == 1
    assert out[0].citation_path == "12 CFR 1005.6(b)(1)"


def test_extracts_free_form_comment_citation() -> None:
    text = "See 12 CFR 1005, Comment 6(b)-1 for the knowledge standard."
    out = CitationExtractor().extract(text)
    assert any(c.citation_path == "12 CFR 1005, Comment 6(b)-1" for c in out)


def test_extracts_multiple_citations() -> None:
    text = "If timely, $50 cap [CFR:1005.6(b)(1)]. If untimely, $500 cap [CFR:1005.6(b)(2)]."
    out = CitationExtractor().extract(text)
    paths = [c.citation_path for c in out]
    assert "12 CFR 1005.6(b)(1)" in paths
    assert "12 CFR 1005.6(b)(2)" in paths


def test_claim_is_surrounding_sentence() -> None:
    text = "First sentence. Liability is $50 [CFR:1005.6(b)(1)]. Third sentence."
    out = CitationExtractor().extract(text)
    assert len(out) == 1
    assert "Liability is $50" in out[0].claim
    assert "First sentence" not in out[0].claim
    assert "Third sentence" not in out[0].claim


def test_canonicalizes_bare_tagged_citation() -> None:
    """A tag containing just the section number (no '12 CFR' prefix) should
    expand to canonical form."""
    text = "See [CFR:1005.6(b)(1)] for details."
    out = CitationExtractor().extract(text)
    assert out[0].citation_path == "12 CFR 1005.6(b)(1)"


def test_canonicalizes_comment_in_tag() -> None:
    text = "See [CFR:Comment 6(b)-1] for the knowledge standard."
    out = CitationExtractor().extract(text)
    assert out[0].citation_path == "12 CFR 1005, Comment 6(b)-1"


def test_no_citations_returns_empty() -> None:
    out = CitationExtractor().extract("This response has no citations at all.")
    assert out == []


def test_no_double_extraction_for_overlapping_patterns() -> None:
    """A free-form '12 CFR 1005.6(b)(1)' inside a tag-friendly response shouldn't
    produce two ExtractedCitations for the same span."""
    text = "[CFR:1005.6(b)(1)]"
    out = CitationExtractor().extract(text)
    # Only the tag matched here; no plain 12 CFR ... pattern present.
    assert len(out) == 1

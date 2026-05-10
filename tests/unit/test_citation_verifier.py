"""Tests for `regrag.citations.verifier.CitationVerifier`.

The verifier is the **fail-closed gate**, so the tests aggressively cover
every way an answer can be rejected: hallucinated citation, citation that
exists but doesn't support the claim, response with no citations at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from regrag.chunking import chunk_nodes
from regrag.citations import (
    CitationExtractor,
    CitationVerifier,
    ExtractedCitation,
    VerificationResult,
)
from regrag.ingest import parse_part_xml

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "reg_e_part_1005_excerpt.xml"


@pytest.fixture(scope="module")
def corpus() -> list:
    nodes = parse_part_xml(FIXTURE.read_bytes())
    return chunk_nodes(nodes)


@pytest.fixture(scope="module")
def verifier(corpus: list) -> CitationVerifier:
    return CitationVerifier(corpus=corpus)


# ---- Existence ----


def test_existing_citation_passes_existence(verifier: CitationVerifier) -> None:
    extracted = [
        ExtractedCitation(
            citation_path="12 CFR 1005.6(b)(1)",
            claim="Timely notice given. The consumer's liability shall not exceed $50.",
            span=(0, 0),
        )
    ]
    result = verifier.verify(extracted)
    assert result.citations[0].exists
    assert result.citations[0].chunk_id is not None


def test_hallucinated_citation_fails_existence(verifier: CitationVerifier) -> None:
    """Citation to a section that doesn't exist in the corpus is the
    canonical hallucination case. Must fail with `unverified_citation`."""
    extracted = [
        ExtractedCitation(
            citation_path="12 CFR 9999.99(z)",
            claim="The model said something with a fake citation.",
            span=(0, 0),
        )
    ]
    result = verifier.verify(extracted)
    assert not result.citations[0].exists
    assert not result.citations[0].passed
    assert result.citations[0].failure_reason == "unverified_citation"


# ---- Support ----


def test_supported_claim_passes(verifier: CitationVerifier) -> None:
    """A claim that quotes language from the cited chunk should pass."""
    extracted = [
        ExtractedCitation(
            citation_path="12 CFR 1005.6(b)(1)",
            claim="Within two business days the consumer's liability is the lesser of $50.",
            span=(0, 0),
        )
    ]
    result = verifier.verify(extracted)
    assert result.citations[0].passed
    assert result.citations[0].support_score >= 0.3


def test_unsupported_claim_fails_support() -> None:
    """A claim that's about totally different content but cites a real
    chunk should fail with `low_support_score`."""
    nodes = parse_part_xml(FIXTURE.read_bytes())
    chunks = chunk_nodes(nodes)
    verifier = CitationVerifier(corpus=chunks, support_threshold=0.5)
    extracted = [
        ExtractedCitation(
            citation_path="12 CFR 1005.6(b)(1)",
            claim="Cryptocurrency exchanges must report quarterly to the SEC.",
            span=(0, 0),
        )
    ]
    result = verifier.verify(extracted)
    assert result.citations[0].exists
    assert not result.citations[0].supports
    assert result.citations[0].failure_reason == "low_support_score"


# ---- Aggregate behavior (this is what the pipeline reads) ----


def test_all_passed_requires_at_least_one_citation(verifier: CitationVerifier) -> None:
    """A response with zero citations is not a pass — bank-grade systems
    don't return claim-shaped text without traceability."""
    result = VerificationResult(citations=[])
    assert not result.all_passed
    assert result.refusal_reason == "no_citations_emitted"


def test_one_failure_makes_whole_response_fail(verifier: CitationVerifier) -> None:
    """If any citation fails, the whole response is rejected. Better one
    refusal than three verified plus one hallucinated."""
    extracted = [
        ExtractedCitation(
            citation_path="12 CFR 1005.6(b)(1)",
            claim="Within two business days the consumer's liability is the lesser of $50.",
            span=(0, 0),
        ),
        ExtractedCitation(
            citation_path="12 CFR 9999.99(z)",  # hallucinated
            claim="And also this fake thing.",
            span=(0, 0),
        ),
    ]
    result = verifier.verify(extracted)
    assert not result.all_passed
    assert result.refusal_reason == "unverified_citation"


def test_extractor_to_verifier_end_to_end(verifier: CitationVerifier) -> None:
    """A realistic round-trip: a model-style response goes through
    extraction then verification, end-to-end."""
    response_text = (
        "Within two business days the consumer's liability is the lesser of $50 "
        "or the unauthorized transfers that occurred [CFR:1005.6(b)(1)]."
    )
    extracted = CitationExtractor().extract(response_text)
    result = verifier.verify(extracted)
    assert result.all_passed


def test_extractor_to_verifier_catches_hallucination(verifier: CitationVerifier) -> None:
    response_text = (
        "Cryptocurrency exchanges have specific obligations under the EFTA [CFR:9999.99(z)]."
    )
    extracted = CitationExtractor().extract(response_text)
    result = verifier.verify(extracted)
    assert not result.all_passed
    assert result.refusal_reason == "unverified_citation"

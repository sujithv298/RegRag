"""End-to-end tests for `regrag.pipeline.answer_query`.

These exercise every module: PII → retrieval → prompt → LLM (FakeAdapter)
→ extract → verify → audit. Same code path as a real production query;
only the model adapter is swapped for a deterministic stand-in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from regrag.audit import AuditLogger
from regrag.chunking import chunk_nodes
from regrag.ingest import parse_part_xml
from regrag.models import FakeAdapter
from regrag.pipeline import answer_query
from regrag.retrieval import (
    BM25Index,
    DenseRetriever,
    HashingEmbedder,
    HybridRetriever,
    LexicalOverlapReranker,
)
from regrag.store import InMemoryVectorStore

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "reg_e_part_1005_excerpt.xml"


@pytest.fixture
def corpus() -> list:
    nodes = parse_part_xml(FIXTURE.read_bytes())
    return chunk_nodes(nodes)


@pytest.fixture
def retriever(corpus: list) -> HybridRetriever:
    h = HybridRetriever(
        bm25=BM25Index(),
        dense=DenseRetriever(embedder=HashingEmbedder(), store=InMemoryVectorStore()),
        reranker=LexicalOverlapReranker(),
    )
    h.add(corpus)
    return h


# ---- Happy path: passing verification → answered ----


def test_answered_when_response_has_verifiable_citation(
    corpus: list, retriever: HybridRetriever, tmp_path: Path
) -> None:
    canned = (
        "Within two business days the consumer's liability is the lesser of $50 "
        "or the unauthorized transfers that occurred [CFR:1005.6(b)(1)]."
    )
    logger = AuditLogger(log_path=tmp_path / "audit.jsonl")

    result = answer_query(
        "What is the consumer's liability for an unauthorized transfer reported within two business days?",
        retriever=retriever,
        model=FakeAdapter(response=canned),
        audit_logger=logger,
        corpus=corpus,
        corpus_snapshot_date="2025-01-01",
    )

    assert result.outcome == "answered"
    assert result.refusal_reason is None
    assert result.answer == canned
    assert any(c.citation_path == "12 CFR 1005.6(b)(1)" and c.passed for c in result.citations)


# ---- Fail-closed paths: refusal ----


def test_refused_when_response_has_hallucinated_citation(
    corpus: list, retriever: HybridRetriever, tmp_path: Path
) -> None:
    bad = "Crypto exchanges have specific reporting obligations [CFR:9999.99(z)]."
    logger = AuditLogger(log_path=tmp_path / "audit.jsonl")

    result = answer_query(
        "What does Reg E say about crypto?",
        retriever=retriever,
        model=FakeAdapter(response=bad),
        audit_logger=logger,
        corpus=corpus,
        corpus_snapshot_date="2025-01-01",
    )

    assert result.outcome == "refused"
    assert result.refusal_reason == "unverified_citation"
    assert "cannot answer" in result.answer.lower()
    # The hallucinated citation is recorded in the audit record, not
    # silently dropped.
    assert any(
        c.citation_path == "12 CFR 9999.99(z)" and not c.exists for c in result.citations
    )


def test_refused_when_response_has_no_citations(
    corpus: list, retriever: HybridRetriever, tmp_path: Path
) -> None:
    """Bank-grade: claim-shaped text with no citation = refusal."""
    no_citations = "The consumer's liability is fifty dollars."
    logger = AuditLogger(log_path=tmp_path / "audit.jsonl")

    result = answer_query(
        "liability?",
        retriever=retriever,
        model=FakeAdapter(response=no_citations),
        audit_logger=logger,
        corpus=corpus,
        corpus_snapshot_date="2025-01-01",
    )

    assert result.outcome == "refused"
    assert result.refusal_reason == "no_citations_emitted"


# ---- Audit invariants ----


def test_every_call_writes_exactly_one_audit_record(
    corpus: list, retriever: HybridRetriever, tmp_path: Path
) -> None:
    """The pipeline's invariant: 1 query → 1 audit record."""
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path=log_path)

    answer_query(
        "q",
        retriever=retriever,
        model=FakeAdapter(
            response="Within two business days the consumer's liability is the lesser of $50 [CFR:1005.6(b)(1)]."
        ),
        audit_logger=logger,
        corpus=corpus,
        corpus_snapshot_date="2025-01-01",
    )
    answer_query(
        "q2",
        retriever=retriever,
        model=FakeAdapter(response="Crypto exchanges have specific obligations [CFR:9999.99(z)]."),
        audit_logger=logger,
        corpus=corpus,
        corpus_snapshot_date="2025-01-01",
    )

    records = logger.read_all()
    assert len(records) == 2
    # Different outcomes recorded for different queries.
    assert {r.outcome for r in records} == {"answered", "refused"}


def test_audit_record_carries_pipeline_metadata(
    corpus: list, retriever: HybridRetriever, tmp_path: Path
) -> None:
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path=log_path)

    answer_query(
        "what does the regulation say",
        retriever=retriever,
        model=FakeAdapter(
            response="text [CFR:1005.6(b)(1)] with two business days liability $50",
            name="fake",
            version="v0",
        ),
        audit_logger=logger,
        corpus=corpus,
        corpus_snapshot_date="2025-01-01",
    )

    record = logger.read_all()[0]
    assert record.model_name == "fake"
    assert record.model_version == "v0"
    assert record.corpus_snapshot_date == "2025-01-01"
    assert record.prompt_template_version == "0.1.0"
    assert record.retrieved_chunk_ids  # non-empty
    assert record.prompt_hash and len(record.prompt_hash) == 64
    assert record.response_hash and len(record.response_hash) == 64


def test_pii_is_scrubbed_before_logging(
    corpus: list, retriever: HybridRetriever, tmp_path: Path
) -> None:
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path=log_path)

    sensitive_question = (
        "My SSN is 123-45-6789 and my email is jane@example.com — what is the "
        "consumer's liability for an unauthorized transfer reported within two "
        "business days?"
    )

    answer_query(
        sensitive_question,
        retriever=retriever,
        model=FakeAdapter(response="Liability is $50 [CFR:1005.6(b)(1)]"),
        audit_logger=logger,
        corpus=corpus,
        corpus_snapshot_date="2025-01-01",
    )

    record = logger.read_all()[0]
    assert "123-45-6789" not in record.scrubbed_input
    assert "jane@example.com" not in record.scrubbed_input
    assert record.pii_redaction_count == 2
    assert record.pii_scrubber == "regex"

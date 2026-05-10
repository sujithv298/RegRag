"""Tests for the audit log."""

from __future__ import annotations

import json
from pathlib import Path

from regrag.audit import AuditLogger, AuditRecord, make_record
from regrag.citations import VerifiedCitation
from regrag.citations.types import VerificationResult


def _passing_verification() -> VerificationResult:
    return VerificationResult(
        citations=[
            VerifiedCitation(
                citation_path="12 CFR 1005.6(b)(1)",
                claim="Within two business days liability is $50.",
                chunk_id="section:1005.6:b.1",
                exists=True,
                supports=True,
                support_score=0.6,
            )
        ]
    )


def _failing_verification() -> VerificationResult:
    return VerificationResult(
        citations=[
            VerifiedCitation(
                citation_path="12 CFR 9999.99(z)",
                claim="Some fake claim.",
                chunk_id=None,
                exists=False,
                supports=False,
                support_score=0.0,
            )
        ]
    )


def _make(verification: VerificationResult, **overrides):
    base = {
        "scrubbed_input": "What is the liability cap for timely notice?",
        "pii_redaction_count": 0,
        "pii_scrubber": "regex",
        "corpus_snapshot_date": "2025-01-01",
        "retrieved_chunk_ids": ["section:1005.6:b.1", "section:1005.6:b"],
        "prompt_template_version": "0.1.0",
        "prompt_text": "System prompt + retrieved chunks + question.",
        "model_name": "claude-sonnet-4-6",
        "model_version": "2025-01-15",
        "response_text": "Liability is $50 [CFR:1005.6(b)(1)].",
        "verification": verification,
        "outcome": "answered",
    }
    base.update(overrides)
    return make_record(**base)


# ---- make_record ----


def test_make_record_populates_id_and_timestamp() -> None:
    record = _make(_passing_verification())
    assert record.query_id  # uuid4 string
    assert len(record.query_id) == 36
    assert record.timestamp is not None


def test_make_record_hashes_prompt_and_response() -> None:
    record = _make(_passing_verification())
    assert len(record.prompt_hash) == 64  # sha256 hex
    assert len(record.response_hash) == 64


def test_make_record_carries_citation_audit_records() -> None:
    record = _make(_passing_verification())
    assert len(record.citations) == 1
    assert record.citations[0].citation_path == "12 CFR 1005.6(b)(1)"
    assert record.citations[0].exists
    assert record.citations[0].supports


def test_failed_verification_record_carries_refusal() -> None:
    record = _make(
        _failing_verification(),
        outcome="refused",
        refusal_reason="unverified_citation",
        response_text="Some fake claim.",
    )
    assert record.outcome == "refused"
    assert record.refusal_reason == "unverified_citation"


# ---- AuditLogger ----


def test_logger_writes_one_line_per_record(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path=log_path)
    logger.write(_make(_passing_verification()))
    logger.write(_make(_passing_verification()))
    lines = log_path.read_text().splitlines()
    assert len(lines) == 2
    # Each line is a valid JSON object (no merging across newlines).
    for line in lines:
        payload = json.loads(line)
        assert "query_id" in payload


def test_logger_round_trips(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path=log_path)
    record = _make(_passing_verification())
    logger.write(record)
    read_back = logger.read_all()
    assert len(read_back) == 1
    assert isinstance(read_back[0], AuditRecord)
    assert read_back[0].query_id == record.query_id
    assert read_back[0].prompt_hash == record.prompt_hash


def test_logger_creates_parent_dir(tmp_path: Path) -> None:
    log_path = tmp_path / "deeply" / "nested" / "audit.jsonl"
    AuditLogger(log_path=log_path)
    assert log_path.parent.exists()


def test_logger_empty_when_no_writes_yet(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path=log_path)
    assert logger.read_all() == []


def test_audit_record_schema_version_set() -> None:
    """If we ever bump the schema_version we want it visible. This test
    pins it so the bump becomes a deliberate choice, not a silent change."""
    record = _make(_passing_verification())
    assert record.schema_version == "1"

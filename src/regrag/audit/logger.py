"""Append-only JSONL audit-log writer.

Default destination is the path passed to the constructor (typically read
from `REGRAG_AUDIT_LOG`). Replace this single class to send records to
S3 / Splunk / Kafka / a SIEM — nothing else in the pipeline cares.

Why JSONL by default:
  - Append-only: every write is one `f.write(line)`, no risk of corrupting
    earlier records on crash.
  - Easy to grep, easy to ship to log aggregators, easy to replay.
  - The schema in `schema.py` is pydantic-validated, so JSONL doesn't lose
    type safety — `read_all()` re-validates on read.

A helper `make_record(...)` builds an AuditRecord from the runtime objects
the pipeline already has, so callers don't have to think about hashing or
schema_version.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from pathlib import Path

from regrag.audit.schema import AuditCitationRecord, AuditRecord, Outcome
from regrag.citations.types import VerificationResult


class AuditLogger:
    """Write `AuditRecord`s as JSONL to a file."""

    def __init__(self, *, log_path: str | Path) -> None:
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._log_path

    def write(self, record: AuditRecord) -> None:
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

    def read_all(self) -> list[AuditRecord]:
        if not self._log_path.exists():
            return []
        return [
            AuditRecord.model_validate_json(line)
            for line in self._log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


# ---- Helpers ----


def make_record(
    *,
    scrubbed_input: str,
    pii_redaction_count: int,
    pii_scrubber: str,
    corpus_snapshot_date: str,
    retrieved_chunk_ids: list[str],
    prompt_template_version: str,
    prompt_text: str,
    model_name: str,
    model_version: str,
    response_text: str,
    verification: VerificationResult,
    outcome: Outcome,
    refusal_reason: str | None = None,
) -> AuditRecord:
    """Assemble an `AuditRecord` from the runtime objects the pipeline has.

    Computes `query_id`, timestamp, prompt_hash, and response_hash so the
    pipeline doesn't have to.
    """
    return AuditRecord(
        query_id=str(uuid.uuid4()),
        timestamp=datetime.now(UTC),
        scrubbed_input=scrubbed_input,
        pii_redaction_count=pii_redaction_count,
        pii_scrubber=pii_scrubber,
        corpus_snapshot_date=corpus_snapshot_date,
        retrieved_chunk_ids=retrieved_chunk_ids,
        prompt_template_version=prompt_template_version,
        model_name=model_name,
        model_version=model_version,
        response_text=response_text,
        citations=[
            AuditCitationRecord(
                citation_path=c.citation_path,
                claim=c.claim,
                chunk_id=c.chunk_id,
                exists=c.exists,
                supports=c.supports,
                support_score=c.support_score,
            )
            for c in verification.citations
        ],
        outcome=outcome,
        refusal_reason=refusal_reason,
        prompt_hash=_sha256(prompt_text),
        response_hash=_sha256(response_text),
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

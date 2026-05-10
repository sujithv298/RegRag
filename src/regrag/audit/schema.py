"""Audit log JSON schema. Versioned. Pydantic-validated.

This is a **contract**. Any breaking change bumps `SCHEMA_VERSION` and
forkers piping records into a SIEM should pin to a major version.

Per-record contents are documented in `docs/audit-log-schema.md`.

The hashing convention:
  - `prompt_hash` is sha256 over the rendered prompt, hex-digest.
  - `response_hash` is sha256 over the raw response text, hex-digest.
Both let an examiner detect tampering: they can re-hash the stored prompt
and response and confirm the audit record is genuine.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1"

Outcome = Literal["answered", "refused"]


class AuditCitationRecord(BaseModel):
    """One per extracted citation in the response."""

    model_config = ConfigDict(frozen=True)

    citation_path: str
    claim: str
    chunk_id: str | None
    exists: bool
    supports: bool
    support_score: float


class AuditRecord(BaseModel):
    """One per query. Append-only, JSONL on disk by default."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    query_id: str = Field(description="UUID v4, unique per query.")
    timestamp: datetime = Field(description="UTC, ISO-8601, when the query started.")

    # Input side
    scrubbed_input: str = Field(description="PII-redacted user query. Raw is never persisted.")
    pii_redaction_count: int = Field(
        description="Number of PII entities Presidio/regex redacted."
    )
    pii_scrubber: str = Field(description="Which scrubber implementation ran. Audit-log provenance.")

    # Corpus + retrieval
    corpus_snapshot_date: str = Field(description="ISO date of the eCFR snapshot the index was built against.")
    retrieved_chunk_ids: list[str] = Field(description="Chunk IDs in rank order, post-rerank.")

    # Prompt + model
    prompt_template_version: str = Field(description="Semver of the prompt template used.")
    model_name: str = Field(description="Adapter-reported name, e.g. 'claude-sonnet-4-6'.")
    model_version: str = Field(description="Adapter-reported version (API metadata where available).")

    # Response
    response_text: str = Field(description="Raw model output, before any refusal conversion.")
    citations: list[AuditCitationRecord] = Field(default_factory=list)
    outcome: Outcome = Field(description="'answered' if all citations verified, else 'refused'.")
    refusal_reason: str | None = Field(
        default=None,
        description="Set when outcome=='refused'. Examples: 'no_citations_emitted', "
        "'unverified_citation', 'low_support_score'.",
    )

    # Tamper detection
    prompt_hash: str = Field(description="SHA-256 hex of the rendered prompt.")
    response_hash: str = Field(description="SHA-256 hex of the response_text.")

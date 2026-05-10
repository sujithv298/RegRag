"""Pydantic models for the citation pipeline.

`ExtractedCitation`     — a citation parsed out of model output, paired with
                          the surrounding sentence (the "claim") it sits next to.
`VerifiedCitation`      — what the verifier returns: existence + support
                          status per extracted citation.
`VerificationResult`    — the aggregate over all citations in one response.
                          Knows whether the response can be returned to the
                          user (`all_passed`) or must be refused.

These are checked-in contracts. The audit log references them by shape.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ExtractedCitation(BaseModel):
    """One citation parsed out of model output."""

    model_config = ConfigDict(frozen=True)

    citation_path: str = Field(description="Canonical citation, e.g. '12 CFR 1005.6(b)(1)'.")
    claim: str = Field(description="The sentence the citation sits next to.")
    span: tuple[int, int] = Field(description="(start, end) char offsets in the source text.")


class VerifiedCitation(BaseModel):
    """Verification outcome for one extracted citation."""

    model_config = ConfigDict(frozen=True)

    citation_path: str
    claim: str
    chunk_id: str | None = Field(
        default=None,
        description="Chunk ID this citation resolved to, or None if not found in corpus.",
    )
    exists: bool = Field(description="True if citation_path resolves to a chunk in the corpus.")
    supports: bool = Field(
        description="True if the resolved chunk's text supports the claim above the threshold."
    )
    support_score: float = Field(
        description="0..1 fraction of meaningful claim tokens found in the chunk's source_text."
    )

    @property
    def passed(self) -> bool:
        return self.exists and self.supports

    @property
    def failure_reason(self) -> str | None:
        if not self.exists:
            return "unverified_citation"
        if not self.supports:
            return "low_support_score"
        return None


class VerificationResult(BaseModel):
    """Aggregate verification outcome over all citations in one response."""

    model_config = ConfigDict(frozen=True)

    citations: list[VerifiedCitation] = Field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """True iff every citation passed AND there was at least one citation.

        An answer with zero citations is treated as not-passing — a bank-grade
        system shouldn't return claim-shaped text with no traceability."""
        return bool(self.citations) and all(c.passed for c in self.citations)

    @property
    def first_failure(self) -> VerifiedCitation | None:
        for c in self.citations:
            if not c.passed:
                return c
        return None

    @property
    def refusal_reason(self) -> str | None:
        """One-line refusal reason for the audit log, or None if the result passed."""
        if not self.citations:
            return "no_citations_emitted"
        first = self.first_failure
        if first is None:
            return None
        return first.failure_reason

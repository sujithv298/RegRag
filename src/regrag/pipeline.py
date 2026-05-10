"""Top-level query pipeline — `answer_query`.

This is where every module finally meets. The function is intentionally
linear and explicit: each step is a named call, and the invariant —
*every call to `answer_query` produces exactly one audit record* — is
enforced by the structure of this file.

Step-by-step:

    1. Scrub PII from the user input.
    2. Retrieve top-k chunks via the hybrid retriever (BM25 + dense + rerank).
    3. Render the versioned prompt template with the retrieved chunks.
    4. Call the LLM adapter.
    5. Extract citations from the model's response.
    6. Verify each citation against the corpus (fail closed).
    7. Decide outcome: answered if every citation passed, else refused.
    8. Build and write the audit record.
    9. Return an `AnswerResult` to the caller.

The contract: the caller never gets the model's raw response if it
contained a hallucinated or unsupported citation. They get the refusal
phrase and a refusal_reason instead. The full raw response is preserved
in the audit log for examiner review.

Future v2 — agentic mode — would replace steps 2-4 with a multi-turn
loop. Steps 5-9 stay identical. That's the whole reason the contract
surface lives here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from regrag.audit import AuditLogger, AuditRecord, make_record
from regrag.chunking import Chunk
from regrag.citations import CitationExtractor, CitationVerifier
from regrag.citations.types import VerifiedCitation
from regrag.models.base import LLMAdapter
from regrag.pii import PIIScrubber, RegexPIIScrubber
from regrag.prompt import (
    PROMPT_TEMPLATE_VERSION,
    REFUSAL_PHRASE,
    SYSTEM_PROMPT,
    render_full_prompt,
    render_user_message,
)
from regrag.retrieval.hybrid import HybridRetriever


class AnswerResult(BaseModel):
    """What the caller of `answer_query` gets back."""

    model_config = ConfigDict(frozen=True)

    answer: str = Field(description="What to show to the user — real answer or refusal phrase.")
    outcome: Literal["answered", "refused"]
    refusal_reason: str | None = Field(default=None)
    audit_query_id: str = Field(description="UUID of the audit record for this query.")
    audit_record: AuditRecord
    citations: list[VerifiedCitation] = Field(default_factory=list)


def answer_query(
    question: str,
    *,
    retriever: HybridRetriever,
    model: LLMAdapter,
    audit_logger: AuditLogger,
    corpus: list[Chunk],
    corpus_snapshot_date: str,
    pii_scrubber: PIIScrubber | None = None,
    top_k: int = 5,
    refusal_message: str = REFUSAL_PHRASE,
) -> AnswerResult:
    """Run one query through the full RAG pipeline.

    Args:
        question: User's natural-language question.
        retriever: Configured `HybridRetriever` (BM25 + dense + reranker).
            Must already have the corpus added.
        model: Any `LLMAdapter`.
        audit_logger: Where the audit record gets written.
        corpus: All chunks in the indexed corpus. The verifier uses this to
            check citation existence and support; the retriever's internal
            store cannot be relied on (e.g., dense store may be a different
            instance than what the verifier needs).
        corpus_snapshot_date: ISO date the corpus was ingested at. Recorded
            in the audit log for replay reproducibility.
        pii_scrubber: Defaults to RegexPIIScrubber for offline-friendliness.
            Production should pass PresidioPIIScrubber.
        top_k: Number of chunks to send to the LLM. Default 5.
        refusal_message: Returned to the caller when verification fails.

    Returns:
        `AnswerResult` with the user-facing answer (real or refusal),
        the outcome, the audit record, and per-citation verification status.
    """
    # 1. Scrub PII before anything else touches the input.
    scrubber = pii_scrubber or RegexPIIScrubber()
    scrub = scrubber.scrub(question)

    # 2. Retrieve top-k chunks.
    retrieved = retriever.search(scrub.text, k=top_k)
    chunks_for_prompt = [r.chunk for r in retrieved]

    # 3. Render the versioned prompt.
    user_message = render_user_message(scrub.text, chunks_for_prompt)
    full_prompt_for_hash = render_full_prompt(scrub.text, chunks_for_prompt)

    # 4. Call the LLM.
    generation = model.generate(system=SYSTEM_PROMPT, user=user_message)

    # 5 + 6. Extract and verify citations. Fail closed.
    extracted = CitationExtractor().extract(generation.text)
    verifier = CitationVerifier(corpus=corpus)
    verification = verifier.verify(extracted)

    # 7. Decide outcome.
    if verification.all_passed:
        outcome: Literal["answered", "refused"] = "answered"
        refusal_reason: str | None = None
        answer_to_user = generation.text
    else:
        outcome = "refused"
        refusal_reason = verification.refusal_reason
        answer_to_user = refusal_message

    # 8. Audit. Always. Same record shape for both outcomes.
    record = make_record(
        scrubbed_input=scrub.text,
        pii_redaction_count=scrub.redaction_count,
        pii_scrubber=scrubber.name,
        corpus_snapshot_date=corpus_snapshot_date,
        retrieved_chunk_ids=[c.chunk_id for c in chunks_for_prompt],
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        prompt_text=full_prompt_for_hash,
        model_name=generation.model_name,
        model_version=generation.model_version,
        response_text=generation.text,
        verification=verification,
        outcome=outcome,
        refusal_reason=refusal_reason,
    )
    audit_logger.write(record)

    # 9. Return to caller.
    return AnswerResult(
        answer=answer_to_user,
        outcome=outcome,
        refusal_reason=refusal_reason,
        audit_query_id=record.query_id,
        audit_record=record,
        citations=list(verification.citations),
    )

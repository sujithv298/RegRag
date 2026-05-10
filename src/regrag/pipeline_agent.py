"""Top-level agentic query pipeline — `answer_query_agentic`.

Mirror of `regrag.pipeline.answer_query` but with an agent loop in the
middle instead of a single LLM call. Same fail-closed verifier, same audit
schema, same return shape.

Step-by-step:

    1. Scrub PII from the user input.
    2. Build the agent's tool set (search, citation lookup, context expand).
    3. Render the agentic system prompt.
    4. Run the agent loop: LLM ↔ tools, until LLM emits a final answer
       or hits `max_turns`.
    5. Extract citations from the final answer.
    6. Verify each citation against the corpus (fail closed).
    7. Decide outcome: answered / refused.
    8. Build and write the audit record (now including the tool-call trace).
    9. Return `AnswerResult`.

The contract surface is identical to v1: the caller sees the same
`AnswerResult`. The only difference is `audit_record.tool_calls` is
populated. That's how examiners can replay every decision the LLM made.
"""

from __future__ import annotations

from typing import Literal

from regrag.agent import (
    AgentBudgetExceeded,
    AgentTrace,
    ExpandChunkContext,
    GetChunkByCitation,
    SearchRegulations,
    Tool,
    run_agent_loop,
)
from regrag.audit import AuditLogger, make_record
from regrag.chunking import Chunk
from regrag.citations import CitationExtractor, CitationVerifier
from regrag.pii import PIIScrubber, RegexPIIScrubber
from regrag.pipeline import AnswerResult
from regrag.prompt import PROMPT_TEMPLATE_VERSION, REFUSAL_PHRASE
from regrag.retrieval.hybrid import HybridRetriever

AGENT_PROMPT_TEMPLATE_VERSION = "agent-0.1.0"

AGENT_SYSTEM_PROMPT = f"""You are a regulatory compliance assistant for US banking regulations.

You answer questions using only regulation text retrieved through your tools. \
You may call tools multiple times to refine your understanding before answering.

Rules:

1. Use ONLY text returned by your tools. Do not draw on prior training knowledge.

2. Every factual claim in your final answer must be followed by an inline \
citation in the format [CFR:<citation>], for example [CFR:1005.6(b)(1)] for a \
rule paragraph or [CFR:Comment 6(b)-1] for an Official Interpretation comment.

3. If your tools do not return enough information to answer, respond with \
exactly this phrase and nothing else: "{REFUSAL_PHRASE}"

4. Paraphrase in plain English. Do not speculate, extrapolate, or give legal advice.

5. Do not invent citations. If language you want to cite is not in tool output, \
do not write it.

6. Be efficient. Most questions need at most 2-3 tool calls. Do not search \
exhaustively when you have enough to answer."""


def answer_query_agentic(
    question: str,
    *,
    retriever: HybridRetriever,
    model,
    audit_logger: AuditLogger,
    corpus: list[Chunk],
    corpus_snapshot_date: str,
    pii_scrubber: PIIScrubber | None = None,
    max_turns: int = 6,
    refusal_message: str = REFUSAL_PHRASE,
) -> AnswerResult:
    """Run one agentic query through PII → agent loop → verify → audit.

    `model` must implement `agent_turn(system, user, tools, history) -> AgentTurn`.
    `FakeAdapter` (with `agent_responder=`) and the upcoming `AnthropicAgentAdapter`
    both qualify.
    """
    # 1. PII scrub.
    scrubber = pii_scrubber or RegexPIIScrubber()
    scrub = scrubber.scrub(question)

    # 2. Build tools.
    tools: list[Tool] = [
        SearchRegulations(retriever=retriever),
        GetChunkByCitation(corpus=corpus),
        ExpandChunkContext(corpus=corpus),
    ]

    # 3 + 4. Run the agent loop.
    final_text: str
    trace: AgentTrace
    budget_exceeded = False
    try:
        loop_result = run_agent_loop(
            adapter=model,
            system=AGENT_SYSTEM_PROMPT,
            user=scrub.text,
            tools=tools,
            max_turns=max_turns,
        )
        final_text = loop_result.final_text
        trace = loop_result.trace
    except AgentBudgetExceeded as exc:
        # Treat budget exhaustion as a refusal with a specific reason. The
        # audit log captures the partial trace.
        final_text = refusal_message
        trace = AgentTrace(steps=[])  # no usable trace; orchestrator already logged
        budget_exceeded = True
        del exc

    # 5 + 6. Extract and verify citations.
    extracted = CitationExtractor().extract(final_text)
    verifier = CitationVerifier(corpus=corpus)
    verification = verifier.verify(extracted)

    # 7. Decide outcome.
    if budget_exceeded:
        outcome: Literal["answered", "refused"] = "refused"
        refusal_reason: str | None = "max_turns_exceeded"
        answer_to_user = refusal_message
    elif verification.all_passed:
        outcome = "answered"
        refusal_reason = None
        answer_to_user = final_text
    else:
        outcome = "refused"
        refusal_reason = verification.refusal_reason
        answer_to_user = refusal_message

    # 8. Audit. Tool-call trace goes into the audit record.
    retrieved_chunk_ids = _chunk_ids_from_trace(trace, corpus)
    record = make_record(
        scrubbed_input=scrub.text,
        pii_redaction_count=scrub.redaction_count,
        pii_scrubber=scrubber.name,
        corpus_snapshot_date=corpus_snapshot_date,
        retrieved_chunk_ids=retrieved_chunk_ids,
        prompt_template_version=AGENT_PROMPT_TEMPLATE_VERSION,
        prompt_text=AGENT_SYSTEM_PROMPT + "\n\n" + scrub.text,
        model_name=getattr(model, "name", "unknown"),
        model_version=getattr(model, "version", "unknown"),
        response_text=final_text,
        verification=verification,
        outcome=outcome,
        refusal_reason=refusal_reason,
    )
    audit_logger.write(record)

    # 9. Return.
    return AnswerResult(
        answer=answer_to_user,
        outcome=outcome,
        refusal_reason=refusal_reason,
        audit_query_id=record.query_id,
        audit_record=record,
        citations=list(verification.citations),
    )


def _chunk_ids_from_trace(trace: AgentTrace, corpus: list[Chunk]) -> list[str]:
    """Extract every chunk_id mentioned in tool results.

    The agent might mention chunks via search_regulations (returns multiple
    chunk_ids) or get_chunk_by_citation (returns one chunk_id). We
    de-duplicate while preserving order of first occurrence.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    valid_ids = {c.chunk_id for c in corpus}
    for step in trace.steps:
        content = step.result.content
        if not isinstance(content, str):
            continue
        # The tool output strings include `chunk_id=...` lines (see
        # tools._format_chunks_for_llm). Parse them out.
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("[") and "chunk_id=" in stripped:
                candidate = stripped.split("chunk_id=", 1)[1].split()[0]
                if candidate in valid_ids and candidate not in seen:
                    seen.add(candidate)
                    ordered.append(candidate)
    return ordered

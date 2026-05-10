"""End-to-end tests for the agent loop + agentic pipeline.

Uses FakeAdapter in agent_responder mode to drive the orchestrator
deterministically. Every test exercises the same code paths a real
LLM-with-tool-use would.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from regrag.agent import (
    AgentBudgetExceeded,
    AgentTurn,
    ExpandChunkContext,
    GetChunkByCitation,
    SearchRegulations,
    ToolCall,
    run_agent_loop,
)
from regrag.audit import AuditLogger
from regrag.chunking import chunk_nodes
from regrag.ingest import parse_part_xml
from regrag.models import FakeAdapter
from regrag.pipeline_agent import answer_query_agentic
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


# ---- Tool implementations ----


def test_search_regulations_returns_chunks(retriever: HybridRetriever) -> None:
    tool = SearchRegulations(retriever=retriever)
    out = tool.run({"query": "consumer liability for unauthorized transfers", "k": 3})
    assert "chunk_id=" in out
    assert "1005.6" in out


def test_search_regulations_returns_response_for_offtopic_query(
    retriever: HybridRetriever,
) -> None:
    """Hybrid retrieval *always* returns top-k from the dense leg even for
    off-topic queries (cosine similarity is never exactly zero between two
    normalized vectors). That's a real property of vector retrievers worth
    pinning here — the agent has to evaluate relevance from the *content*
    of returned chunks, not from a "no matches" signal at the retrieval layer.
    """
    tool = SearchRegulations(retriever=retriever)
    out = tool.run({"query": "xylophone narwhal blockchain"})
    # Tool must always return a string response, never crash.
    assert isinstance(out, str)
    assert len(out) > 0


def test_search_regulations_requires_query(retriever: HybridRetriever) -> None:
    tool = SearchRegulations(retriever=retriever)
    out = tool.run({})
    assert "Error" in out


def test_get_chunk_by_citation_finds_existing(corpus: list) -> None:
    tool = GetChunkByCitation(corpus=corpus)
    out = tool.run({"citation_path": "12 CFR 1005.6(b)(1)"})
    assert "1005.6(b)(1)" in out
    assert "Timely notice given" in out


def test_get_chunk_by_citation_handles_missing(corpus: list) -> None:
    tool = GetChunkByCitation(corpus=corpus)
    out = tool.run({"citation_path": "12 CFR 9999.99(z)"})
    assert "No chunk found" in out


def test_expand_chunk_context_returns_neighbors(corpus: list) -> None:
    tool = ExpandChunkContext(corpus=corpus)
    target = next(c for c in corpus if c.section == "1005.6")
    out = tool.run({"chunk_id": target.chunk_id, "neighbors": 1})
    assert target.citation_path in out
    assert "neighbor" in out.lower() or len(out.splitlines()) > 5


# ---- Agent orchestration loop ----


def _scripted_responder(script: list[AgentTurn]):
    """Build an agent_responder that returns turns from `script` in order."""
    iterator = iter(script)

    def respond(system, user, tools, history):
        try:
            return next(iterator)
        except StopIteration as exc:
            raise AssertionError(
                f"agent ran past end of script after {len(history)} steps"
            ) from exc

    return respond


def test_orchestrator_runs_one_tool_then_returns(retriever: HybridRetriever) -> None:
    tool = SearchRegulations(retriever=retriever)
    script = [
        AgentTurn(
            tool_call=ToolCall(
                name="search_regulations",
                arguments={"query": "consumer liability"},
                call_id="call-1",
            )
        ),
        AgentTurn(text="Final answer with [CFR:1005.6(b)(1)]"),
    ]
    adapter = FakeAdapter(agent_responder=_scripted_responder(script))
    result = run_agent_loop(adapter=adapter, system="sys", user="usr", tools=[tool], max_turns=5)
    assert result.final_text == "Final answer with [CFR:1005.6(b)(1)]"
    assert result.trace.num_tool_calls == 1
    assert result.trace.steps[0].call.name == "search_regulations"
    assert "1005.6" in str(result.trace.steps[0].result.content)


def test_orchestrator_handles_unknown_tool(retriever: HybridRetriever) -> None:
    """LLM hallucinating a tool name should produce an error result, not crash."""
    tool = SearchRegulations(retriever=retriever)
    script = [
        AgentTurn(tool_call=ToolCall(name="this_tool_does_not_exist", arguments={}, call_id="c1")),
        AgentTurn(text="recovered with [CFR:1005.6(b)(1)]"),
    ]
    adapter = FakeAdapter(agent_responder=_scripted_responder(script))
    result = run_agent_loop(adapter=adapter, system="sys", user="usr", tools=[tool], max_turns=5)
    assert result.trace.steps[0].result.is_error
    assert "unknown tool" in str(result.trace.steps[0].result.content)


def test_orchestrator_raises_on_budget_exceeded(retriever: HybridRetriever) -> None:
    tool = SearchRegulations(retriever=retriever)
    script = [
        AgentTurn(
            tool_call=ToolCall(name="search_regulations", arguments={"query": "x"}, call_id=f"c{i}")
        )
        for i in range(10)
    ]
    adapter = FakeAdapter(agent_responder=_scripted_responder(script))
    with pytest.raises(AgentBudgetExceeded):
        run_agent_loop(adapter=adapter, system="sys", user="usr", tools=[tool], max_turns=3)


def test_orchestrator_requires_at_least_one_tool() -> None:
    adapter = FakeAdapter(agent_responder=lambda *a, **k: AgentTurn(text="x"))
    with pytest.raises(ValueError):
        run_agent_loop(adapter=adapter, system="s", user="u", tools=[], max_turns=3)


# ---- End-to-end agentic pipeline ----


def test_agentic_pipeline_answered_path(corpus: list, retriever: HybridRetriever, tmp_path) -> None:
    """The headline test: question → agent searches → final answer with
    verifiable citation → outcome=answered, audit record written."""

    def responder(system, user, tools, history):
        if not history:
            return AgentTurn(
                tool_call=ToolCall(
                    name="search_regulations",
                    arguments={"query": "consumer liability two business days"},
                    call_id="c1",
                )
            )
        # After one tool call, produce a final answer that cites the canonical paragraph.
        return AgentTurn(
            text=(
                "Within two business days, the consumer's liability is the lesser "
                "of $50 or the unauthorized transfers that occurred [CFR:1005.6(b)(1)]."
            )
        )

    logger = AuditLogger(log_path=tmp_path / "audit.jsonl")
    result = answer_query_agentic(
        "What is the consumer's max liability for an unauthorized transfer reported within two business days?",
        retriever=retriever,
        model=FakeAdapter(agent_responder=responder, name="fake-agent", version="v0"),
        audit_logger=logger,
        corpus=corpus,
        corpus_snapshot_date="2025-01-01",
    )

    assert result.outcome == "answered"
    assert "[CFR:1005.6(b)(1)]" in result.answer
    assert any(c.citation_path == "12 CFR 1005.6(b)(1)" and c.passed for c in result.citations)


def test_agentic_pipeline_fail_closed_on_hallucination(
    corpus: list, retriever: HybridRetriever, tmp_path
) -> None:
    """Even after multiple agent turns, hallucinated citations must trigger refusal."""

    def responder(system, user, tools, history):
        if not history:
            return AgentTurn(
                tool_call=ToolCall(
                    name="search_regulations",
                    arguments={"query": "cryptocurrency reporting"},
                    call_id="c1",
                )
            )
        return AgentTurn(text="Crypto exchanges have reporting obligations [CFR:9999.99(z)].")

    logger = AuditLogger(log_path=tmp_path / "audit.jsonl")
    result = answer_query_agentic(
        "What does Reg E say about cryptocurrency reporting?",
        retriever=retriever,
        model=FakeAdapter(agent_responder=responder),
        audit_logger=logger,
        corpus=corpus,
        corpus_snapshot_date="2025-01-01",
    )

    assert result.outcome == "refused"
    assert result.refusal_reason == "unverified_citation"
    assert "cannot answer" in result.answer.lower()


def test_agentic_pipeline_max_turns_refuses(
    corpus: list, retriever: HybridRetriever, tmp_path
) -> None:
    """Agent that loops forever calling tools without ever finalizing should
    refuse with refusal_reason=max_turns_exceeded."""

    def responder(system, user, tools, history):
        return AgentTurn(
            tool_call=ToolCall(name="search_regulations", arguments={"query": "x"}, call_id="c")
        )

    logger = AuditLogger(log_path=tmp_path / "audit.jsonl")
    result = answer_query_agentic(
        "test",
        retriever=retriever,
        model=FakeAdapter(agent_responder=responder),
        audit_logger=logger,
        corpus=corpus,
        corpus_snapshot_date="2025-01-01",
        max_turns=2,
    )
    assert result.outcome == "refused"
    assert result.refusal_reason == "max_turns_exceeded"


def test_agentic_pipeline_writes_audit_record(
    corpus: list, retriever: HybridRetriever, tmp_path
) -> None:
    log_path = tmp_path / "audit.jsonl"

    def responder(system, user, tools, history):
        if not history:
            return AgentTurn(
                tool_call=ToolCall(
                    name="search_regulations",
                    arguments={"query": "consumer liability"},
                    call_id="c1",
                )
            )
        return AgentTurn(text="Within two business days liability is $50 [CFR:1005.6(b)(1)].")

    logger = AuditLogger(log_path=log_path)
    answer_query_agentic(
        "q",
        retriever=retriever,
        model=FakeAdapter(agent_responder=responder, name="fake-agent", version="v0"),
        audit_logger=logger,
        corpus=corpus,
        corpus_snapshot_date="2025-01-01",
    )

    records = logger.read_all()
    assert len(records) == 1
    record = records[0]
    assert record.model_name == "fake-agent"
    assert record.prompt_template_version == "agent-0.1.0"
    assert record.outcome == "answered"
    assert record.retrieved_chunk_ids  # populated from tool trace

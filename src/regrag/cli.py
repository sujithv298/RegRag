"""Command-line interface for RegRAG.

Three top-level commands map to the three things a forker actually does:

    regrag ingest --part 1005             # offline, one-shot: pull eCFR XML, parse, chunk
    regrag query "..."                    # online, per request: answer with verified citations
    regrag eval --gold evals/gold/...     # batch: run the gold set, emit a metrics report

Two debugging utilities:

    regrag inspect --xml path/to.xml          # parse a local XML file, print first N nodes
    regrag inspect-chunks --nodes path.jsonl  # chunk an existing parsed-nodes JSONL
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import click

# Load .env early so subsequent SDK clients pick up keys automatically.
# This is a no-op if python-dotenv isn't installed (e.g., minimal CI install).
try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

from datetime import UTC

from regrag.agent import AgentTurn, ToolCall
from regrag.audit import AuditLogger
from regrag.chunking import Chunk, chunk_nodes
from regrag.ingest import fetch_part_xml, parse_part_xml
from regrag.models import FakeAdapter
from regrag.pipeline import answer_query
from regrag.pipeline_agent import answer_query_agentic
from regrag.retrieval import (
    BM25Index,
    DenseRetriever,
    HashingEmbedder,
    HybridRetriever,
    LexicalOverlapReranker,
)
from regrag.store import InMemoryVectorStore

DEFAULT_SNAPSHOT_DATE = "2025-01-01"


@click.group()
@click.version_option()
def main() -> None:
    """RegRAG — verified-citation RAG over US banking regulations."""


@main.command()
@click.option("--part", type=int, required=True, help="CFR Part number, e.g. 1005 for Reg E.")
@click.option("--title", type=int, default=12, show_default=True, help="CFR Title number.")
@click.option(
    "--date",
    default=DEFAULT_SNAPSHOT_DATE,
    show_default=True,
    help="eCFR snapshot date (ISO-8601). Pinned for audit-log reproducibility.",
)
@click.option(
    "--out",
    type=click.Path(),
    default="data",
    show_default=True,
    help="Where to write the raw XML, parsed-nodes JSONL, and chunks JSONL.",
)
def ingest(part: int, title: int, date: str, out: str) -> None:
    """Fetch a CFR Part, parse it into hierarchy nodes, chunk for retrieval.

    Phase 2 lands fetch + parse. Phase 3 adds chunking. Indexing into Chroma
    + BM25 lands in Phase 4.
    """
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"title-{title}-part-{part}-{date}"
    xml_path = out_dir / f"{base}.xml"
    nodes_path = out_dir / f"{base}.nodes.jsonl"
    chunks_path = out_dir / f"{base}.chunks.jsonl"

    click.echo(f"Fetching 12 CFR Part {part} as of {date} ...")
    xml_bytes = asyncio.run(fetch_part_xml(title=title, part=part, date=date))
    xml_path.write_bytes(xml_bytes)
    click.echo(f"  Wrote {len(xml_bytes):,} bytes to {xml_path}")

    click.echo("Parsing into hierarchy nodes ...")
    nodes = parse_part_xml(xml_bytes, title=title)
    with nodes_path.open("w") as f:
        for node in nodes:
            f.write(node.model_dump_json() + "\n")
    click.echo(f"  Wrote {len(nodes)} nodes to {nodes_path}")

    click.echo("Chunking for retrieval ...")
    chunks = chunk_nodes(nodes)
    with chunks_path.open("w") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")
    click.echo(f"  Wrote {len(chunks)} chunks to {chunks_path}")
    split_count = sum(1 for c in chunks if c.is_split)
    if split_count:
        click.echo(f"  ({split_count} chunks resulted from splitting long paragraphs)")


@main.command()
@click.option("--xml", "xml_path", type=click.Path(exists=True), required=True)
@click.option("--limit", type=int, default=10, show_default=True)
def inspect(xml_path: str, limit: int) -> None:
    """Parse a local XML file and print the first N parsed nodes. Debugging aid."""
    xml_bytes = Path(xml_path).read_bytes()
    nodes = parse_part_xml(xml_bytes)
    click.echo(f"Parsed {len(nodes)} nodes from {xml_path}\n")
    for node in nodes[:limit]:
        click.echo(
            json.dumps(
                {
                    "citation_path": node.citation_path,
                    "chunk_id": node.chunk_id,
                    "is_interpretation": node.is_interpretation,
                    "text_preview": node.text[:120] + ("..." if len(node.text) > 120 else ""),
                },
                indent=2,
            )
        )


@main.command(name="inspect-chunks")
@click.option("--xml", "xml_path", type=click.Path(exists=True), required=True)
@click.option("--limit", type=int, default=3, show_default=True)
def inspect_chunks(xml_path: str, limit: int) -> None:
    """Parse a local XML file, chunk it, and print the first N chunks in full."""
    xml_bytes = Path(xml_path).read_bytes()
    nodes = parse_part_xml(xml_bytes)
    chunks = chunk_nodes(nodes)
    click.echo(f"Parsed {len(nodes)} nodes -> {len(chunks)} chunks\n")
    for chunk in chunks[:limit]:
        click.echo(f"--- chunk_id: {chunk.chunk_id} ---")
        click.echo(chunk.text)
        click.echo()


@main.command()
@click.argument("question")
@click.option(
    "--chunks",
    "chunks_path",
    type=click.Path(exists=True),
    required=True,
    help="Chunks JSONL file produced by `regrag ingest`.",
)
@click.option("--k", type=int, default=5, show_default=True, help="Top-k results to return.")
@click.option(
    "--retriever",
    type=click.Choice(["bm25", "dense", "hybrid", "all"]),
    default="all",
    show_default=True,
    help="Which retriever(s) to run. 'all' shows BM25, dense, and hybrid side-by-side.",
)
def search(question: str, chunks_path: str, k: int, retriever: str) -> None:
    """Search a chunks JSONL with BM25, dense, hybrid, or all side-by-side."""
    chunks = [
        Chunk.model_validate_json(line)
        for line in Path(chunks_path).read_text().splitlines()
        if line.strip()
    ]
    click.echo(f'Search "{question}"  (corpus: {len(chunks)} chunks, top {k})\n')

    if retriever in ("bm25", "all"):
        bm25 = BM25Index()
        bm25.add(chunks)
        click.echo("--- BM25 (keyword) ---")
        _render_results(bm25.search(question, k=k))

    if retriever in ("dense", "all"):
        dense = DenseRetriever(
            embedder=HashingEmbedder(),  # offline stub; swap to BGEEmbedder in production
            store=InMemoryVectorStore(),
        )
        dense.add(chunks)
        click.echo("--- dense (HashingEmbedder + InMemoryVectorStore) ---")
        _render_results(dense.search(question, k=k))

    if retriever in ("hybrid", "all"):
        hybrid = HybridRetriever(
            bm25=BM25Index(),
            dense=DenseRetriever(embedder=HashingEmbedder(), store=InMemoryVectorStore()),
            reranker=LexicalOverlapReranker(),  # offline stub; swap to CrossEncoderReranker
        )
        hybrid.add(chunks)
        click.echo("--- hybrid (BM25 + dense → RRF → lexical-overlap rerank) ---")
        _render_results(hybrid.search(question, k=k))


def _render_results(results: list) -> None:
    if not results:
        click.echo("  no matches.\n")
        return
    for i, sc in enumerate(results, 1):
        preview = sc.chunk.source_text[:140] + ("..." if len(sc.chunk.source_text) > 140 else "")
        click.echo(f"{i}. [{sc.retriever} {sc.score:.3f}] {sc.chunk.citation_path}")
        click.echo(f"   {preview}")
    click.echo()


@main.command()
@click.argument("question")
@click.option(
    "--chunks",
    "chunks_path",
    type=click.Path(exists=True),
    required=True,
    help="Chunks JSONL file produced by `regrag ingest`.",
)
@click.option(
    "--model",
    type=click.Choice(["fake-good", "fake-bad", "anthropic", "openai"]),
    default="fake-good",
    show_default=True,
    help="Which model adapter to use. fake-* run offline; anthropic/openai need API keys.",
)
@click.option(
    "--mode",
    type=click.Choice(["deterministic", "agent"]),
    default="deterministic",
    show_default=True,
    help="Pipeline shape: one-shot RAG (deterministic) or tool-using agent loop.",
)
@click.option(
    "--audit-log",
    type=click.Path(),
    default="audit.jsonl",
    show_default=True,
)
@click.option("--snapshot-date", default=DEFAULT_SNAPSHOT_DATE, show_default=True)
def query(
    question: str,
    chunks_path: str,
    model: str,
    mode: str,
    audit_log: str,
    snapshot_date: str,
) -> None:
    """Answer a question end-to-end: PII scrub → retrieve → LLM → verify → audit."""
    chunks = [
        Chunk.model_validate_json(line)
        for line in Path(chunks_path).read_text().splitlines()
        if line.strip()
    ]

    retriever = HybridRetriever(
        bm25=BM25Index(),
        dense=DenseRetriever(embedder=HashingEmbedder(), store=InMemoryVectorStore()),
        reranker=LexicalOverlapReranker(),
    )
    retriever.add(chunks)

    logger = AuditLogger(log_path=audit_log)

    if mode == "agent":
        adapter = _build_agent_adapter(model)
        result = answer_query_agentic(
            question,
            retriever=retriever,
            model=adapter,
            audit_logger=logger,
            corpus=chunks,
            corpus_snapshot_date=snapshot_date,
        )
    else:
        adapter = _build_adapter(model, retriever, question)
        result = answer_query(
            question,
            retriever=retriever,
            model=adapter,
            audit_logger=logger,
            corpus=chunks,
            corpus_snapshot_date=snapshot_date,
        )

    click.echo(f"Question: {question}\n")
    click.echo(f"Outcome: {result.outcome}")
    if result.refusal_reason:
        click.echo(f"Refusal reason: {result.refusal_reason}")
    click.echo(f"\nAnswer:\n{result.answer}\n")
    click.echo("Citations:")
    for c in result.citations:
        status = "PASS" if c.passed else f"FAIL ({c.failure_reason})"
        click.echo(
            f"  - {c.citation_path}  exists={c.exists}  supports={c.supports}  "
            f"score={c.support_score:.2f}  → {status}"
        )
    click.echo(f"\nAudit record: {result.audit_query_id}  →  {audit_log}")


_EXCERPT1_RE = re.compile(
    r"\[Excerpt 1\]\s+Citation:\s+([^\n]+)\n[^\n]+\n\n([\s\S]+?)(?=\n\n---|\Z)"
)
_QUESTION_RE = re.compile(r"Question:\s*(.+?)(?=\n\n|\Z)", re.DOTALL)

# Below this fraction of question tokens overlapping the top chunk body,
# fake-good emits the refusal phrase. Calibrated to refuse genuinely
# out-of-scope queries (Reg Z, FDIC, etc.) without false-positive-refusing
# in-scope ones. A real LLM does this implicitly by following the prompt;
# fake-good needs an explicit threshold.
_FAKE_GOOD_REFUSAL_THRESHOLD = 0.2

_REFUSAL = "I cannot answer this question based on the provided regulatory text."


def _fake_good_responder(system: str, user: str) -> str:
    """Echo the top retrieved chunk with citation, OR refuse if the chunk
    doesn't substantively overlap the question (mimics LLM-following-prompt)."""
    del system  # unused

    excerpt_match = _EXCERPT1_RE.search(user)
    question_match = _QUESTION_RE.search(user)
    if not excerpt_match or not question_match:
        return _REFUSAL

    citation = excerpt_match.group(1).strip()
    body = excerpt_match.group(2).strip()
    question = question_match.group(1).strip()

    # Lazy-import to avoid a top-level retrieval dep cycle.
    from regrag.retrieval import tokenize, tokenize_query

    q_tokens = set(tokenize_query(question))
    body_tokens = set(tokenize(body))
    if not q_tokens:
        return _REFUSAL
    overlap = len(q_tokens & body_tokens) / len(q_tokens)
    if overlap < _FAKE_GOOD_REFUSAL_THRESHOLD:
        return _REFUSAL

    citation_payload = citation.replace("12 CFR ", "")
    body_clean = body.rstrip(".").rstrip()
    return f"{body_clean} [CFR:{citation_payload}]."


def _build_adapter(model_choice: str, retriever, question: str):
    """Build an LLM adapter for the chosen mode.

    `fake-good` constructs a plausible response using the top retrieved
    chunk, demonstrating the answered path. `fake-bad` emits a hallucinated
    citation, demonstrating the refusal path. The real adapters require
    API keys and live network access.
    """
    if model_choice == "fake-good":
        # Use a callable so the response is built per-query at generate-time.
        # This matters for the eval harness where one adapter handles many
        # questions; the rendered user message tells us what was retrieved
        # for *this* question.
        return FakeAdapter(
            response=_fake_good_responder,
            name="fake-good",
            version="v0",
        )
    if model_choice == "fake-bad":
        return FakeAdapter(
            response=(
                "Cryptocurrency exchanges have specific obligations to report "
                "quarterly under the Electronic Fund Transfer Act [CFR:9999.99(z)]."
            ),
            name="fake-bad",
            version="v0",
        )
    if model_choice == "anthropic":
        from regrag.models import AnthropicAdapter

        return AnthropicAdapter()
    if model_choice == "openai":
        from regrag.models import OpenAIAdapter

        return OpenAIAdapter()
    raise click.UsageError(f"Unknown --model {model_choice}")


# ---- Agent-mode adapter builders ----


_TOOL_CITATION_RE = re.compile(r"citation:\s+([^\n]+)")
_TOOL_TEXT_RE = re.compile(r"text:\s+([\s\S]+?)(?=\n\[|\Z)")


def _fake_good_agent_responder(system: str, user: str, tools, history) -> AgentTurn:
    """Two-step agent: search_regulations once, then finalize with the top chunk.

    Mirrors the simplest realistic agent pattern. Real Claude with tool-use
    will produce more interesting trajectories; this stub demonstrates the
    loop end-to-end offline.
    """
    del system, tools  # unused in this stub
    if not history:
        return AgentTurn(
            tool_call=ToolCall(
                name="search_regulations",
                arguments={"query": user, "k": 3},
                call_id="call-1",
            )
        )
    last_result_text = str(history[-1].result.content)
    citation_match = _TOOL_CITATION_RE.search(last_result_text)
    text_match = _TOOL_TEXT_RE.search(last_result_text)
    if not citation_match or not text_match:
        return AgentTurn(
            text="I cannot answer this question based on the provided regulatory text."
        )
    citation = citation_match.group(1).strip()
    body = text_match.group(1).strip().rstrip(".").rstrip()
    citation_payload = citation.replace("12 CFR ", "")
    return AgentTurn(text=f"{body} [CFR:{citation_payload}].")


def _fake_bad_agent_responder(system: str, user: str, tools, history) -> AgentTurn:
    """Agent that searches once, then hallucinates a citation. Demonstrates
    the fail-closed verifier still works in agent mode."""
    del system, user, tools  # unused
    if not history:
        return AgentTurn(
            tool_call=ToolCall(
                name="search_regulations",
                arguments={"query": "cryptocurrency", "k": 3},
                call_id="call-1",
            )
        )
    return AgentTurn(
        text=(
            "Cryptocurrency exchanges have specific obligations to report quarterly "
            "under the Electronic Fund Transfer Act [CFR:9999.99(z)]."
        )
    )


def _build_agent_adapter(model_choice: str):
    if model_choice == "fake-good":
        return FakeAdapter(
            agent_responder=_fake_good_agent_responder,
            name="fake-good-agent",
            version="v0",
        )
    if model_choice == "fake-bad":
        return FakeAdapter(
            agent_responder=_fake_bad_agent_responder,
            name="fake-bad-agent",
            version="v0",
        )
    if model_choice == "anthropic":
        from regrag.models import AnthropicAgentAdapter

        return AnthropicAgentAdapter()
    if model_choice == "openai":
        from regrag.models import OpenAIAgentAdapter

        return OpenAIAgentAdapter()
    raise click.UsageError(f"Unknown --model {model_choice}")


@main.command(name="eval-compare")
@click.option("--gold", type=click.Path(exists=True), required=True)
@click.option("--chunks", "chunks_path", type=click.Path(exists=True), required=True)
@click.option(
    "--model",
    type=click.Choice(["fake-good"]),
    default="fake-good",
    show_default=True,
    help="Model adapter for both pipelines (real LLMs need Phase 8b's Anthropic agent adapter wired in).",
)
@click.option(
    "--audit-log", type=click.Path(), default="audit-eval-compare.jsonl", show_default=True
)
@click.option("--out", type=click.Path(), default="evals/reports", show_default=True)
@click.option("--snapshot-date", default=DEFAULT_SNAPSHOT_DATE, show_default=True)
def eval_compare(
    gold: str,
    chunks_path: str,
    model: str,
    audit_log: str,
    out: str,
    snapshot_date: str,
) -> None:
    """Run the gold set through deterministic AND agent pipelines; print a side-by-side comparison."""
    from datetime import datetime

    from evals import (
        compare_reports,
        format_comparison,
        load_gold_set,
        run_eval,
    )
    from regrag.pipeline_agent import answer_query_agentic

    chunks = [
        Chunk.model_validate_json(line)
        for line in Path(chunks_path).read_text().splitlines()
        if line.strip()
    ]
    gold_set = load_gold_set(gold)

    retriever = HybridRetriever(
        bm25=BM25Index(),
        dense=DenseRetriever(embedder=HashingEmbedder(), store=InMemoryVectorStore()),
        reranker=LexicalOverlapReranker(),
    )
    retriever.add(chunks)
    logger = AuditLogger(log_path=audit_log)

    click.echo("Running deterministic pipeline ...")
    det_report = run_eval(
        gold_set,
        retriever=retriever,
        model=_build_adapter(model, retriever, ""),
        audit_logger=logger,
        corpus=chunks,
        corpus_snapshot_date=snapshot_date,
        gold_set_path=gold,
    )

    click.echo("Running agent pipeline ...")
    agent_report = run_eval(
        gold_set,
        retriever=retriever,
        model=_build_agent_adapter(model),
        audit_logger=logger,
        corpus=chunks,
        corpus_snapshot_date=snapshot_date,
        gold_set_path=gold,
        pipeline_fn=answer_query_agentic,
        prompt_template_version="agent-0.1.0",
    )

    comparison = compare_reports(deterministic=det_report, agent=agent_report, gold_set_path=gold)
    click.echo("\n" + format_comparison(comparison))

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = out_dir / f"comparison-{stamp}.json"
    report_path.write_text(comparison.model_dump_json(indent=2))
    click.echo(f"\nReport: {report_path}")


@main.command(name="eval")
@click.option("--gold", type=click.Path(exists=True), required=True, help="Gold-set JSONL file.")
@click.option(
    "--chunks",
    "chunks_path",
    type=click.Path(exists=True),
    required=True,
    help="Chunks JSONL produced by `regrag ingest`.",
)
@click.option(
    "--model",
    type=click.Choice(["fake-good", "anthropic", "openai"]),
    default="fake-good",
    show_default=True,
)
@click.option("--audit-log", type=click.Path(), default="audit-eval.jsonl", show_default=True)
@click.option(
    "--out",
    type=click.Path(),
    default="evals/reports",
    show_default=True,
    help="Directory to write the JSON report into.",
)
@click.option("--snapshot-date", default=DEFAULT_SNAPSHOT_DATE, show_default=True)
def eval_(
    gold: str,
    chunks_path: str,
    model: str,
    audit_log: str,
    out: str,
    snapshot_date: str,
) -> None:
    """Run the eval harness against a gold set; print metrics; write JSON report."""
    # Local imports so click sees the command without paying the eval-runner
    # import cost on every CLI invocation.
    from datetime import datetime

    from evals import load_gold_set, run_eval

    chunks = [
        Chunk.model_validate_json(line)
        for line in Path(chunks_path).read_text().splitlines()
        if line.strip()
    ]
    gold_set = load_gold_set(gold)

    retriever = HybridRetriever(
        bm25=BM25Index(),
        dense=DenseRetriever(embedder=HashingEmbedder(), store=InMemoryVectorStore()),
        reranker=LexicalOverlapReranker(),
    )
    retriever.add(chunks)

    adapter = _build_adapter(model, retriever, "")  # question is per-case in eval
    logger = AuditLogger(log_path=audit_log)

    report = run_eval(
        gold_set,
        retriever=retriever,
        model=adapter,
        audit_logger=logger,
        corpus=chunks,
        corpus_snapshot_date=snapshot_date,
        gold_set_path=gold,
    )

    # Pretty-print the headline.
    click.echo(f"\nGold set: {gold}  ({report.overall.n_cases} cases)")
    click.echo(f"Model:    {report.model_name}")
    click.echo(f"Prompt:   v{report.prompt_template_version}\n")

    click.echo("Overall:")
    _print_metrics(report.overall, indent=2)

    click.echo("\nBy category:")
    for cat, metrics in report.by_category.items():
        click.echo(f"  [{cat}]")
        _print_metrics(metrics, indent=4)

    click.echo("\nFailing cases:")
    failing = [c for c in report.cases if not c.passed]
    if not failing:
        click.echo("  (none)")
    else:
        for c in failing:
            click.echo(f"  - {c.entry_id} ({c.category}): outcome={c.actual_outcome}")
            click.echo(
                f"      cit_correct={c.citations_correct}  "
                f"kw_present={c.keywords_present}  "
                f"outcome_match={c.outcome_matches_expected}"
            )

    # Write the report to disk.
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = out_dir / f"report-{stamp}.json"
    report_path.write_text(report.model_dump_json(indent=2))
    click.echo(f"\nReport: {report_path}")


def _print_metrics(m, *, indent: int = 0) -> None:
    pad = " " * indent
    click.echo(f"{pad}n_cases:                       {m.n_cases}")
    click.echo(f"{pad}citation_accuracy:             {m.citation_accuracy:.2f}")
    click.echo(f"{pad}answer_correctness:            {m.answer_correctness:.2f}")
    click.echo(f"{pad}refusal_rate_correct:          {m.refusal_rate_correct:.2f}")
    click.echo(f"{pad}refusal_rate_false_positive:   {m.refusal_rate_false_positive:.2f}")
    click.echo(f"{pad}pass_rate:                     {m.pass_rate:.2f}")


if __name__ == "__main__":
    main()

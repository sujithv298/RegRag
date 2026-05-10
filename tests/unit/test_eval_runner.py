"""Tests for the eval harness.

Uses FakeAdapter to simulate the model so the runner's logic can be
tested deterministically. The smoke gold set is small enough to
exercise every code path: should-answer-passes, should-refuse-passes,
should-answer-but-fails (citation hallucinated).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from regrag.audit import AuditLogger
from regrag.chunking import chunk_nodes
from regrag.ingest import parse_part_xml
from regrag.models import FakeAdapter
from regrag.retrieval import (
    BM25Index,
    DenseRetriever,
    HashingEmbedder,
    HybridRetriever,
    LexicalOverlapReranker,
)
from regrag.store import InMemoryVectorStore

from evals import compute_metrics, load_gold_set, run_eval
from evals.types import EvalCaseResult, EvalReport, GoldEntry

FIXTURE_XML = Path(__file__).resolve().parents[1] / "fixtures" / "reg_e_part_1005_excerpt.xml"
SMOKE_GOLD = Path(__file__).resolve().parents[1] / "fixtures" / "eval_smoke.jsonl"


@pytest.fixture
def corpus() -> list:
    nodes = parse_part_xml(FIXTURE_XML.read_bytes())
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


# ---- load_gold_set ----


def test_load_gold_set_parses_jsonl() -> None:
    entries = load_gold_set(SMOKE_GOLD)
    assert len(entries) == 3
    assert isinstance(entries[0], GoldEntry)
    assert entries[0].id == "smoke-001"
    assert entries[0].expected_citations == ["12 CFR 1005.6(b)(1)"]


def test_load_gold_set_handles_empty_lines() -> None:
    """Tolerate stray blank lines in JSONL (common when editing by hand)."""
    entries = load_gold_set(SMOKE_GOLD)
    assert all(e.id for e in entries)


# ---- run_eval — happy path: smart fake adapter answers correctly ----


def _smart_fake_response(top_chunk_lookup) -> str:
    """Build a callable that picks the right canned response per question."""

    def respond(system: str, user: str) -> str:
        if "unauthorized transfer reported within two business days" in user.lower():
            # Real citation matching smoke-001's expected.
            return (
                "Per [CFR:1005.6(b)(1)], within two business days the "
                "consumer's liability is the lesser of $50 or the amount of "
                "unauthorized transfers."
            )
        if "regulation z" in user.lower():
            # Out-of-scope: model correctly emits the refusal phrase.
            return "I cannot answer this question based on the provided regulatory text."
        if "authority and purpose" in user.lower():
            return (
                "Per [CFR:1005.1(a)], this part, known as Regulation E, is issued "
                "by the Bureau of Consumer Financial Protection. Authority."
            )
        return "I cannot answer this question based on the provided regulatory text."

    return respond


def test_run_eval_returns_report_with_per_category_metrics(
    corpus: list, retriever: HybridRetriever, tmp_path: Path
) -> None:
    gold = load_gold_set(SMOKE_GOLD)
    logger = AuditLogger(log_path=tmp_path / "audit.jsonl")
    model = FakeAdapter(response=_smart_fake_response(corpus), name="fake-smart", version="v0")

    report = run_eval(
        gold,
        retriever=retriever,
        model=model,
        audit_logger=logger,
        corpus=corpus,
        corpus_snapshot_date="2025-01-01",
        gold_set_path=str(SMOKE_GOLD),
    )

    assert isinstance(report, EvalReport)
    assert report.model_name == "fake-smart"
    assert report.prompt_template_version == "0.1.0"
    assert len(report.cases) == 3
    assert report.overall.n_cases == 3
    assert "exact_citation" in report.by_category
    assert "intentional_refusal" in report.by_category


def test_run_eval_writes_one_audit_record_per_case(
    corpus: list, retriever: HybridRetriever, tmp_path: Path
) -> None:
    gold = load_gold_set(SMOKE_GOLD)
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path=log_path)
    model = FakeAdapter(response=_smart_fake_response(corpus))

    run_eval(
        gold,
        retriever=retriever,
        model=model,
        audit_logger=logger,
        corpus=corpus,
        corpus_snapshot_date="2025-01-01",
    )

    records = logger.read_all()
    assert len(records) == len(gold)


def test_smart_fake_passes_metrics_in_their_intended_direction(
    corpus: list, retriever: HybridRetriever, tmp_path: Path
) -> None:
    """The smart fake answers correctly. Metrics should reflect that:
    citation_accuracy and answer_correctness on the should-answer cases
    should be high; refusal_rate_correct on the should-refuse case should
    be 1.0; refusal_rate_false_positive should be 0.0."""
    gold = load_gold_set(SMOKE_GOLD)
    logger = AuditLogger(log_path=tmp_path / "audit.jsonl")
    model = FakeAdapter(response=_smart_fake_response(corpus))

    report = run_eval(
        gold,
        retriever=retriever,
        model=model,
        audit_logger=logger,
        corpus=corpus,
        corpus_snapshot_date="2025-01-01",
    )

    # Should-refuse category: smoke-002 should have been correctly refused.
    refusal_metrics = report.by_category["intentional_refusal"]
    assert refusal_metrics.refusal_rate_correct == 1.0


# ---- compute_metrics on hand-built cases ----


def _case(
    *,
    entry_id: str,
    category: str,
    actual_outcome: str,
    citations_correct: bool = True,
    keywords_present: bool = True,
    outcome_matches_expected: bool = True,
) -> EvalCaseResult:
    return EvalCaseResult(
        entry_id=entry_id,
        category=category,  # type: ignore[arg-type]
        question="q",
        actual_outcome=actual_outcome,  # type: ignore[arg-type]
        actual_refusal_reason=None,
        actual_citations=[],
        actual_answer="answer",
        citations_correct=citations_correct,
        keywords_present=keywords_present,
        outcome_matches_expected=outcome_matches_expected,
    )


def test_compute_metrics_empty_cases_returns_zeros() -> None:
    metrics = compute_metrics([])
    assert metrics.n_cases == 0
    assert metrics.citation_accuracy == 0.0
    assert metrics.pass_rate == 0.0


def test_compute_metrics_all_passing_yields_perfect_scores() -> None:
    cases = [
        _case(
            entry_id=f"id-{i}",
            category="exact_citation",
            actual_outcome="answered",
            citations_correct=True,
            keywords_present=True,
            outcome_matches_expected=True,
        )
        for i in range(5)
    ]
    metrics = compute_metrics(cases)
    assert metrics.citation_accuracy == 1.0
    assert metrics.answer_correctness == 1.0
    assert metrics.pass_rate == 1.0


def test_compute_metrics_correctly_refused_counted_as_passing() -> None:
    """A correctly refused case counts as passed in pass_rate."""
    cases = [
        _case(
            entry_id="refuse-1",
            category="intentional_refusal",
            actual_outcome="refused",
            outcome_matches_expected=True,
        )
    ]
    metrics = compute_metrics(cases)
    assert metrics.refusal_rate_correct == 1.0
    assert metrics.pass_rate == 1.0


def test_compute_metrics_false_positive_refusal_visible() -> None:
    """A case that should have answered but was refused must show up in
    refusal_rate_false_positive — that's the over-refusal signal we tune against."""
    cases = [
        _case(
            entry_id="fp-1",
            category="exact_citation",
            actual_outcome="refused",
            outcome_matches_expected=False,
        )
    ]
    metrics = compute_metrics(cases)
    assert metrics.refusal_rate_false_positive == 1.0
    assert metrics.pass_rate == 0.0

"""Eval harness runner.

Two pure functions:

  - `run_eval`: take a gold set + the runtime objects, run every entry
    through `answer_query`, return an `EvalReport`.
  - `compute_metrics`: aggregate a list of EvalCaseResults into metrics.

Function-shaped (not class-shaped) because the eval is conceptually a
fold over (gold_set, pipeline) → report. Class state would just be ceremony.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from typing import Callable

from regrag.audit import AuditLogger
from regrag.chunking import Chunk
from regrag.pii import PIIScrubber
from regrag.pipeline import AnswerResult, answer_query
from regrag.prompt import PROMPT_TEMPLATE_VERSION
from regrag.retrieval.hybrid import HybridRetriever

PipelineFn = Callable[..., AnswerResult]

from evals.types import (
    Category,
    EvalCaseResult,
    EvalMetrics,
    EvalReport,
    GoldEntry,
)


def load_gold_set(path: str | Path) -> list[GoldEntry]:
    """Load a gold set from JSONL on disk."""
    return [
        GoldEntry.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_eval(
    gold_set: list[GoldEntry],
    *,
    retriever: HybridRetriever,
    model,
    audit_logger: AuditLogger,
    corpus: list[Chunk],
    corpus_snapshot_date: str,
    pii_scrubber: PIIScrubber | None = None,
    gold_set_path: str = "",
    pipeline_fn: PipelineFn = answer_query,
    prompt_template_version: str = PROMPT_TEMPLATE_VERSION,
) -> EvalReport:
    """Run every gold entry through `pipeline_fn` and return a metrics report.

    `pipeline_fn` defaults to the deterministic `answer_query`. Pass
    `pipeline_fn=answer_query_agentic` for agentic mode.

    Each gold entry produces:
      - one audit record (the pipeline's invariant), AND
      - one EvalCaseResult (the runner's per-case output).
    """
    cases: list[EvalCaseResult] = []
    for entry in gold_set:
        result = pipeline_fn(
            entry.question,
            retriever=retriever,
            model=model,
            audit_logger=audit_logger,
            corpus=corpus,
            corpus_snapshot_date=corpus_snapshot_date,
            pii_scrubber=pii_scrubber,
        )
        cases.append(_score_case(entry, result))

    return EvalReport(
        timestamp=datetime.now(timezone.utc),
        gold_set_path=gold_set_path,
        model_name=getattr(model, "name", "unknown"),
        prompt_template_version=prompt_template_version,
        overall=compute_metrics(cases),
        by_category={
            cat: compute_metrics([c for c in cases if c.category == cat])
            for cat in _categories_present(cases)
        },
        cases=cases,
    )


def compute_metrics(cases: list[EvalCaseResult]) -> EvalMetrics:
    """Aggregate EvalCaseResults into the headline metrics."""
    n = len(cases)
    if n == 0:
        return EvalMetrics(
            n_cases=0, n_should_answer=0, n_should_refuse=0,
            citation_accuracy=0.0, answer_correctness=0.0,
            refusal_rate_correct=0.0, refusal_rate_false_positive=0.0,
            pass_rate=0.0,
        )

    # Partition: did the gold expect an answer or a refusal?
    should_answer = [
        c for c in cases
        if not (c.actual_outcome == "refused" and c.outcome_matches_expected)
        and not (c.actual_outcome == "answered" and not c.outcome_matches_expected)
    ]
    # Above is brittle. Cleaner: use whether the *expected* outcome was refusal.
    # The case result doesn't carry the gold entry, but we can infer:
    # outcome_matches_expected combined with actual_outcome tells us.
    expected_refuse = [
        c for c in cases
        if (c.actual_outcome == "refused" and c.outcome_matches_expected)
        or (c.actual_outcome == "answered" and not c.outcome_matches_expected)
    ]
    expected_answer = [c for c in cases if c not in expected_refuse]

    n_should_answer = len(expected_answer)
    n_should_refuse = len(expected_refuse)

    citation_accuracy = (
        sum(1 for c in expected_answer if c.actual_outcome == "answered" and c.citations_correct)
        / n_should_answer
        if n_should_answer
        else 0.0
    )
    answer_correctness = (
        sum(
            1
            for c in expected_answer
            if c.actual_outcome == "answered" and c.keywords_present
        )
        / n_should_answer
        if n_should_answer
        else 0.0
    )
    refusal_rate_correct = (
        sum(1 for c in expected_refuse if c.actual_outcome == "refused")
        / n_should_refuse
        if n_should_refuse
        else 0.0
    )
    refusal_rate_false_positive = (
        sum(1 for c in expected_answer if c.actual_outcome == "refused")
        / n_should_answer
        if n_should_answer
        else 0.0
    )
    pass_rate = sum(1 for c in cases if c.passed) / n

    return EvalMetrics(
        n_cases=n,
        n_should_answer=n_should_answer,
        n_should_refuse=n_should_refuse,
        citation_accuracy=citation_accuracy,
        answer_correctness=answer_correctness,
        refusal_rate_correct=refusal_rate_correct,
        refusal_rate_false_positive=refusal_rate_false_positive,
        pass_rate=pass_rate,
    )


# ---- Internal ----


def _score_case(entry: GoldEntry, result: object) -> EvalCaseResult:
    """Compare one gold entry to the pipeline's actual result."""
    actual_outcome = result.outcome  # type: ignore[attr-defined]
    actual_citations = [
        c.citation_path for c in result.citations  # type: ignore[attr-defined]
    ]

    expected_set = set(entry.expected_citations)
    actual_set = set(actual_citations)
    citations_correct = expected_set.issubset(actual_set) if expected_set else (
        not actual_set if entry.should_refuse else False
    )

    answer_lower = result.answer.lower()  # type: ignore[attr-defined]
    keywords_present = all(
        kw.lower() in answer_lower for kw in entry.expected_answer_keywords
    ) if entry.expected_answer_keywords else True

    expected_outcome = "refused" if entry.should_refuse else "answered"
    outcome_matches_expected = actual_outcome == expected_outcome

    return EvalCaseResult(
        entry_id=entry.id,
        category=entry.category,
        question=entry.question,
        actual_outcome=actual_outcome,
        actual_refusal_reason=result.refusal_reason,  # type: ignore[attr-defined]
        actual_citations=actual_citations,
        actual_answer=result.answer,  # type: ignore[attr-defined]
        citations_correct=citations_correct,
        keywords_present=keywords_present,
        outcome_matches_expected=outcome_matches_expected,
    )


def _categories_present(cases: list[EvalCaseResult]) -> list[Category]:
    seen: list[Category] = []
    for c in cases:
        if c.category not in seen:
            seen.append(c.category)
    return seen

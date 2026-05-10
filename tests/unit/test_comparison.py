"""Tests for the eval comparison machinery."""

from __future__ import annotations

from datetime import UTC, datetime

from evals.comparison import compare_reports, format_comparison
from evals.types import EvalCaseResult, EvalMetrics, EvalReport


def _case(
    *,
    entry_id: str,
    category: str = "exact_citation",
    actual_outcome: str = "answered",
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


def _metrics(*, n: int = 1, pass_rate: float = 1.0) -> EvalMetrics:
    return EvalMetrics(
        n_cases=n,
        n_should_answer=n,
        n_should_refuse=0,
        citation_accuracy=pass_rate,
        answer_correctness=pass_rate,
        refusal_rate_correct=0.0,
        refusal_rate_false_positive=0.0,
        pass_rate=pass_rate,
    )


def _report(cases: list[EvalCaseResult], *, model_name: str) -> EvalReport:
    return EvalReport(
        timestamp=datetime.now(UTC),
        gold_set_path="test",
        model_name=model_name,
        prompt_template_version="0.1.0",
        overall=_metrics(n=len(cases), pass_rate=sum(1 for c in cases if c.passed) / len(cases) if cases else 0),
        by_category={},
        cases=cases,
    )


# ---- compare_reports ----


def test_compare_reports_pairs_cases_by_entry_id() -> None:
    det = _report([_case(entry_id="a"), _case(entry_id="b")], model_name="det")
    agent = _report([_case(entry_id="a"), _case(entry_id="b")], model_name="agent")
    report = compare_reports(deterministic=det, agent=agent)
    assert len(report.case_comparisons) == 2
    ids = [c.entry_id for c in report.case_comparisons]
    assert ids == ["a", "b"]


def test_compare_reports_skips_nonmatching_ids() -> None:
    det = _report([_case(entry_id="a"), _case(entry_id="only_in_det")], model_name="det")
    agent = _report([_case(entry_id="a"), _case(entry_id="only_in_agent")], model_name="agent")
    report = compare_reports(deterministic=det, agent=agent)
    assert len(report.case_comparisons) == 1
    assert report.case_comparisons[0].entry_id == "a"


# ---- delta classification ----


def test_both_passed() -> None:
    det = _report([_case(entry_id="a")], model_name="det")
    agent = _report([_case(entry_id="a")], model_name="agent")
    report = compare_reports(deterministic=det, agent=agent)
    assert report.n_both_passed == 1
    assert report.n_both_failed == 0
    assert report.n_progressed == 0
    assert report.n_regressed == 0


def test_progressed_when_agent_passes_and_det_fails() -> None:
    det = _report(
        [_case(entry_id="a", citations_correct=False)], model_name="det"
    )
    agent = _report([_case(entry_id="a")], model_name="agent")
    report = compare_reports(deterministic=det, agent=agent)
    assert report.n_progressed == 1
    assert report.n_regressed == 0


def test_regressed_when_det_passes_and_agent_fails() -> None:
    det = _report([_case(entry_id="a")], model_name="det")
    agent = _report(
        [_case(entry_id="a", citations_correct=False)], model_name="agent"
    )
    report = compare_reports(deterministic=det, agent=agent)
    assert report.n_regressed == 1
    assert report.n_progressed == 0


def test_both_failed() -> None:
    det = _report(
        [_case(entry_id="a", citations_correct=False)], model_name="det"
    )
    agent = _report(
        [_case(entry_id="a", citations_correct=False)], model_name="agent"
    )
    report = compare_reports(deterministic=det, agent=agent)
    assert report.n_both_failed == 1


# ---- format_comparison ----


def test_format_comparison_includes_both_model_names() -> None:
    det = _report([_case(entry_id="a")], model_name="claude-sonnet")
    agent = _report([_case(entry_id="a")], model_name="claude-sonnet-agent")
    report = compare_reports(deterministic=det, agent=agent)
    output = format_comparison(report)
    assert "claude-sonnet" in output
    assert "claude-sonnet-agent" in output


def test_format_comparison_lists_regressions_when_present() -> None:
    det = _report([_case(entry_id="a")], model_name="det")
    agent = _report(
        [_case(entry_id="a", citations_correct=False)], model_name="agent"
    )
    report = compare_reports(deterministic=det, agent=agent)
    output = format_comparison(report)
    assert "Regressions" in output
    assert "a" in output


def test_format_comparison_no_regressions_section_when_clean() -> None:
    det = _report([_case(entry_id="a")], model_name="det")
    agent = _report([_case(entry_id="a")], model_name="agent")
    report = compare_reports(deterministic=det, agent=agent)
    output = format_comparison(report)
    assert "Regressions" not in output

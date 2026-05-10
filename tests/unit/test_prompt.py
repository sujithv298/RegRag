"""Tests for `regrag.prompt`."""

from __future__ import annotations

from pathlib import Path

import pytest

from regrag.chunking import chunk_nodes
from regrag.ingest import parse_part_xml
from regrag.prompt import (
    PROMPT_TEMPLATE_VERSION,
    REFUSAL_PHRASE,
    SYSTEM_PROMPT,
    render_full_prompt,
    render_user_message,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "reg_e_part_1005_excerpt.xml"


@pytest.fixture(scope="module")
def chunks() -> list:
    nodes = parse_part_xml(FIXTURE.read_bytes())
    return chunk_nodes(nodes)


# ---- Pinned constants ----


def test_template_version_is_set() -> None:
    """Bumping this is a deliberate choice; this test pins the current value
    so changes show up in PRs."""
    assert PROMPT_TEMPLATE_VERSION == "0.1.0"


def test_refusal_phrase_in_system_prompt() -> None:
    """The model must know the exact refusal phrase the verifier expects."""
    assert REFUSAL_PHRASE in SYSTEM_PROMPT


def test_system_prompt_requires_inline_citations() -> None:
    assert "[CFR:" in SYSTEM_PROMPT
    assert "ONLY" in SYSTEM_PROMPT  # forbid prior knowledge


# ---- User message rendering ----


def test_user_message_includes_question(chunks: list) -> None:
    msg = render_user_message("What is the consumer's liability?", chunks[:2])
    assert "What is the consumer's liability?" in msg


def test_user_message_includes_each_chunks_citation(chunks: list) -> None:
    msg = render_user_message("question", chunks[:3])
    for c in chunks[:3]:
        assert c.citation_path in msg


def test_user_message_includes_each_chunks_source_text(chunks: list) -> None:
    msg = render_user_message("question", chunks[:3])
    for c in chunks[:3]:
        assert c.source_text in msg


def test_user_message_handles_empty_chunks() -> None:
    msg = render_user_message("question", [])
    assert "no regulatory text" in msg.lower()
    assert "question" in msg


def test_user_message_marks_interpretation_chunks(chunks: list) -> None:
    """Interpretation chunks should be labeled distinctly in the prompt so
    the model knows it's reading a Comment, not a rule."""
    interp_chunks = [c for c in chunks if c.is_interpretation]
    assert interp_chunks  # fixture has interpretations
    msg = render_user_message("question", interp_chunks)
    assert "Official Interpretation" in msg


# ---- Full prompt for hashing ----


def test_full_prompt_contains_both_system_and_user(chunks: list) -> None:
    full = render_full_prompt("question", chunks[:1])
    assert "<<system>>" in full
    assert "<<user>>" in full
    assert SYSTEM_PROMPT in full

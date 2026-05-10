"""Tests for `regrag.models.FakeAdapter`. Real adapters are integration-only."""

from __future__ import annotations

from regrag.models import FakeAdapter, GenerationResult, LLMAdapter


def test_implements_protocol() -> None:
    assert isinstance(FakeAdapter(response=""), LLMAdapter)


def test_fixed_string_response() -> None:
    adapter = FakeAdapter(response="hello")
    out = adapter.generate(system="sys", user="usr")
    assert isinstance(out, GenerationResult)
    assert out.text == "hello"


def test_callable_response_sees_prompts() -> None:
    """A callable response should receive system and user as args, so test
    fixtures can construct context-aware fake responses."""
    received: dict[str, str] = {}

    def respond(system: str, user: str) -> str:
        received["system"] = system
        received["user"] = user
        return f"saw {len(user)} chars of user"

    adapter = FakeAdapter(response=respond)
    out = adapter.generate(system="sys-content", user="user-content")
    assert out.text == "saw 12 chars of user"
    assert received == {"system": "sys-content", "user": "user-content"}


def test_name_and_version_are_carried() -> None:
    adapter = FakeAdapter(response="x", name="canned-fake", version="2026.05.02")
    assert adapter.name == "canned-fake"
    assert adapter.version == "2026.05.02"
    out = adapter.generate(system="", user="")
    assert out.model_name == "canned-fake"
    assert out.model_version == "2026.05.02"

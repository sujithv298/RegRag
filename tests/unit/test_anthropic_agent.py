"""Tests for AnthropicAgentAdapter.

We don't hit the real Anthropic API in unit tests — instead we hand-craft
a fake response object that matches Anthropic SDK shape and verify the
adapter parses tool_use and text blocks correctly. The real-API path is
covered by manual smoke testing on the user's machine.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from regrag.agent.types import AgentStep, ToolCall, ToolResult
from regrag.models.anthropic import (
    AnthropicAgentAdapter,
    _parse_response,
    _tool_to_anthropic_spec,
)


# ---- Tool spec translation ----


def test_tool_to_anthropic_spec_has_required_fields() -> None:
    fake_tool = SimpleNamespace(
        name="search_regulations",
        description="Search the corpus.",
        schema={"type": "object", "properties": {}},
    )
    spec = _tool_to_anthropic_spec(fake_tool)
    assert spec["name"] == "search_regulations"
    assert spec["description"] == "Search the corpus."
    assert spec["input_schema"] == {"type": "object", "properties": {}}


# ---- Response parsing ----


def test_parse_response_with_tool_use_returns_toolcall() -> None:
    fake_response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                id="toolu_abc",
                name="search_regulations",
                input={"query": "consumer liability", "k": 3},
            )
        ]
    )
    turn = _parse_response(fake_response)
    assert turn.is_tool_call
    assert turn.tool_call is not None
    assert turn.tool_call.name == "search_regulations"
    assert turn.tool_call.arguments == {"query": "consumer liability", "k": 3}
    assert turn.tool_call.call_id == "toolu_abc"


def test_parse_response_with_text_returns_final_answer() -> None:
    fake_response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Final answer [CFR:1005.6(b)(1)].")]
    )
    turn = _parse_response(fake_response)
    assert turn.is_final
    assert turn.text == "Final answer [CFR:1005.6(b)(1)]."


def test_parse_response_prefers_tool_use_over_text() -> None:
    """Anthropic can emit text + tool_use in the same response. The
    orchestrator wants the tool_use; the text is intermediate reasoning."""
    fake_response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="I'll search for that."),
            SimpleNamespace(
                type="tool_use",
                id="toolu_xyz",
                name="search_regulations",
                input={"query": "x"},
            ),
        ]
    )
    turn = _parse_response(fake_response)
    assert turn.is_tool_call
    assert turn.tool_call.name == "search_regulations"


def test_parse_response_empty_content_returns_empty_text() -> None:
    fake_response = SimpleNamespace(content=[])
    turn = _parse_response(fake_response)
    assert turn.is_final
    assert turn.text == ""


# ---- Message-history reconstruction ----


def test_build_messages_includes_history_as_tool_use_pairs() -> None:
    adapter = AnthropicAgentAdapter()
    history = [
        AgentStep(
            call=ToolCall(name="search_regulations", arguments={"query": "x"}, call_id="t1"),
            result=ToolResult(call_id="t1", tool_name="search_regulations", content="results"),
        )
    ]
    messages = adapter._build_messages(user="What is X?", history=history)

    assert len(messages) == 3
    assert messages[0] == {"role": "user", "content": "What is X?"}
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"][0]["type"] == "tool_use"
    assert messages[1]["content"][0]["name"] == "search_regulations"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"][0]["type"] == "tool_result"
    assert messages[2]["content"][0]["tool_use_id"] == "t1"


# ---- agent_turn dispatching (full mock) ----


def test_agent_turn_calls_anthropic_with_tools_spec() -> None:
    """Verify agent_turn passes our tools through to Anthropic correctly."""
    adapter = AnthropicAgentAdapter()
    mock_client = MagicMock()
    mock_client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="done [CFR:1005.6(b)(1)]")]
    )
    adapter._client = mock_client  # bypass lazy load

    fake_tool = SimpleNamespace(
        name="search_regulations",
        description="Search.",
        schema={"type": "object"},
    )
    turn = adapter.agent_turn(system="sys", user="usr", tools=[fake_tool], history=[])

    assert turn.is_final
    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == AnthropicAgentAdapter.DEFAULT_MODEL
    assert call_kwargs["system"] == "sys"
    assert len(call_kwargs["tools"]) == 1
    assert call_kwargs["tools"][0]["name"] == "search_regulations"

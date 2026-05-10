"""Tests for OpenAIAgentAdapter.

Mirrors test_anthropic_agent.py — we don't hit the real OpenAI API. A
hand-crafted fake response object that matches the OpenAI SDK shape
verifies the adapter parses tool_calls and message content correctly.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from regrag.agent.types import AgentStep, ToolCall, ToolResult
from regrag.models.openai import (
    OpenAIAgentAdapter,
    _parse_openai_response,
    _tool_to_openai_spec,
)

# ---- Tool spec translation ----


def test_tool_to_openai_spec_wraps_in_function_block() -> None:
    fake_tool = SimpleNamespace(
        name="search_regulations",
        description="Search the corpus.",
        schema={"type": "object", "properties": {}},
    )
    spec = _tool_to_openai_spec(fake_tool)
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "search_regulations"
    assert spec["function"]["description"] == "Search the corpus."
    assert spec["function"]["parameters"] == {"type": "object", "properties": {}}


# ---- Response parsing ----


def test_parse_response_with_tool_call_returns_toolcall() -> None:
    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_xyz",
                            type="function",
                            function=SimpleNamespace(
                                name="search_regulations",
                                arguments=json.dumps({"query": "consumer liability", "k": 3}),
                            ),
                        )
                    ],
                )
            )
        ]
    )
    turn = _parse_openai_response(fake_response)
    assert turn.is_tool_call
    assert turn.tool_call is not None
    assert turn.tool_call.name == "search_regulations"
    assert turn.tool_call.arguments == {"query": "consumer liability", "k": 3}
    assert turn.tool_call.call_id == "call_xyz"


def test_parse_response_with_text_returns_final_answer() -> None:
    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Final answer with [CFR:1005.6(b)(1)].",
                    tool_calls=None,
                )
            )
        ]
    )
    turn = _parse_openai_response(fake_response)
    assert turn.is_final
    assert turn.text == "Final answer with [CFR:1005.6(b)(1)]."


def test_parse_response_prefers_tool_call_over_text() -> None:
    """OpenAI can emit content + tool_calls together; we want the tool_call."""
    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Let me search.",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            type="function",
                            function=SimpleNamespace(
                                name="search_regulations",
                                arguments=json.dumps({"query": "x"}),
                            ),
                        )
                    ],
                )
            )
        ]
    )
    turn = _parse_openai_response(fake_response)
    assert turn.is_tool_call
    assert turn.tool_call.name == "search_regulations"


def test_parse_response_handles_malformed_arguments() -> None:
    """If the model emits invalid JSON for tool args (rare but possible),
    fall back to empty arguments rather than crashing."""
    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_bad",
                            type="function",
                            function=SimpleNamespace(
                                name="search_regulations",
                                arguments="not valid json {",
                            ),
                        )
                    ],
                )
            )
        ]
    )
    turn = _parse_openai_response(fake_response)
    assert turn.is_tool_call
    assert turn.tool_call.arguments == {}


def test_parse_response_no_choices_returns_empty_text() -> None:
    fake_response = SimpleNamespace(choices=[])
    turn = _parse_openai_response(fake_response)
    assert turn.is_final
    assert turn.text == ""


# ---- Message-history reconstruction ----


def test_build_messages_includes_system_user_and_history() -> None:
    adapter = OpenAIAgentAdapter()
    history = [
        AgentStep(
            call=ToolCall(name="search_regulations", arguments={"query": "x"}, call_id="t1"),
            result=ToolResult(call_id="t1", tool_name="search_regulations", content="results"),
        )
    ]
    messages = adapter._build_messages(
        system="You are a regulatory assistant.",
        user="What is X?",
        history=history,
    )

    assert len(messages) == 4
    assert messages[0] == {"role": "system", "content": "You are a regulatory assistant."}
    assert messages[1] == {"role": "user", "content": "What is X?"}

    # Assistant turn that called the tool
    assert messages[2]["role"] == "assistant"
    assert messages[2]["tool_calls"][0]["function"]["name"] == "search_regulations"
    assert json.loads(messages[2]["tool_calls"][0]["function"]["arguments"]) == {"query": "x"}

    # Tool result
    assert messages[3]["role"] == "tool"
    assert messages[3]["tool_call_id"] == "t1"
    assert messages[3]["content"] == "results"


# ---- agent_turn dispatching ----


def test_agent_turn_calls_openai_with_tools_spec() -> None:
    adapter = OpenAIAgentAdapter()
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="done [CFR:1005.6(b)(1)]",
                    tool_calls=None,
                )
            )
        ]
    )
    adapter._client = mock_client  # bypass lazy load

    fake_tool = SimpleNamespace(
        name="search_regulations",
        description="Search.",
        schema={"type": "object"},
    )
    turn = adapter.agent_turn(system="sys", user="usr", tools=[fake_tool], history=[])

    assert turn.is_final
    mock_client.chat.completions.create.assert_called_once()
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == OpenAIAgentAdapter.DEFAULT_MODEL
    assert kwargs["tool_choice"] == "auto"
    assert len(kwargs["tools"]) == 1
    assert kwargs["tools"][0]["type"] == "function"
    assert kwargs["tools"][0]["function"]["name"] == "search_regulations"
    # System prompt is in messages[0], not as a separate parameter (unlike Anthropic).
    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["messages"][0]["content"] == "sys"

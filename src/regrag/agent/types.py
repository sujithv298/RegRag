"""Core types for the agent loop.

`ToolCall`     — structured request from the LLM to invoke one of our tools.
                 Roughly mirrors Anthropic's `tool_use` content block shape so
                 a real-LLM adapter is a thin translation layer.
`ToolResult`   — what the tool returns, fed back to the LLM next turn.
`AgentTurn`    — one model output: either a ToolCall (continue) or a final
                 text answer (terminate).
`AgentTrace`   — the ordered history of (ToolCall, ToolResult) pairs.
                 Persisted into the audit log so an examiner can replay
                 every decision the LLM made.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolCall(BaseModel):
    """A request from the LLM to invoke a tool by name with arguments."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Tool name, e.g. 'search_regulations'.")
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str = Field(
        default="",
        description="LLM-assigned id for matching results back to calls. "
        "Anthropic returns this in `tool_use.id`; we mirror that.",
    )


class ToolResult(BaseModel):
    """What a tool returns. Whatever it is, the orchestrator serializes
    `content` to a string before handing it back to the LLM."""

    model_config = ConfigDict(frozen=True)

    call_id: str
    tool_name: str
    content: Any = Field(description="Whatever the tool returned. Stringified for the LLM.")
    is_error: bool = Field(default=False)


class AgentTurn(BaseModel):
    """One LLM output. Exactly one of `text` or `tool_call` is set."""

    model_config = ConfigDict(frozen=True)

    text: str | None = Field(default=None, description="Final answer if set.")
    tool_call: ToolCall | None = Field(default=None, description="Tool to invoke if set.")

    @property
    def is_final(self) -> bool:
        return self.text is not None

    @property
    def is_tool_call(self) -> bool:
        return self.tool_call is not None


class AgentStep(BaseModel):
    """One (call, result) pair — used for both the in-flight history the
    LLM sees and the audit-log trace."""

    model_config = ConfigDict(frozen=True)

    call: ToolCall
    result: ToolResult


class AgentTrace(BaseModel):
    """Ordered sequence of agent steps for one query."""

    model_config = ConfigDict(frozen=True)

    steps: list[AgentStep] = Field(default_factory=list)

    @property
    def num_tool_calls(self) -> int:
        return len(self.steps)

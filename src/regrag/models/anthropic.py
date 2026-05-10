"""Anthropic Claude adapter.

Two classes:
  - `AnthropicAdapter`       — single-shot LLMAdapter (used by the
                                deterministic pipeline).
  - `AnthropicAgentAdapter`  — agent loop adapter that uses Anthropic's
                                `tool_use` API (used by the agent pipeline).

Both lazy-load — importing this module never opens an HTTP client; only
the first call does. API key resolves from `ANTHROPIC_API_KEY` env var
by default, or can be passed explicitly.
"""

from __future__ import annotations

from typing import Any

from regrag.agent.types import AgentStep, AgentTurn, ToolCall
from regrag.models.types import GenerationResult


class AnthropicAdapter:
    """LLMAdapter implementation backed by the `anthropic` SDK."""

    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        api_key: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key
        self._client: object | None = None

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def version(self) -> str:
        # Claude API returns the model id as `model` in responses; that
        # equals self._model_name for our purposes. Adapters that need a
        # finer build identifier can override.
        return self._model_name

    def generate(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> GenerationResult:
        self._ensure_loaded()
        assert self._client is not None
        message = self._client.messages.create(  # type: ignore[attr-defined]
            model=self._model_name,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user}],
        )
        text = message.content[0].text if message.content else ""
        return GenerationResult(
            text=text,
            model_name=self._model_name,
            model_version=getattr(message, "model", self._model_name),
            input_tokens=getattr(message.usage, "input_tokens", None)
            if getattr(message, "usage", None)
            else None,
            output_tokens=getattr(message.usage, "output_tokens", None)
            if getattr(message, "usage", None)
            else None,
        )

    def _ensure_loaded(self) -> None:
        if self._client is not None:
            return
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "AnthropicAdapter requires the `anthropic` package. "
                "Install via `uv sync` (it's in the default deps)."
            ) from exc
        self._client = Anthropic(api_key=self._api_key) if self._api_key else Anthropic()


class AnthropicAgentAdapter:
    """Anthropic-backed adapter for the agent loop.

    Translates between our `Tool`/`AgentTurn` types and Anthropic's
    `tool_use` API:

      - On each turn, builds a `messages` list from the orchestrator's
        history and calls `messages.create(tools=[...])`.
      - Reads back the response. If it contains a `tool_use` block, returns
        a `ToolCall`. If it contains a `text` block (and no tool_use),
        returns a final-answer `AgentTurn`.

    The orchestrator in `regrag.agent.orchestrator.run_agent_loop` doesn't
    know which adapter is in use — it just calls `agent_turn(...)`. That's
    the point of the adapter abstraction.
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client: object | None = None

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def version(self) -> str:
        return self._model_name

    def agent_turn(
        self,
        *,
        system: str,
        user: str,
        tools: list[Any],
        history: list[AgentStep],
    ) -> AgentTurn:
        """One turn of the agent loop. Calls Anthropic with `tool_use` enabled."""
        self._ensure_loaded()
        assert self._client is not None

        messages = self._build_messages(user=user, history=history)
        anthropic_tools = [_tool_to_anthropic_spec(t) for t in tools]

        response = self._client.messages.create(  # type: ignore[attr-defined]
            model=self._model_name,
            system=system,
            tools=anthropic_tools,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return _parse_response(response)

    # ---- Internal ----

    def _build_messages(self, *, user: str, history: list[AgentStep]) -> list[dict[str, Any]]:
        """Reconstruct the conversation: original user message + each
        (assistant tool_use, user tool_result) pair from history."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        for step in history:
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": step.call.call_id or "tool_call",
                            "name": step.call.name,
                            "input": dict(step.call.arguments),
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": step.call.call_id or "tool_call",
                            "content": str(step.result.content),
                            "is_error": step.result.is_error,
                        }
                    ],
                }
            )
        return messages

    def _ensure_loaded(self) -> None:
        if self._client is not None:
            return
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "AnthropicAgentAdapter requires the `anthropic` package. Install via `uv sync`."
            ) from exc
        self._client = Anthropic(api_key=self._api_key) if self._api_key else Anthropic()


# ---- Module-level helpers ----


def _tool_to_anthropic_spec(tool: Any) -> dict[str, Any]:
    """Translate one of our `Tool` instances to Anthropic's tool spec shape."""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.schema,
    }


def _parse_response(response: Any) -> AgentTurn:
    """Convert an Anthropic Message into an `AgentTurn`.

    Prefer `tool_use` blocks — if the model wants a tool, that's the next
    step. Fall back to `text` for final answers. If neither is present
    (rare), return an empty-text turn so the orchestrator can decide.
    """
    content_blocks = getattr(response, "content", []) or []
    text_chunks: list[str] = []
    for block in content_blocks:
        block_type = getattr(block, "type", None)
        if block_type == "tool_use":
            return AgentTurn(
                tool_call=ToolCall(
                    name=getattr(block, "name", ""),
                    arguments=dict(getattr(block, "input", {}) or {}),
                    call_id=getattr(block, "id", "") or "",
                )
            )
        if block_type == "text":
            text_chunks.append(getattr(block, "text", ""))
    return AgentTurn(text="\n".join(text_chunks).strip() or "")

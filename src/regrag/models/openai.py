"""OpenAI GPT-4 adapter.

Two classes:
  - `OpenAIAdapter`       — single-shot LLMAdapter (deterministic pipeline).
  - `OpenAIAgentAdapter`  — agent loop adapter using OpenAI's chat completions
                            tool-calling API (agent pipeline).

Both lazy-load. API key resolves from `OPENAI_API_KEY` env var by default.
"""

from __future__ import annotations

import json
from typing import Any

from regrag.agent.types import AgentStep, AgentTurn, ToolCall
from regrag.models.types import GenerationResult


class OpenAIAdapter:
    """LLMAdapter implementation backed by the `openai` SDK."""

    DEFAULT_MODEL = "gpt-4o"

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
        completion = self._client.chat.completions.create(  # type: ignore[attr-defined]
            model=self._model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        choice = completion.choices[0]
        text = choice.message.content or ""
        usage = getattr(completion, "usage", None)
        return GenerationResult(
            text=text,
            model_name=self._model_name,
            model_version=getattr(completion, "model", self._model_name),
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        )

    def _ensure_loaded(self) -> None:
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "OpenAIAdapter requires the `openai` package. "
                "Install via `uv sync` (it's in the default deps)."
            ) from exc
        self._client = OpenAI(api_key=self._api_key) if self._api_key else OpenAI()


class OpenAIAgentAdapter:
    """OpenAI-backed adapter for the agent loop.

    Translates between our `Tool`/`AgentTurn` types and OpenAI's chat
    completions tool-calling API:

      - On each turn, builds a `messages` list from the orchestrator's
        history and calls `chat.completions.create(tools=[...])`.
      - Reads back the assistant message. If it contains `tool_calls`,
        returns a `ToolCall` (we pick the first call; OpenAI can emit
        parallel calls but our orchestrator handles one at a time).
        Otherwise returns the message text as a final answer.

    The OpenAI tool-calling shape differs from Anthropic's:
      - Tools live under {"type": "function", "function": {...}}, not flat.
      - Arguments come back as a JSON *string* and need parsing.
      - Tool results are sent back as messages with role="tool" rather
        than as content blocks inside a user message.

    The orchestrator in `regrag.agent.orchestrator.run_agent_loop` doesn't
    care about any of this — it just calls `agent_turn(...)` on whichever
    adapter you pass.
    """

    DEFAULT_MODEL = "gpt-4o"

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
        self._ensure_loaded()
        assert self._client is not None

        messages = self._build_messages(system=system, user=user, history=history)
        openai_tools = [_tool_to_openai_spec(t) for t in tools]

        response = self._client.chat.completions.create(  # type: ignore[attr-defined]
            model=self._model_name,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return _parse_openai_response(response)

    # ---- Internal ----

    def _build_messages(
        self, *, system: str, user: str, history: list[AgentStep]
    ) -> list[dict[str, Any]]:
        """Reconstruct the conversation: system + user + each (assistant
        tool_calls, tool result) pair from history."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        for step in history:
            call_id = step.call.call_id or "tool_call"
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": step.call.name,
                                "arguments": json.dumps(step.call.arguments),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": str(step.result.content),
                }
            )
        return messages

    def _ensure_loaded(self) -> None:
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "OpenAIAgentAdapter requires the `openai` package. Install via `uv sync`."
            ) from exc
        self._client = OpenAI(api_key=self._api_key) if self._api_key else OpenAI()


# ---- Module-level helpers ----


def _tool_to_openai_spec(tool: Any) -> dict[str, Any]:
    """Translate one of our `Tool` instances to OpenAI's tool spec shape."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.schema,
        },
    }


def _parse_openai_response(response: Any) -> AgentTurn:
    """Convert an OpenAI ChatCompletion into an `AgentTurn`.

    Prefer tool_calls — if the model wants a tool, that's the next step.
    Fall back to message content for final answers. If there are multiple
    tool_calls in one response (parallel tool use), we return the first;
    the orchestrator runs them serially anyway.
    """
    choices = getattr(response, "choices", []) or []
    if not choices:
        return AgentTurn(text="")
    message = choices[0].message
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        first = tool_calls[0]
        try:
            args = json.loads(first.function.arguments)
        except (json.JSONDecodeError, AttributeError):
            args = {}
        return AgentTurn(
            tool_call=ToolCall(
                name=getattr(first.function, "name", "") or "",
                arguments=args if isinstance(args, dict) else {},
                call_id=getattr(first, "id", "") or "",
            )
        )
    text = getattr(message, "content", None) or ""
    return AgentTurn(text=text)

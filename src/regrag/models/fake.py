"""FakeAdapter — a deterministic LLM stand-in for tests and offline demos.

Three modes:

  - **Fixed-string mode**: pass `response="..."` and every `generate` call
    returns that exact text.

  - **Callable mode**: pass `response=fn` where `fn(system, user) -> str`.
    Lets a test construct a plausible response referencing actual chunks.

  - **Agent mode**: pass `agent_responder=fn` where
    `fn(system, user, tools, history) -> AgentTurn`. The orchestrator in
    `regrag.agent.orchestrator.run_agent_loop` calls this. The fn typically
    inspects `history` length to decide whether to call a tool or return a
    final answer.

The FakeAdapter is *not* a mock — it's a real adapter that happens to be
deterministic. The pipeline can't tell it apart from `AnthropicAdapter`.
"""

from __future__ import annotations

from typing import Any, Callable, Union

from regrag.agent.types import AgentStep, AgentTurn
from regrag.models.types import GenerationResult

FakeResponse = Union[str, Callable[[str, str], str]]
FakeAgentResponder = Callable[[str, str, list[Any], list[AgentStep]], AgentTurn]


class FakeAdapter:
    """LLM adapter that returns a fixed string or a callable's output."""

    def __init__(
        self,
        *,
        response: FakeResponse | None = None,
        agent_responder: FakeAgentResponder | None = None,
        name: str = "fake",
        version: str = "v0",
    ) -> None:
        if response is None and agent_responder is None:
            raise ValueError("FakeAdapter requires response= or agent_responder=")
        self._response = response
        self._agent_responder = agent_responder
        self._name = name
        self._version = version

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def generate(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> GenerationResult:
        del max_tokens, temperature  # unused; kept for interface parity
        if self._response is None:
            raise RuntimeError(
                "FakeAdapter constructed in agent-mode only; call agent_turn() instead."
            )
        if callable(self._response):
            text = self._response(system, user)
        else:
            text = self._response
        return GenerationResult(
            text=text,
            model_name=self._name,
            model_version=self._version,
        )

    def agent_turn(
        self,
        *,
        system: str,
        user: str,
        tools: list[Any],
        history: list[AgentStep],
    ) -> AgentTurn:
        """Drive one turn of the agent loop. Required by `run_agent_loop`."""
        if self._agent_responder is None:
            raise RuntimeError(
                "FakeAdapter constructed without agent_responder=; "
                "use generate() instead, or pass agent_responder= at construction."
            )
        return self._agent_responder(system, user, tools, history)

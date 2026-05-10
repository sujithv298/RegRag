"""The agent loop.

Pseudocode:

    history = []
    for turn in range(max_turns):
        turn_output = adapter.agent_turn(system, user, tools, history)
        if turn_output.is_final:
            return (turn_output.text, history)
        # Tool call
        tool = lookup_by_name(turn_output.tool_call.name)
        result = tool.run(turn_output.tool_call.arguments)
        history.append(AgentStep(call=turn_output.tool_call, result=...))
    raise BudgetExceeded("max_turns reached without final answer")

The orchestrator deliberately knows nothing about RAG, citations, audit, or
PII. Those concerns live in `pipeline_agent.py`. This file is the pure
agent loop; it could orchestrate any LLM + tools.
"""

from __future__ import annotations

from regrag.agent.tools import Tool
from regrag.agent.types import AgentStep, AgentTrace, ToolCall, ToolResult

DEFAULT_MAX_TURNS = 6


class AgentBudgetExceeded(RuntimeError):  # noqa: N818 — exported name; keep stable for callers + tests
    """Raised when the loop hits max_turns without a final answer."""


class AgentLoopResult:
    """What the orchestrator returns: final text + the trace of decisions."""

    def __init__(self, *, final_text: str, trace: AgentTrace) -> None:
        self.final_text = final_text
        self.trace = trace


def run_agent_loop(
    *,
    adapter,
    system: str,
    user: str,
    tools: list[Tool],
    max_turns: int = DEFAULT_MAX_TURNS,
) -> AgentLoopResult:
    """Drive the LLM through tool calls until it returns a final answer.

    `adapter` must expose `agent_turn(system, user, tools, history) -> AgentTurn`.
    We don't import the protocol here to keep the import graph shallow;
    duck-typing is fine for one method.
    """
    if not tools:
        raise ValueError("agent loop requires at least one tool")
    tools_by_name: dict[str, Tool] = {t.name: t for t in tools}
    steps: list[AgentStep] = []

    for turn in range(max_turns):
        agent_turn = adapter.agent_turn(
            system=system, user=user, tools=tools, history=steps
        )

        if agent_turn.is_final:
            assert agent_turn.text is not None
            return AgentLoopResult(
                final_text=agent_turn.text,
                trace=AgentTrace(steps=steps),
            )

        if not agent_turn.is_tool_call:
            raise RuntimeError(
                f"Agent turn {turn} was neither a final answer nor a tool call. "
                "Adapter contract violated."
            )

        call: ToolCall = agent_turn.tool_call  # type: ignore[assignment]
        tool = tools_by_name.get(call.name)
        if tool is None:
            steps.append(
                AgentStep(
                    call=call,
                    result=ToolResult(
                        call_id=call.call_id,
                        tool_name=call.name,
                        content=f"Error: unknown tool '{call.name}'.",
                        is_error=True,
                    ),
                )
            )
            continue

        try:
            content = tool.run(call.arguments)
            is_error = False
        except Exception as exc:
            content = f"Error executing tool '{call.name}': {exc}"
            is_error = True

        steps.append(
            AgentStep(
                call=call,
                result=ToolResult(
                    call_id=call.call_id,
                    tool_name=call.name,
                    content=content,
                    is_error=is_error,
                ),
            )
        )

    raise AgentBudgetExceeded(
        f"agent loop hit max_turns={max_turns} without a final answer "
        f"(executed {len(steps)} tool calls)"
    )

"""Agent loop: tool definitions + orchestrator.

Public surface:

  - `Tool` protocol + concrete tools (SearchRegulations, GetChunkByCitation,
    ExpandChunkContext)
  - `AgentTurn`, `ToolCall`, `ToolResult`, `AgentStep`, `AgentTrace` types
  - `run_agent_loop` — the orchestration function

The top-level `pipeline_agent.answer_query_agentic` wraps this with PII,
citation verification, and audit logging — same regulator-friendly
guarantees as the deterministic pipeline.
"""

from __future__ import annotations

from regrag.agent.orchestrator import (
    DEFAULT_MAX_TURNS,
    AgentBudgetExceeded,
    AgentLoopResult,
    run_agent_loop,
)
from regrag.agent.tools import (
    ExpandChunkContext,
    GetChunkByCitation,
    SearchRegulations,
    Tool,
)
from regrag.agent.types import (
    AgentStep,
    AgentTrace,
    AgentTurn,
    ToolCall,
    ToolResult,
)

__all__ = [
    "DEFAULT_MAX_TURNS",
    "AgentBudgetExceeded",
    "AgentLoopResult",
    "AgentStep",
    "AgentTrace",
    "AgentTurn",
    "ExpandChunkContext",
    "GetChunkByCitation",
    "SearchRegulations",
    "Tool",
    "ToolCall",
    "ToolResult",
    "run_agent_loop",
]

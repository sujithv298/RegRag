"""LLM adapters.

Same `LLMAdapter` protocol, four implementations:

  - `FakeAdapter`        — deterministic stand-in for tests and offline demos
  - `AnthropicAdapter`   — Claude (cloud)
  - `OpenAIAdapter`      — GPT-4 (cloud)
  - `LlamaLocalAdapter`  — local Llama via llama-cpp-python (on-prem)

A bank with data-residency restrictions swaps the cloud adapter for the
local one and nothing else in the pipeline changes. That's what
"model-agnostic" means as code.
"""

from __future__ import annotations

from regrag.models.anthropic import AnthropicAdapter, AnthropicAgentAdapter
from regrag.models.base import LLMAdapter
from regrag.models.fake import FakeAdapter
from regrag.models.llama_local import LlamaLocalAdapter
from regrag.models.openai import OpenAIAdapter, OpenAIAgentAdapter
from regrag.models.types import GenerationResult

__all__ = [
    "AnthropicAdapter",
    "AnthropicAgentAdapter",
    "FakeAdapter",
    "GenerationResult",
    "LLMAdapter",
    "LlamaLocalAdapter",
    "OpenAIAdapter",
    "OpenAIAgentAdapter",
]

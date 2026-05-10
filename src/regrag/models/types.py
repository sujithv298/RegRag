"""Shared types for the model-adapter layer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GenerationResult(BaseModel):
    """One LLM completion. The audit log uses every field except `text`."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="Raw model output. Pre-verification, pre-refusal.")
    model_name: str = Field(description="Adapter-reported model name.")
    model_version: str = Field(
        description="Adapter-reported version. Often equals model_name; some APIs "
        "return a more specific build identifier in their response metadata."
    )
    input_tokens: int | None = Field(
        default=None, description="Provider-reported input token count, where available."
    )
    output_tokens: int | None = Field(
        default=None, description="Provider-reported output token count, where available."
    )

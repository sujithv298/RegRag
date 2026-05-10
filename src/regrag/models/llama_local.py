"""Local Llama adapter via llama-cpp-python.

This is the **data-residency path**. A bank that cannot send queries to a
cloud LLM points `REGRAG_DEFAULT_MODEL=llama_local` and the entire pipeline
runs on-prem. No external network calls anywhere in `answer_query`.

Only available with the `local` extra installed:
    uv sync --extra local

Lazy-loaded — importing this module without the extra installed is fine;
only `generate` raises.
"""

from __future__ import annotations

from regrag.models.types import GenerationResult


class LlamaLocalAdapter:
    """LLMAdapter for local Llama via llama-cpp-python."""

    def __init__(
        self,
        *,
        model_path: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
    ) -> None:
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._model: object | None = None

    @property
    def name(self) -> str:
        # The path's basename is what humans recognize ("llama-3-70b.Q4_K_M.gguf").
        from pathlib import Path  # noqa: PLC0415

        return Path(self._model_path).name

    @property
    def version(self) -> str:
        return self.name

    def generate(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> GenerationResult:
        self._ensure_loaded()
        assert self._model is not None  # noqa: S101
        # llama-cpp's chat API takes OpenAI-shaped messages.
        result = self._model.create_chat_completion(  # type: ignore[attr-defined]
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = result["choices"][0]
        text = choice["message"]["content"] or ""
        usage = result.get("usage") or {}
        return GenerationResult(
            text=text,
            model_name=self.name,
            model_version=self.version,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from llama_cpp import Llama  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "LlamaLocalAdapter requires `llama-cpp-python`. "
                "Install via `uv sync --extra local`."
            ) from exc
        self._model = Llama(
            model_path=self._model_path,
            n_ctx=self._n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            verbose=False,
        )

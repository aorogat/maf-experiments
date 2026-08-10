"""Instrumented LLM client wrapping OpenAI SDK (same pattern as overhead runners)."""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from frameworks.gabm_skeleton.metrics import InvocationRecorder, Metrics


def _normalize_model(model: str) -> str:
    if model.startswith("openai/"):
        return model.split("openai/", 1)[1]
    return model


def record_usage(
    metrics: Metrics | None,
    recorder: InvocationRecorder | None,
    *,
    input_tokens: int,
    output_tokens: int,
    llm_time_s: float = 0.0,
) -> None:
    """Update shared Metrics and optional per-node recorder for one LLM call."""
    if metrics is not None:
        metrics.llm_calls += 1
        metrics.llm_time_s += llm_time_s
        metrics.input_tokens += input_tokens
        metrics.output_tokens += output_tokens
    if recorder is not None:
        recorder.add(
            llm_calls=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            llm_time_s=llm_time_s,
        )


class InstrumentedOpenAIClient:
    """OpenAI client that records token usage and wall-clock LLM time."""

    def __init__(self, model: str = "gpt-4o-mini", metrics: Metrics | None = None):
        load_dotenv()
        self.model = _normalize_model(model)
        self.metrics = metrics
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in environment or .env")
        self.client = OpenAI(api_key=api_key)

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0,
        recorder: InvocationRecorder | None = None,
    ) -> str:
        t0 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed = time.perf_counter() - t0

        usage = response.usage
        in_tok = (getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
        out_tok = (getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0
        record_usage(
            self.metrics, recorder,
            input_tokens=in_tok, output_tokens=out_tok, llm_time_s=elapsed,
        )

        content = response.choices[0].message.content
        return (content or "").strip()


class StubLLMClient:
    """Deterministic LLM stub for tests (no network)."""

    def __init__(
        self,
        responses: list[str] | None = None,
        input_tokens: int = 10,
        output_tokens: int = 5,
        metrics: Metrics | None = None,
    ):
        self.responses = list(responses or [])
        self._index = 0
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.metrics = metrics
        self.prompts: list[str] = []

    def reset_for_run(self) -> None:
        """Reset per-run state so repeated run() calls are independent."""
        self._index = 0
        self.prompts.clear()
        if self.metrics is not None:
            self.metrics.reset()

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0,
        recorder: InvocationRecorder | None = None,
    ) -> str:
        self.prompts.append(prompt)
        record_usage(
            self.metrics, recorder,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )

        if self._index < len(self.responses):
            response = self.responses[self._index]
            self._index += 1
            return response
        return self.responses[-1] if self.responses else "{}"

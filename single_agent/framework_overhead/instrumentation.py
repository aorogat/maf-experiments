"""
Shared API-boundary instrumentation for framework overhead experiments.

Captures LLM calls, token usage, and wall-clock LLM time at the OpenAI SDK
and litellm boundaries so every framework reports comparable decomposition fields.
"""

from __future__ import annotations

import functools
import inspect
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

_SENTINEL = "__gabm_instrumented__"
EPS_MS = 1.0  # tolerance for llm_ms > total_ms overlap detection

_tls = threading.local()


@dataclass
class _Accumulator:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_time_s: float = 0.0
    active_calls: int = 0
    max_concurrency: int = 0
    _span_start: float | None = None
    _span_end: float | None = None

    @property
    def wall_llm_span_s(self) -> float:
        if self._span_start is None or self._span_end is None:
            return 0.0
        return max(0.0, self._span_end - self._span_start)

    def reset(self) -> None:
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_time_s = 0.0
        self.active_calls = 0
        self.max_concurrency = 0
        self._span_start = None
        self._span_end = None


@dataclass
class MeasureResult:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_time_s: float = 0.0
    wall_llm_span_s: float = 0.0
    max_concurrency: int = 0


def _get_acc() -> _Accumulator:
    acc = getattr(_tls, "acc", None)
    if acc is None:
        acc = _Accumulator()
        _tls.acc = acc
    return acc


def _normalize_usage(usage: Any) -> tuple[int, int]:
    if usage is None:
        return 0, 0
    in_tok = getattr(usage, "prompt_tokens", None)
    if in_tok is None:
        in_tok = getattr(usage, "input_tokens", 0)
    out_tok = getattr(usage, "completion_tokens", None)
    if out_tok is None:
        out_tok = getattr(usage, "output_tokens", 0)
    return int(in_tok or 0), int(out_tok or 0)


def _extract_usage_from_result(result: Any) -> tuple[int, int]:
    if result is None:
        return 0, 0
    usage = getattr(result, "usage", None)
    if usage is not None:
        return _normalize_usage(usage)
    if isinstance(result, tuple) and len(result) >= 2:
        return _extract_usage_from_result(result[1])
    return 0, 0


def _record_call(acc: _Accumulator, result: Any, elapsed_s: float) -> None:
    acc.llm_calls += 1
    acc.llm_time_s += elapsed_s
    in_tok, out_tok = _extract_usage_from_result(result)
    acc.input_tokens += in_tok
    acc.output_tokens += out_tok


def _enter_call(acc: _Accumulator) -> float:
    acc.active_calls += 1
    acc.max_concurrency = max(acc.max_concurrency, acc.active_calls)
    t0 = time.perf_counter()
    if acc._span_start is None:
        acc._span_start = t0
    return t0


def _leave_call(acc: _Accumulator, t0: float) -> float:
    t1 = time.perf_counter()
    if acc._span_end is None or t1 > acc._span_end:
        acc._span_end = t1
    acc.active_calls = max(0, acc.active_calls - 1)
    return t1 - t0


def _wrap_callable(fn: Callable) -> Callable:
    """Wrap sync or async-returning OpenAI SDK methods."""
    if getattr(fn, _SENTINEL, False):
        return fn

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_fn_wrapper(*args: Any, **kwargs: Any) -> Any:
            acc = _get_acc()
            t0 = _enter_call(acc)
            try:
                result = await fn(*args, **kwargs)
            except Exception:
                _leave_call(acc, t0)
                raise
            elapsed = _leave_call(acc, t0)
            _record_call(acc, result, elapsed)
            return result

        setattr(async_fn_wrapper, _SENTINEL, True)
        return async_fn_wrapper

    @functools.wraps(fn)
    def universal_wrapper(*args: Any, **kwargs: Any) -> Any:
        acc = _get_acc()
        t0 = _enter_call(acc)
        try:
            raw = fn(*args, **kwargs)
        except Exception:
            _leave_call(acc, t0)
            raise

        if inspect.isawaitable(raw):

            async def await_wrapper() -> Any:
                try:
                    result = await raw
                except Exception:
                    _leave_call(acc, t0)
                    raise
                elapsed = _leave_call(acc, t0)
                _record_call(acc, result, elapsed)
                return result

            return await_wrapper()

        elapsed = _leave_call(acc, t0)
        _record_call(acc, raw, elapsed)
        return raw

    setattr(universal_wrapper, _SENTINEL, True)
    return universal_wrapper


def _patch_method(owner: type, method_name: str) -> None:
    original = getattr(owner, method_name, None)
    if original is None or getattr(original, _SENTINEL, False):
        return
    wrapped = _wrap_callable(original)
    setattr(owner, method_name, wrapped)


_patches_installed = False


def install_patches() -> None:
    """Install idempotent OpenAI SDK and litellm boundary patches."""
    global _patches_installed
    if _patches_installed:
        return

    from openai.resources.chat.completions import AsyncCompletions, Completions
    from openai.resources.responses import AsyncResponses, Responses

    _patch_method(Completions, "create")
    _patch_method(AsyncCompletions, "create")
    _patch_method(Responses, "create")
    _patch_method(AsyncResponses, "create")

    try:
        from litellm.llms.openai.openai import OpenAIChatCompletion

        _patch_method(OpenAIChatCompletion, "make_sync_openai_chat_completion_request")
        _patch_method(OpenAIChatCompletion, "make_openai_chat_completion_request")
    except ImportError:
        pass

    _patches_installed = True


@contextmanager
def measure():
    """Reset thread-local counters; populate MeasureResult on exit."""
    acc = _Accumulator()
    _tls.acc = acc
    result = MeasureResult()
    try:
        yield result
    finally:
        result.llm_calls = acc.llm_calls
        result.input_tokens = acc.input_tokens
        result.output_tokens = acc.output_tokens
        result.llm_time_s = acc.llm_time_s
        result.wall_llm_span_s = acc.wall_llm_span_s
        result.max_concurrency = acc.max_concurrency


def overlap_flag(m: MeasureResult, total_ms: float) -> bool:
    llm_ms = m.llm_time_s * 1000.0
    return m.max_concurrency > 1 or llm_ms > total_ms + EPS_MS


def residual_ms(m: MeasureResult, total_ms: float, *, use_span_on_overlap: bool = True) -> tuple[float, bool]:
    """
    Compute framework residual in ms.

    Returns (residual_ms, residual_valid).
    When overlap is detected, falls back to wall_llm_span_s for the LLM numerator.
    """
    llm_ms = m.llm_time_s * 1000.0
    if overlap_flag(m, total_ms):
        span_ms = m.wall_llm_span_s * 1000.0
        if use_span_on_overlap and span_ms > 0:
            return total_ms - span_ms, False
        return total_ms - llm_ms, False
    return total_ms - llm_ms, True

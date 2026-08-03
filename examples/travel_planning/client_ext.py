"""Instrumented LLM client extensions for agent-bound (function-calling) tools."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from frameworks.gabm_skeleton.llm_client import InstrumentedOpenAIClient, StubLLMClient

from examples.travel_planning.tools import execute_cal, execute_web
from examples.travel_planning.trace import ToolCall


@dataclass
class ToolCallResult:
    """Result of a complete_with_tools round-trip."""

    content: str
    tool_calls: list[ToolCall]


class ToolCallingClient(InstrumentedOpenAIClient):
    """OpenAI function-calling client reusing the shared Metrics counters.

    NOTE: Used for agent-bound (probabilistic) tool mode to keep token/call
    counting identical across paradigms. This is NOT LangGraph's native
    ChatOpenAI.bind_tools path — see langgraph_impl.py for rationale.
    """

    def complete_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        *,
        max_tokens: int = 256,
        temperature: float = 0,
        force_tool: str | None = None,
    ) -> ToolCallResult:
        """One LLM call; may return tool calls (probabilistic) or plain content."""
        t0 = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            if force_tool:
                kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": force_tool},
                }
            else:
                kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - t0

        if self.metrics is not None:
            self.metrics.llm_calls += 1
            self.metrics.llm_time_s += elapsed
            usage = response.usage
            if usage is not None:
                self.metrics.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self.metrics.output_tokens += getattr(usage, "completion_tokens", 0) or 0

        message = response.choices[0].message
        recorded: list[ToolCall] = []
        content = (message.content or "").strip()

        if message.tool_calls:
            for tc in message.tool_calls:
                fn = tc.function
                args = json.loads(fn.arguments or "{}")
                if fn.name == "Web":
                    result = execute_web(args)
                elif fn.name == "Cal":
                    result = execute_cal(args)
                else:
                    result = {"error": f"unknown tool {fn.name}"}
                recorded.append(ToolCall(
                    tool=fn.name,
                    args=args,
                    result=result,
                    invocation="probabilistic",
                    invoked_by="agent",
                ))

        return ToolCallResult(content=content, tool_calls=recorded)


@dataclass
class StubToolResponse:
    """Canned stub response: either plain content or a tool call."""

    content: str = ""
    tool_name: str | None = None
    tool_args: dict | None = None


class ToolCallingStub(StubLLMClient):
    """Deterministic stub supporting tool-call or plain-content responses."""

    def __init__(
        self,
        responses: list[str | StubToolResponse] | None = None,
        input_tokens: int = 10,
        output_tokens: int = 5,
        metrics=None,
    ):
        super().__init__(
            responses=[],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metrics=metrics,
        )
        self._tool_responses: list[StubToolResponse] = []
        for r in responses or []:
            if isinstance(r, StubToolResponse):
                self._tool_responses.append(r)
            else:
                self._tool_responses.append(StubToolResponse(content=r))
        self._tool_index = 0

    def reset_for_run(self) -> None:
        super().reset_for_run()
        self._tool_index = 0

    def complete_with_tools(
        self,
        prompt: str,
        tools: list[dict] | None = None,
        *,
        max_tokens: int = 256,
        temperature: float = 0,
        force_tool: str | None = None,
    ) -> ToolCallResult:
        self.prompts.append(prompt)
        if self.metrics is not None:
            self.metrics.llm_calls += 1
            self.metrics.input_tokens += self.input_tokens
            self.metrics.output_tokens += self.output_tokens

        if self._tool_index < len(self._tool_responses):
            stub = self._tool_responses[self._tool_index]
            self._tool_index += 1
        elif self._tool_responses:
            stub = self._tool_responses[-1]
        else:
            return ToolCallResult(content="{}", tool_calls=[])

        recorded: list[ToolCall] = []
        if stub.tool_name and stub.tool_args is not None:
            if stub.tool_name == "Web":
                result = execute_web(stub.tool_args)
            elif stub.tool_name == "Cal":
                result = execute_cal(stub.tool_args)
            else:
                result = {}
            recorded.append(ToolCall(
                tool=stub.tool_name,
                args=stub.tool_args,
                result=result,
                invocation="probabilistic",
                invoked_by="agent",
            ))
        return ToolCallResult(content=stub.content, tool_calls=recorded)

    def complete(self, prompt: str, *, max_tokens: int = 256, temperature: float = 0) -> str:
        result = self.complete_with_tools(prompt, tools=None, max_tokens=max_tokens, temperature=temperature)
        if result.tool_calls:
            return json.dumps({"tool_called": result.tool_calls[0].tool})
        return result.content or "{}"

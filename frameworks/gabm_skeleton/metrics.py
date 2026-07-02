"""Instrumentation container for GABM skeleton runs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Metrics:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_time_s: float = 0.0
    total_time_s: float = 0.0
    output_chars: int = 0

    def reset(self) -> None:
        """Zero all counters (fresh trial)."""
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_time_s = 0.0
        self.total_time_s = 0.0
        self.output_chars = 0

    @property
    def framework_residual_s(self) -> float:
        return self.total_time_s - self.llm_time_s


@dataclass(frozen=True)
class RoundTrace:
    prompts_issued: list[str]
    llm_calls: int
    state_snapshot: dict
    communication: str
    tool_invocations: list[dict]


@dataclass
class RunResult:
    answer: str
    final_state: dict
    metrics: Metrics
    trace: list[RoundTrace] = field(default_factory=list)

    def trace_table(self) -> list[dict]:
        rows = []
        for i, rt in enumerate(self.trace):
            rows.append({
                "round": i + 1,
                "llm_calls": rt.llm_calls,
                "communication": rt.communication,
                "tool_count": len(rt.tool_invocations),
                "tools": [t.get("tool") for t in rt.tool_invocations],
                "state_keys": sorted(rt.state_snapshot.keys()),
            })
        return rows

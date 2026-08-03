"""Unified trace schema and side-by-side table emitters."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from frameworks.gabm_skeleton.metrics import Metrics, RoundTrace

from examples.travel_planning.task import (
    AGENT_BUDGET,
    AGENT_FLIGHT,
    AGENT_HOTEL,
    AGENT_PLANNER,
    budget_satisfied,
    format_itinerary,
    plan_complete,
    resolve_budget_status,
)


@dataclass
class ToolCall:
    tool: str  # "Web" | "Cal"
    args: dict
    result: Any
    invocation: str  # "deterministic" | "probabilistic"
    invoked_by: str  # "graph_node" | "agent" | "environment"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraceStep:
    agent: str
    prompt_issued: str
    llm_calls: int
    input_tokens: int
    output_tokens: int
    state_snapshot: dict
    communication: str  # "graph_edge" | "role_handoff" | "gm_mediated"
    tool_invocations: list[ToolCall] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tool_invocations"] = [t.to_dict() for t in self.tool_invocations]
        return d


@dataclass
class TravelTrace:
    paradigm: str
    tool_mode: str
    steps: list[TraceStep] = field(default_factory=list)
    metrics: Metrics | None = None
    final_state: dict = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Per-paradigm totals for side-by-side comparison."""
        total_llm = sum(s.llm_calls for s in self.steps)
        total_in = sum(s.input_tokens for s in self.steps)
        total_out = sum(s.output_tokens for s in self.steps)
        all_tools = [t for s in self.steps for t in s.tool_invocations]
        det = sum(1 for t in all_tools if t.invocation == "deterministic")
        prob = sum(1 for t in all_tools if t.invocation == "probabilistic")
        web_invoked = any(t.tool == "Web" for t in all_tools)
        cal_invoked = any(t.tool == "Cal" for t in all_tools)
        complete = plan_complete(self.final_state)
        return {
            "paradigm": self.paradigm,
            "tool_mode": self.tool_mode,
            "total_llm_calls": total_llm,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tool_calls": len(all_tools),
            "deterministic_tool_calls": det,
            "probabilistic_tool_calls": prob,
            "steps": len(self.steps),
            "web_invoked": web_invoked,
            "cal_invoked": cal_invoked,
            "plan_complete": complete,
            "final_itinerary": format_itinerary(self.final_state),
            "budget_satisfied": budget_satisfied(self.final_state),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "paradigm": self.paradigm,
            "tool_mode": self.tool_mode,
            "steps": [s.to_dict() for s in self.steps],
            "summary": self.summary(),
            "final_state": self.final_state,
        }

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def skeleton_tool_to_toolcall(inv: dict) -> ToolCall:
    """Map skeleton {tool,args,result,deterministic} dict to unified ToolCall."""
    raw_name = inv.get("tool", "")
    if raw_name in ("search_flights", "search_hotels"):
        tool = "Web"
    elif raw_name in ("calculate_total", "suggest_upgrade"):
        tool = "Cal"
    else:
        tool = raw_name
    return ToolCall(
        tool=tool,
        args=inv.get("args", {}),
        result=inv.get("result"),
        invocation="deterministic" if inv.get("deterministic", True) else "probabilistic",
        invoked_by="environment",
    )


def from_round_traces(
    paradigm: str,
    tool_mode: str,
    round_traces: list[RoundTrace],
    agent_ids: list[str],
    metrics: Metrics,
    final_state: dict,
) -> TravelTrace:
    """Convert GABM skeleton RoundTrace list to per-agent TraceStep list."""
    steps: list[TraceStep] = []
    tokens_per_call_in = metrics.input_tokens // max(metrics.llm_calls, 1)
    tokens_per_call_out = metrics.output_tokens // max(metrics.llm_calls, 1)

    for rt in round_traces:
        tools_by_agent: dict[str, list[ToolCall]] = {a: [] for a in agent_ids}
        tool_idx = 0
        agent_tool_map = {
            "flight": AGENT_FLIGHT,
            "hotel": AGENT_HOTEL,
            "budget": AGENT_BUDGET,
            "planner": AGENT_PLANNER,
        }
        for inv in rt.tool_invocations:
            tc = skeleton_tool_to_toolcall(inv)
            if tool_idx < len(agent_ids):
                aid = agent_ids[tool_idx] if tool_idx < len(rt.prompts_issued) else agent_ids[-1]
            else:
                aid = agent_ids[-1]
            # Map by tool type to agent
            raw = inv.get("tool", "")
            if "flight" in raw:
                aid = "Flight"
            elif "hotel" in raw:
                aid = "Hotel"
            elif "total" in raw or "upgrade" in raw or "cal" in raw.lower():
                aid = "Budget"
            tools_by_agent.setdefault(aid, []).append(tc)
            tool_idx += 1

        for i, agent_id in enumerate(agent_ids):
            prompt = rt.prompts_issued[i] if i < len(rt.prompts_issued) else ""
            display_agent = agent_tool_map.get(agent_id, agent_id)
            if isinstance(display_agent, str) and display_agent[0].islower():
                display_agent = display_agent.capitalize()
            steps.append(TraceStep(
                agent=display_agent if display_agent in ("Planner", "Flight", "Hotel", "Budget") else agent_id,
                prompt_issued=prompt,
                llm_calls=1,
                input_tokens=tokens_per_call_in,
                output_tokens=tokens_per_call_out,
                state_snapshot=dict(rt.state_snapshot),
                communication=rt.communication,
                tool_invocations=tools_by_agent.get(display_agent, tools_by_agent.get(agent_id, [])),
            ))

    return TravelTrace(
        paradigm=paradigm,
        tool_mode=tool_mode,
        steps=steps,
        metrics=metrics,
        final_state=final_state,
    )


SIDE_BY_SIDE_COLUMNS = [
    "Paradigm",
    "Tool mode",
    "Steps",
    "LLM calls",
    "Input tokens",
    "Output tokens",
    "Tool calls (det/prob)",
    "Plan complete",
    "Budget satisfied",
]


def side_by_side(traces: dict[str, TravelTrace]) -> tuple[str, str]:
    """Return (markdown_table, latex_table) for supplement."""
    rows = []
    for name, trace in traces.items():
        s = trace.summary()
        rows.append({
            "Paradigm": s["paradigm"],
            "Tool mode": s["tool_mode"],
            "Steps": s["steps"],
            "LLM calls": s["total_llm_calls"],
            "Input tokens": s["total_input_tokens"],
            "Output tokens": s["total_output_tokens"],
            "Tool calls (det/prob)": f"{s['deterministic_tool_calls']}/{s['probabilistic_tool_calls']}",
            "Plan complete": str(s["plan_complete"]),
            "Budget satisfied": str(s["budget_satisfied"]),
        })

    # Markdown
    md_lines = ["| " + " | ".join(SIDE_BY_SIDE_COLUMNS) + " |"]
    md_lines.append("| " + " | ".join("---" for _ in SIDE_BY_SIDE_COLUMNS) + " |")
    for row in rows:
        md_lines.append("| " + " | ".join(str(row[c]) for c in SIDE_BY_SIDE_COLUMNS) + " |")
    md = "\n".join(md_lines)

    # LaTeX
    col_spec = "l" + "r" * (len(SIDE_BY_SIDE_COLUMNS) - 1)
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Travel-planning trace comparison across paradigms.}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(SIDE_BY_SIDE_COLUMNS) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        latex_lines.append(" & ".join(str(row[c]) for c in SIDE_BY_SIDE_COLUMNS) + " \\\\")
    latex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    latex = "\n".join(latex_lines)

    return md, latex


# Frozen column order for W2-W2 citation — do not reorder or rename.
TOOL_INVOCATION_COLUMNS = [
    "Paradigm",
    "Tool mode",
    "Tool-call count",
    "Invoked?",
    "LLM calls",
    "Tokens (in/out)",
]


def tool_invocation_comparison(traces: dict[str, TravelTrace]) -> str:
    """LaTeX table: deterministic vs probabilistic tool invocation (W2-W2)."""
    rows = []
    for _name, trace in traces.items():
        s = trace.summary()
        invoked = (
            f"Web={'Y' if s['web_invoked'] else 'N'}, "
            f"Cal={'Y' if s['cal_invoked'] else 'N'}, "
            f"Plan={'Y' if s['plan_complete'] else 'N'}"
        )
        rows.append({
            "Paradigm": s["paradigm"],
            "Tool mode": s["tool_mode"],
            "Tool-call count": s["total_tool_calls"],
            "Invoked?": invoked,
            "LLM calls": s["total_llm_calls"],
            "Tokens (in/out)": f"{s['total_input_tokens']}/{s['total_output_tokens']}",
        })

    lines = [
        "% Frozen column order for W2-W2 — do not reorder or rename.",
        "% Invoked? includes Plan=Y/N: whether flight+hotel were selected.",
        "% LangGraph agent-bound may show Web=N, Cal=Y, Plan=N — probabilistic skip",
        "% of required tools (incomplete plan), not a harness bug. Graph-node cannot skip.",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Tool invocation comparison: deterministic (graph-node) vs probabilistic (agent-bound).}",
        "\\begin{tabular}{llrlrl}",
        "\\toprule",
        " & ".join(TOOL_INVOCATION_COLUMNS) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(row[c]) for c in TOOL_INVOCATION_COLUMNS) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)

"""GABM skeleton implementation (env-executed tools)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from frameworks.gabm_skeleton.agent import SkeletonAgent
from frameworks.gabm_skeleton.environment import Environment
from frameworks.gabm_skeleton.llm_client import InstrumentedOpenAIClient, StubLLMClient
from frameworks.gabm_skeleton.metrics import Metrics, RunResult
from frameworks.gabm_skeleton.runner import GABMSkeletonRunner

from examples.travel_planning.task import (
    AGENT_BUDGET,
    AGENT_FLIGHT,
    AGENT_HOTEL,
    AGENT_PLANNER,
    STATE_SCHEMA,
    TripRequest,
    initial_state,
)
from examples.travel_planning.tools import (
    cal_args_from_state,
    execute_cal,
    execute_web,
    invoke_as_environment,
    web_args_for_request,
)
from examples.travel_planning.trace import ToolCall, TraceStep, TravelTrace, skeleton_tool_to_toolcall


def _gabm_transition(state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    new_state = dict(state)
    for key in ("chosen_flight", "chosen_hotel", "running_cost", "status"):
        if key in action:
            new_state[key] = action[key]
    return new_state


class Fig1GabmTask:
    """Four-agent Fig-1 travel task on the GABM skeleton (Planner/Flight/Hotel/Budget)."""

    AGENT_PLANNER = "planner"
    AGENT_FLIGHT = "flight"
    AGENT_HOTEL = "hotel"
    AGENT_BUDGET = "budget"

    def __init__(
        self,
        request: TripRequest | None = None,
        max_rounds: int = 1,
    ):
        self.request = request or TripRequest()
        self.max_rounds = max_rounds

    @property
    def agent_ids(self) -> list[str]:
        return [self.AGENT_PLANNER, self.AGENT_FLIGHT, self.AGENT_HOTEL, self.AGENT_BUDGET]

    def make_environment(self) -> Environment:
        return Environment(STATE_SCHEMA, initial_state(self.request), _gabm_transition)

    def visible_fields(self, agent_id: str) -> set[str]:
        if agent_id == self.AGENT_PLANNER:
            return {"origin", "destination", "dates", "budget", "status"}
        if agent_id == self.AGENT_FLIGHT:
            return {"origin", "destination", "dates", "budget", "running_cost"}
        if agent_id == self.AGENT_HOTEL:
            return {"destination", "dates", "budget", "chosen_flight"}
        if agent_id == self.AGENT_BUDGET:
            return {"chosen_flight", "chosen_hotel", "budget", "running_cost", "status"}
        return set()

    def process_action(
        self,
        agent_id: str,
        action: dict[str, Any],
        state: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict]]:
        """Environment executes tools; agents only emit JSON actions."""
        new_state = dict(state)
        tools: list[dict] = []

        if agent_id == self.AGENT_PLANNER:
            new_state["status"] = "delegating"

        elif agent_id == self.AGENT_FLIGHT:
            args = web_args_for_request(self.request, kind="flight")
            tc: ToolCall = invoke_as_environment("Web", args, execute_web)
            tools.append({
                "tool": "search_flights",
                "args": tc.args,
                "result": tc.result.get("options", [{}])[0] if tc.result else {},
                "deterministic": True,
            })
            options = tc.result.get("options", [])
            if options:
                new_state["chosen_flight"] = options[0]

        elif agent_id == self.AGENT_HOTEL:
            args = web_args_for_request(self.request, kind="hotel")
            tc = invoke_as_environment("Web", args, execute_web)
            tools.append({
                "tool": "search_hotels",
                "args": tc.args,
                "result": tc.result.get("options", [{}])[0] if tc.result else {},
                "deterministic": True,
            })
            options = tc.result.get("options", [])
            if options:
                new_state["chosen_hotel"] = options[0]

        elif agent_id == self.AGENT_BUDGET:
            args = cal_args_from_state(state)
            tc = invoke_as_environment("Cal", args, execute_cal)
            tools.append({
                "tool": "calculate_total",
                "args": tc.args,
                "result": tc.result,
                "deterministic": True,
            })
            new_state["running_cost"] = tc.result.get("total", 0)
            total = new_state["running_cost"]
            within = tc.result.get("within_budget", total <= state["budget"])
            from examples.travel_planning.task import resolve_budget_status
            new_state["status"] = resolve_budget_status(new_state, total, within_budget=within)

        else:
            raise ValueError(f"Unknown agent: {agent_id}")

        return new_state, tools

    def is_complete(self, state: dict[str, Any]) -> bool:
        from examples.travel_planning.task import budget_satisfied
        return budget_satisfied(state)

    def extract_answer(self, state: dict[str, Any]) -> str:
        from examples.travel_planning.task import format_itinerary
        return format_itinerary(state)

    def make_agents(self, llm_client) -> list[SkeletonAgent]:
        return [
            SkeletonAgent(
                agent_id=self.AGENT_PLANNER,
                system_prompt='You are the Planner. Return JSON: {"action": "delegate"}',
                llm_client=llm_client,
                fallback_action={"action": "delegate"},
            ),
            SkeletonAgent(
                agent_id=self.AGENT_FLIGHT,
                system_prompt='You are the Flight agent. Return JSON: {"action": "select_flight"}',
                llm_client=llm_client,
                fallback_action={"action": "select_flight"},
            ),
            SkeletonAgent(
                agent_id=self.AGENT_HOTEL,
                system_prompt='You are the Hotel agent. Return JSON: {"action": "select_hotel"}',
                llm_client=llm_client,
                fallback_action={"action": "select_hotel"},
            ),
            SkeletonAgent(
                agent_id=self.AGENT_BUDGET,
                system_prompt='You are the Budget agent. Return JSON: {"action": "check_budget"}',
                llm_client=llm_client,
                fallback_action={"action": "check_budget"},
            ),
        ]


_AGENT_DISPLAY = {
    "planner": AGENT_PLANNER,
    "flight": AGENT_FLIGHT,
    "hotel": AGENT_HOTEL,
    "budget": AGENT_BUDGET,
}


def run_result_to_travel_trace(result: RunResult, task: Fig1GabmTask) -> TravelTrace:
    """Convert GABM RunResult (per-round traces) to per-agent TraceStep list."""
    metrics = result.metrics
    tokens_in = metrics.input_tokens // max(metrics.llm_calls, 1)
    tokens_out = metrics.output_tokens // max(metrics.llm_calls, 1)
    steps: list[TraceStep] = []

    for rt in result.trace:
        tools_by_idx: list[list[ToolCall]] = [[] for _ in task.agent_ids]
        tool_assignments = {
            "search_flights": task.AGENT_FLIGHT,
            "search_hotels": task.AGENT_HOTEL,
            "calculate_total": task.AGENT_BUDGET,
        }
        for inv in rt.tool_invocations:
            tc = skeleton_tool_to_toolcall(inv)
            tc.invoked_by = "environment"
            raw = inv.get("tool", "")
            aid = tool_assignments.get(raw, task.AGENT_BUDGET)
            idx = task.agent_ids.index(aid)
            tools_by_idx[idx].append(tc)

        for i, agent_id in enumerate(task.agent_ids):
            prompt = rt.prompts_issued[i] if i < len(rt.prompts_issued) else ""
            steps.append(TraceStep(
                agent=_AGENT_DISPLAY.get(agent_id, agent_id),
                prompt_issued=prompt,
                llm_calls=1,
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                state_snapshot=deepcopy(rt.state_snapshot),
                communication="gm_mediated",
                tool_invocations=tools_by_idx[i],
            ))

    return TravelTrace(
        paradigm="GABM",
        tool_mode="environment-executed",
        steps=steps,
        metrics=metrics,
        final_state=result.final_state,
    )


def run_gabm(
    request: TripRequest | None = None,
    llm_client: InstrumentedOpenAIClient | StubLLMClient | None = None,
    model: str = "openai/gpt-4o-mini",
) -> TravelTrace:
    """Execute travel planning via GABM skeleton."""
    task = Fig1GabmTask(request=request, max_rounds=1)
    runner = GABMSkeletonRunner(model=model, llm_client=llm_client)
    result = runner.run(task)
    assert isinstance(result, RunResult)
    return run_result_to_travel_trace(result, task)

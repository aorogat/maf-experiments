"""Role-based implementation (agent-bound tools)."""

from __future__ import annotations

import json
from copy import deepcopy

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from frameworks.gabm_skeleton.llm_client import InstrumentedOpenAIClient, StubLLMClient
from frameworks.gabm_skeleton.metrics import Metrics
from single_agent.framework_overhead.instrumentation import install_patches, measure

from examples.travel_planning.client_ext import ToolCallingClient, ToolCallingStub
from examples.travel_planning.task import (
    AGENT_BUDGET,
    AGENT_FLIGHT,
    AGENT_HOTEL,
    AGENT_PLANNER,
    TripRequest,
    initial_state,
    plan_complete,
    resolve_budget_status,
)
from examples.travel_planning.tools import execute_cal, execute_web, invoke_as_agent
from examples.travel_planning.trace import ToolCall, TraceStep, TravelTrace

# R1.O3: CrewAI internal planner prompts are not fully retrievable; we record
# task descriptions. Live runs use install_patches()+measure() at API boundaries
# for token counts; stub runs use InstrumentedCrewLLM directly.


class _ToolLog:
    """Per-step log for agent-bound tool invocations during a CrewAI run."""

    def __init__(self) -> None:
        self.by_agent: dict[str, list[ToolCall]] = {}

    def record(self, agent: str, tc: ToolCall) -> None:
        self.by_agent.setdefault(agent, []).append(tc)

    def reset_agent(self, agent: str) -> None:
        self.by_agent[agent] = []


class WebInput(BaseModel):
    query: str = Field(..., description="Search query")
    kind: str = Field("flight", description="flight or hotel")
    destination: str = Field("NYC", description="Destination city code")


class CalInput(BaseModel):
    flight_price: int = Field(0, description="Flight price")
    hotel_price: int = Field(0, description="Hotel price")
    budget: int = Field(0, description="Total budget")


def _make_web_tool(log: _ToolLog, agent: str) -> BaseTool:
    class WebTool(BaseTool):
        name: str = "Web"
        description: str = "Search for flights or hotels."
        args_schema: type[BaseModel] = WebInput

        def _run(self, query: str, kind: str = "flight", destination: str = "NYC") -> str:
            args = {"query": query, "kind": kind, "destination": destination}
            tc = invoke_as_agent("Web", args, execute_web)
            log.record(agent, tc)
            return json.dumps(tc.result)

    return WebTool()


def _make_cal_tool(log: _ToolLog, agent: str) -> BaseTool:
    class CalTool(BaseTool):
        name: str = "Cal"
        description: str = "Calculate total trip cost and check budget."
        args_schema: type[BaseModel] = CalInput

        def _run(self, flight_price: int = 0, hotel_price: int = 0, budget: int = 0) -> str:
            args = {"flight_price": flight_price, "hotel_price": hotel_price, "budget": budget}
            tc = invoke_as_agent("Cal", args, execute_cal)
            log.record(agent, tc)
            return json.dumps(tc.result)

    return CalTool()


class InstrumentedCrewLLM:
    """Wraps shared instrumented client for CrewAI Agent llm= (stub / test path)."""

    def __init__(self, client: InstrumentedOpenAIClient | StubLLMClient | ToolCallingClient | ToolCallingStub):
        self._client = client
        self.model = getattr(client, "model", "gpt-4o-mini")

    def call(self, messages: list | str, **kwargs) -> str:
        if isinstance(messages, list):
            prompt = messages[-1].get("content", "") if messages else ""
        else:
            prompt = str(messages)
        return self._client.complete(prompt)

    def __call__(self, prompt: str) -> str:
        return self._client.complete(prompt)


def _snapshot_metrics(metrics: Metrics) -> tuple[int, int, int]:
    return metrics.llm_calls, metrics.input_tokens, metrics.output_tokens


def _metrics_delta(metrics: Metrics, before: tuple[int, int, int]) -> tuple[int, int, int]:
    return (
        metrics.llm_calls - before[0],
        metrics.input_tokens - before[1],
        metrics.output_tokens - before[2],
    )


def _apply_tool_results(state: dict, agent: str, tools: list[ToolCall]) -> dict:
    new_state = deepcopy(state)
    for tc in tools:
        if tc.tool == "Web" and tc.result:
            options = tc.result.get("options", [])
            if options:
                if agent == AGENT_FLIGHT:
                    new_state["chosen_flight"] = options[0]
                elif agent == AGENT_HOTEL:
                    new_state["chosen_hotel"] = options[0]
        elif tc.tool == "Cal" and tc.result:
            total = tc.result.get("total", 0)
            within = tc.result.get("within_budget", total <= new_state.get("budget", 0))
            if plan_complete(new_state):
                new_state["running_cost"] = total
            new_state["status"] = resolve_budget_status(new_state, total, within_budget=within)
    return new_state


def _task_description(agent_name: str, state: dict, req: TripRequest) -> str:
    """Build task prompt from current shared state (not frozen at trip start)."""
    trip_desc = (
        f"Plan a trip from {req.origin} to {req.destination}, "
        f"dates {req.dates}, budget ${req.budget}."
    )
    if agent_name == AGENT_PLANNER:
        return (
            f"{trip_desc} As Planner, outline the plan and delegate to Flight, Hotel, "
            "and Budget agents. Respond briefly with your coordination plan."
        )
    if agent_name == AGENT_FLIGHT:
        return (
            f"Search flights to {req.destination} using the Web tool if needed. "
            f"Context: {json.dumps(state)}"
        )
    if agent_name == AGENT_HOTEL:
        return (
            f"Search hotels in {req.destination} using the Web tool if needed. "
            f"Context: {json.dumps(state)}"
        )
    flight_p = (state.get("chosen_flight") or {}).get("price", 0)
    hotel_p = (state.get("chosen_hotel") or {}).get("price", 0)
    return (
        f"Calculate total cost with Cal tool. Flight price: {flight_p}, "
        f"hotel price: {hotel_p}, budget ${req.budget}."
    )


def _normalize_crewai_model(model: str) -> str:
    if model.startswith("openai/"):
        return model
    bare = model.split("openai/", 1)[-1] if "openai/" in model else model
    return f"openai/{bare}"


def run_crewai(
    request: TripRequest | None = None,
    llm_client: InstrumentedOpenAIClient | StubLLMClient | ToolCallingClient | ToolCallingStub | None = None,
    model: str = "openai/gpt-4o-mini",
) -> TravelTrace:
    """Execute travel planning via CrewAI role-conditioned crew."""
    load_dotenv()
    req = request or TripRequest()
    metrics = Metrics()
    tool_log = _ToolLog()
    stub_mode = isinstance(llm_client, (StubLLMClient, ToolCallingStub))

    if llm_client is None:
        llm_client = InstrumentedOpenAIClient(model=model, metrics=metrics)
    else:
        llm_client.metrics = metrics

    if stub_mode:
        agent_llm = InstrumentedCrewLLM(llm_client)
    else:
        install_patches()
        agent_llm = LLM(model=_normalize_crewai_model(model), temperature=0)

    state = initial_state(req)
    steps: list[TraceStep] = []

    web_flight = _make_web_tool(tool_log, AGENT_FLIGHT)
    web_hotel = _make_web_tool(tool_log, AGENT_HOTEL)
    cal_tool = _make_cal_tool(tool_log, AGENT_BUDGET)

    planner = Agent(
        role="Planner Agent",
        goal="Coordinate trip planning and delegate to specialists.",
        backstory="Central planner for the trip itinerary.",
        memory=False,
        verbose=False,
        llm=agent_llm,
    )
    flight_agent = Agent(
        role="Flight Agent",
        goal="Find and select flights.",
        backstory="Flight specialist.",
        tools=[web_flight],
        memory=False,
        verbose=False,
        llm=agent_llm,
    )
    hotel_agent = Agent(
        role="Hotel Agent",
        goal="Find and book accommodation.",
        backstory="Hotel specialist.",
        tools=[web_hotel],
        memory=False,
        verbose=False,
        llm=agent_llm,
    )
    budget_agent = Agent(
        role="Budget Agent",
        goal="Ensure the trip meets the budget constraint.",
        backstory="Budget reviewer.",
        tools=[cal_tool],
        memory=False,
        verbose=False,
        llm=agent_llm,
    )

    agent_order = [
        (AGENT_PLANNER, planner),
        (AGENT_FLIGHT, flight_agent),
        (AGENT_HOTEL, hotel_agent),
        (AGENT_BUDGET, budget_agent),
    ]

    for agent_name, agent in agent_order:
        tool_log.reset_agent(agent_name)
        description = _task_description(agent_name, state, req)
        task = Task(description=description, expected_output="Brief JSON or text summary.", agent=agent)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)

        if stub_mode:
            before = _snapshot_metrics(metrics)
            crew.kickoff()
            llm_c, in_t, out_t = _metrics_delta(metrics, before)
        else:
            with measure() as m:
                crew.kickoff()
            llm_c, in_t, out_t = m.llm_calls, m.input_tokens, m.output_tokens
            metrics.llm_calls += llm_c
            metrics.input_tokens += in_t
            metrics.output_tokens += out_t

        agent_tools = list(tool_log.by_agent.get(agent_name, []))
        if agent_name == AGENT_PLANNER:
            state["status"] = "delegating"
        else:
            state = _apply_tool_results(state, agent_name, agent_tools)

        steps.append(TraceStep(
            agent=agent_name,
            prompt_issued=description,
            llm_calls=llm_c,
            input_tokens=in_t,
            output_tokens=out_t,
            state_snapshot=deepcopy(state),
            communication="role_handoff",
            tool_invocations=agent_tools,
        ))

    if not plan_complete(state):
        state["status"] = "incomplete"

    return TravelTrace(
        paradigm="CrewAI",
        tool_mode="probabilistic (agent-bound)",
        steps=steps,
        metrics=metrics,
        final_state=state,
    )

"""Directed-graph implementation (+ tool-node and agent-bound variants)."""

from __future__ import annotations

from copy import deepcopy
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from frameworks.gabm_skeleton.llm_client import InstrumentedOpenAIClient, StubLLMClient
from frameworks.gabm_skeleton.metrics import Metrics

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
from examples.travel_planning.tools import (
    CAL_TOOL_SCHEMA,
    WEB_TOOL_SCHEMA,
    cal_args_from_state,
    execute_cal,
    execute_web,
    invoke_as_graph_node,
    web_args_for_request,
)
from examples.travel_planning.trace import TraceStep, TravelTrace


class GraphState(TypedDict, total=False):
    """LangGraph channel state; nodes return partial updates for parallel merge."""

    origin: str
    destination: str
    dates: str
    budget: int
    chosen_flight: dict
    chosen_hotel: dict
    running_cost: int
    status: str


def _metrics_delta(metrics: Metrics, before: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return (llm_calls, input_tokens, output_tokens) since before snapshot."""
    return (
        metrics.llm_calls - before[0],
        metrics.input_tokens - before[1],
        metrics.output_tokens - before[2],
    )


def _snapshot_metrics(metrics: Metrics) -> tuple[int, int, int]:
    return metrics.llm_calls, metrics.input_tokens, metrics.output_tokens


def _merged(state: GraphState, update: dict) -> dict:
    """Full state after applying a partial node update (for trace snapshots)."""
    return {**state, **update}


class _TraceCollector:
    """Accumulates TraceStep entries during a LangGraph run."""

    def __init__(self) -> None:
        self.steps: list[TraceStep] = []

    def record(
        self,
        agent: str,
        prompt: str,
        metrics: Metrics,
        before: tuple[int, int, int],
        state: dict,
        tools: list,
    ) -> None:
        llm_c, in_t, out_t = _metrics_delta(metrics, before)
        self.steps.append(TraceStep(
            agent=agent,
            prompt_issued=prompt,
            llm_calls=llm_c,
            input_tokens=in_t,
            output_tokens=out_t,
            state_snapshot=deepcopy(state),
            communication="graph_edge",
            tool_invocations=list(tools),
        ))


def _wire_parallel_fig1(workflow: StateGraph) -> None:
    """Figure 1(a): Planner fans out to Flight and Hotel in parallel, join before Budget."""
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "flight_agent")
    workflow.add_edge("planner", "hotel_agent")
    workflow.add_edge("flight_agent", "web_flight")
    workflow.add_edge("hotel_agent", "web_hotel")
    workflow.add_edge("web_flight", "budget_agent")
    workflow.add_edge("web_hotel", "budget_agent")
    workflow.add_edge("budget_agent", "cal_tool")
    workflow.add_edge("cal_tool", END)


def _wire_parallel_fig1_agent(workflow: StateGraph) -> None:
    """Agent-bound variant: parallel Flight/Hotel agents, join at Budget."""
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "flight_agent")
    workflow.add_edge("planner", "hotel_agent")
    workflow.add_edge("flight_agent", "budget_agent")
    workflow.add_edge("hotel_agent", "budget_agent")
    workflow.add_edge("budget_agent", END)


def build_graph(
    tool_mode: Literal["node", "agent"] = "node",
    llm_client: InstrumentedOpenAIClient | StubLLMClient | ToolCallingClient | ToolCallingStub | None = None,
    model: str = "openai/gpt-4o-mini",
):
    """Build LangGraph workflow for travel planning.

    Topology matches Figure 1(a): Planner -> {Flight, Hotel} (parallel) -> Budget.
    tool_mode='node': Cal/Web as graph tool nodes (deterministic, native LangGraph).
    tool_mode='agent': tools via ToolCallingClient (probabilistic; NOT native bind_tools).
    """
    metrics = Metrics()
    if llm_client is None:
        llm_client = InstrumentedOpenAIClient(model=model, metrics=metrics)
    else:
        llm_client.metrics = metrics

    collector = _TraceCollector()
    request_holder: dict[str, TripRequest] = {"request": TripRequest()}

    # --- Node mode: faithful native graph tool nodes ---
    if tool_mode == "node":

        def planner_node(state: GraphState) -> dict:
            prompt = (
                f"You are the Planner agent. Coordinate a trip from {state['origin']} to "
                f"{state['destination']} on {state['dates']} with budget ${state['budget']}. "
                "Return JSON: {\"action\": \"delegate\"}"
            )
            before = _snapshot_metrics(metrics)
            llm_client.complete(prompt)
            update = {"status": "delegating"}
            collector.record(AGENT_PLANNER, prompt, metrics, before, _merged(state, update), [])
            return update

        def flight_agent_node(state: GraphState) -> dict:
            prompt = (
                f"You are the Flight agent. Select a flight to {state['destination']}. "
                "Return JSON: {\"action\": \"search_flight\"}"
            )
            before = _snapshot_metrics(metrics)
            llm_client.complete(prompt)
            collector.record(AGENT_FLIGHT, prompt, metrics, before, dict(state), [])
            return {}

        def web_flight_tool_node(state: GraphState) -> dict:
            req = request_holder["request"]
            args = web_args_for_request(req, kind="flight")
            tc = invoke_as_graph_node("Web", args, execute_web)
            options = tc.result.get("options", [])
            update: dict = {}
            if options:
                update["chosen_flight"] = options[0]
            collector.record(
                AGENT_FLIGHT, "[graph tool node: Web flight search]", metrics,
                _snapshot_metrics(metrics), _merged(state, update), [tc],
            )
            return update

        def hotel_agent_node(state: GraphState) -> dict:
            prompt = (
                f"You are the Hotel agent. Book accommodation in {state['destination']}. "
                "Return JSON: {\"action\": \"search_hotel\"}"
            )
            before = _snapshot_metrics(metrics)
            llm_client.complete(prompt)
            collector.record(AGENT_HOTEL, prompt, metrics, before, dict(state), [])
            return {}

        def web_hotel_tool_node(state: GraphState) -> dict:
            req = request_holder["request"]
            args = web_args_for_request(req, kind="hotel")
            tc = invoke_as_graph_node("Web", args, execute_web)
            options = tc.result.get("options", [])
            update = {}
            if options:
                update["chosen_hotel"] = options[0]
            collector.record(
                AGENT_HOTEL, "[graph tool node: Web hotel search]", metrics,
                _snapshot_metrics(metrics), _merged(state, update), [tc],
            )
            return update

        def budget_agent_node(state: GraphState) -> dict:
            prompt = (
                f"You are the Budget agent. Review costs against budget ${state['budget']}. "
                "Return JSON: {\"action\": \"check_budget\"}"
            )
            before = _snapshot_metrics(metrics)
            llm_client.complete(prompt)
            collector.record(AGENT_BUDGET, prompt, metrics, before, dict(state), [])
            return {}

        def cal_tool_node(state: GraphState) -> dict:
            args = cal_args_from_state(state)
            tc = invoke_as_graph_node("Cal", args, execute_cal)
            total = tc.result.get("total", 0)
            within = tc.result.get("within_budget", total <= state.get("budget", 0))
            update = {
                "running_cost": total,
                "status": resolve_budget_status(state, total, within_budget=within),
            }
            collector.record(
                AGENT_BUDGET, "[graph tool node: Cal budget check]", metrics,
                _snapshot_metrics(metrics), _merged(state, update), [tc],
            )
            return update

        workflow = StateGraph(GraphState)
        workflow.add_node("planner", planner_node)
        workflow.add_node("flight_agent", flight_agent_node)
        workflow.add_node("web_flight", web_flight_tool_node)
        workflow.add_node("hotel_agent", hotel_agent_node)
        workflow.add_node("web_hotel", web_hotel_tool_node)
        workflow.add_node("budget_agent", budget_agent_node)
        workflow.add_node("cal_tool", cal_tool_node)
        _wire_parallel_fig1(workflow)

    else:
        # --- Agent mode: probabilistic tool calls via ToolCallingClient ---
        # NOT LangGraph's native ChatOpenAI.bind_tools path — we hold the shared
        # instrumented client constant for token/call parity across paradigms.
        # This isolates invocation mode; it is not a faithful reproduction of how
        # a real LangGraph user binds agent tools. R3.Q3 leans on tool_mode='node'.
        if not isinstance(llm_client, (ToolCallingClient, ToolCallingStub)):
            llm_client = ToolCallingClient(model=getattr(llm_client, "model", model), metrics=metrics)
            llm_client.metrics = metrics

        def planner_node(state: GraphState) -> dict:
            prompt = (
                f"You are the Planner. Plan trip {state['origin']} -> {state['destination']}, "
                f"dates {state['dates']}, budget ${state['budget']}. JSON: {{\"action\":\"delegate\"}}"
            )
            before = _snapshot_metrics(metrics)
            llm_client.complete(prompt)
            update = {"status": "delegating"}
            collector.record(AGENT_PLANNER, prompt, metrics, before, _merged(state, update), [])
            return update

        def flight_agent_node(state: GraphState) -> dict:
            prompt = (
                f"You are the Flight agent for {state['destination']}. "
                "Optionally call Web to search flights, or respond with JSON only."
            )
            before = _snapshot_metrics(metrics)
            result = llm_client.complete_with_tools(prompt, [WEB_TOOL_SCHEMA])
            tools = list(result.tool_calls)
            update: dict = {}
            if tools and tools[0].result.get("options"):
                update["chosen_flight"] = tools[0].result["options"][0]
            collector.record(AGENT_FLIGHT, prompt, metrics, before, _merged(state, update), tools)
            return update

        def hotel_agent_node(state: GraphState) -> dict:
            prompt = (
                f"You are the Hotel agent for {state['destination']}. "
                "Optionally call Web to search hotels, or respond with JSON only."
            )
            before = _snapshot_metrics(metrics)
            result = llm_client.complete_with_tools(prompt, [WEB_TOOL_SCHEMA])
            tools = list(result.tool_calls)
            update = {}
            if tools and tools[0].result.get("options"):
                update["chosen_hotel"] = tools[0].result["options"][0]
            collector.record(AGENT_HOTEL, prompt, metrics, before, _merged(state, update), tools)
            return update

        def budget_agent_node(state: GraphState) -> dict:
            prompt = (
                f"You are the Budget agent. Budget is ${state['budget']}. "
                "Optionally call Cal to sum costs, or respond with JSON only."
            )
            before = _snapshot_metrics(metrics)
            result = llm_client.complete_with_tools(prompt, [CAL_TOOL_SCHEMA])
            tools = list(result.tool_calls)
            update = {}
            if tools and "total" in (tools[0].result or {}):
                total = tools[0].result["total"]
                within = tools[0].result.get("within_budget", total <= state.get("budget", 0))
                if plan_complete(state):
                    update["running_cost"] = total
                update["status"] = resolve_budget_status(state, total, within_budget=within)
            collector.record(AGENT_BUDGET, prompt, metrics, before, _merged(state, update), tools)
            return update

        workflow = StateGraph(GraphState)
        workflow.add_node("planner", planner_node)
        workflow.add_node("flight_agent", flight_agent_node)
        workflow.add_node("hotel_agent", hotel_agent_node)
        workflow.add_node("budget_agent", budget_agent_node)
        _wire_parallel_fig1_agent(workflow)

    compiled = workflow.compile()
    return compiled, collector, metrics, request_holder


def run_langgraph(
    request: TripRequest | None = None,
    tool_mode: Literal["node", "agent"] = "node",
    llm_client: InstrumentedOpenAIClient | StubLLMClient | ToolCallingClient | ToolCallingStub | None = None,
    model: str = "openai/gpt-4o-mini",
) -> TravelTrace:
    """Execute travel planning via LangGraph and return unified trace."""
    req = request or TripRequest()
    graph, collector, metrics, request_holder = build_graph(
        tool_mode=tool_mode, llm_client=llm_client, model=model,
    )
    request_holder["request"] = req
    final_state = dict(graph.invoke(initial_state(req)))
    if tool_mode == "agent" and not plan_complete(final_state):
        final_state["status"] = "incomplete"

    mode_label = "deterministic (graph-node)" if tool_mode == "node" else "probabilistic (agent-bound)"
    return TravelTrace(
        paradigm="LangGraph",
        tool_mode=mode_label,
        steps=collector.steps,
        metrics=metrics,
        final_state=final_state,
    )

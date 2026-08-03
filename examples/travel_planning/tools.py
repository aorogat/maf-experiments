"""Web (search stub) + Cal (calculator); deterministic vs agent-bound wrappers."""

from __future__ import annotations

from typing import Any, Callable

from frameworks.gabm_skeleton.tasks.travel import FLIGHTS, HOTELS

from examples.travel_planning.task import TripRequest
from examples.travel_planning.trace import ToolCall


def web_search(query: str, *, kind: str = "flight", destination: str = "NYC") -> dict[str, Any]:
    """Stub Web tool: returns fixed flight or hotel options for known destinations."""
    dest = destination
    if "hotel" in query.lower() or kind == "hotel":
        options = [HOTELS.get(dest, HOTELS["NYC"])]
        return {"options": options, "kind": "hotel"}
    options = [FLIGHTS.get(dest, FLIGHTS["NYC"])]
    return {"options": options, "kind": "flight"}


def cal_sum(*, flight_price: int = 0, hotel_price: int = 0, budget: int | None = None) -> dict[str, Any]:
    """Stub Cal tool: sums costs and optionally checks budget."""
    total = flight_price + hotel_price
    result: dict[str, Any] = {"total": total}
    if budget is not None:
        result["within_budget"] = total <= budget
    return result


def invoke_as_graph_node(
    tool_name: str,
    args: dict[str, Any],
    executor: Callable[[dict[str, Any]], Any],
) -> ToolCall:
    """Deterministic graph tool node — edge guarantees invocation."""
    result = executor(args)
    return ToolCall(
        tool=tool_name,
        args=args,
        result=result,
        invocation="deterministic",
        invoked_by="graph_node",
    )


def invoke_as_agent(
    tool_name: str,
    args: dict[str, Any],
    executor: Callable[[dict[str, Any]], Any],
) -> ToolCall:
    """Probabilistic agent-bound tool — LLM chose to call."""
    result = executor(args)
    return ToolCall(
        tool=tool_name,
        args=args,
        result=result,
        invocation="probabilistic",
        invoked_by="agent",
    )


def invoke_as_environment(
    tool_name: str,
    args: dict[str, Any],
    executor: Callable[[dict[str, Any]], Any],
) -> ToolCall:
    """Environment-executed tool — mediator runs it and folds result into state."""
    result = executor(args)
    return ToolCall(
        tool=tool_name,
        args=args,
        result=result,
        invocation="deterministic",
        invoked_by="environment",
    )


def web_args_for_request(request: TripRequest, *, kind: str = "flight") -> dict[str, Any]:
    return {"query": f"{kind} {request.destination}", "destination": request.destination, "kind": kind}


def cal_args_from_state(state: dict[str, Any]) -> dict[str, Any]:
    flight = state.get("chosen_flight") or {}
    hotel = state.get("chosen_hotel") or {}
    return {
        "flight_price": flight.get("price", 0),
        "hotel_price": hotel.get("price", 0),
        "budget": state.get("budget"),
    }


def execute_web(args: dict[str, Any]) -> dict[str, Any]:
    return web_search(
        args.get("query", ""),
        kind=args.get("kind", "flight"),
        destination=args.get("destination", "NYC"),
    )


def execute_cal(args: dict[str, Any]) -> dict[str, Any]:
    return cal_sum(
        flight_price=args.get("flight_price", 0),
        hotel_price=args.get("hotel_price", 0),
        budget=args.get("budget"),
    )


# OpenAI function-calling schemas for agent-bound mode
WEB_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "Web",
        "description": "Search for flights or hotels.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "kind": {"type": "string", "enum": ["flight", "hotel"]},
                "destination": {"type": "string"},
            },
            "required": ["query", "kind", "destination"],
        },
    },
}

CAL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "Cal",
        "description": "Calculate total trip cost and check budget.",
        "parameters": {
            "type": "object",
            "properties": {
                "flight_price": {"type": "integer"},
                "hotel_price": {"type": "integer"},
                "budget": {"type": "integer"},
            },
            "required": ["flight_price", "hotel_price"],
        },
    },
}

TOOL_SCHEMAS = [WEB_TOOL_SCHEMA, CAL_TOOL_SCHEMA]

"""Trip request, budget constraint, and shared schema (paradigm-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from frameworks.gabm_skeleton.tasks.travel import FLIGHTS, HOTELS

# Fixed Fig-1 test request (reproducible stubs key off destination)
DEFAULT_ORIGIN = "BOS"
DEFAULT_DESTINATION = "NYC"
DEFAULT_DATES = "2026-08-01/2026-08-05"
DEFAULT_BUDGET = 500

AGENT_PLANNER = "Planner"
AGENT_FLIGHT = "Flight"
AGENT_HOTEL = "Hotel"
AGENT_BUDGET = "Budget"

ALL_AGENTS = [AGENT_PLANNER, AGENT_FLIGHT, AGENT_HOTEL, AGENT_BUDGET]


@dataclass(frozen=True)
class TripRequest:
    """Fixed trip-planning input from Figure 1."""

    origin: str = DEFAULT_ORIGIN
    destination: str = DEFAULT_DESTINATION
    dates: str = DEFAULT_DATES
    budget: int = DEFAULT_BUDGET

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin,
            "destination": self.destination,
            "dates": self.dates,
            "budget": self.budget,
        }


def initial_state(request: TripRequest | None = None) -> dict[str, Any]:
    """Shared trip state schema used by all three paradigms."""
    req = request or TripRequest()
    return {
        "origin": req.origin,
        "destination": req.destination,
        "dates": req.dates,
        "budget": req.budget,
        "chosen_flight": {},
        "chosen_hotel": {},
        "running_cost": 0,
        "status": "planning",
    }


STATE_SCHEMA: dict[str, type] = {
    "origin": str,
    "destination": str,
    "dates": str,
    "budget": int,
    "chosen_flight": dict,
    "chosen_hotel": dict,
    "running_cost": int,
    "status": str,
}


def plan_complete(state: dict[str, Any]) -> bool:
    """True when both flight and hotel have been selected."""
    return bool(state.get("chosen_flight")) and bool(state.get("chosen_hotel"))


def budget_satisfied(state: dict[str, Any]) -> bool:
    """True when flight + hotel are chosen and total cost is within budget."""
    has_flight = bool(state.get("chosen_flight"))
    has_hotel = bool(state.get("chosen_hotel"))
    within_budget = state.get("running_cost", 0) <= state.get("budget", 0)
    approved = state.get("status") in ("approved", "complete")
    return has_flight and has_hotel and within_budget and approved


def resolve_budget_status(
    state: dict[str, Any],
    total: int,
    *,
    within_budget: bool | None = None,
) -> str:
    """Derive status after Cal; never 'approved' without flight+hotel (R3.Q3)."""
    if not plan_complete(state):
        return "incomplete"
    if within_budget is None:
        within_budget = total <= state.get("budget", 0)
    return "approved" if within_budget else "over_budget"


def format_itinerary(state: dict[str, Any]) -> str:
    """Human-readable proposed itinerary (output artifact)."""
    flight = state.get("chosen_flight") or {}
    hotel = state.get("chosen_hotel") or {}
    return (
        f"Trip {state.get('origin', '?')} -> {state.get('destination', '?')} "
        f"({state.get('dates', '?')}): "
        f"flight={flight.get('id', 'none')} (${flight.get('price', 0)}), "
        f"hotel={hotel.get('id', 'none')} (${hotel.get('price', 0)}), "
        f"total=${state.get('running_cost', 0)}, "
        f"budget=${state.get('budget', 0)}, status={state.get('status', '?')}"
    )


def expected_flight(destination: str) -> dict[str, Any]:
    """Deterministic stub lookup for tests."""
    return dict(FLIGHTS.get(destination, FLIGHTS["NYC"]))


def expected_hotel(destination: str) -> dict[str, Any]:
    """Deterministic stub lookup for tests."""
    return dict(HOTELS.get(destination, HOTELS["NYC"]))


def expected_total(destination: str) -> int:
    """Expected running cost for the fixed test request."""
    return expected_flight(destination)["price"] + expected_hotel(destination)["price"]

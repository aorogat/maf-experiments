"""Travel planning task: flight, hotel, and budget agents coordinating via GM."""

from __future__ import annotations

import random
from typing import Any

from frameworks.gabm_skeleton.agent import SkeletonAgent
from frameworks.gabm_skeleton.environment import Environment

# Deterministic lookup tables
FLIGHTS = {
    "NYC": {"id": "AA100", "price": 300, "airline": "AA"},
    "LAX": {"id": "UA200", "price": 250, "airline": "UA"},
    "CHI": {"id": "DL300", "price": 200, "airline": "DL"},
}

HOTELS = {
    "NYC": {"id": "H1", "price": 150, "name": "City Inn"},
    "LAX": {"id": "H2", "price": 120, "name": "Beach Lodge"},
    "CHI": {"id": "H3", "price": 100, "name": "Windy Stay"},
}

UPGRADE_OPTIONS = ["economy_plus", "business", "first"]


def search_flights(args: dict) -> dict:
    dest = args.get("destination", "NYC")
    flight = FLIGHTS.get(dest, FLIGHTS["NYC"])
    return {"tool": "search_flights", "args": args, "result": flight, "deterministic": True}


def search_hotels(args: dict) -> dict:
    dest = args.get("destination", "NYC")
    hotel = HOTELS.get(dest, HOTELS["NYC"])
    return {"tool": "search_hotels", "args": args, "result": hotel, "deterministic": True}


def calculate_total(args: dict) -> dict:
    flight_price = args.get("flight_price", 0)
    hotel_price = args.get("hotel_price", 0)
    total = flight_price + hotel_price
    return {
        "tool": "calculate_total",
        "args": args,
        "result": {"total": total},
        "deterministic": True,
    }


def suggest_upgrade(args: dict, rng: random.Random) -> dict:
    choice = rng.choice(UPGRADE_OPTIONS)
    surcharge = rng.randint(50, 200)
    return {
        "tool": "suggest_upgrade",
        "args": args,
        "result": {"upgrade": choice, "surcharge": surcharge},
        "deterministic": False,
    }


def _travel_transition(state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    # Used only for direct Environment.apply in unit tests
    new_state = dict(state)
    for key in ("selected_flight", "selected_hotel", "total_cost", "status"):
        if key in action:
            new_state[key] = action[key]
    return new_state


class TravelTask:
    """Three agents coordinate on a trip plan within budget."""

    SCHEMA = {
        "destination": str,
        "dates": str,
        "budget": int,
        "selected_flight": dict,
        "selected_hotel": dict,
        "total_cost": int,
        "status": str,
    }

    AGENT_FLIGHT = "flight"
    AGENT_HOTEL = "hotel"
    AGENT_BUDGET = "budget"

    def __init__(
        self,
        destination: str = "NYC",
        dates: str = "2026-08-01/2026-08-05",
        budget: int = 500,
        max_rounds: int = 5,
        seed: int = 42,
    ):
        self.destination = destination
        self.dates = dates
        self.budget = budget
        self.max_rounds = max_rounds
        self._rng = random.Random(seed)

    @property
    def agent_ids(self) -> list[str]:
        return [self.AGENT_FLIGHT, self.AGENT_HOTEL, self.AGENT_BUDGET]

    def initial_state(self) -> dict[str, Any]:
        return {
            "destination": self.destination,
            "dates": self.dates,
            "budget": self.budget,
            "selected_flight": {},
            "selected_hotel": {},
            "total_cost": 0,
            "status": "planning",
        }

    def make_environment(self) -> Environment:
        return Environment(self.SCHEMA, self.initial_state(), _travel_transition)

    def visible_fields(self, agent_id: str) -> set[str]:
        if agent_id == self.AGENT_FLIGHT:
            return {"destination", "dates", "budget", "total_cost"}
        if agent_id == self.AGENT_HOTEL:
            return {"destination", "dates", "budget", "selected_flight"}
        if agent_id == self.AGENT_BUDGET:
            return {"selected_flight", "selected_hotel", "budget", "total_cost", "status"}
        return set()

    def process_action(
        self,
        agent_id: str,
        action: dict[str, Any],
        state: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict]]:
        new_state = dict(state)
        tools: list[dict] = []

        if agent_id == self.AGENT_FLIGHT:
            inv = search_flights({"destination": state["destination"]})
            tools.append(inv)
            new_state["selected_flight"] = inv["result"]

        elif agent_id == self.AGENT_HOTEL:
            inv = search_hotels({"destination": state["destination"]})
            tools.append(inv)
            new_state["selected_hotel"] = inv["result"]

        elif agent_id == self.AGENT_BUDGET:
            flight = state.get("selected_flight") or {}
            hotel = state.get("selected_hotel") or {}
            calc = calculate_total({
                "flight_price": flight.get("price", 0),
                "hotel_price": hotel.get("price", 0),
            })
            tools.append(calc)
            new_state["total_cost"] = calc["result"]["total"]

            if new_state["total_cost"] > state["budget"]:
                upgrade = suggest_upgrade({"destination": state["destination"]}, self._rng)
                tools.append(upgrade)
                new_state["status"] = "over_budget"
            else:
                new_state["status"] = "approved"

        else:
            raise ValueError(f"Unknown agent: {agent_id}")

        return new_state, tools

    def is_complete(self, state: dict[str, Any]) -> bool:
        has_flight = bool(state.get("selected_flight"))
        has_hotel = bool(state.get("selected_hotel"))
        within_budget = state.get("total_cost", 0) <= state.get("budget", 0)
        return has_flight and has_hotel and within_budget and state.get("status") == "approved"

    def extract_answer(self, state: dict[str, Any]) -> str:
        return (
            f"Trip to {state['destination']}: "
            f"flight={state['selected_flight'].get('id', 'none')}, "
            f"hotel={state['selected_hotel'].get('id', 'none')}, "
            f"total=${state['total_cost']}, status={state['status']}"
        )

    def make_agents(self, llm_client) -> list[SkeletonAgent]:
        return [
            SkeletonAgent(
                agent_id=self.AGENT_FLIGHT,
                system_prompt='Select a flight. Return JSON: {"action": "select_flight"}',
                llm_client=llm_client,
                fallback_action={"action": "select_flight"},
            ),
            SkeletonAgent(
                agent_id=self.AGENT_HOTEL,
                system_prompt='Select a hotel. Return JSON: {"action": "select_hotel"}',
                llm_client=llm_client,
                fallback_action={"action": "select_hotel"},
            ),
            SkeletonAgent(
                agent_id=self.AGENT_BUDGET,
                system_prompt='Review budget. Return JSON: {"action": "review_budget"}',
                llm_client=llm_client,
                fallback_action={"action": "review_budget"},
            ),
        ]

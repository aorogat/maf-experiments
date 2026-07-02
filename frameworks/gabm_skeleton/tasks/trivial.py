"""Trivial single-agent workload: What is 2+2?"""

from __future__ import annotations

from typing import Any

from frameworks.gabm_skeleton.agent import SkeletonAgent
from frameworks.gabm_skeleton.environment import Environment


def _trivial_transition(state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    if "answer" not in action:
        raise ValueError("Action must contain 'answer'")
    if not isinstance(action["answer"], str):
        raise TypeError("answer must be a string")
    new_state = dict(state)
    new_state["answer"] = action["answer"]
    return new_state


class TrivialTask:
    """One agent, one round: answer a math question."""

    SCHEMA = {"query": str, "answer": str}
    AGENT_ID = "solver"

    def __init__(self, question: str = "What is 2+2?"):
        self.question = question
        self.max_rounds = 1
        self.llm_max_tokens = 16  # parity with direct_llm overhead baseline

    @property
    def agent_ids(self) -> list[str]:
        return [self.AGENT_ID]

    def initial_state(self) -> dict[str, Any]:
        return {"query": self.question, "answer": ""}

    def make_environment(self) -> Environment:
        return Environment(self.SCHEMA, self.initial_state(), _trivial_transition)

    def visible_fields(self, agent_id: str) -> set[str]:
        if agent_id == self.AGENT_ID:
            return {"query"}
        return set()

    def process_action(
        self,
        agent_id: str,
        action: dict[str, Any],
        state: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict]]:
        if agent_id != self.AGENT_ID:
            raise ValueError(f"Unknown agent: {agent_id}")
        if "answer" not in action:
            raise ValueError("Action must contain 'answer'")
        new_state = dict(state)
        new_state["answer"] = str(action["answer"])
        return new_state, []

    def is_complete(self, state: dict[str, Any]) -> bool:
        return bool(state.get("answer"))

    def extract_answer(self, state: dict[str, Any]) -> str:
        return state.get("answer", "")

    def make_agents(self, llm_client) -> list[SkeletonAgent]:
        return [
            SkeletonAgent(
                agent_id=self.AGENT_ID,
                system_prompt="Answer the math query. Return JSON: {\"answer\": \"<number>\"}",
                llm_client=llm_client,
                fallback_action={"answer": "4"},
                max_tokens=self.llm_max_tokens,
            )
        ]

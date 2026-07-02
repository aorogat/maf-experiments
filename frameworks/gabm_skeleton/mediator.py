"""Game Master: observation filtering and action application."""

from __future__ import annotations

from typing import Any, Protocol

from frameworks.gabm_skeleton.environment import Environment, InvalidActionError


class TaskProtocol(Protocol):
    def visible_fields(self, agent_id: str) -> set[str]: ...
    def process_action(
        self,
        agent_id: str,
        action: dict[str, Any],
        state: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict]]: ...


class GameMaster:
    """Mediator between agents and the shared environment."""

    def __init__(self, task: TaskProtocol, environment: Environment):
        self.task = task
        self.environment = environment

    def observe(self, agent_id: str) -> dict[str, Any]:
        """Return filtered projection of E for the given agent."""
        state = self.environment.snapshot()
        visible = self.task.visible_fields(agent_id)
        return {k: v for k, v in state.items() if k in visible}

    def apply_round(self, actions: dict[str, dict[str, Any]]) -> list[dict]:
        """Apply one action per agent; return all tool invocations for the round."""
        all_tools: list[dict] = []
        state = self.environment.snapshot()

        for agent_id, action in actions.items():
            new_state, tool_invocations = self.task.process_action(agent_id, action, state)
            all_tools.extend(tool_invocations)
            state = new_state

        # Commit merged state atomically
        self.environment.replace(state)
        return all_tools

    def apply_single(self, agent_id: str, action: dict[str, Any]) -> list[dict]:
        """Apply a single agent action (used when testing mediator in isolation)."""
        state = self.environment.snapshot()
        new_state, tool_invocations = self.task.process_action(agent_id, action, state)
        self.environment.replace(new_state)
        return tool_invocations

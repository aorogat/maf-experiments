"""Shared environment state E with grounded variables and transitions."""

from __future__ import annotations

import copy
from typing import Any, Callable


class InvalidActionError(ValueError):
    """Raised when an action cannot be applied to the environment."""


class Environment:
    """Shared state object holding grounded variables."""

    def __init__(
        self,
        schema: dict[str, type],
        initial: dict[str, Any],
        transition: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    ):
        self._schema = schema
        self._transition = transition
        self._state = self._validate(initial)

    def _validate(self, state: dict[str, Any]) -> dict[str, Any]:
        for key in state:
            if key not in self._schema:
                raise KeyError(f"Unknown key: {key}")
        for key, expected_type in self._schema.items():
            if key in state and not isinstance(state[key], expected_type):
                raise TypeError(
                    f"Key {key!r} expected {expected_type.__name__}, "
                    f"got {type(state[key]).__name__}"
                )
        return dict(state)

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self._state)

    def replace(self, new_state: dict[str, Any]) -> None:
        """Replace entire state after validation (used by GM after a round)."""
        self._state = self._validate(new_state)

    def apply(self, action: dict[str, Any]) -> None:
        """Apply action via transition function; reject invalid actions without mutation."""
        current = copy.deepcopy(self._state)
        try:
            new_state = self._transition(current, action)
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidActionError(str(exc)) from exc
        self._validate(new_state)
        self._state = new_state

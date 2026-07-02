"""Skeleton agent: observe -> one LLM call -> action."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol


class LLMClient(Protocol):
    def complete(self, prompt: str, *, max_tokens: int = 256, temperature: float = 0) -> str: ...


class SkeletonAgent:
    """Agent that produces an action via exactly one LLM call per act()."""

    def __init__(
        self,
        agent_id: str,
        system_prompt: str,
        llm_client: LLMClient,
        fallback_action: dict[str, Any] | None = None,
        max_tokens: int = 256,
    ):
        self.agent_id = agent_id
        self.system_prompt = system_prompt
        self.llm_client = llm_client
        self.fallback_action = fallback_action or {}
        self.max_tokens = max_tokens
        self.last_prompt: str = ""

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        obs_json = json.dumps(observation, sort_keys=True)
        self.last_prompt = (
            f"{self.system_prompt}\n"
            f"Observation: {obs_json}\n"
            f"Respond with a single JSON object for your action."
        )
        raw = self.llm_client.complete(self.last_prompt, max_tokens=self.max_tokens)
        return self._parse_action(raw)

    def _parse_action(self, raw: str) -> dict[str, Any]:
        text = raw.strip()
        # Try direct JSON parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        # Try extracting JSON object from response
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return dict(self.fallback_action)

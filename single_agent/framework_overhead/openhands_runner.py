"""
OpenHands SDK baseline runner for framework overhead experiments.

Uses the OpenHands Software Agent SDK with tools disabled so the harness
measures orchestration + system-prompt overhead on a trivial QA task, comparable
to other frameworks in this suite.

Docs: https://docs.openhands.dev/sdk/getting-started

Usage:
    python -m single_agent.framework_overhead.openhands_runner
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time

from dotenv import load_dotenv
from pydantic import SecretStr

# Suppress SDK banner before import side effects.
os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

from openhands.sdk import Agent, Conversation, LLM
from openhands.sdk.conversation.title_utils import extract_message_text
from openhands.sdk.event import MessageEvent

QUESTION = "What is 2+2?"

logging.getLogger("openhands").setLevel(logging.WARNING)


class OpenHandsRunner:
    """Minimal OpenHands SDK runner (no tools, fresh conversation per trial)."""

    def __init__(self, model: str = "openai/gpt-4o-mini"):
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self.model = model
        self._api_key = api_key

    def run(self, question: str = QUESTION) -> tuple[str, float]:
        """Run one OpenHands conversation and return (answer, latency_ms)."""
        workspace = tempfile.mkdtemp(prefix="openhands_overhead_")
        start = time.perf_counter()

        try:
            llm = LLM(
                model=self.model,
                api_key=SecretStr(self._api_key),
                log_completions=False,
            )
            agent = Agent(
                llm=llm,
                tools=[],
                include_default_tools=[],
                system_prompt_kwargs={"cli_mode": True},
            )
            conversation = Conversation(
                agent=agent,
                workspace=workspace,
                max_iteration_per_run=1,
                visualizer=None,
                stuck_detection=False,
            )
            conversation.send_message(question)
            conversation.run()

            answer = ""
            for event in conversation.state.events:
                if (
                    isinstance(event, MessageEvent)
                    and event.source == "agent"
                    and event.llm_message
                ):
                    text = extract_message_text(event)
                    if text:
                        answer = text
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

        latency_ms = (time.perf_counter() - start) * 1000.0
        return answer.strip(), latency_ms


if __name__ == "__main__":
    runner = OpenHandsRunner(model="openai/gpt-4o-mini")

    print("=== OpenHands SDK (No-Tools Overhead Test) ===")
    for i in range(3):
        resp, latency = runner.run(QUESTION)
        print(f"Run {i+1}: Q={QUESTION} | A={resp} | {latency:.2f} ms")

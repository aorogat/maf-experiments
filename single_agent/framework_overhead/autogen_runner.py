"""
AutoGen baseline runner for framework overhead experiments.

Uses autogen-agentchat 0.7.5 (AssistantAgent + OpenAIChatCompletionClient).
No memory, no tools, no multi-agent coordination — single trivial query only.

Usage:
    python -m single_agent.framework_overhead.autogen_runner \
        --model gpt-4o-mini
"""

import asyncio
import time
import os
import argparse

from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

QUESTION = "What is 2+2?"


class AutoGenRunner:
    """Minimal AutoGen runner (no memory, no tools, single agent)."""

    def __init__(self, model: str = "gpt-4o-mini"):
        load_dotenv()

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set")

        if model.startswith("openai/"):
            model = model.split("openai/", 1)[1]
        self.model = model

        self._model_client = OpenAIChatCompletionClient(
            model=self.model,
            temperature=0,
            max_tokens=16,
        )

    def run(self, question: str = QUESTION) -> tuple[str, float]:
        """Run a single AutoGen call and measure latency."""
        start = time.perf_counter()
        result = asyncio.run(self._run_once(question))
        end = time.perf_counter()

        answer = self._extract_answer(result)
        return answer.strip(), (end - start) * 1000

    async def _run_once(self, question: str):
        agent = AssistantAgent(
            name="OverheadAgent",
            model_client=self._model_client,
            system_message="Answer the question briefly and correctly.",
        )
        return await agent.run(task=question)

    @staticmethod
    def _extract_answer(result) -> str:
        for message in reversed(result.messages):
            if isinstance(message, TextMessage) and message.source == "assistant":
                return message.content or ""
        if result.messages:
            content = getattr(result.messages[-1], "content", "")
            return str(content) if content is not None else ""
        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name (e.g., gpt-4o-mini)",
    )
    args = parser.parse_args()

    runner = AutoGenRunner(model=args.model)

    print("=== AutoGen (No-Memory Overhead Test) ===")
    for i in range(3):
        resp, latency = runner.run(QUESTION)
        print(f"Run {i+1}: Q={QUESTION} | A={resp} | ⏱️ {latency:.2f} ms")

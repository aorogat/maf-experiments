"""
Direct LLM baseline runner (Ollama).
Usage:
    python -m single_agent.framework_overhead.direct_llm
"""

import time
from llms.local_llm import LocalOllamaLLM

QUESTION = "What is 2+2?"


class DirectLLMRunner:
    """Reusable Direct LLM runner for overhead experiments."""

    def __init__(self, model="deepseek-llm:7b"):
        # Initialize local Ollama model once
        self.llm = LocalOllamaLLM(model)

    def run(self, question: str = QUESTION):
        """Run a single LLM call with timing."""
        start = time.perf_counter()
        response = self.llm.generate(question)
        end = time.perf_counter()
        return str(response).strip(), (end - start) * 1000  # return ms latency


if __name__ == "__main__":
    runner = DirectLLMRunner()
    print("=== Direct LLM Test ===")
    for i in range(3):
        resp, latency = runner.run(QUESTION)
        print(f"Run {i+1}: Q={QUESTION} | A={resp} | ⏱️ {latency:.2f} ms")

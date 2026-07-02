"""
GABM Skeleton baseline runner for framework overhead experiments.

Minimal environment-mediated GABM: one agent, one round, one LLM call on the
trivial "What is 2+2?" workload.

Usage:
    python -m single_agent.framework_overhead.gabm_skeleton_runner --model openai/gpt-4o-mini
"""

import argparse

from dotenv import load_dotenv

from frameworks.gabm_skeleton.runner import GABMSkeletonRunner as _CoreRunner

QUESTION = "What is 2+2?"


class GABMSkeletonRunner:
    """Overhead harness adapter for the minimal GABM skeleton framework."""

    def __init__(self, model: str = "openai/gpt-4o-mini"):
        load_dotenv()
        self._runner = _CoreRunner(model=model)

    def run(self, question: str = QUESTION) -> tuple[str, float]:
        return self._runner.run(question)

    @property
    def last_result(self):
        """Last RunResult (metrics, trace) from the most recent run()."""
        return self._runner.last_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help="Model to use (e.g. openai/gpt-4o-mini)",
    )
    args = parser.parse_args()

    runner = GABMSkeletonRunner(model=args.model)
    print("=== GABM Skeleton Overhead Test ===")
    for i in range(3):
        resp, latency = runner.run(QUESTION)
        m = runner.last_result.metrics if runner.last_result else None
        tokens = f" tokens={m.input_tokens}/{m.output_tokens}" if m else ""
        print(f"Run {i+1}: Q={QUESTION} | A={resp} | ⏱️ {latency:.2f} ms{tokens}")

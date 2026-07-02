"""Integration tests for trivial task runner (stub LLM, no network)."""

import unittest

from frameworks.gabm_skeleton.llm_client import StubLLMClient
from frameworks.gabm_skeleton.metrics import Metrics
from frameworks.gabm_skeleton.runner import GABMSkeletonRunner
from frameworks.gabm_skeleton.tasks.trivial import TrivialTask


class TestRunnerTrivial(unittest.TestCase):
    def test_trivial_yields_stub_answer(self):
        metrics = Metrics()
        stub = StubLLMClient(
            responses=['{"answer": "4"}'],
            metrics=metrics,
        )
        runner = GABMSkeletonRunner(llm_client=stub)
        result = runner.run(TrivialTask())

        self.assertEqual(result.final_state["answer"], "4")
        self.assertEqual(result.answer, "4")

    def test_metrics_llm_calls_equals_one(self):
        metrics = Metrics()
        stub = StubLLMClient(responses=['{"answer": "4"}'], metrics=metrics)
        runner = GABMSkeletonRunner(llm_client=stub)
        result = runner.run(TrivialTask())

        self.assertEqual(result.metrics.llm_calls, 1)
        if result.metrics.llm_calls > 1:
            self.fail(f"FAIL: llm_calls={result.metrics.llm_calls}, expected 1")

    def test_all_metric_fields_populated(self):
        metrics = Metrics()
        stub = StubLLMClient(responses=['{"answer": "4"}'], metrics=metrics)
        runner = GABMSkeletonRunner(llm_client=stub)
        result = runner.run(TrivialTask())
        m = result.metrics

        self.assertGreaterEqual(m.llm_calls, 0)
        self.assertGreaterEqual(m.input_tokens, 0)
        self.assertGreaterEqual(m.output_tokens, 0)
        self.assertGreaterEqual(m.llm_time_s, 0)
        self.assertGreater(m.total_time_s, 0)
        self.assertGreaterEqual(m.framework_residual_s, 0)
        self.assertGreater(m.output_chars, 0)

    def test_guardrail_n_times_r(self):
        metrics = Metrics()
        stub = StubLLMClient(responses=['{"answer": "4"}'], metrics=metrics)
        runner = GABMSkeletonRunner(llm_client=stub)
        result = runner.run(TrivialTask())

        num_agents = 1
        rounds = len(result.trace)
        self.assertEqual(result.metrics.llm_calls, num_agents * rounds)

    def test_dual_api_string_returns_tuple(self):
        metrics = Metrics()
        stub = StubLLMClient(responses=['{"answer": "4"}'], metrics=metrics)
        runner = GABMSkeletonRunner(llm_client=stub)
        answer, latency_ms = runner.run("What is 2+2?")

        self.assertEqual(answer, "4")
        self.assertIsInstance(latency_ms, float)
        self.assertIsNotNone(runner.last_result)
        # Returned latency must match total_time_s (harness uses runner-reported value)
        self.assertAlmostEqual(
            latency_ms,
            runner.last_result.metrics.total_time_s * 1000.0,
            places=3,
        )

    def test_metrics_reset_across_repeated_runs(self):
        """50-trial harness reuses one runner; each run must have independent metrics."""
        stub = StubLLMClient(responses=['{"answer": "4"}'] * 10)
        runner = GABMSkeletonRunner(llm_client=stub)

        for i in range(3):
            result = runner.run(TrivialTask())
            self.assertEqual(result.metrics.llm_calls, 1, f"run {i+1} should have exactly 1 LLM call")
            self.assertEqual(result.metrics.input_tokens, 10)
            self.assertEqual(len(result.trace), 1)


if __name__ == "__main__":
    unittest.main()

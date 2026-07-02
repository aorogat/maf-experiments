"""Integration tests for travel task runner (stub LLM, no network)."""

import unittest

from frameworks.gabm_skeleton.llm_client import StubLLMClient
from frameworks.gabm_skeleton.runner import GABMSkeletonRunner
from frameworks.gabm_skeleton.tasks.travel import TravelTask

TRAVEL_RESPONSES = [
    '{"action": "select_flight"}',
    '{"action": "select_hotel"}',
    '{"action": "review_budget"}',
]


class TestTravel(unittest.TestCase):
    def _make_runner(self, max_rounds: int, budget: int = 500, seed: int = 42):
        responses = TRAVEL_RESPONSES * max_rounds
        stub = StubLLMClient(responses=responses)
        runner = GABMSkeletonRunner(llm_client=stub)
        task = TravelTask(budget=budget, max_rounds=max_rounds, seed=seed)
        return runner, task

    def test_three_agents_per_round_llm_calls(self):
        runner, task = self._make_runner(max_rounds=2, budget=50)
        result = runner.run(task)

        rounds = len(result.trace)
        self.assertEqual(result.metrics.llm_calls, 3 * rounds)

    def test_trace_has_all_five_fields_per_round(self):
        runner, task = self._make_runner(max_rounds=2, budget=50)
        result = runner.run(task)

        self.assertEqual(len(result.trace), 2)
        for rt in result.trace:
            self.assertIsInstance(rt.prompts_issued, list)
            self.assertIsInstance(rt.llm_calls, int)
            self.assertIsInstance(rt.state_snapshot, dict)
            self.assertEqual(rt.communication, "gm_mediated")
            self.assertIsInstance(rt.tool_invocations, list)
            self.assertEqual(len(rt.prompts_issued), 3)

    def test_tool_invocation_has_deterministic_flag(self):
        runner, task = self._make_runner(max_rounds=1, budget=500)
        result = runner.run(task)

        all_tools = [t for rt in result.trace for t in rt.tool_invocations]
        self.assertGreaterEqual(len(all_tools), 1)
        for tool in all_tools:
            self.assertIn("tool", tool)
            self.assertIn("deterministic", tool)
        deterministic_tools = [t for t in all_tools if t["deterministic"] is True]
        self.assertGreaterEqual(len(deterministic_tools), 1)

    def test_stops_at_max_rounds_when_budget_not_satisfied(self):
        runner, task = self._make_runner(max_rounds=2, budget=50)
        result = runner.run(task)

        self.assertEqual(len(result.trace), 2)
        self.assertFalse(task.is_complete(result.final_state))
        self.assertEqual(result.metrics.llm_calls, 3 * 2)

    def test_stops_when_budget_predicate_satisfied(self):
        runner, task = self._make_runner(max_rounds=5, budget=500)
        result = runner.run(task)

        self.assertTrue(task.is_complete(result.final_state))
        self.assertEqual(len(result.trace), 1)
        self.assertEqual(result.metrics.llm_calls, 3)

    def test_guardrail_n_times_r(self):
        runner, task = self._make_runner(max_rounds=3, budget=50)
        result = runner.run(task)

        rounds = len(result.trace)
        self.assertEqual(result.metrics.llm_calls, 3 * rounds)

    def test_trace_table_returns_rows(self):
        runner, task = self._make_runner(max_rounds=1, budget=500)
        result = runner.run(task)

        rows = result.trace_table()
        self.assertEqual(len(rows), 1)
        self.assertIn("round", rows[0])
        self.assertIn("llm_calls", rows[0])


if __name__ == "__main__":
    unittest.main()

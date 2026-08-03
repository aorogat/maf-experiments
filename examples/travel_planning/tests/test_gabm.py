"""Tests for GABM skeleton travel-planning implementation."""

import unittest

from frameworks.gabm_skeleton.llm_client import StubLLMClient

from examples.travel_planning.gabm_impl import run_gabm
from examples.travel_planning.task import TripRequest, budget_satisfied


GABM_RESPONSES = [
    '{"action": "delegate"}',
    '{"action": "select_flight"}',
    '{"action": "select_hotel"}',
    '{"action": "check_budget"}',
]


class TestGABM(unittest.TestCase):
    def test_runs_on_skeleton(self):
        stub = StubLLMClient(responses=GABM_RESPONSES)
        trace = run_gabm(TripRequest(), llm_client=stub)
        self.assertEqual(len(trace.steps), 4)

    def test_communication_gm_mediated(self):
        stub = StubLLMClient(responses=GABM_RESPONSES)
        trace = run_gabm(TripRequest(), llm_client=stub)
        for step in trace.steps:
            self.assertEqual(step.communication, "gm_mediated")

    def test_tools_invoked_by_environment(self):
        stub = StubLLMClient(responses=GABM_RESPONSES)
        trace = run_gabm(TripRequest(), llm_client=stub)
        all_tools = [t for s in trace.steps for t in s.tool_invocations]
        self.assertGreaterEqual(len(all_tools), 2)
        for t in all_tools:
            self.assertEqual(t.invoked_by, "environment")

    def test_guardrail_n_times_r(self):
        stub = StubLLMClient(responses=GABM_RESPONSES)
        trace = run_gabm(TripRequest(), llm_client=stub)
        self.assertEqual(trace.metrics.llm_calls, 4)

    def test_valid_itinerary_budget_satisfied(self):
        """Final state must have flight + hotel and satisfy budget — not just 'ran'."""
        stub = StubLLMClient(responses=GABM_RESPONSES)
        trace = run_gabm(TripRequest(), llm_client=stub)
        state = trace.final_state
        self.assertTrue(state.get("chosen_flight"))
        self.assertTrue(state.get("chosen_hotel"))
        self.assertEqual(state.get("status"), "approved")
        self.assertTrue(budget_satisfied(state))
        self.assertLessEqual(state["running_cost"], state["budget"])


if __name__ == "__main__":
    unittest.main()

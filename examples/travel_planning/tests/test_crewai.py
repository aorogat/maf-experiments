"""Tests for CrewAI travel-planning implementation."""

import unittest

from frameworks.gabm_skeleton.llm_client import StubLLMClient

from examples.travel_planning.crewai_impl import run_crewai
from examples.travel_planning.task import TripRequest


class TestCrewAI(unittest.TestCase):
    def test_crew_runs_with_role_handoff(self):
        stub = StubLLMClient(
            responses=['{"plan":"delegate"}'] * 8,
            input_tokens=10,
            output_tokens=5,
        )
        trace = run_crewai(TripRequest(), llm_client=stub)
        self.assertGreaterEqual(len(trace.steps), 2)
        agents = {s.agent for s in trace.steps}
        self.assertIn("Planner", agents)
        self.assertTrue(agents & {"Flight", "Hotel", "Budget"})

    def test_communication_role_handoff(self):
        stub = StubLLMClient(responses=['{}'] * 8)
        trace = run_crewai(TripRequest(), llm_client=stub)
        for step in trace.steps:
            self.assertEqual(step.communication, "role_handoff")

    def test_five_fields_populated(self):
        stub = StubLLMClient(responses=['{}'] * 8)
        trace = run_crewai(TripRequest(), llm_client=stub)
        for step in trace.steps:
            self.assertTrue(step.agent)
            self.assertIsInstance(step.prompt_issued, str)
            self.assertIsInstance(step.llm_calls, int)
            self.assertIsInstance(step.state_snapshot, dict)

    def test_budget_task_description_uses_current_state(self):
        """Budget prompt must reflect flight/hotel prices from shared state."""
        from examples.travel_planning.crewai_impl import _task_description
        from examples.travel_planning.task import AGENT_BUDGET, initial_state

        state = initial_state()
        state["chosen_flight"] = {"id": "AA100", "price": 300}
        state["chosen_hotel"] = {"id": "H1", "price": 150}
        desc = _task_description(AGENT_BUDGET, state, TripRequest())
        self.assertIn("Flight price: 300", desc)
        self.assertIn("hotel price: 150", desc)


if __name__ == "__main__":
    unittest.main()

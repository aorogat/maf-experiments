"""Tests for LangGraph travel-planning implementation."""

import unittest

from frameworks.gabm_skeleton.llm_client import StubLLMClient
from frameworks.gabm_skeleton.metrics import Metrics

from examples.travel_planning.client_ext import StubToolResponse, ToolCallingStub
from examples.travel_planning.langgraph_impl import run_langgraph
from examples.travel_planning.task import AGENT_BUDGET, AGENT_FLIGHT, AGENT_HOTEL, AGENT_PLANNER, TripRequest, budget_satisfied


class TestLangGraph(unittest.TestCase):
    def _stub_plain(self) -> StubLLMClient:
        responses = ['{"action":"delegate"}'] * 10
        return StubLLMClient(responses=responses)

    def test_node_mode_end_to_end(self):
        trace = run_langgraph(TripRequest(), tool_mode="node", llm_client=self._stub_plain())
        self.assertTrue(budget_satisfied(trace.final_state))
        self.assertGreater(len(trace.steps), 0)

    def test_node_mode_deterministic_tools_always_present(self):
        trace = run_langgraph(TripRequest(), tool_mode="node", llm_client=self._stub_plain())
        all_tools = [t for s in trace.steps for t in s.tool_invocations]
        web_calls = [t for t in all_tools if t.tool == "Web"]
        cal_calls = [t for t in all_tools if t.tool == "Cal"]
        self.assertGreaterEqual(len(web_calls), 2)
        self.assertGreaterEqual(len(cal_calls), 1)
        for t in all_tools:
            self.assertEqual(t.invocation, "deterministic")
            self.assertEqual(t.invoked_by, "graph_node")

    def test_agent_mode_can_have_zero_tool_calls(self):
        metrics = Metrics()
        stub = ToolCallingStub(
            responses=[
                StubToolResponse(content='{"action":"delegate"}'),
                StubToolResponse(content='{"action":"skip_tools"}'),
                StubToolResponse(content='{"action":"skip_tools"}'),
                StubToolResponse(content='{"action":"skip_tools"}'),
            ],
            metrics=metrics,
        )
        trace = run_langgraph(TripRequest(), tool_mode="agent", llm_client=stub)
        all_tools = [t for s in trace.steps for t in s.tool_invocations]
        self.assertEqual(len(all_tools), 0)
        self.assertFalse(trace.summary()["plan_complete"])
        self.assertFalse(trace.summary()["budget_satisfied"])
        self.assertEqual(trace.final_state.get("status"), "incomplete")

    def test_each_step_has_five_fields(self):
        trace = run_langgraph(TripRequest(), tool_mode="node", llm_client=self._stub_plain())
        for step in trace.steps:
            self.assertTrue(step.agent)
            self.assertIsInstance(step.prompt_issued, str)
            self.assertIsInstance(step.llm_calls, int)
            self.assertIsInstance(step.input_tokens, int)
            self.assertIsInstance(step.output_tokens, int)
            self.assertIsInstance(step.state_snapshot, dict)
            self.assertEqual(step.communication, "graph_edge")
            self.assertIsInstance(step.tool_invocations, list)

    def test_parallel_flight_hotel_branches(self):
        """Figure 1(a): Flight and Hotel run in parallel after Planner, join at Budget."""
        trace = run_langgraph(TripRequest(), tool_mode="node", llm_client=self._stub_plain())
        agents = [s.agent for s in trace.steps]
        self.assertIn(AGENT_PLANNER, agents)
        self.assertIn(AGENT_FLIGHT, agents)
        self.assertIn(AGENT_HOTEL, agents)
        self.assertIn(AGENT_BUDGET, agents)
        # Budget steps occur after both Flight and Hotel branch activity
        last_flight = max(i for i, a in enumerate(agents) if a == AGENT_FLIGHT)
        last_hotel = max(i for i, a in enumerate(agents) if a == AGENT_HOTEL)
        first_budget = min(i for i, a in enumerate(agents) if a == AGENT_BUDGET)
        self.assertGreater(first_budget, last_flight)
        self.assertGreater(first_budget, last_hotel)

    def test_per_node_attribution_matches_client_totals(self):
        """Parallel Flight∥Hotel must not double-count via shared Metrics deltas."""
        stub = StubLLMClient(responses=['{"action":"delegate"}'] * 10)
        trace = run_langgraph(TripRequest(), tool_mode="node", llm_client=stub)
        client = trace.metrics
        assert client is not None
        step_calls = sum(s.llm_calls for s in trace.steps)
        step_in = sum(s.input_tokens for s in trace.steps)
        step_out = sum(s.output_tokens for s in trace.steps)
        self.assertEqual(step_calls, client.llm_calls)
        self.assertEqual(step_in, client.input_tokens)
        self.assertEqual(step_out, client.output_tokens)
        # Four agent LLM nodes; tool nodes contribute 0
        self.assertEqual(client.llm_calls, 4)
        agent_steps = [s for s in trace.steps if s.llm_calls > 0]
        self.assertEqual(len(agent_steps), 4)
        for s in agent_steps:
            self.assertEqual(s.llm_calls, 1)


if __name__ == "__main__":
    unittest.main()

"""Cross-paradigm trace comparison tests."""

import unittest

from frameworks.gabm_skeleton.llm_client import StubLLMClient

from examples.travel_planning.client_ext import StubToolResponse, ToolCallingStub
from examples.travel_planning.crewai_impl import run_crewai
from examples.travel_planning.gabm_impl import run_gabm
from examples.travel_planning.langgraph_impl import run_langgraph
from examples.travel_planning.task import TripRequest
from examples.travel_planning.tests.test_gabm import GABM_RESPONSES
from examples.travel_planning.trace import (
    TOOL_INVOCATION_COLUMNS,
    side_by_side,
    tool_invocation_comparison,
)


class TestCrossParadigm(unittest.TestCase):
    def _all_traces(self):
        stub = StubLLMClient(responses=['{}'] * 20)
        tool_stub = ToolCallingStub(
            responses=[StubToolResponse(content='{}')] * 10,
        )
        return {
            "langgraph_node": run_langgraph(TripRequest(), tool_mode="node", llm_client=stub),
            "langgraph_agent": run_langgraph(TripRequest(), tool_mode="agent", llm_client=tool_stub),
            "crewai": run_crewai(TripRequest(), llm_client=stub),
            "gabm": run_gabm(TripRequest(), llm_client=StubLLMClient(responses=GABM_RESPONSES)),
        }

    def test_side_by_side_one_row_per_paradigm(self):
        traces = self._all_traces()
        md, latex = side_by_side(traces)
        self.assertIn("LangGraph", md)
        self.assertIn("CrewAI", md)
        self.assertIn("GABM", md)
        for trace in traces.values():
            summary = trace.summary()
            self.assertIn("total_llm_calls", summary)
            self.assertIn("budget_satisfied", summary)

    def test_tool_invocation_comparison_deterministic_gt_zero_for_node(self):
        traces = self._all_traces()
        tex = tool_invocation_comparison(traces)
        for col in TOOL_INVOCATION_COLUMNS:
            self.assertIn(col, tex)
        node_summary = traces["langgraph_node"].summary()
        self.assertGreater(node_summary["deterministic_tool_calls"], 0)
        self.assertTrue(node_summary["plan_complete"])
        agent_summary = traces["langgraph_agent"].summary()
        self.assertEqual(agent_summary["tool_mode"], "probabilistic (agent-bound)")
        self.assertFalse(agent_summary["plan_complete"])


if __name__ == "__main__":
    unittest.main()

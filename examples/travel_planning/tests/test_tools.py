"""Tests for Web/Cal stubs and invocation wrappers."""

import unittest

from examples.travel_planning.tools import (
    cal_sum,
    execute_cal,
    execute_web,
    invoke_as_agent,
    invoke_as_environment,
    invoke_as_graph_node,
    web_search,
)


class TestTools(unittest.TestCase):
    def test_web_returns_fixed_flight_options(self):
        result = web_search("flight NYC", kind="flight", destination="NYC")
        self.assertEqual(result["kind"], "flight")
        self.assertEqual(len(result["options"]), 1)
        self.assertEqual(result["options"][0]["id"], "AA100")

    def test_web_returns_fixed_hotel_options(self):
        result = web_search("hotel NYC", kind="hotel", destination="NYC")
        self.assertEqual(result["options"][0]["id"], "H1")

    def test_cal_sums_correctly(self):
        result = cal_sum(flight_price=300, hotel_price=150, budget=500)
        self.assertEqual(result["total"], 450)
        self.assertTrue(result["within_budget"])

    def test_graph_node_wrapper_tags(self):
        tc = invoke_as_graph_node("Web", {"query": "flight"}, execute_web)
        self.assertEqual(tc.invocation, "deterministic")
        self.assertEqual(tc.invoked_by, "graph_node")

    def test_agent_wrapper_tags(self):
        tc = invoke_as_agent("Cal", {"flight_price": 100, "hotel_price": 50}, execute_cal)
        self.assertEqual(tc.invocation, "probabilistic")
        self.assertEqual(tc.invoked_by, "agent")

    def test_environment_wrapper_tags(self):
        tc = invoke_as_environment("Web", {"query": "flight", "destination": "NYC"}, execute_web)
        self.assertEqual(tc.invocation, "deterministic")
        self.assertEqual(tc.invoked_by, "environment")


if __name__ == "__main__":
    unittest.main()

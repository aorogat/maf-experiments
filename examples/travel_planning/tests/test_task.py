"""Tests for trip request and budget predicate."""

import unittest

from examples.travel_planning.task import (
    TripRequest,
    budget_satisfied,
    expected_total,
    format_itinerary,
    initial_state,
    plan_complete,
    resolve_budget_status,
)


class TestTask(unittest.TestCase):
    def test_trip_request_defaults(self):
        req = TripRequest()
        self.assertEqual(req.origin, "BOS")
        self.assertEqual(req.destination, "NYC")
        self.assertEqual(req.budget, 500)

    def test_initial_state_well_formed(self):
        state = initial_state()
        self.assertIn("origin", state)
        self.assertIn("chosen_flight", state)
        self.assertIn("running_cost", state)
        self.assertEqual(state["status"], "planning")

    def test_budget_satisfied_true(self):
        state = initial_state()
        state["chosen_flight"] = {"id": "AA100", "price": 300}
        state["chosen_hotel"] = {"id": "H1", "price": 150}
        state["running_cost"] = 450
        state["status"] = "approved"
        self.assertTrue(budget_satisfied(state))

    def test_plan_complete_requires_flight_and_hotel(self):
        state = initial_state()
        self.assertFalse(plan_complete(state))
        state["chosen_flight"] = {"id": "AA100"}
        self.assertFalse(plan_complete(state))
        state["chosen_hotel"] = {"id": "H1"}
        self.assertTrue(plan_complete(state))

    def test_resolve_budget_status_incomplete_without_selections(self):
        state = initial_state()
        self.assertEqual(resolve_budget_status(state, 0, within_budget=True), "incomplete")

    def test_resolve_budget_status_approved_when_complete_and_within_budget(self):
        state = initial_state()
        state["chosen_flight"] = {"id": "AA100", "price": 300}
        state["chosen_hotel"] = {"id": "H1", "price": 150}
        self.assertEqual(resolve_budget_status(state, 450, within_budget=True), "approved")

    def test_budget_satisfied_false_over_budget(self):
        state = initial_state()
        state["chosen_flight"] = {"id": "AA100", "price": 300}
        state["chosen_hotel"] = {"id": "H1", "price": 150}
        state["running_cost"] = 450
        state["status"] = "over_budget"
        self.assertFalse(budget_satisfied(state))

    def test_expected_total_for_nyc(self):
        self.assertEqual(expected_total("NYC"), 450)

    def test_format_itinerary(self):
        state = initial_state()
        state["chosen_flight"] = {"id": "AA100", "price": 300}
        state["chosen_hotel"] = {"id": "H1", "price": 150}
        state["running_cost"] = 450
        text = format_itinerary(state)
        self.assertIn("AA100", text)
        self.assertIn("H1", text)


if __name__ == "__main__":
    unittest.main()

"""Tests for GABM skeleton Game Master mediator."""

import unittest

from frameworks.gabm_skeleton.environment import Environment
from frameworks.gabm_skeleton.mediator import GameMaster
from frameworks.gabm_skeleton.tasks.travel import TravelTask
from frameworks.gabm_skeleton.tasks.trivial import TrivialTask


class TestMediator(unittest.TestCase):
    def test_observation_is_filtered_projection(self):
        task = TravelTask(destination="NYC", budget=500)
        env = task.make_environment()
        gm = GameMaster(task, env)

        flight_obs = gm.observe("flight")
        self.assertIn("destination", flight_obs)
        self.assertIn("budget", flight_obs)
        self.assertNotIn("selected_flight", flight_obs)
        self.assertNotIn("selected_hotel", flight_obs)

        hotel_obs = gm.observe("hotel")
        self.assertIn("selected_flight", hotel_obs)
        self.assertNotIn("selected_hotel", hotel_obs)
        self.assertNotIn("total_cost", hotel_obs)

        budget_obs = gm.observe("budget")
        self.assertIn("selected_flight", budget_obs)
        self.assertIn("selected_hotel", budget_obs)
        self.assertNotIn("destination", budget_obs)

    def test_apply_round_applies_one_action_per_agent(self):
        task = TrivialTask()
        env = task.make_environment()
        gm = GameMaster(task, env)

        actions = {"solver": {"answer": "4"}}
        tools = gm.apply_round(actions)
        self.assertEqual(env.state["answer"], "4")
        self.assertEqual(tools, [])

    def test_travel_round_applies_all_three_agents(self):
        task = TravelTask(destination="NYC", budget=500)
        env = task.make_environment()
        gm = GameMaster(task, env)

        actions = {
            "flight": {"action": "select_flight"},
            "hotel": {"action": "select_hotel"},
            "budget": {"action": "review_budget"},
        }
        tools = gm.apply_round(actions)

        self.assertEqual(len(actions), 3)
        self.assertTrue(env.state["selected_flight"])
        self.assertTrue(env.state["selected_hotel"])
        self.assertGreater(env.state["total_cost"], 0)
        self.assertGreaterEqual(len(tools), 3)


if __name__ == "__main__":
    unittest.main()

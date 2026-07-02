"""Tests for GABM skeleton environment."""

import unittest

from frameworks.gabm_skeleton.environment import Environment, InvalidActionError


def _counter_transition(state, action):
    if "increment" not in action:
        raise ValueError("Action must contain 'increment'")
    new_state = dict(state)
    new_state["count"] = state["count"] + action["increment"]
    return new_state


class TestEnvironment(unittest.TestCase):
    def test_initializes_with_declared_variables(self):
        env = Environment(
            schema={"count": int, "label": str},
            initial={"count": 0, "label": "test"},
            transition=_counter_transition,
        )
        self.assertEqual(env.state["count"], 0)
        self.assertEqual(env.state["label"], "test")

    def test_rejects_unknown_keys_on_init(self):
        with self.assertRaises(KeyError):
            Environment(
                schema={"count": int},
                initial={"count": 0, "extra": 1},
                transition=_counter_transition,
            )

    def test_transition_applies_valid_action(self):
        env = Environment(
            schema={"count": int},
            initial={"count": 0},
            transition=_counter_transition,
        )
        env.apply({"increment": 3})
        self.assertEqual(env.state["count"], 3)

    def test_invalid_action_rejected_without_mutation(self):
        env = Environment(
            schema={"count": int},
            initial={"count": 5},
            transition=_counter_transition,
        )
        with self.assertRaises(InvalidActionError):
            env.apply({"bad_key": 1})
        self.assertEqual(env.state["count"], 5)

    def test_snapshot_is_independent_copy(self):
        env = Environment(
            schema={"count": int},
            initial={"count": 1},
            transition=_counter_transition,
        )
        snap = env.snapshot()
        snap["count"] = 99
        self.assertEqual(env.state["count"], 1)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from runtime.policies import fixed as fixed_policy


class FixedPolicyTest(unittest.TestCase):
    def test_actions_are_independently_configurable(self):
        env = {
            "NQO_FIXED_DEC": "apply",
            "NQO_FIXED_SCHED_ALPHA": "0.75",
            "NQO_FIXED_ENUM": "top5",
            "NQO_FIXED_FILTER": "selective",
            "NQO_FIXED_AJOIN": "conservative",
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertEqual(
                fixed_policy.predict({"request_type": "dec", "remaining_splits": 2})[
                    "dec_action"
                ],
                "apply",
            )
            self.assertEqual(
                fixed_policy.predict({"request_type": "sched"})["sched_alpha"],
                0.75,
            )
            self.assertEqual(
                fixed_policy.predict({"request_type": "enum"})["enum_action"],
                "top5",
            )
            adapt = fixed_policy.predict({"request_type": "adapt"})
            self.assertEqual(adapt["filter_action"], "selective")
            self.assertEqual(adapt["ajoin_action"], "conservative")

    def test_split_round_limit_stops_further_decomposition(self):
        env = {
            "NQO_FIXED_DEC": "apply",
            "NQO_FIXED_DEC_ROUNDS": "2",
        }
        with patch.dict(os.environ, env, clear=False):
            action = fixed_policy.predict(
                {"request_type": "dec", "round": 2, "remaining_splits": 3}
            )
        self.assertEqual(action["dec_action"], "skip")

    def test_alpha_sequence_selects_a_per_round_action(self):
        env = {
            "NQO_FIXED_SCHED_ALPHA": "0.5",
            "NQO_FIXED_SCHED_ALPHA_SEQUENCE": "0.75,0.25",
        }
        with patch.dict(os.environ, env, clear=False):
            first = fixed_policy.predict({"request_type": "sched", "round": 0})
            second = fixed_policy.predict({"request_type": "sched", "round": 1})
            fallback = fixed_policy.predict({"request_type": "sched", "round": 2})
        self.assertEqual(first["sched_alpha"], 0.75)
        self.assertEqual(second["sched_alpha"], 0.25)
        self.assertEqual(fallback["sched_alpha"], 0.5)

    def test_legacy_environment_and_request_names_are_read_only_aliases(self):
        with patch.dict(
            os.environ,
            {"NQO_FIXED_HIGH": "split", "NQO_FIXED_SPLIT_ROUNDS": "1"},
            clear=True,
        ):
            action = fixed_policy.predict(
                {"request_type": "high", "round": 0, "remaining_splits": 1}
            )
        self.assertEqual(action["dec_action"], "apply")
        self.assertNotIn("high_action", action)


if __name__ == "__main__":
    unittest.main()

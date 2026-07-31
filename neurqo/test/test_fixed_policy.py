#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1] / "server"
sys.path.insert(0, str(SERVER_DIR))

import fixed_policy  # noqa: E402


class FixedPolicyTest(unittest.TestCase):
    def test_actions_are_independently_configurable(self):
        env = {
            "NEURQO_FIXED_HIGH": "split",
            "NEURQO_FIXED_ALPHA": "0.75",
            "NEURQO_FIXED_SEARCH": "top5",
            "NEURQO_FIXED_LIP": "selective",
            "NEURQO_FIXED_AJA": "conservative",
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertEqual(
                fixed_policy.predict({"request_type": "high", "remaining_splits": 2})[
                    "high_action"
                ],
                "split",
            )
            self.assertEqual(
                fixed_policy.predict({"request_type": "select"})["schedule_alpha"],
                0.75,
            )
            self.assertEqual(
                fixed_policy.predict({"request_type": "search"})["search_label"],
                "top5",
            )
            low = fixed_policy.predict({"request_type": "low"})
            self.assertEqual(low["lip_action"], "selective")
            self.assertEqual(low["aja_level"], "conservative")

    def test_split_round_limit_stops_further_decomposition(self):
        env = {
            "NEURQO_FIXED_HIGH": "split",
            "NEURQO_FIXED_SPLIT_ROUNDS": "2",
        }
        with patch.dict(os.environ, env, clear=False):
            action = fixed_policy.predict(
                {"request_type": "high", "round": 2, "remaining_splits": 3}
            )
        self.assertEqual(action["high_action"], "stop")

    def test_alpha_sequence_selects_a_per_round_action(self):
        env = {
            "NEURQO_FIXED_ALPHA": "0.5",
            "NEURQO_FIXED_ALPHA_SEQUENCE": "0.75,0.25",
        }
        with patch.dict(os.environ, env, clear=False):
            first = fixed_policy.predict({"request_type": "select", "round": 0})
            second = fixed_policy.predict({"request_type": "select", "round": 1})
            fallback = fixed_policy.predict({"request_type": "select", "round": 2})
        self.assertEqual(first["schedule_alpha"], 0.75)
        self.assertEqual(second["schedule_alpha"], 0.25)
        self.assertEqual(fallback["schedule_alpha"], 0.5)


if __name__ == "__main__":
    unittest.main()

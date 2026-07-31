#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1] / "server"
sys.path.insert(0, str(SERVER_DIR))

import online_trainer  # noqa: E402


class _FakeHRL:
    SCHEDULE_ALPHA_VALUES = (0.0, 0.5, 1.0)
    SEARCH_LABELS = ["default", "top5"]
    ACTION_LABELS = [
        "none",
        "lip_sel",
        "aja_conservative",
        "lip_sel+aja_conservative",
    ]


class OnlineTrainerPhaseTest(unittest.TestCase):
    def setUp(self):
        self.trainer = online_trainer.OnlineCheckpointTrainer.__new__(
            online_trainer.OnlineCheckpointTrainer
        )
        self.trainer.hrl = _FakeHRL()

    def test_split_updates_each_existing_head_from_selected_subquery(self):
        event = {
            "phase": "split",
            "stop": False,
            "state": {"request_type": "high"},
            "decision_states": {
                "high": {"request_type": "high"},
                "select": {
                    "request_type": "select",
                    "candidates": [{"candidate_id": 2}],
                },
                "search": {"request_type": "search"},
                "low": {
                    "request_type": "low",
                    "plan_json": {"Node Type": "Hash Join"},
                },
            },
            "action": {
                "high_action": "split",
                "candidate_id": 2,
                "schedule_idx": 1,
                "schedule_alpha": 0.5,
                "search_strategy": "topk",
                "search_k": 5,
                "execution_action": "conservative",
                "lip_action": "none",
            },
            "timing_ms": {"total": 10.0},
        }

        transition = online_trainer.event_to_transition(event)

        self.assertIsNotNone(transition)
        self.assertEqual(
            self.trainer._target_indices(transition),
            {"high": 1, "select": 1, "search": 1, "low": 2},
        )

    def test_final_updates_each_head_from_its_own_state(self):
        event = {
            "phase": "final",
            "stop": True,
            "state": {"request_type": "high"},
            "decision_states": {
                "high": {"request_type": "high"},
                "search": {"request_type": "search"},
                "low": {
                    "request_type": "low",
                    "plan_json": {"Node Type": "Hash Join"},
                },
            },
            "action": {
                "high_action": "stop",
                "search_strategy": "topk",
                "search_k": 5,
                "execution_action": "conservative",
                "lip_action": "selective",
            },
            "timing_ms": {"total": 20.0},
        }

        transition = online_trainer.event_to_transition(event)

        self.assertIsNotNone(transition)
        self.assertEqual(
            self.trainer._target_indices(transition),
            {"high": 0, "search": 1, "low": 3},
        )

    def test_old_combined_log_remains_readable(self):
        state = {"request_type": "round", "plan_json": {"Node Type": "Seq Scan"}}

        transition = online_trainer.event_to_transition(
            {
                "phase": "final",
                "stop": True,
                "state": state,
                "action": {},
                "timing_ms": {"total": 1.0},
            }
        )

        self.assertEqual(
            transition["decision_states"],
            {"high": state, "search": state, "low": state},
        )

    def test_conservative_aja_maps_to_separate_low_action(self):
        transition = {
            "action": {
                "execution_action": "conservative",
                "lip_action": "selective",
            },
            "decision_states": {
                "low": {
                    "request_type": "low",
                    "plan_json": {"Node Type": "Hash Join"},
                },
            },
        }

        self.assertEqual(
            self.trainer._target_indices(transition),
            {"low": 3},
        )


if __name__ == "__main__":
    unittest.main()

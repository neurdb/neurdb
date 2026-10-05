#!/usr/bin/env python3
import unittest

from training import incremental_trainer


class _FakeHRL:
    SCHED_ALPHA_VALUES = (0.0, 0.5, 1.0)
    ENUM_LABELS = ["native", "top5"]
    ADAPT_LABELS = [
        "none",
        "filter_selective",
        "ajoin_conservative",
        "filter_selective+ajoin_conservative",
    ]


class IncrementalTrainerPhaseTest(unittest.TestCase):
    def setUp(self):
        self.trainer = incremental_trainer.IncrementalCheckpointTrainer.__new__(
            incremental_trainer.IncrementalCheckpointTrainer
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
                "sched_alpha": 0.5,
                "search_strategy": "topk",
                "search_k": 5,
                "execution_action": "conservative",
                "lip_action": "none",
            },
            "timing_ms": {"total": 10.0},
        }

        transition = incremental_trainer.event_to_transition(event)

        self.assertIsNotNone(transition)
        self.assertEqual(
            self.trainer._target_indices(transition),
            {"dec": 1, "sched": 1, "enum": 1, "adapt": 2},
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

        transition = incremental_trainer.event_to_transition(event)

        self.assertIsNotNone(transition)
        self.assertEqual(
            self.trainer._target_indices(transition),
            {"dec": 0, "enum": 1, "adapt": 3},
        )

    def test_old_combined_log_remains_readable(self):
        state = {"request_type": "round", "plan_json": {"Node Type": "Seq Scan"}}

        transition = incremental_trainer.event_to_transition(
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
            {"dec": state, "enum": state, "adapt": state},
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
            {"adapt": 3},
        )


if __name__ == "__main__":
    unittest.main()

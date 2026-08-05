#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SERVER_DIR = Path(__file__).resolve().parents[1] / "server"
sys.path.insert(0, str(SERVER_DIR))

import ai_server  # noqa: E402


class FakeAdapter:
    source = "test"

    def predict(self, _state):
        return {
            "high_action": "split",
            "search_label": "split",
            "low_label": "aja",
            "note": "combined prediction must be phase-filtered",
        }


class FakeSelectionAdapter:
    source = "selector-test"

    def predict(self, state):
        if state.get("request_type") == "select":
            return {
                "candidate_id": 2,
                "selection_strategy": "model",
                "note": "test selector",
            }
        return {}


class FakeAlphaAdapter:
    source = "alpha-test"

    def predict(self, state):
        if state.get("request_type") == "select":
            return {
                "schedule_idx": 2,
                "schedule_alpha": 1.0,
                "selection_strategy": "alpha_1.00",
            }
        return {}


class FakeTrainingAdapter:
    source = "training-test"

    def predict(self, _state):
        return {
            "high_action": "stop",
            "action_index": 0,
            "action_mask": [True, False],
            "action_probability": 0.75,
            "log_probability": -0.287682,
            "predicted_value": 1.25,
            "policy_version": "checkpoint-7",
            "inference_mode": "stochastic",
            "high_split_probability": 0.25,
        }


class PhaseDecisionTest(unittest.TestCase):
    def setUp(self):
        self.old_adapter = ai_server.ADAPTER
        ai_server.ADAPTER = FakeAdapter()

    def tearDown(self):
        ai_server.ADAPTER = self.old_adapter

    def test_high_returns_only_decomposition_action(self):
        action = ai_server.decide_action(
            {
                "request_type": "high",
                "base_rels": 4,
                "remaining_splits": 2,
            }
        )

        self.assertEqual(action["action"], "split")
        self.assertFalse(action["stop"])
        self.assertNotIn("search_strategy", action)
        self.assertNotIn("execution_action", action)
        self.assertNotIn("lip_action", action)

    def test_search_maps_split_label_to_top_one(self):
        action = ai_server.decide_action({"request_type": "search"})

        self.assertEqual(action["action"], "search")
        self.assertEqual(action["search_strategy"], "topk")
        self.assertEqual(action["search_k"], 1)
        self.assertNotIn("execution_action", action)
        self.assertNotIn("lip_action", action)

    def test_select_uses_phi4_until_a_candidate_model_is_connected(self):
        action = ai_server.decide_action(
            {
                "request_type": "select",
                "candidates": [
                    {
                        "candidate_id": 0,
                        "plan_total_cost": 20,
                        "plan_rows": 100,
                    },
                    {
                        "candidate_id": 1,
                        "plan_total_cost": 50,
                        "plan_rows": 10,
                    },
                    {
                        "candidate_id": 2,
                        "plan_total_cost": 5,
                        "plan_rows": 200,
                    },
                ],
            }
        )

        self.assertEqual(action["action"], "select")
        self.assertEqual(action["candidate_id"], 1)
        self.assertEqual(action["selection_strategy"], "phi4")
        self.assertNotIn("search_strategy", action)
        self.assertNotIn("execution_action", action)

    def test_select_accepts_model_candidate_id(self):
        ai_server.ADAPTER = FakeSelectionAdapter()

        action = ai_server.decide_action(
            {
                "request_type": "select",
                "candidates": [
                    {"candidate_id": 0, "plan_total_cost": 1, "plan_rows": 1},
                    {"candidate_id": 2, "plan_total_cost": 100, "plan_rows": 100},
                ],
            }
        )

        self.assertEqual(action["candidate_id"], 2)
        self.assertEqual(action["selection_strategy"], "model")

    def test_select_uses_predicted_alpha_to_rank_candidates(self):
        ai_server.ADAPTER = FakeAlphaAdapter()

        action = ai_server.decide_action(
            {
                "request_type": "select",
                "candidates": [
                    {
                        "candidate_id": 0,
                        "plan_total_cost": 20,
                        "plan_rows": 1,
                    },
                    {
                        "candidate_id": 1,
                        "plan_total_cost": 5,
                        "plan_rows": 1000,
                    },
                ],
            }
        )

        self.assertEqual(action["candidate_id"], 1)
        self.assertEqual(action["schedule_idx"], 2)
        self.assertEqual(action["schedule_alpha"], 1.0)
        self.assertEqual(action["selection_strategy"], "alpha_1.00")

    def test_low_returns_plan_driven_execution_action(self):
        action = ai_server.decide_action(
            {
                "request_type": "low",
                "aliases": ["a", "b", "c"],
                "plan_rows": 50_000,
                "plan_summary": {"joins": 2},
            }
        )

        self.assertEqual(action["action"], "low")
        self.assertEqual(action["execution_action"], "aggressive")
        self.assertEqual(action["lip_action"], "none")
        self.assertNotIn("join_method", action)
        self.assertNotIn("aja_hint", action)
        self.assertNotIn("search_strategy", action)

    def test_low_accepts_explicit_adaptive_threshold_level(self):
        class ConservativeAdapter:
            source = "conservative-test"

            def predict(self, _state):
                return {
                    "aja_level": "conservative",
                    "lip_action": "selective",
                }

        ai_server.ADAPTER = ConservativeAdapter()
        action = ai_server.decide_action({"request_type": "low"})

        self.assertEqual(action["execution_action"], "conservative")
        self.assertEqual(action["lip_action"], "selective")
        self.assertNotIn("join_method", action)
        self.assertNotIn("aja_hint", action)

    def test_training_metadata_survives_phase_filtering(self):
        ai_server.ADAPTER = FakeTrainingAdapter()
        action = ai_server.decide_action({"request_type": "high"})

        self.assertEqual(action["action"], "stop")
        self.assertEqual(action["action_index"], 0)
        self.assertEqual(action["action_mask"], [True, False])
        self.assertEqual(action["action_probability"], 0.75)
        self.assertAlmostEqual(action["log_probability"], -0.287682)
        self.assertEqual(action["predicted_value"], 1.25)
        self.assertEqual(action["policy_version"], "checkpoint-7")
        self.assertEqual(action["inference_mode"], "stochastic")
        self.assertEqual(action["high_split_probability"], 0.25)

    def test_deterministic_low_action_uses_masked_argmax(self):
        adapter = ai_server.PolicyAdapter()
        adapter._torch = torch
        adapter._device = torch.device("cpu")

        action, metadata = adapter._masked_action(
            torch.tensor([0.0, 0.4, -1.0]), [True, True, True], "low"
        )

        self.assertEqual(action, 1)
        self.assertEqual(metadata["action_index"], 1)
        self.assertEqual(metadata["inference_mode"], "deterministic")

    def test_exact_structured_state_encodings_are_reused(self):
        adapter = ai_server.PolicyAdapter()
        adapter._torch = torch
        adapter._device = torch.device("cpu")
        calls = []
        adapter._model = SimpleNamespace(
            encode_state_obj=lambda state, _device: (
                calls.append(state.cache_key) or torch.tensor([float(len(calls))])
            )
        )
        first_state = SimpleNamespace(cache_key=("online", "high", "q1", (0.0,)))
        same_state = SimpleNamespace(cache_key=("online", "high", "q1", (0.0,)))
        later_state = SimpleNamespace(cache_key=("online", "high", "q1", (1.0,)))

        first = adapter._encode(first_state)
        repeated = adapter._encode(same_state)
        later = adapter._encode(later_state)

        self.assertIs(first, repeated)
        self.assertEqual(later.item(), 2.0)
        self.assertEqual(len(calls), 2)

    def test_stochastic_search_samples_without_confidence_guard(self):
        adapter = ai_server.PolicyAdapter(
            inference_mode="stochastic",
            stochastic_heads="search",
            sampling_seed=7,
        )
        adapter._torch = torch
        adapter._device = torch.device("cpu")
        torch.manual_seed(7)

        _action, metadata = adapter._masked_action(
            torch.tensor([0.0, 0.4]), [True, True], "search"
        )

        self.assertNotIn("search_abstained", metadata)
        self.assertEqual(metadata["inference_mode"], "stochastic")

    def test_coverage_sampling_prefers_underexplored_valid_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coverage.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "counts": {"search": {"state": [100, 0]}},
                    }
                ),
                encoding="utf-8",
            )
            adapter = ai_server.PolicyAdapter(
                inference_mode="stochastic",
                stochastic_heads="search",
                coverage_counts_path=str(path),
                coverage_mix=0.3,
                coverage_power=0.5,
            )
        adapter._torch = torch
        adapter._device = torch.device("cpu")
        adapter._hrl = SimpleNamespace(coverage_state_hash=lambda _state: "state")
        _action, metadata = adapter._masked_action(
            torch.tensor([0.0, 0.0]), [True, True], "search", {"sql": "select 1"}
        )
        self.assertGreater(
            metadata["coverage_probabilities"][1],
            metadata["coverage_probabilities"][0],
        )
        self.assertEqual(metadata["coverage_mix"], 0.3)

    def test_tpch_model_inference_masks_query_split(self):
        adapter = ai_server.PolicyAdapter(workload="tpch")
        adapter._torch = torch
        adapter._hrl = SimpleNamespace(
            apply_action_ablation_mask=lambda mask, _phase, _ablation: np.asarray(
                mask, dtype=np.float32
            )
        )
        adapter._query_graph_state = lambda _state, _level: object()
        adapter._encode = lambda _state: torch.zeros(2)
        adapter._model = SimpleNamespace(
            high_actor=lambda _encoded: torch.tensor([0.0, 10.0]),
            high_critic=lambda _encoded: torch.tensor([0.0]),
        )
        observed = {}

        def masked_action(_logits, mask, _phase, _state):
            observed["mask"] = mask
            return 0, {
                "action_index": 0,
                "action_mask": mask,
            }

        adapter._masked_action = masked_action
        action = adapter._predict_hrl(
            {
                "request_type": "high",
                "base_rels": 8,
                "remaining_splits": 7,
            }
        )

        self.assertEqual(observed["mask"], [1.0, 0.0])
        self.assertEqual(action["high_action"], "stop")

    def test_runtime_relation_plan_contains_temp_table_statistics(self):
        plan = ai_server._runtime_relation_plan(
            {
                "relations": [
                    {
                        "alias": "temp1",
                        "relname": "temp1",
                        "estimated_rows": 41840,
                        "pages": 321,
                        "is_temporary": True,
                    }
                ]
            }
        )

        node = plan["Plan"]["Plans"][0]
        self.assertEqual(node["Relation Name"], "temp1")
        self.assertEqual(node["Plan Rows"], 41840)
        self.assertEqual(node["Total Cost"], 321)

    def test_normalized_nested_loop_is_recognized_as_a_join(self):
        plan = {
            "Node Type": "Aggregate",
            "Plans": [{"Node Type": "Nested Loop"}],
        }

        self.assertTrue(ai_server._plan_contains_join(plan))
        self.assertFalse(ai_server._plan_contains_join(plan, "hashjoin"))

    def test_policy_adapter_delegates_high_context_to_shared_builder(self):
        received = {}
        empty_plan = object()

        class FakeTransfer:
            HIGH_CTX_DIM = 2
            np = np

            @staticmethod
            def parse_query_graph(_sql):
                return object()

            @staticmethod
            def build_transfer_graph_state(_sql, _graph, _catalog, plan_json=None):
                received["graph_plan_json"] = plan_json
                return object(), None

            @staticmethod
            def build_high_context(**kwargs):
                received.update(kwargs)
                return np.arange(2, dtype=np.float32)

            @staticmethod
            def empty_plan_tree():
                return empty_plan

            @staticmethod
            def StructuredState(**kwargs):
                return SimpleNamespace(**kwargs)

        adapter = object.__new__(ai_server.PolicyAdapter)
        adapter._transfer = FakeTransfer
        adapter._catalog = object()
        adapter._query_graph_cache = {}
        adapter._plan_tree_cache = {}
        state = adapter._query_graph_state(
            {
                "sql": "SELECT * FROM title t",
                "plan_json": {"Plan": {"Total Cost": 123}},
                "plan_total_cost": 123,
                "plan_rows": 7,
                "plan_summary": {"joins": 2, "max_depth": 4},
                "cumulative_cost_ms": 9,
                "round": 1,
                "max_split_rounds": 5,
            },
            "high",
        )

        self.assertEqual(state.ctx.tolist(), [0, 1])
        self.assertEqual(received["graph_plan_json"], None)
        self.assertEqual(received["cumulative_ms"], 9)
        self.assertEqual(received["round_index"], 1)
        self.assertNotIn("plan_total_cost", received)
        self.assertIs(state.current_plan, empty_plan)

    def test_low_state_uses_plan_and_execution_context_only(self):
        empty_query_graph = object()
        plan_tree = object()
        received = {}

        class FakeTransfer:
            LOW_CTX_DIM = 7
            np = np

            @staticmethod
            def build_low_context(**kwargs):
                received.update(kwargs)
                return np.arange(7, dtype=np.float32)

            @staticmethod
            def plan_to_tree(_plan, catalog=None):
                return plan_tree

            @staticmethod
            def empty_plan_tree():
                return object()

            @staticmethod
            def empty_query_graph_state():
                return empty_query_graph

            @staticmethod
            def StructuredState(**kwargs):
                return SimpleNamespace(**kwargs)

        adapter = object.__new__(ai_server.PolicyAdapter)
        adapter._transfer = FakeTransfer
        adapter._catalog = object()
        adapter._plan_tree_cache = {}

        state = adapter._plan_state(
            {
                "sql": "SELECT t.id FROM title AS t",
                "plan_json": {"Plan": {"Node Type": "Seq Scan"}},
                "cumulative_cost_ms": 25,
                "round": 2,
                "max_split_rounds": 8,
                "is_split_execution": True,
                "search_strategy": "topk",
                "search_k": 5,
            }
        )

        self.assertIs(state.query_graph, empty_query_graph)
        self.assertIs(state.current_plan, plan_tree)
        self.assertEqual(state.ctx.tolist(), list(range(7)))
        self.assertEqual(received["cumulative_ms"], 25)
        self.assertEqual(received["round_index"], 2)
        self.assertEqual(received["max_rounds"], 8)
        self.assertTrue(received["is_split_execution"])
        self.assertEqual(received["search_strategy"], "topk")
        self.assertEqual(received["search_k"], 5)

    def test_checkpoint_query_topology_ablation_is_applied_at_inference(self):
        raw_graph = object()
        bag_graph = object()

        class FakeTransfer:
            HIGH_CTX_DIM = 2
            np = np

            @staticmethod
            def parse_query_graph(_sql):
                return object()

            @staticmethod
            def build_transfer_graph_state(_sql, _graph, _catalog, plan_json=None):
                return raw_graph, None

            @staticmethod
            def remove_query_graph_topology(graph):
                self.assertIs(graph, raw_graph)
                return bag_graph

            @staticmethod
            def build_high_context(**_kwargs):
                return np.zeros(2, dtype=np.float32)

            @staticmethod
            def empty_plan_tree():
                return object()

            @staticmethod
            def StructuredState(**kwargs):
                return SimpleNamespace(**kwargs)

        adapter = object.__new__(ai_server.PolicyAdapter)
        adapter._transfer = FakeTransfer
        adapter._catalog = object()
        adapter._query_graph_cache = {}
        adapter.state_ablation = "no_query_topology"
        state = adapter._query_graph_state({"sql": "SELECT * FROM title"}, "high")
        self.assertIs(state.query_graph, bag_graph)

    def test_checkpoint_plan_topology_ablation_is_applied_at_inference(self):
        raw_tree = object()
        flat_tree = object()

        class FakeTransfer:
            LOW_CTX_DIM = 7
            np = np

            @staticmethod
            def build_low_context(**_kwargs):
                return np.zeros(7, dtype=np.float32)

            @staticmethod
            def plan_to_tree(_plan, catalog=None):
                return raw_tree

            @staticmethod
            def flatten_plan_tree_topology(tree):
                self.assertIs(tree, raw_tree)
                return flat_tree

            @staticmethod
            def empty_plan_tree():
                return object()

            @staticmethod
            def empty_query_graph_state():
                return object()

            @staticmethod
            def StructuredState(**kwargs):
                return SimpleNamespace(**kwargs)

        adapter = object.__new__(ai_server.PolicyAdapter)
        adapter._transfer = FakeTransfer
        adapter._catalog = object()
        adapter._plan_tree_cache = {}
        adapter.state_ablation = "no_plan_topology"
        state = adapter._plan_state(
            {"plan_json": {"Plan": {"Node Type": "Hash Join"}}}
        )
        self.assertIs(state.current_plan, flat_tree)

    def test_legacy_combined_and_standalone_aja_requests_are_rejected(self):
        for request_type in ("round", "aja"):
            with self.subTest(request_type=request_type):
                with self.assertRaisesRegex(ValueError, "unknown request_type"):
                    ai_server.decide_action({"request_type": request_type})


if __name__ == "__main__":
    unittest.main()

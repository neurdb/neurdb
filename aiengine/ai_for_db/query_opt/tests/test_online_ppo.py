from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
from experience.store import ExperienceStore, content_hash
from model.encoders.query_graph import CatalogInfo
from model.encoders.state import (
    ADAPT_CTX_DIM,
    COLUMN_IDX_CONST_POS,
    DEC_CTX_DIM,
    build_adapt_context,
    build_dec_context,
    build_transfer_graph_state,
    empty_structured_state,
    flatten_plan_tree_topology,
    parse_query_graph,
    plan_to_tree,
    remove_query_graph_topology,
)
from model.policy.action_space import (
    ADAPT_LABELS,
    ENUM_LABELS,
    N_ADAPT,
    N_SCHED,
    Transition,
    apply_action_ablation_mask,
)
from model.policy.hierarchical_actor_critic import (
    HACNetwork,
    _mixed_masked_categorical,
    ppo_update,
)
from optimization.state import runtime_relation_plan
from training.experience_trainer import (
    FROZEN_INDEPENDENT_PRIOR_ENCODER_PHASES,
    FROZEN_REPLAY_ENCODER_PHASES,
    ReplayTarget,
    _action_cost_regression_target,
    _action_cost_target,
    _balanced_replay_weights,
    _phase_action_index,
    _replay_state_identity,
    _sched_cost_target,
    collect_independent_action_targets,
    collect_replay_targets,
    collect_residual_split_prior_targets,
    collect_transitions,
    expand_legacy_checkpoint_tensors,
    initialize_policy_profile,
    initialize_safe_policy,
    joint_dec_prior_update,
    portable_numpy_rng_state,
    replay_policy_update,
)
from training.state_builder import ExecutionStateBuilder


def _test_catalog() -> CatalogInfo:
    """Small deterministic catalog fixture independent of a workload dump."""
    return CatalogInfo(
        {
            "tables": [
                {"table_name": "title", "row_count": 2_528_312},
                {"table_name": "movie_info", "row_count": 14_835_720},
                {"table_name": "company_name", "row_count": 234_997},
            ],
            "column_statistics": [
                {
                    "table_name": "title",
                    "column_name": "id",
                    "data_type": "integer",
                    "n_distinct": -1.0,
                },
                {
                    "table_name": "title",
                    "column_name": "production_year",
                    "data_type": "integer",
                    "n_distinct": 123.0,
                },
                {
                    "table_name": "movie_info",
                    "column_name": "movie_id",
                    "data_type": "integer",
                    "n_distinct": 356_180.0,
                },
                {
                    "table_name": "company_name",
                    "column_name": "country_code",
                    "data_type": "character varying(255)",
                    "n_distinct": 164.0,
                    "most_common_vals": ["[us]"],
                    "most_common_freqs": [0.36016667],
                },
            ],
            "primary_keys": [
                {"table_name": "title", "column_name": "id"},
            ],
            "foreign_keys": [
                {"table_name": "movie_info", "column_name": "movie_id"},
            ],
            "indexes": [],
        }
    )


def _append_compact_execution(
    store: ExperienceStore,
    *,
    query_id: str,
    decisions: list[dict],
    runtime_ms: float,
    episode_id: str | None = None,
    created_at_ms: int | None = None,
) -> None:
    """Write a test trajectory using the production one-table format."""
    source_episode_id = episode_id or f"episode-{query_id}-{store.now_ms()}"
    trajectory = []
    for decision in decisions:
        state = decision["state"]
        action = dict(decision["action"])
        policy = dict(action)
        for key in (
            "policy_version",
            "action_probability",
            "log_probability",
            "predicted_value",
            "action_mask",
        ):
            if key in decision:
                policy[key] = decision[key]
        trajectory.append(
            {
                "round_index": int(decision.get("round_index", 0)),
                "phase": decision["phase"],
                "state": state,
                "state_hash": content_hash(state),
                "action": action,
                "policy": policy,
                "runtime_ms": float(decision.get("runtime_ms", runtime_ms)),
                "charged_runtime_ms": float(
                    decision.get(
                        "charged_runtime_ms",
                        decision.get("runtime_ms", runtime_ms),
                    )
                ),
                "is_timeout": False,
                "observed_at_ms": int(created_at_ms or store.now_ms()),
                "implementation_version": "test-v1",
            }
        )
    store.append_execution(
        query_id=query_id,
        sql_hash=content_hash(query_id),
        trajectory=trajectory,
        db_events=[],
        status="ok",
        first_runtime_ms=runtime_ms,
        charged_runtime_ms=runtime_ms,
        timeout_limit_ms=60_000,
        action_config_hash="test-config",
        source_episode_id=source_episode_id,
        created_at_ms=created_at_ms,
    )


def nonempty_adapt_state():
    state = empty_structured_state(level="adapt", ctx_dim=ADAPT_CTX_DIM)
    state.current_plan = plan_to_tree(
        {
            "Plan": {
                "Node Type": "Hash Join",
                "Plan Rows": 100,
                "Plan Width": 16,
                "Startup Cost": 2,
                "Total Cost": 20,
                "Plans": [
                    {
                        "Node Type": "Seq Scan",
                        "Plan Rows": 10,
                        "Plan Width": 8,
                        "Total Cost": 5,
                    },
                    {
                        "Node Type": "Seq Scan",
                        "Plan Rows": 20,
                        "Plan Width": 8,
                        "Total Cost": 7,
                    },
                ],
            }
        }
    )
    return state


def test_query_topology_ablation_preserves_nodes_and_removes_joins() -> None:
    catalog = _test_catalog()
    sql = (
        "SELECT * FROM title AS t, movie_info AS mi "
        "WHERE t.id = mi.movie_id AND t.production_year > 2000"
    )
    graph, _stats = build_transfer_graph_state(
        sql,
        parse_query_graph(sql),
        catalog,
    )
    ablated = remove_query_graph_topology(graph)

    assert np.array_equal(ablated.table_node_features, graph.table_node_features)
    assert np.array_equal(ablated.column_node_features, graph.column_node_features)
    assert np.array_equal(ablated.membership_edges, graph.membership_edges)
    assert graph.table_join_edges.shape[0] > 0
    assert ablated.join_edges.shape == (0, 2)
    assert ablated.table_join_edges.shape == (0, 2)


def test_plan_topology_ablation_preserves_all_nodes_as_a_flat_bag() -> None:
    original = nonempty_adapt_state().current_plan
    flattened = flatten_plan_tree_topology(original)

    def nodes(tree):
        return [tree, *(node for child in tree.children for node in nodes(child))]

    original_nodes = nodes(original)
    flattened_nodes = nodes(flattened)
    assert len(flattened_nodes) == len(original_nodes)
    assert len(flattened.children) == len(original_nodes) - 1
    assert all(not child.children for child in flattened.children)
    assert sorted(node.op_type_id for node in flattened_nodes) == sorted(
        node.op_type_id for node in original_nodes
    )


def test_action_ablation_masks_optimizer_mechanisms():
    no_split = apply_action_ablation_mask(
        np.ones(2, dtype=np.float32),
        "dec",
        "no_split",
    )
    assert no_split.tolist() == [1.0, 0.0]

    search = apply_action_ablation_mask(
        np.ones(len(ENUM_LABELS), dtype=np.float32),
        "enum",
        "no_topk",
    )
    assert search.tolist() == [1.0, 0.0]

    no_filter = apply_action_ablation_mask(
        np.ones(len(ADAPT_LABELS), dtype=np.float32),
        "adapt",
        "no_filter",
    )
    no_ajoin = apply_action_ablation_mask(
        np.ones(len(ADAPT_LABELS), dtype=np.float32),
        "adapt",
        "no_ajoin",
    )
    assert [ADAPT_LABELS[index] for index in np.flatnonzero(no_filter)] == [
        "none",
        "ajoin_conservative",
    ]
    assert [ADAPT_LABELS[index] for index in np.flatnonzero(no_ajoin)] == [
        "none",
        "filter_selective",
    ]


def test_low_context_captures_execution_scope_and_enum_action() -> None:
    ctx = build_adapt_context(
        cumulative_ms=25,
        round_index=2,
        max_rounds=8,
        is_split_execution=True,
        enum_action="top5",
        enum_k=5,
    )

    assert ctx.shape == (ADAPT_CTX_DIM,)
    assert np.allclose(
        ctx[:DEC_CTX_DIM],
        build_dec_context(
            cumulative_ms=25,
            round_index=2,
            max_rounds=8,
        ),
    )
    assert ctx[DEC_CTX_DIM] == 1.0
    assert ctx[DEC_CTX_DIM + 1 :].tolist() == [0.0, 0.0, 1.0, 0.0]


def test_online_action_space_uses_selected_dataset_wide_mechanisms() -> None:
    assert ADAPT_LABELS == [
        "none",
        "filter_selective",
        "ajoin_conservative",
        "filter_selective+ajoin_conservative",
    ]
    assert N_ADAPT == 4
    assert _phase_action_index(
        "adapt",
        {
            "filter_action": "selective",
            "ajoin_action": "conservative",
        },
    ) == ADAPT_LABELS.index("filter_selective+ajoin_conservative")


def test_standalone_top10_history_maps_to_learned_topk_mechanism() -> None:
    assert _phase_action_index(
        "enum",
        {
            "enum_k": 10,
            "enum_action": "top10",
        },
    ) == ENUM_LABELS.index("top5")


def test_standalone_low_variants_map_to_learned_mechanisms() -> None:
    assert _phase_action_index(
        "adapt",
        {"filter_action": "full", "ajoin_action": "aggressive"},
    ) == ADAPT_LABELS.index("filter_selective+ajoin_conservative")
    assert _phase_action_index(
        "adapt",
        {"filter_action": "none", "ajoin_action": "aggressive"},
    ) == ADAPT_LABELS.index("ajoin_conservative")


def test_legacy_action_heads_are_remapped_by_semantic_index() -> None:
    model = HACNetwork(hidden=16)
    legacy = model.state_dict()
    legacy["sched_actor.bias"] = torch.arange(5, dtype=torch.float32)
    legacy["enum_actor.bias"] = torch.arange(4, dtype=torch.float32)
    legacy["adapt_actor.bias"] = torch.arange(9, dtype=torch.float32)
    legacy["adapt_cost_head.bias"] = torch.arange(9, dtype=torch.float32)

    migrated, keys = expand_legacy_checkpoint_tensors(model, legacy)

    assert migrated["sched_actor.bias"].tolist() == [0.0, 2.0, 4.0]
    assert migrated["enum_actor.bias"].tolist() == [0.0, 2.0]
    assert migrated["adapt_actor.bias"].tolist() == [0.0, 2.0, 6.0, 8.0]
    assert migrated["adapt_cost_head.bias"].tolist() == [0.0, 2.0, 6.0, 8.0]
    assert {
        "sched_actor.bias",
        "enum_actor.bias",
        "adapt_actor.bias",
        "adapt_cost_head.bias",
    }.issubset(keys)


def test_schedule_replay_can_preserve_shared_high_encoder() -> None:
    model = HACNetwork(32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    target = ReplayTarget(
        state=empty_structured_state(level="dec", ctx_dim=DEC_CTX_DIM),
        target=2,
        mask=np.ones(N_SCHED, dtype=np.float32),
        weight=1.0,
        phase="sched",
        state_hash="state",
        query_id="query",
        action_costs_ms={0: 2.0, 2: 1.0},
    )
    encoder_before = {
        key: value.detach().clone() for key, value in model.encoder.state_dict().items()
    }
    actor_before = model.sched_actor.weight.detach().clone()

    replay_policy_update(
        model,
        optimizer,
        [target],
        "sched",
        torch.device("cpu"),
        epochs=2,
        freeze_encoder=True,
    )

    assert all(
        torch.equal(encoder_before[key], value)
        for key, value in model.encoder.state_dict().items()
    )
    assert not torch.equal(actor_before, model.sched_actor.weight)


def test_downstream_replay_phases_preserve_shared_high_encoder() -> None:
    assert FROZEN_REPLAY_ENCODER_PHASES == {
        "sched",
        "enum",
        "adapt",
    }


def test_independent_adapt_prior_trains_plan_encoder() -> None:
    assert FROZEN_INDEPENDENT_PRIOR_ENCODER_PHASES == {
        "sched",
        "enum",
    }
    model = HACNetwork(32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    target = ReplayTarget(
        state=nonempty_adapt_state(),
        target=2,
        mask=np.ones(N_ADAPT, dtype=np.float32),
        weight=1.0,
        phase="adapt",
        state_hash="low-independent-prior-state",
        query_id="query",
        action_costs_ms={0: 2.0, 2: 1.0},
    )
    plan_before = {
        key: value.detach().clone()
        for key, value in model.encoder.plan_encoder.state_dict().items()
    }

    replay_policy_update(
        model,
        optimizer,
        [target],
        "adapt",
        torch.device("cpu"),
        epochs=2,
        freeze_encoder=("adapt" in FROZEN_INDEPENDENT_PRIOR_ENCODER_PHASES),
    )

    assert any(
        not torch.equal(plan_before[key], value)
        for key, value in model.encoder.plan_encoder.state_dict().items()
    )


def test_low_replay_trains_private_trunk_without_shared_plan_encoder() -> None:
    model = HACNetwork(32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    target = ReplayTarget(
        state=nonempty_adapt_state(),
        target=2,
        mask=np.ones(N_ADAPT, dtype=np.float32),
        weight=1.0,
        phase="adapt",
        state_hash="low-state",
        query_id="query",
        action_costs_ms={0: 2.0, 2: 1.0},
    )
    graph_before = {
        key: value.detach().clone()
        for key, value in model.encoder.graph_encoder.state_dict().items()
    }
    plan_before = {
        key: value.detach().clone()
        for key, value in model.encoder.plan_encoder.state_dict().items()
    }
    adapt_trunk_before = model.encoder.adapt_trunk[0].weight.detach().clone()

    replay_policy_update(
        model,
        optimizer,
        [target],
        "adapt",
        torch.device("cpu"),
        epochs=2,
        freeze_encoder=True,
    )

    assert all(
        torch.equal(graph_before[key], value)
        for key, value in model.encoder.graph_encoder.state_dict().items()
    )
    assert all(
        torch.equal(plan_before[key], value)
        for key, value in model.encoder.plan_encoder.state_dict().items()
    )
    assert not torch.equal(
        adapt_trunk_before,
        model.encoder.adapt_trunk[0].weight,
    )


def test_low_ppo_trains_private_trunk_without_shared_plan_encoder() -> None:
    model = HACNetwork(32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    transition = Transition(
        state=nonempty_adapt_state(),
        action=0,
        reward=-1.0,
        done=True,
        log_prob=0.0,
        value=0.0,
        mask=np.ones(N_ADAPT, dtype=np.float32),
    )
    graph_before = {
        key: value.detach().clone()
        for key, value in model.encoder.graph_encoder.state_dict().items()
    }
    plan_before = {
        key: value.detach().clone()
        for key, value in model.encoder.plan_encoder.state_dict().items()
    }
    adapt_trunk_before = model.encoder.adapt_trunk[0].weight.detach().clone()

    diagnostics = {}
    ppo_update(
        model,
        optimizer,
        [transition],
        "adapt",
        torch.device("cpu"),
        n_epochs=2,
        freeze_encoder=True,
        diagnostics=diagnostics,
    )

    assert all(
        torch.equal(graph_before[key], value)
        for key, value in model.encoder.graph_encoder.state_dict().items()
    )
    assert all(
        torch.equal(plan_before[key], value)
        for key, value in model.encoder.plan_encoder.state_dict().items()
    )
    assert not torch.equal(
        adapt_trunk_before,
        model.encoder.adapt_trunk[0].weight,
    )
    assert diagnostics["transitions"] == 1
    assert diagnostics["updates"] == 2
    assert diagnostics["reward_sum"] == -1.0
    assert diagnostics["entropy"] >= 0.0
    assert diagnostics["clip_fraction"] >= 0.0
    assert "explained_variance" in diagnostics


def test_high_replay_balances_runtime_importance_per_round_scope() -> None:
    targets = []
    for scope, round_fraction in (("root", 0.0), ("residual", 0.25)):
        for action, weight in (
            (0, 1.0),
            (1, 100.0 if scope == "root" else 1.0),
        ):
            state = empty_structured_state(level="dec", ctx_dim=DEC_CTX_DIM)
            state.ctx[1] = round_fraction
            targets.append(
                ReplayTarget(
                    state=state,
                    target=action,
                    mask=np.ones(2, dtype=np.float32),
                    weight=weight,
                    phase="dec",
                    state_hash=f"{scope}:{action}",
                    query_id=f"{scope}:{action}",
                    action_costs_ms={0: 2.0, 1: 1.0},
                )
            )

    weights = _balanced_replay_weights(
        targets,
        importance_power=0.5,
        balance_scopes=True,
    )

    for round_fraction in (0.0, 0.25):
        for action in (0, 1):
            total = sum(
                float(weights[index])
                for index, target in enumerate(targets)
                if target.target == action
                and math.isclose(float(target.state.ctx[1]), round_fraction)
            )
            assert math.isclose(total, 1.0)


def test_sched_cost_target_softens_near_equal_alpha_runtimes() -> None:
    target = ReplayTarget(
        state=empty_structured_state(level="dec", ctx_dim=DEC_CTX_DIM),
        target=0,
        mask=np.ones(N_SCHED, dtype=np.float32),
        weight=1.0,
        phase="sched",
        state_hash="state",
        query_id="query",
        action_costs_ms={
            0: 100.0,
            1: 101.0,
            2: 400.0,
        },
    )

    probabilities = _sched_cost_target(
        target,
        torch.device("cpu"),
        temperature=0.1,
    )

    assert probabilities[0] > probabilities[1] > probabilities[2]
    assert probabilities[1] > 0.2
    assert probabilities[2] < 1e-6
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0))


def test_action_cost_target_uses_relative_regret_and_ignores_unmeasured() -> None:
    target = ReplayTarget(
        state=empty_structured_state(level="dec", ctx_dim=DEC_CTX_DIM),
        target=1,
        mask=np.ones(4, dtype=np.float32),
        weight=1.0,
        phase="adapt",
        state_hash="state",
        query_id="query",
        action_costs_ms={0: 110.0, 1: 100.0, 3: 500.0},
    )

    probabilities = _action_cost_target(
        target,
        torch.device("cpu"),
        size=4,
        temperature=0.1,
    )

    assert probabilities[1] > probabilities[0] > probabilities[3]
    assert probabilities[2] < 1e-8
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0))


def test_action_cost_regression_target_preserves_runtime_magnitude() -> None:
    target = ReplayTarget(
        state=empty_structured_state(level="dec", ctx_dim=DEC_CTX_DIM),
        target=1,
        mask=np.ones(4, dtype=np.float32),
        weight=1.0,
        phase="adapt",
        state_hash="state",
        query_id="query",
        action_costs_ms={0: 200.0, 1: 100.0, 3: 1_000.0},
    )

    scores, measured = _action_cost_regression_target(
        target,
        torch.device("cpu"),
        size=4,
    )

    assert torch.isclose(scores[1], torch.tensor(0.0))
    assert torch.isclose(scores[0], torch.tensor(-math.log(2.0)))
    assert torch.isclose(scores[3], torch.tensor(-math.log(10.0)))
    assert measured.tolist() == [1.0, 1.0, 0.0, 1.0]


def test_runtime_relation_plan_uses_analyzed_temp_rows() -> None:
    plan = runtime_relation_plan(
        {
            "relations": [
                {
                    "relname": "temp1",
                    "alias": "temp1",
                    "estimated_rows": 77,
                    "pages": 2,
                }
            ]
        }
    )
    assert plan["Plan"]["Plans"][0]["Plan Rows"] == 77


def test_online_low_state_uses_only_selected_plan() -> None:
    builder = ExecutionStateBuilder("job", catalog=_test_catalog())
    state = builder.build(
        "adapt",
        "low-state",
        {
            "sql": "SELECT t.id FROM title AS t WHERE t.id = 1",
            "plan_json": {
                "Plan": {
                    "Node Type": "Seq Scan",
                    "Relation Name": "title",
                    "Alias": "t",
                    "Plan Rows": 1,
                    "Plan Width": 4,
                    "Startup Cost": 0,
                    "Total Cost": 1,
                }
            },
        },
    )

    assert state.query_graph.table_node_features.shape[0] == 1
    assert np.count_nonzero(state.query_graph.table_node_features) == 0
    assert not state.current_plan.is_sentinel


def test_online_high_state_excludes_baseline_plan_tree() -> None:
    builder = ExecutionStateBuilder("job", catalog=_test_catalog())
    state = builder.build(
        "dec",
        "high-state",
        {
            "sql": (
                "SELECT min(t.title) FROM title AS t "
                "JOIN movie_info AS mi ON mi.movie_id=t.id"
            ),
            "plan_json": {
                "Plan": {
                    "Node Type": "Hash Join",
                    "Plan Rows": 5,
                    "Plan Width": 8,
                    "Startup Cost": 1,
                    "Total Cost": 10,
                    "Plans": [
                        {
                            "Node Type": "Seq Scan",
                            "Relation Name": "title",
                            "Alias": "t",
                            "Plan Rows": 10,
                        },
                        {
                            "Node Type": "Seq Scan",
                            "Relation Name": "movie_info",
                            "Alias": "mi",
                            "Plan Rows": 20,
                        },
                    ],
                }
            },
        },
    )

    assert state.current_plan.is_sentinel
    assert len(state.ctx) == DEC_CTX_DIM


def test_query_graph_parser_handles_postgres_deparsed_aliases() -> None:
    graph = parse_query_graph(
        """
        SELECT min(t.title) AS movie
        FROM title t, movie_info AS mi
        WHERE t.id = mi.movie_id
          AND mi.info ~~ '%release%'::text
        """
    )
    assert graph.aliases == {"t": "title", "mi": "movie_info"}
    assert graph.edges == [("t", "mi", "id", "movie_id")]

    joined = parse_query_graph(
        """
        SELECT *
        FROM public.title t
        LEFT JOIN movie_keyword mk ON mk.movie_id = t.id
        JOIN keyword AS k ON k.id = mk.keyword_id
        """
    )
    assert joined.aliases == {
        "t": "title",
        "mk": "movie_keyword",
        "k": "keyword",
    }
    assert len(joined.edges) == 2


def test_high_query_graph_uses_baseline_scan_estimates() -> None:
    sql = "SELECT * FROM title t WHERE t.production_year = 2005"
    graph = parse_query_graph(sql)
    catalog = _test_catalog()
    query_graph, _stats = build_transfer_graph_state(
        sql,
        graph,
        catalog,
        plan_json={
            "Plan": {
                "Node Type": "Seq Scan",
                "Relation Name": "title",
                "Alias": "t",
                "Plan Rows": 3,
                "Plan Width": 64,
            }
        },
    )
    assert math.isclose(
        float(query_graph.table_node_features[0, 0]),
        math.log1p(3) / 18.0,
        rel_tol=1e-6,
    )


def test_high_query_graph_encodes_categorical_literal_frequency() -> None:
    catalog = _test_catalog()

    def literal_frequency(country_code: str) -> float:
        sql = (
            "SELECT * FROM company_name cn "
            f"WHERE (cn.country_code)::text ='{country_code}'::text"
        )
        graph = parse_query_graph(sql)
        query_graph, _stats = build_transfer_graph_state(sql, graph, catalog)
        return float(query_graph.column_node_features[0, COLUMN_IDX_CONST_POS])

    assert math.isclose(literal_frequency("[us]"), 0.36016667, rel_tol=1e-5)
    assert math.isclose(literal_frequency("[sm]"), 1.0 / 164.0, rel_tol=1e-5)


def test_high_replay_identity_ignores_unstable_baseline_plan_cost() -> None:
    first = {
        "request_type": "dec",
        "sql": "SELECT * FROM temp1",
        "remaining_splits": 2,
        "plan_total_cost": 100.0,
        "plan_json": {"node": "Hash Join", "total_cost": 100.0},
    }
    repeated = {
        **first,
        "remaining_splits": 0,
        "plan_total_cost": 130.0,
        "plan_json": {"node": "Hash Join", "total_cost": 130.0},
    }

    assert _replay_state_identity("dec", first) == _replay_state_identity(
        "dec", repeated
    )
    assert _replay_state_identity("adapt", first) != _replay_state_identity(
        "adapt", repeated
    )


def test_high_context_contains_only_round_features() -> None:
    context = build_dec_context(
        cumulative_ms=120.0,
        round_index=2,
        max_rounds=8,
    )
    assert len(context) == 2
    assert math.isclose(float(context[0]), math.log1p(120) / 12.0, rel_tol=1e-6)
    assert math.isclose(float(context[1]), 0.25, rel_tol=1e-6)


def test_collect_transitions_reads_stochastic_policy_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with ExperienceStore(Path(tmp) / "experience.sqlite") as store:
            _append_compact_execution(
                store,
                query_id="1a",
                runtime_ms=80.0,
                decisions=[
                    {
                        "round_index": 0,
                        "phase": "adapt",
                        "policy_version": "iter-0",
                        "action_probability": 0.25,
                        "log_probability": -1.386294,
                        "predicted_value": -0.5,
                        "action_mask": [True] * N_ADAPT,
                        "state": {
                            "request_type": "adapt",
                            "plan_json": {
                                "node": "SeqScan",
                                "rows": 10,
                                "total_cost": 5,
                            },
                        },
                        "action": {
                            "action_index": 0,
                            "inference_mode": "stochastic",
                            "filter_action": "none",
                            "ajoin_action": "off",
                            "temperature": 1.5,
                            "exploration_epsilon": 0.2,
                        },
                    }
                ],
            )

            buffers = collect_transitions(
                store,
                workload="job",
                policy_version="iter-0",
                reward_scale_ms=100.0,
                catalog=_test_catalog(),
            )
            assert len(buffers["adapt"]) == 1
            transition = buffers["adapt"][0]
            assert transition.action == 0
            assert transition.reward == -0.8
            assert transition.log_prob == -1.386294
            assert transition.temperature == 1.5
            assert transition.exploration_epsilon == 0.2


def test_mixed_policy_reconstructs_online_sampling_probability() -> None:
    logits = torch.tensor([[1.2, -0.4, 0.7]], dtype=torch.float32)
    masks = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)
    temperature = torch.tensor([1.5], dtype=torch.float32)
    epsilon = torch.tensor([0.2], dtype=torch.float32)

    dist = _mixed_masked_categorical(
        logits,
        masks,
        temperature,
        epsilon,
    )
    valid_logits = torch.tensor([1.2, 0.7]) / 1.5
    base = torch.softmax(valid_logits, dim=0)
    expected_action_0 = 0.8 * float(base[0]) + 0.2 * 0.5

    assert math.isclose(
        float(dist.probs[0, 0]),
        expected_action_0,
        rel_tol=1e-6,
    )
    assert float(dist.probs[0, 1]) == 0.0


def test_mixed_policy_reconstructs_coverage_guided_probability() -> None:
    logits = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    masks = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    coverage = torch.tensor([[0.1, 0.9]], dtype=torch.float32)
    dist = _mixed_masked_categorical(
        logits,
        masks,
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        torch.tensor([0.3]),
        coverage,
    )
    assert torch.allclose(dist.probs, torch.tensor([[0.38, 0.62]]), atol=1e-6)


def test_schedule_transitions_propagate_reward_across_split_rounds() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with ExperienceStore(Path(tmp) / "experience.sqlite") as store:
            decisions = []
            for round_index in range(2):
                decisions.append(
                    {
                        "round_index": round_index,
                        "phase": "sched",
                        "policy_version": "iter-0",
                        "action_mask": [True] * N_SCHED,
                        "state": {
                            "request_type": "sched",
                            "round": round_index,
                        },
                        "action": {
                            "action_index": 2,
                            "inference_mode": "stochastic",
                        },
                        "runtime_ms": 20.0,
                    }
                )
            _append_compact_execution(
                store,
                query_id="1a",
                runtime_ms=40.0,
                decisions=decisions,
            )

            transitions = collect_transitions(
                store,
                workload="job",
                policy_version="iter-0",
                catalog=_test_catalog(),
            )["sched"]
            assert [transition.done for transition in transitions] == [
                False,
                True,
            ]


def test_safe_initial_policy_prefers_postgres_equivalent_actions() -> None:
    model = HACNetwork(hidden=16)
    initialize_safe_policy(model, sched_alpha=0.5)

    assert int(model.dec_actor.bias.argmax()) == 0
    assert int(model.sched_actor.bias.argmax()) == 1
    assert int(model.enum_actor.bias.argmax()) == 0
    assert int(model.adapt_actor.bias.argmax()) == 0


def test_independent_action_initial_policy_profiles() -> None:
    expected = {
        "query_split": (1, 1, 0, 0),
        "top5": (0, 1, 1, 0),
        "lip_selective": (0, 1, 0, ADAPT_LABELS.index("filter_selective")),
        "aja_conservative": (
            0,
            1,
            0,
            ADAPT_LABELS.index("ajoin_conservative"),
        ),
    }
    for profile, actions in expected.items():
        model = HACNetwork(hidden=16)
        initialize_policy_profile(
            model,
            profile=profile,
            sched_alpha=0.5,
        )
        actual = (
            int(model.dec_actor.bias.argmax()),
            int(model.sched_actor.bias.argmax()),
            int(model.enum_actor.bias.argmax()),
            int(model.adapt_actor.bias.argmax()),
        )
        assert actual == actions


def test_independent_action_prior_uses_whitelist_and_root_default_state() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        summary = {}
        costs = {
            "query_split": {"1a": 50.0, "held_out": 1.0},
            "top5": {"1a": 120.0, "held_out": 1.0},
            "lip_selective": {"1a": 60.0, "held_out": 1.0},
            "aja_conservative": {"1a": 80.0, "held_out": 1.0},
        }
        for profile, values in costs.items():
            summary[profile] = {
                "queries": {
                    query_id: {"median_charged_ms": runtime_ms}
                    for query_id, runtime_ms in values.items()
                },
                "per_query_speedups": {
                    query_id: 100.0 / runtime_ms
                    for query_id, runtime_ms in values.items()
                },
            }
        summary_path = root / "independent.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        with ExperienceStore(root / "experience.sqlite") as store:

            def add_default_root(query_id: str) -> None:
                _append_compact_execution(
                    store,
                    query_id=query_id,
                    runtime_ms=100.0,
                    decisions=[
                        {
                            "round_index": 0,
                            "policy_version": "default",
                            "phase": "dec",
                            "action_mask": [True, True],
                            "state": {
                                "request_type": "dec",
                                "sql": "SELECT * FROM title AS t",
                                "round": 0,
                            },
                            "action": {"action_index": 0, "dec_action": "skip"},
                        },
                        {
                            "round_index": 0,
                            "policy_version": "default",
                            "phase": "enum",
                            "action_mask": [True, True],
                            "state": {
                                "request_type": "enum",
                                "sql": "SELECT * FROM title AS t",
                            },
                            "action": {"action_index": 0, "enum_action": "native"},
                        },
                        {
                            "round_index": 0,
                            "policy_version": "default",
                            "phase": "adapt",
                            "action_mask": [True] * N_ADAPT,
                            "state": {
                                "request_type": "adapt",
                                "plan_json": {
                                    "Plan": {
                                        "Node Type": "Seq Scan",
                                        "Relation Name": "title",
                                        "Alias": "t",
                                        "Plan Rows": 10,
                                    }
                                },
                            },
                            "action": {
                                "action_index": 0,
                                "filter_action": "none",
                                "ajoin_action": "off",
                            },
                        },
                    ],
                )

            add_default_root("1a")
            add_default_root("held_out")
            targets = collect_independent_action_targets(
                store,
                workload="job",
                summary_path=summary_path,
                reward_scale_ms=100.0,
                query_ids=["1a"],
                catalog=_test_catalog(),
            )

            assert [(target.query_id, target.target) for target in targets["dec"]] == [
                ("1a", 1)
            ]
            assert [target.target for target in targets["enum"]] == [0]
            assert [target.target for target in targets["adapt"]] == [
                ADAPT_LABELS.index("filter_selective")
            ]
            assert not targets["sched"]
            assert all(
                target.query_id != "held_out"
                for phase_targets in targets.values()
                for target in phase_targets
            )


def test_tpch_independent_prior_does_not_require_query_split_profile() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        summary = {
            profile: {
                "queries": {"1": {"median_charged_ms": 10.0}},
                "per_query_speedups": {"1": 1.0},
            }
            for profile in ("top5", "lip_selective", "aja_conservative")
        }
        summary_path = root / "independent.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        with ExperienceStore(root / "experience.sqlite") as store:
            targets = collect_independent_action_targets(
                store,
                workload="tpch",
                summary_path=summary_path,
                reward_scale_ms=10.0,
                query_ids=["1"],
                catalog=_test_catalog(),
            )
        assert targets == {
            "dec": [],
            "sched": [],
            "enum": [],
            "adapt": [],
        }


def test_residual_split_prior_is_train_only_and_deduplicated() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with ExperienceStore(Path(tmp) / "experience.sqlite") as store:
            for query_id in ("1a", "held_out"):
                _append_compact_execution(
                    store,
                    query_id=query_id,
                    runtime_ms=100.0,
                    decisions=[
                        {
                            "round_index": 1,
                            "phase": "dec",
                            "action_mask": [True, True],
                            "state": {
                                "request_type": "dec",
                                "sql": "SELECT * FROM title AS t",
                                "round": 1,
                                "cumulative_ms": 10.0,
                            },
                            "action": {
                                "action_index": 1,
                                "dec_action": "apply",
                            },
                            "policy_version": "query-split",
                        },
                        {
                            "round_index": 1,
                            "phase": "dec",
                            "action_mask": [True, True],
                            "state": {
                                "request_type": "dec",
                                "sql": "SELECT * FROM title AS t",
                                "round": 1,
                                "cumulative_ms": 10.0,
                            },
                            "action": {"action_index": 1, "dec_action": "apply"},
                            "policy_version": "query-split",
                        },
                    ],
                )

            targets = collect_residual_split_prior_targets(
                store,
                workload="job",
                query_ids=["1a"],
                catalog=_test_catalog(),
            )

            assert len(targets) == 1
            assert targets[0].query_id == "1a"
            assert targets[0].target == 1
            assert targets[0].mask.tolist() == [1.0, 1.0]


def test_joint_high_prior_supports_conservative_hard_root_labels() -> None:
    model = HACNetwork(hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    root_targets = [
        ReplayTarget(
            state=empty_structured_state(level="dec", ctx_dim=DEC_CTX_DIM),
            target=0,
            mask=np.ones(2, dtype=np.float32),
            weight=1.0,
            phase="dec",
            state_hash="root",
            query_id="1a",
            action_costs_ms={0: 10.0, 1: 20.0},
        )
    ]
    residual_targets = [
        ReplayTarget(
            state=empty_structured_state(level="dec", ctx_dim=DEC_CTX_DIM),
            target=1,
            mask=np.ones(2, dtype=np.float32),
            weight=1.0,
            phase="dec",
            state_hash="residual",
            query_id="1a",
            action_costs_ms={1: 1.0},
        )
    ]

    losses = joint_dec_prior_update(
        model,
        optimizer,
        root_targets,
        residual_targets,
        torch.device("cpu"),
        epochs=2,
        root_cost_regression=False,
    )

    assert math.isfinite(losses["root_cost_regression"])
    assert math.isfinite(losses["residual_split"])


def test_numpy_rng_checkpoint_state_is_portable() -> None:
    state = portable_numpy_rng_state()

    assert state["bit_generator"] == "MT19937"
    assert isinstance(state["keys"], list)
    assert len(state["keys"]) == 624
    assert isinstance(state["position"], int)
    assert isinstance(state["cached_gaussian"], float)


def test_runtime_replay_uses_query_whitelist_across_all_buffer_entries() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with ExperienceStore(Path(tmp) / "experience.sqlite") as store:

            def add_high_sample(
                query_id: str,
                runtime_ms: float,
            ) -> None:
                for action_index, measured_ms in ((0, 100.0), (1, runtime_ms)):
                    _append_compact_execution(
                        store,
                        query_id=query_id,
                        runtime_ms=measured_ms,
                        episode_id=f"episode-{query_id}-{action_index}",
                        decisions=[
                            {
                                "round_index": 0,
                                "phase": "dec",
                                "policy_version": "iter-0",
                                "action_mask": [True, True],
                                "state": {
                                    "request_type": "dec",
                                    "sql": f"SELECT * FROM {query_id}",
                                    "round": 0,
                                },
                                "action": {
                                    "action_index": action_index,
                                    "dec_action": (
                                        "stop" if action_index == 0 else "split"
                                    ),
                                    "inference_mode": "stochastic",
                                },
                            }
                        ],
                    )

            add_high_sample("fast_query", 50.0)
            add_high_sample("noisy_query", 97.0)
            add_high_sample("slow_query", 103.0)
            add_high_sample("validation_query", 1.0)
            add_high_sample("pooled_train_query", 40.0)
            add_high_sample("historical_test_query", 1.0)
            add_high_sample("current_test_query", 1.0)

            targets = collect_replay_targets(
                store,
                workload="job",
                cutoff_ms=store.now_ms(),
                reward_scale_ms=100.0,
                query_ids=(
                    "fast_query",
                    "historical_test_query",
                    "noisy_query",
                    "pooled_train_query",
                    "slow_query",
                    "validation_query",
                ),
                minimum_samples=1,
                catalog=_test_catalog(),
            )["dec"]

            assert {target.query_id: target.target for target in targets} == {
                "fast_query": 1,
                "historical_test_query": 1,
                "noisy_query": 1,
                "pooled_train_query": 1,
                "slow_query": 0,
                "validation_query": 1,
            }
            assert all(target.query_id != "current_test_query" for target in targets)
            weights = _balanced_replay_weights(
                targets,
                importance_power=0.0,
            )
            class_totals = {
                action: sum(
                    float(weights[index])
                    for index, target in enumerate(targets)
                    if target.target == action
                )
                for action in (0, 1)
            }
            assert math.isclose(
                class_totals[0],
                class_totals[1],
                rel_tol=1e-6,
            )


def test_runtime_replay_prefers_measured_default_over_virtual_pg() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with ExperienceStore(Path(tmp) / "experience.sqlite") as store:
            for action_index, runtime_ms in ((0, 120.0), (1, 96.0)):
                _append_compact_execution(
                    store,
                    query_id="measured_default",
                    runtime_ms=runtime_ms,
                    episode_id=f"episode-action-{action_index}",
                    decisions=[
                        {
                            "round_index": 0,
                            "phase": "dec",
                            "policy_version": "iter-0",
                            "action_mask": [True, True],
                            "state": {
                                "request_type": "dec",
                                "sql": "SELECT * FROM measured_default",
                                "round": 0,
                            },
                            "action": {
                                "action_index": action_index,
                                "dec_action": (
                                    "skip" if action_index == 0 else "apply"
                                ),
                                "inference_mode": "stochastic",
                            },
                        }
                    ],
                )

            targets = collect_replay_targets(
                store,
                workload="job",
                cutoff_ms=store.now_ms(),
                reward_scale_ms=100.0,
                query_ids=("measured_default",),
                minimum_samples=1,
                catalog=_test_catalog(),
            )["dec"]

            assert len(targets) == 1
            assert targets[0].action_costs_ms == {0: 120.0, 1: 96.0}
            assert targets[0].target == 1

            reused = collect_replay_targets(
                store,
                workload="job",
                cutoff_ms=store.now_ms(),
                reward_scale_ms=100.0,
                query_ids=("measured_default",),
                minimum_samples=1,
                catalog=_test_catalog(),
            )["dec"]
            assert len(reused) == 1
            assert reused[0].target == 1


def test_runtime_replay_deduplicates_complete_trajectory_labels() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with ExperienceStore(Path(tmp) / "experience.sqlite") as store:
            split_decision = {
                "round_index": 0,
                "phase": "dec",
                "action_mask": [True, True],
                "state": {
                    "request_type": "dec",
                    "sql": "SELECT * FROM shared_query",
                    "round": 0,
                },
                "action": {"action_index": 1, "dec_action": "apply"},
            }
            for index, runtime_ms in enumerate((40.0, 90.0), start=1):
                _append_compact_execution(
                    store,
                    query_id="shared_query",
                    runtime_ms=runtime_ms,
                    episode_id=f"episode-split-{index}",
                    created_at_ms=index,
                    decisions=[split_decision],
                )
            _append_compact_execution(
                store,
                query_id="shared_query",
                runtime_ms=100.0,
                episode_id="episode-default",
                created_at_ms=3,
                decisions=[
                    {
                        "round_index": 0,
                        "phase": "dec",
                        "action_mask": [True, True],
                        "state": {
                            "request_type": "dec",
                            "sql": "SELECT * FROM shared_query",
                            "round": 0,
                        },
                        "action": {
                            "action_index": 0,
                            "dec_action": "skip",
                        },
                    }
                ],
            )

            targets = collect_replay_targets(
                store,
                workload="job",
                cutoff_ms=store.now_ms(),
                reward_scale_ms=100.0,
                query_ids=("shared_query",),
                minimum_samples=1,
                catalog=_test_catalog(),
            )["dec"]

            assert len(targets) == 1
            assert targets[0].action_costs_ms == {0: 100.0, 1: 40.0}


def test_replay_does_not_attribute_downstream_search_to_high() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with ExperienceStore(Path(tmp) / "experience.sqlite") as store:
            for suffix, search_action, runtime_ms in (
                ("default", 0, 100.0),
                ("top5", 1, 20.0),
            ):
                decisions = []
                for phase, action_index, mask in (
                    ("dec", 0, [True, True]),
                    ("enum", search_action, [True, True]),
                    ("adapt", 0, [True] * N_ADAPT),
                ):
                    decisions.append(
                        {
                            "round_index": 0,
                            "phase": phase,
                            "action_mask": mask,
                            "state": {
                                "request_type": phase,
                                "sql": "SELECT * FROM hierarchy_query",
                                "round": 0,
                            },
                            "action": {"action_index": action_index},
                            "runtime_ms": runtime_ms,
                        }
                    )
                _append_compact_execution(
                    store,
                    query_id="hierarchy_query",
                    runtime_ms=runtime_ms,
                    episode_id=f"episode-{suffix}",
                    decisions=decisions,
                )

            targets = collect_replay_targets(
                store,
                workload="job",
                cutoff_ms=store.now_ms(),
                reward_scale_ms=100.0,
                query_ids=("hierarchy_query",),
                minimum_samples=1,
                catalog=_test_catalog(),
            )

            # Dec receives only the default downstream trajectory, so it has
            # no fabricated split-vs-stop counterfactual target.
            assert targets["dec"] == []
            assert len(targets["enum"]) == 1
            assert targets["enum"][0].target == 1
            assert targets["enum"][0].action_costs_ms == {
                0: 100.0,
                1: 20.0,
            }


def test_high_replay_backpropagates_non_greedy_prefix_value() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with ExperienceStore(Path(tmp) / "experience.sqlite") as store:

            def add_episode(
                suffix: str,
                rounds: list[tuple[str, int, float]],
            ) -> None:
                decisions = []
                for round_index, (state_name, action_index, runtime_ms) in enumerate(
                    rounds
                ):
                    decisions.append(
                        {
                            "round_index": round_index,
                            "phase": "dec",
                            "action_mask": [True, True],
                            "state": {
                                "request_type": "dec",
                                "sql": f"SELECT * FROM {state_name}",
                                "round": round_index,
                            },
                            "action": {
                                "action_index": action_index,
                                "dec_action": ("apply" if action_index else "skip"),
                            },
                            "runtime_ms": runtime_ms,
                        }
                    )
                _append_compact_execution(
                    store,
                    query_id="prefix_query",
                    runtime_ms=sum(runtime for _state, _action, runtime in rounds),
                    episode_id=f"episode-{suffix}",
                    decisions=decisions,
                )

            add_episode(
                "always-split",
                [
                    ("root", 1, 10.0),
                    ("residual", 1, 10.0),
                    ("final", 0, 1.0),
                ],
            )
            add_episode(
                "prefix-one",
                [("root", 1, 10.0), ("residual", 0, 100.0)],
            )
            add_episode("stop", [("root", 0, 80.0)])

            targets = collect_replay_targets(
                store,
                workload="job",
                cutoff_ms=store.now_ms(),
                reward_scale_ms=100.0,
                query_ids=("prefix_query",),
                minimum_samples=1,
                catalog=_test_catalog(),
            )["dec"]
            by_sql = {target.state.cache_key: target for target in targets}

            assert len(targets) == 2
            root = next(
                target
                for target in targets
                if target.action_costs_ms == {0: 80.0, 1: 21.0}
            )
            residual = next(
                target
                for target in targets
                if target.action_costs_ms == {0: 100.0, 1: 11.0}
            )
            assert root.target == 1
            assert residual.target == 1
            assert len(by_sql) == 2

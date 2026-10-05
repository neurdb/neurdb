#!/usr/bin/env python3
"""Joint policy training over execution experience from multiple workloads."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from experience.store import ExperienceStore
from model.encoders.query_graph import CatalogInfo
from model.policy.action_space import ADAPT_LABELS, ENUM_LABELS, SCHED_ALPHA_VALUES
from model.policy.hierarchical_actor_critic import ppo_update
from optimization.decomposition_eligibility import workload_supports_decomposition
from training.experience_trainer import (
    FROZEN_INDEPENDENT_PRIOR_ENCODER_PHASES,
    FROZEN_REPLAY_ENCODER_PHASES,
    PHASE_LEVEL,
    collect_independent_action_targets,
    collect_residual_split_prior_targets,
    collect_transitions,
    initialize_policy_profile,
    joint_dec_prior_update,
    load_model_checkpoint,
    portable_numpy_rng_state,
    replay_policy_update,
    save_checkpoint_atomic,
)


def load_sources(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("sources") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("source manifest must contain a non-empty sources list")
    result = []
    for row in rows:
        workload = str(row["workload"]).lower()
        if workload not in {"job", "stack", "tpch"}:
            raise ValueError(f"unsupported workload {workload!r}")
        item = dict(row)
        item["workload"] = workload
        item["experience_db"] = Path(item["experience_db"])
        item["catalog_path"] = Path(item["catalog_path"])
        if not item["catalog_path"].is_file():
            raise ValueError(f"catalog snapshot does not exist: {item['catalog_path']}")
        item["catalog"] = CatalogInfo(item["catalog_path"])
        item["training_query_ids"] = [
            str(value) for value in item["training_query_ids"]
        ]
        if item.get("independent_action_summary"):
            item["independent_action_summary"] = Path(
                item["independent_action_summary"]
            )
        result.append(item)
    return result


def restore_optimizer(
    model: torch.nn.Module,
    state: Any,
    *,
    learning_rate: float,
) -> tuple[torch.optim.Optimizer, bool]:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    if state is None:
        return optimizer, False
    try:
        optimizer.load_state_dict(state)
    except ValueError:
        return optimizer, False
    return optimizer, True


def action_counts(targets: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for target in targets:
        counts[str(target.target)] += 1
    return dict(sorted(counts.items()))


def mixed_pretrain(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sources: list[dict[str, Any]],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    combined = {phase: [] for phase in PHASE_LEVEL}
    residual = []
    source_counts: dict[str, dict[str, int]] = {}
    for source in sources:
        summary = source.get("independent_action_summary")
        if summary is None:
            raise ValueError(
                f"{source['workload']} source lacks independent_action_summary"
            )
        with ExperienceStore(source["experience_db"], read_only=True) as store:
            targets = collect_independent_action_targets(
                store,
                workload=source["workload"],
                summary_path=summary,
                reward_scale_ms=float(source["reward_scale_ms"]),
                query_ids=source["training_query_ids"],
                action_ablation="none",
                prior_mode="direct",
                crossfit_confidence_z=1.645,
                crossfit_trees=100,
                seed=args.seed,
                catalog=source["catalog"],
            )
            for phase in PHASE_LEVEL:
                combined[phase].extend(targets[phase])
            if workload_supports_decomposition(source["workload"]):
                residual.extend(
                    collect_residual_split_prior_targets(
                        store,
                        workload=source["workload"],
                        query_ids=source["training_query_ids"],
                        action_ablation="none",
                        catalog=source["catalog"],
                    )
                )
        source_counts[source["workload"]] = {
            phase: len(targets[phase]) for phase in PHASE_LEVEL
        }

    configured_epochs = {
        "dec": args.dec_prior_epochs,
        "sched": args.sched_prior_epochs,
        "enum": args.enum_prior_epochs,
        "adapt": args.adapt_prior_epochs,
    }
    losses = {}
    for phase, targets in combined.items():
        epochs = configured_epochs[phase]
        losses[phase] = replay_policy_update(
            model,
            optimizer,
            targets,
            PHASE_LEVEL[phase],
            device,
            epochs=epochs,
            batch_size=args.replay_batch_size,
            importance_power=args.replay_importance_power,
            freeze_encoder=(phase in FROZEN_INDEPENDENT_PRIOR_ENCODER_PHASES),
            sched_cost_temperature=0.1,
            action_cost_temperature=None,
            action_cost_regression=(phase != "sched"),
            balance_classes=False,
        )
    residual_losses = {"root_cost_regression": 0.0, "residual_split": 0.0}
    if args.residual_split_prior_epochs > 0 and residual:
        residual_losses = joint_dec_prior_update(
            model,
            optimizer,
            combined["dec"],
            residual,
            device,
            epochs=args.residual_split_prior_epochs,
            root_cost_regression=True,
        )
    return {
        "source_target_counts": source_counts,
        "combined_target_counts": {
            phase: len(targets) for phase, targets in combined.items()
        },
        "combined_action_counts": {
            phase: action_counts(targets) for phase, targets in combined.items()
        },
        "phase_epochs": configured_epochs,
        "losses": losses,
        "residual_target_count": len(residual),
        "residual_epochs": args.residual_split_prior_epochs,
        "residual_losses": residual_losses,
    }


def mixed_ppo_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sources: list[dict[str, Any]],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    combined = {phase: [] for phase in PHASE_LEVEL}
    source_counts: dict[str, dict[str, int]] = {}
    experience_cutoffs_ms: dict[str, int] = {}
    for source in sources:
        with ExperienceStore(source["experience_db"], read_only=True) as store:
            cutoff_ms = store.now_ms()
            buffers = collect_transitions(
                store,
                workload=source["workload"],
                policy_version=source["source_policy_version"],
                query_ids=source["training_query_ids"],
                cutoff_ms=cutoff_ms,
                require_stochastic=True,
                reward_scale_ms=float(source["reward_scale_ms"]),
                reward_clip=args.reward_clip,
                action_ablation="none",
                catalog=source["catalog"],
            )
            for phase in PHASE_LEVEL:
                combined[phase].extend(buffers[phase])
            experience_cutoffs_ms[source["workload"]] = cutoff_ms
            source_counts[source["workload"]] = {
                phase: len(buffers[phase]) for phase in PHASE_LEVEL
            }

    losses = {}
    diagnostics: dict[str, dict[str, Any]] = {phase: {} for phase in PHASE_LEVEL}
    for phase, transitions in combined.items():
        losses[phase] = ppo_update(
            model,
            optimizer,
            transitions,
            PHASE_LEVEL[phase],
            device,
            method="standardmdp_rl",
            n_epochs=args.ppo_epochs,
            batch_size=args.ppo_batch_size,
            ent_coef=args.entropy,
            gamma=args.gamma,
            lam=args.lambda_gae,
            freeze_encoder=(phase in FROZEN_REPLAY_ENCODER_PHASES),
            diagnostics=diagnostics[phase],
        )
    if not any(combined.values()):
        raise RuntimeError("mixed PPO found no eligible on-policy transitions")
    return {
        "source_transition_counts": source_counts,
        "combined_transition_counts": {
            phase: len(transitions) for phase, transitions in combined.items()
        },
        "losses": losses,
        "diagnostics": diagnostics,
        "experience_cutoffs_ms": experience_cutoffs_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pretrain", "ppo"), required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--policy-version", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-batch-size", type=int, default=64)
    parser.add_argument("--entropy", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda-gae", type=float, default=0.95)
    parser.add_argument("--reward-clip", type=float, default=30.0)
    parser.add_argument("--dec-prior-epochs", type=int, default=96)
    parser.add_argument("--sched-prior-epochs", type=int, default=0)
    parser.add_argument("--enum-prior-epochs", type=int, default=96)
    parser.add_argument("--adapt-prior-epochs", type=int, default=96)
    parser.add_argument("--residual-split-prior-epochs", type=int, default=8)
    parser.add_argument("--replay-batch-size", type=int, default=64)
    parser.add_argument("--replay-importance-power", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sources = load_sources(args.source_manifest)
    device = torch.device(args.device)
    model, prior_metadata, optimizer_state = load_model_checkpoint(
        args.base_checkpoint,
        hidden=args.hidden,
        device=device,
    )
    if args.base_checkpoint is None:
        initialize_policy_profile(
            model,
            profile="query_split",
            sched_alpha=0.5,
            action_bias=0.5,
        )
    optimizer, optimizer_restored = restore_optimizer(
        model,
        optimizer_state,
        learning_rate=args.lr,
    )

    if args.mode == "pretrain":
        update = mixed_pretrain(model, optimizer, sources, device, args)
    else:
        update = mixed_ppo_update(model, optimizer, sources, device, args)

    metadata = {
        **prior_metadata,
        "architecture_version": "online-four-head-v10-paper-actions",
        "iteration": args.iteration,
        "policy_version": args.policy_version,
        "workload": "mixed_job_stack_tpch",
        "source_workloads": [source["workload"] for source in sources],
        "hidden": args.hidden,
        "adapt_labels": ADAPT_LABELS,
        "enum_labels": ENUM_LABELS,
        "sched_alphas": SCHED_ALPHA_VALUES,
        "optimizer_state_restored": optimizer_restored,
        "mixed_update": {"mode": args.mode, **update},
    }
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metadata": metadata,
        "rng_state": {
            "python": random.getstate(),
            "numpy": portable_numpy_rng_state(),
            "torch": torch.get_rng_state(),
        },
    }
    save_checkpoint_atomic(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "checkpoint": str(args.output.resolve()),
                "policy_version": args.policy_version,
                "iteration": args.iteration,
                "mode": args.mode,
                "update": update,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

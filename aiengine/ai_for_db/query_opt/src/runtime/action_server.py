#!/usr/bin/env python3
"""
NQO hierarchical policy action server.

The DB-side NQO path calls this service across the paper's three stages:
  1. pre-planning: choose decomposition and schedule a candidate subquery
  2. planning: choose the plan-search action for the executable unit
  3. execution: choose filter and adaptive-join actions for the selected plan

The wire protocol uses the paper's Dec, Sched, Enum, Filter, and AJoin names.
The request body is JSON, and the response is line-oriented key=value fields
that the C code can parse:

    action=skip
    dec_action=skip
    stop=1

or, for an Enum request:

    action=enum
    enum_action=top5
    enum_k=5

or, for a Sched request:

    action=sched
    candidate_id=2
    sched_alpha=0.5
    selection_strategy=phi4

or, for an Adapt request:

    action=adapt
    ajoin_action=aggressive
    filter_action=full
    note=model: ...

Inference is layered:
  * --model-module path_or_module:callable lets a real controller plug in now.
  * --model-path loads the existing HRL checkpoint format when available.
  * without either, the server falls back to the deterministic stub policy.

Decision events can be appended to JSONL via --trajectory-log. DB-side timing
events are logged separately by the C path when nqo.trajectory_log is set.

Run from the repository root:
    PYTHONPATH=./src python3 -m runtime --host 127.0.0.1 --port 8088
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import importlib.util
import json
import math
import os
import signal
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from experience.collector import ExperienceCollector, default_data_dir
from optimization.action_vocabulary import (
    ADAPT_PHASE,
    DEC_PHASE,
    DECISION_PHASES,
    ENUM_PHASE,
    LEGACY_ACTION_ABLATIONS,
    SCHED_PHASE,
    adapt_label,
    canonical_ajoin_action,
    canonical_dec_action,
    canonical_enum_action,
    canonical_filter_action,
    canonical_phase,
    enum_action_to_strategy,
    normalize_policy_action,
    normalize_policy_state,
    split_adapt_label,
)
from optimization.decomposition_eligibility import workload_supports_decomposition
from optimization.query_compatibility import querysplit_compatible
from optimization.state import runtime_relation_plan

PLAN_NODE_NAME_TO_EXPLAIN = {
    "Agg": "Aggregate",
    "Append": "Append",
    "BitmapHeapScan": "Bitmap Heap Scan",
    "BitmapIndexScan": "Bitmap Index Scan",
    "CteScan": "CTE Scan",
    "FunctionScan": "Function Scan",
    "Gather": "Gather",
    "GatherMerge": "Gather Merge",
    "Group": "Group",
    "Hash": "Hash",
    "HashJoin": "Hash Join",
    "IndexOnlyScan": "Index Only Scan",
    "IndexScan": "Index Scan",
    "Limit": "Limit",
    "Material": "Materialize",
    "MergeAppend": "Merge Append",
    "MergeJoin": "Merge Join",
    "NestLoop": "Nested Loop",
    "Result": "Result",
    "SeqScan": "Seq Scan",
    "Sort": "Sort",
    "SubqueryScan": "Subquery Scan",
    "TidScan": "Tid Scan",
    "ValuesScan": "Values Scan",
}

TRAJECTORY_LOG_LOCK = threading.Lock()
COLLECTOR: ExperienceCollector | None = None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}", flush=True)


def _append_jsonl(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with TRAJECTORY_LOG_LOCK:
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _explain_node_name(name: Any) -> str:
    text = str(name or "Other")
    return PLAN_NODE_NAME_TO_EXPLAIN.get(text, text)


def _normalize_plan_json(plan: Any) -> Any:
    """Convert DB-side lightweight plan JSON to the plan encoder schema."""
    if plan is None:
        return None
    if not isinstance(plan, dict):
        return plan
    if "Plan" in plan:
        return {
            **plan,
            "Plan": _normalize_plan_json(plan.get("Plan")),
        }
    if "Node Type" in plan:
        out = dict(plan)
        if "Plans" in out and isinstance(out["Plans"], list):
            out["Plans"] = [_normalize_plan_json(child) for child in out["Plans"]]
        return out
    if "node" not in plan:
        return plan

    out = {
        "Node Type": _explain_node_name(plan.get("node")),
        "Plan Rows": float(plan.get("rows") or 0.0),
        "Startup Cost": float(plan.get("startup_cost") or 0.0),
        "Total Cost": float(plan.get("total_cost") or 0.0),
        "Plan Width": int(plan.get("width") or 0),
    }
    if plan.get("alias"):
        out["Alias"] = str(plan["alias"])
    children = plan.get("children") or []
    if isinstance(children, list) and children:
        out["Plans"] = [_normalize_plan_json(child) for child in children]
    return out


def _state_for_model(state: dict[str, Any]) -> dict[str, Any]:
    """Return a copy whose plan_json is in the schema used by the HRL code."""
    raw_plan = state.get("plan_json")
    if raw_plan is None:
        raw_plan = state.get("plan")
    if raw_plan is None:
        return state

    normalized = _normalize_plan_json(raw_plan)
    out = dict(state)
    out.setdefault("db_plan_json", raw_plan)
    out["plan_json"] = normalized
    return out


def _plan_contains_join(plan: Any, join_name: str | None = None) -> bool:
    if not isinstance(plan, dict):
        return False
    node_name = str(plan.get("Node Type") or plan.get("node") or "").lower()
    compact_name = node_name.replace(" ", "")
    is_join = "join" in node_name or compact_name in {"nestloop", "nestedloop"}
    if is_join and (join_name is None or join_name.lower() in compact_name):
        return True
    children = plan.get("Plans") or plan.get("children") or []
    return any(_plan_contains_join(child, join_name) for child in children)


class HierarchicalPolicyController:
    """Load the learned controller and map stage states to policy actions."""

    def __init__(
        self,
        *,
        model_module: str | None = None,
        model_path: str | None = None,
        model_method: str = "standardmdp_rl",
        model_hidden: int = 128,
        workload: str = "job",
        catalog_path: str | None = None,
        device: str = "cpu",
        nqo_src: str | None = None,
        inference_mode: str = "deterministic",
        temperature: float = 1.0,
        exploration_epsilon: float = 0.0,
        coverage_counts_path: str | None = None,
        coverage_mix: float = 0.0,
        coverage_power: float = 0.5,
        stochastic_heads: str | list[str] | tuple[str, ...] | None = None,
        sampling_seed: int = 42,
        policy_version: str | None = None,
        torch_threads: int = 1,
        action_ablation: str = "none",
        fixed_sched_alpha: float | None = None,
    ) -> None:
        self.model_module = model_module
        self.model_path = model_path
        self.model_method = model_method
        self.model_hidden = model_hidden
        self.workload = workload
        self.decomposition_enabled = workload_supports_decomposition(workload)
        self.catalog_path = catalog_path
        self.device_name = device
        self.nqo_src = nqo_src
        self.inference_mode = inference_mode.strip().lower()
        self.temperature = float(temperature)
        self.exploration_epsilon = float(exploration_epsilon)
        self.coverage_counts_path = coverage_counts_path
        self.coverage_mix = float(coverage_mix)
        self.coverage_power = float(coverage_power)
        self.coverage_counts: dict[str, dict[str, list[int]]] = {}
        self.sampling_seed = int(sampling_seed)
        valid_heads = set(DECISION_PHASES)
        if stochastic_heads is None:
            parsed_heads = valid_heads
        elif isinstance(stochastic_heads, str):
            parsed_heads = {
                canonical_phase(item)
                for item in stochastic_heads.split(",")
                if item.strip()
            }
        else:
            parsed_heads = {
                canonical_phase(item) for item in stochastic_heads if str(item).strip()
            }
        unknown_heads = parsed_heads - valid_heads
        if unknown_heads:
            raise ValueError(f"unknown stochastic heads: {sorted(unknown_heads)}")
        self.stochastic_heads = frozenset(parsed_heads)
        self.policy_version = policy_version
        self.torch_threads = int(torch_threads)
        self.action_ablation = LEGACY_ACTION_ABLATIONS.get(
            str(action_ablation).strip().lower(),
            str(action_ablation).strip().lower(),
        )
        self.fixed_sched_alpha = (
            None if fixed_sched_alpha is None else float(fixed_sched_alpha)
        )
        if self.action_ablation not in {
            "none",
            "no_dec",
            "no_enum",
            "no_filter",
            "no_ajoin",
        }:
            raise ValueError(f"unknown action ablation {self.action_ablation!r}")
        if (
            self.fixed_sched_alpha is not None
            and not 0.0 <= self.fixed_sched_alpha <= 1.0
        ):
            raise ValueError("fixed_sched_alpha must be in [0, 1]")
        if self.inference_mode not in {"deterministic", "stochastic"}:
            raise ValueError("inference_mode must be 'deterministic' or 'stochastic'")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be greater than zero")
        if not 0.0 <= self.exploration_epsilon < 1.0:
            raise ValueError("exploration_epsilon must be in [0, 1)")
        if not 0.0 <= self.coverage_mix < 1.0:
            raise ValueError("coverage_mix must be in [0, 1)")
        if self.coverage_power < 0.0:
            raise ValueError("coverage_power must be nonnegative")
        if self.coverage_mix > 0.0:
            if not self.coverage_counts_path:
                raise ValueError(
                    "coverage_counts_path is required when coverage_mix > 0"
                )
            coverage_path = Path(self.coverage_counts_path)
            payload = json.loads(coverage_path.read_text(encoding="utf-8"))
            if int(payload.get("schema_version") or 0) != 1:
                raise ValueError(f"unsupported coverage snapshot: {coverage_path}")
            counts = payload.get("counts") or {}
            if not isinstance(counts, dict):
                raise ValueError("coverage snapshot counts must be an object")
            self.coverage_counts = {
                canonical_phase(phase): value
                for phase, value in counts.items()
                if isinstance(value, dict)
            }
        if self.torch_threads < 1:
            raise ValueError("torch_threads must be at least 1")
        self.source = "stub"
        self._callable: Callable[[dict[str, Any]], Any] | None = None
        self._torch = None
        self._model = None
        self._device = None
        self._hrl = None
        self._transfer = None
        self._catalog = None
        self._sched_trained = False
        self.state_ablation = "none"
        self.checkpoint_metadata: dict[str, Any] = {}
        self._query_graph_cache: dict[str, Any] = {}
        self._plan_tree_cache: dict[str, Any] = {}
        # Structured encodings are pure functions of a fixed checkpoint and
        # the canonical model state.  Reuse exact matches across hierarchy
        # heads and repeated executions while still recomputing the actor
        # distribution (and therefore preserving stochastic sampling).
        self._encoded_state_cache: dict[Any, Any] = {}
        # Context (round/cumulative runtime) may change while the underlying
        # query graph or plan tree stays identical.  Cache those expensive
        # encoder outputs separately and always run the context-aware trunk.
        self._shared_embedding_cache: dict[Any, Any] = {}

        if model_module:
            self._load_callable(model_module)
        elif model_path:
            self._load_hrl_checkpoint(model_path)

    def _load_callable(self, spec: str) -> None:
        module_name, sep, attr = spec.partition(":")
        if not sep or not attr:
            raise ValueError("--model-module must be 'module_or_path:callable'")

        module_path = Path(module_name)
        if module_path.exists():
            import_name = f"nqo_online_policy_{abs(hash(str(module_path)))}"
            mod_spec = importlib.util.spec_from_file_location(import_name, module_path)
            if mod_spec is None or mod_spec.loader is None:
                raise ImportError(f"cannot import policy module from {module_path}")
            module = importlib.util.module_from_spec(mod_spec)
            mod_spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_name)

        fn = getattr(module, attr)
        if not callable(fn):
            raise TypeError(f"{spec} is not callable")
        self._callable = fn
        self.source = f"module:{spec}"
        if self.policy_version is None:
            self.policy_version = self.source
        log(f"loaded model callable {spec}")

    def _load_hrl_checkpoint(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            log(f"model checkpoint not found: {path}; using stub policy")
            return

        candidate_src = [self.nqo_src, os.environ.get("NQO_SRC")]
        for src in candidate_src:
            if src and Path(src).exists() and src not in sys.path:
                sys.path.insert(0, src)

        try:
            import torch  # type: ignore
            from model import HierarchicalActorCritic  # type: ignore
            from model.encoders import state as state_encoder  # type: ignore
            from model.encoders.query_graph import CatalogInfo  # type: ignore
            from model.policy import action_space as policy  # type: ignore
        except Exception as exc:  # noqa: BLE001
            log(f"failed to import HRL model code ({exc!r}); using stub policy")
            traceback.print_exc()
            return
        torch.set_num_threads(self.torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        if self.model_method != "standardmdp_rl":
            log(
                f"unsupported model method {self.model_method}; "
                "the NQO runtime requires standardmdp_rl"
            )
            return
        model = HierarchicalActorCritic(hidden=self.model_hidden)

        device = torch.device(
            self.device_name
            if self.device_name != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        try:
            # NQO checkpoints are generated by the local trainer and carry
            # policy metadata in addition to tensors.
            checkpoint = torch.load(
                path,
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(path, map_location=device)
        if (
            isinstance(checkpoint, dict)
            and "model_state" in checkpoint
            and isinstance(checkpoint["model_state"], dict)
        ):
            state_dict = checkpoint["model_state"]
            metadata = checkpoint.get("metadata") or {}
        elif (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
            and isinstance(checkpoint["model_state_dict"], dict)
        ):
            state_dict = checkpoint["model_state_dict"]
            metadata = checkpoint.get("metadata") or {}
        else:
            state_dict = checkpoint
            metadata = {}
        if not isinstance(state_dict, dict):
            raise TypeError(f"unsupported checkpoint payload in {path}")

        state_dict, migrated_tensors = policy.migrate_action_space_checkpoint_tensors(
            model, state_dict
        )
        if migrated_tensors:
            log(
                "migrated legacy action-space checkpoint tensors: "
                + ",".join(migrated_tensors)
            )
        incompatible = model.load_state_dict(state_dict, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = list(incompatible.missing_keys)
        if unexpected or missing:
            raise RuntimeError(
                "checkpoint architecture mismatch: "
                f"missing={missing} unexpected={unexpected}"
            )

        trained_heads = {
            canonical_phase(head) for head in (metadata.get("trained_heads") or [])
        }
        self._sched_trained = SCHED_PHASE in trained_heads
        self.state_ablation = (
            str(metadata.get("state_ablation") or "none").strip().lower()
        )
        if self.state_ablation not in {
            "none",
            "no_query_topology",
            "no_plan_topology",
        }:
            raise ValueError(
                f"unknown checkpoint state ablation {self.state_ablation!r}"
            )
        self.checkpoint_metadata = dict(metadata)
        if self.policy_version is None:
            self.policy_version = str(
                metadata.get("policy_version")
                or metadata.get("checkpoint_id")
                or path.stem
            )
        model.to(device)
        model.eval()

        if not self.catalog_path:
            raise ValueError(
                "learned inference requires --catalog-path with a "
                "database-derived catalog snapshot"
            )
        catalog = CatalogInfo(self.catalog_path)

        self._torch = torch
        self._model = model
        self._device = device
        self._hrl = policy
        self._transfer = state_encoder
        self._catalog = catalog
        self.source = f"checkpoint:{path}"
        self._warmup_hrl()
        torch.manual_seed(self.sampling_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.sampling_seed)
        log(f"loaded HRL checkpoint {path} method={self.model_method} device={device}")

    def _decomposition_allowed(self, state: dict[str, Any]) -> bool:
        if not self.decomposition_enabled:
            return False
        sql = str(state.get("sql") or state.get("original_sql") or "")
        return bool(sql) and querysplit_compatible(sql)

    def predict(self, state: dict[str, Any]) -> dict[str, Any]:
        state = _state_for_model(normalize_policy_state(state))
        if canonical_phase(
            state.get("request_type")
        ) == DEC_PHASE and not self._decomposition_allowed(state):
            return {
                "action": "skip",
                "stop": True,
                "dec_action": "skip",
                "order_decision": "only_cost",
                "note": "query is not QuerySplit SPJ-compatible",
                "model_source": self.source,
            }

        if self._callable is not None:
            raw = self._callable(state)
            if raw is None:
                return {}
            if not isinstance(raw, dict):
                raise TypeError("model callable must return a dict")
            raw = dict(raw)
            raw.setdefault("note", f"model callable {self.source}")
            raw.setdefault("model_source", self.source)
            raw.setdefault("policy_version", self.policy_version or self.source)
            return normalize_policy_action(
                raw,
                phase=state.get("request_type"),
            )

        if self._model is not None:
            return self._predict_hrl(state)

        return {}

    def _plan_tree(self, plan_json: Any):
        transfer = self._transfer
        if plan_json is None or self._catalog is None:
            return transfer.empty_plan_tree()
        cache_key = hashlib.sha256(
            json.dumps(
                plan_json,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        plan_tree = self._plan_tree_cache.get(cache_key)
        if plan_tree is None:
            try:
                plan_tree = transfer.plan_to_tree(
                    plan_json,
                    catalog=self._catalog,
                )
                if getattr(self, "state_ablation", "none") == "no_plan_topology":
                    plan_tree = transfer.flatten_plan_tree_topology(plan_tree)
            except Exception:
                plan_tree = transfer.empty_plan_tree()
            if len(self._plan_tree_cache) >= 512:
                self._plan_tree_cache.pop(next(iter(self._plan_tree_cache)))
            self._plan_tree_cache[cache_key] = plan_tree
        return plan_tree

    def _query_graph_state(self, state: dict[str, Any], level: str):
        level = canonical_phase(level)
        state = normalize_policy_state(state)
        transfer = self._transfer
        sql = state.get("sql") or state.get("original_sql") or ""
        plan_json = None
        if level != "dec":
            plan_json = (
                state.get("plan_json")
                or state.get("plan")
                or runtime_relation_plan(state)
            )
        if not sql or self._catalog is None:
            ctx_dim = transfer.DEC_CTX_DIM if level == "dec" else 0
            return transfer.empty_structured_state(level=level, ctx_dim=ctx_dim)

        cache_key = hashlib.sha256(
            json.dumps(
                {"sql": sql, "plan": plan_json},
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        qgraph = self._query_graph_cache.get(cache_key)
        if qgraph is None:
            try:
                graph = transfer.parse_query_graph(sql)
                qgraph, _stats = transfer.build_transfer_graph_state(
                    sql, graph, self._catalog, plan_json=plan_json
                )
                if getattr(self, "state_ablation", "none") == "no_query_topology":
                    qgraph = transfer.remove_query_graph_topology(qgraph)
            except Exception:
                ctx_dim = transfer.DEC_CTX_DIM if level == "dec" else 0
                return transfer.empty_structured_state(level=level, ctx_dim=ctx_dim)
            if len(self._query_graph_cache) >= 512:
                self._query_graph_cache.pop(next(iter(self._query_graph_cache)))
            self._query_graph_cache[cache_key] = qgraph

        ctx = transfer.np.zeros(
            transfer.DEC_CTX_DIM if level == "dec" else 0,
            dtype=transfer.np.float32,
        )
        if level == "dec":
            ctx = transfer.build_dec_context(
                cumulative_ms=float(state.get("cumulative_cost_ms") or 0.0),
                round_index=float(state.get("round") or 0.0),
                max_rounds=float(state.get("max_split_rounds") or 1.0),
            )
        return transfer.StructuredState(
            level=level,
            query_graph=qgraph,
            current_plan=transfer.empty_plan_tree(),
            ctx=ctx,
            cache_key=(
                "online",
                level,
                getattr(self, "state_ablation", "none"),
                hash(sql),
                tuple(ctx),
            ),
        )

    def _plan_state(self, state: dict[str, Any]):
        state = normalize_policy_state(state)
        transfer = self._transfer
        plan_json = state.get("plan_json") or state.get("plan")
        ctx = transfer.build_adapt_context(
            cumulative_ms=float(state.get("cumulative_cost_ms") or 0.0),
            round_index=float(state.get("round") or 0.0),
            max_rounds=float(state.get("max_split_rounds") or 1.0),
            is_split_execution=bool(state.get("is_split_execution", False)),
            enum_action=str(state.get("enum_action") or "native"),
            enum_k=int(state.get("enum_k") or 0),
        )
        state_key = hashlib.sha256(
            json.dumps(
                {
                    "plan": plan_json,
                    "context": ctx.tolist(),
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        plan_tree = self._plan_tree(plan_json)
        return transfer.StructuredState(
            level="adapt",
            query_graph=transfer.empty_query_graph_state(),
            current_plan=plan_tree,
            ctx=ctx,
            cache_key=(
                "online",
                "adapt",
                getattr(self, "state_ablation", "none"),
                state_key,
                tuple(ctx),
            ),
        )

    def _warmup_hrl(self) -> None:
        if self._model is None:
            return
        transfer = self._transfer
        model = self._model
        with self._torch.no_grad():
            dec = transfer.empty_structured_state(
                level="dec",
                ctx_dim=transfer.DEC_CTX_DIM,
            )
            dec_encoded = self._encode(dec)
            enum = transfer.empty_structured_state(level="enum", ctx_dim=0)
            enum_encoded = self._encode(enum)
            adapt = transfer.empty_structured_state(
                level="adapt",
                ctx_dim=transfer.ADAPT_CTX_DIM,
            )
            adapt_encoded = self._encode(adapt)
            model.dec_actor(dec_encoded)
            model.dec_critic(dec_encoded)
            sched_encoded = model.sched_features(dec_encoded)
            model.sched_actor(sched_encoded)
            model.sched_critic(sched_encoded)
            model.enum_actor(enum_encoded)
            model.enum_critic(enum_encoded)
            model.adapt_actor(adapt_encoded)
            model.adapt_critic(adapt_encoded)

    def _masked_action(
        self,
        logits,
        mask,
        phase: str,
        state: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        phase = canonical_phase(phase)
        torch = self._torch
        logits = logits.reshape(-1)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=self._device)
        masked_logits = logits / self.temperature + (mask_t - 1.0) * 1e9
        base_probs = torch.softmax(masked_logits, dim=-1)
        valid_probs = mask_t / mask_t.sum().clamp_min(1.0)
        policy_probs = (
            1.0 - self.exploration_epsilon
        ) * base_probs + self.exploration_epsilon * valid_probs
        stochastic = (
            self.inference_mode == "stochastic" and phase in self.stochastic_heads
        )
        coverage_state_hash = ""
        coverage_probs = valid_probs
        effective_coverage_mix = self.coverage_mix if stochastic else 0.0
        if effective_coverage_mix > 0.0:
            coverage_state_hash = self._hrl.coverage_state_hash(state or {})
            raw_counts = self.coverage_counts.get(phase, {}).get(
                coverage_state_hash, []
            )
            counts = torch.zeros_like(mask_t)
            for index, count in enumerate(raw_counts[: len(mask)]):
                counts[index] = max(float(count), 0.0)
            weights = torch.pow(counts + 1.0, -self.coverage_power) * mask_t
            coverage_probs = weights / weights.sum().clamp_min(1.0)
        mixed_probs = (
            1.0 - effective_coverage_mix
        ) * policy_probs + effective_coverage_mix * coverage_probs
        dist = torch.distributions.Categorical(probs=mixed_probs)
        if stochastic:
            stable_state = {
                key: value
                for key, value in (state or {}).items()
                if key
                not in {
                    "pid",
                    "run_id",
                    "relid",
                    "cumulative_cost_ms",
                    "plan_state_ms",
                }
            }
            seed_material = json.dumps(
                {
                    "seed": self.sampling_seed,
                    "policy": self.policy_version or self.source,
                    "phase": phase,
                    "state": stable_state,
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
            sample_seed = int.from_bytes(
                hashlib.sha256(seed_material).digest()[:8],
                "big",
            )
            generator = torch.Generator(device=self._device)
            generator.manual_seed(sample_seed)
            action_t = torch.multinomial(
                dist.probs,
                1,
                generator=generator,
            ).reshape(())
        else:
            action_t = masked_logits.argmax()
        action = int(action_t.item())
        metadata = {
            "action_index": action,
            "action_mask": [bool(value) for value in mask],
            "action_probability": float(dist.probs[action_t].item()),
            "log_probability": float(dist.log_prob(action_t).item()),
            "policy_entropy": float(dist.entropy().item()),
            "policy_version": self.policy_version or self.source,
            "inference_mode": "stochastic" if stochastic else "deterministic",
            "temperature": self.temperature,
            "exploration_epsilon": self.exploration_epsilon,
            "coverage_mix": effective_coverage_mix,
            "coverage_power": self.coverage_power,
            "coverage_probabilities": [
                float(value) for value in coverage_probs.detach().cpu().tolist()
            ],
            "coverage_state_hash": coverage_state_hash,
            "stochastic_heads": sorted(self.stochastic_heads),
            "sampling_seed": self.sampling_seed,
        }
        if phase == "dec" and len(mask) >= 2:
            metadata["dec_apply_probability"] = float(dist.probs[1].item())
        return action, metadata

    def _encode(self, structured_state):
        cache_key = getattr(structured_state, "cache_key", None)
        if cache_key is not None:
            try:
                cached = self._encoded_state_cache.get(cache_key)
            except TypeError:
                cache_key = None
            else:
                if cached is not None:
                    return cached
        with self._torch.no_grad():
            encoder = getattr(self._model, "encoder", None)
            level = getattr(structured_state, "level", None)
            can_reuse_shared = encoder is not None and callable(
                getattr(structured_state, "tensor", None)
            )
            if can_reuse_shared and level == "adapt":
                shared_key = ("plan", id(structured_state.current_plan))
                plan_embedding = self._shared_embedding_cache.get(shared_key)
                if plan_embedding is None:
                    plan_embedding = encoder.plan_encoder.encode_tree(
                        structured_state.current_plan,
                        self._device,
                    )
                    self._cache_shared_embedding(shared_key, plan_embedding)
                ctx = structured_state.tensor("ctx", structured_state.ctx, self._device)
                encoded = encoder.adapt_trunk(
                    self._torch.cat([plan_embedding, ctx], dim=0)
                )
            elif can_reuse_shared and level in {"dec", "enum"}:
                shared_key = ("graph", id(structured_state.query_graph))
                graph_embedding = self._shared_embedding_cache.get(shared_key)
                if graph_embedding is None:
                    graph_embedding = encoder.graph_encoder(
                        structured_state.query_graph,
                        self._device,
                    )
                    self._cache_shared_embedding(shared_key, graph_embedding)
                if level == "dec":
                    ctx = structured_state.tensor(
                        "ctx", structured_state.ctx, self._device
                    )
                    encoded = encoder.dec_trunk(
                        self._torch.cat([graph_embedding, ctx], dim=0)
                    )
                else:
                    encoded = encoder.enum_trunk(graph_embedding)
            else:
                encoded = self._model.encode_state_obj(
                    structured_state,
                    self._device,
                )
        if cache_key is not None:
            if len(self._encoded_state_cache) >= 2048:
                self._encoded_state_cache.pop(next(iter(self._encoded_state_cache)))
            self._encoded_state_cache[cache_key] = encoded
        return encoded

    def _cache_shared_embedding(self, key, embedding) -> None:
        if len(self._shared_embedding_cache) >= 2048:
            self._shared_embedding_cache.pop(next(iter(self._shared_embedding_cache)))
        self._shared_embedding_cache[key] = embedding

    def _predict_hrl(self, state: dict[str, Any]) -> dict[str, Any]:
        hrl = self._hrl
        model = self._model
        request_type = canonical_phase(state.get("request_type") or DEC_PHASE)
        base_rels = int(state.get("base_rels") or 0)
        remaining = int(state.get("remaining_splits") or 0)

        if request_type == DEC_PHASE:
            structured_state = self._query_graph_state(state, DEC_PHASE)
            apply_allowed = (
                self._decomposition_allowed(state) and base_rels > 2 and remaining > 0
            )
            mask = [1.0, 1.0 if apply_allowed else 0.0]
            mask = hrl.apply_action_ablation_mask(
                mask, DEC_PHASE, self.action_ablation
            ).tolist()
            with self._torch.no_grad():
                encoded = self._encode(structured_state)
                logits = model.dec_actor(encoded)
                predicted_value = float(model.dec_critic(encoded).squeeze().item())
                dec_idx, policy_meta = self._masked_action(
                    logits, mask, DEC_PHASE, state
                )
            dec_action = "apply" if dec_idx == 1 else "skip"
            return {
                "action": dec_action,
                "stop": dec_idx == 0,
                "order_decision": "only_cost",
                "dec_action": dec_action,
                "predicted_value": predicted_value,
                "model_source": self.source,
                "note": f"model Dec inference: dec={dec_idx}",
                **policy_meta,
            }

        if request_type == SCHED_PHASE:
            if not self._sched_trained and self.fixed_sched_alpha is None:
                # Legacy checkpoints keep the established phi4/alpha=0.5
                # scheduler instead of activating a randomly initialized head.
                return {}
            predicted_value = None
            policy_meta: dict[str, Any] = {}
            if self._sched_trained:
                structured_state = self._query_graph_state(state, DEC_PHASE)
                mask = [1.0] * len(hrl.SCHED_ALPHA_VALUES)
                mask = hrl.apply_action_ablation_mask(
                    mask, SCHED_PHASE, self.action_ablation
                ).tolist()
                with self._torch.no_grad():
                    encoded = self._encode(structured_state)
                    sched_encoded = model.sched_features(encoded)
                    logits = model.sched_actor(sched_encoded)
                    predicted_value = float(
                        model.sched_critic(sched_encoded).squeeze().item()
                    )
                    sched_idx, policy_meta = self._masked_action(
                        logits, mask, SCHED_PHASE, state
                    )
            if self.fixed_sched_alpha is None:
                alpha = hrl.SCHED_ALPHA_VALUES[sched_idx]
            else:
                alpha = self.fixed_sched_alpha
                sched_idx = min(
                    range(len(hrl.SCHED_ALPHA_VALUES)),
                    key=lambda index: abs(hrl.SCHED_ALPHA_VALUES[index] - alpha),
                )
            return {
                "action": SCHED_PHASE,
                "stop": False,
                "sched_idx": sched_idx,
                "sched_alpha": alpha,
                "selection_strategy": f"alpha_{alpha:.2f}",
                "predicted_value": predicted_value,
                "model_source": self.source,
                "note": f"model Sched inference: alpha={alpha:.2f}",
                **policy_meta,
            }

        if request_type == ENUM_PHASE:
            structured_state = self._query_graph_state(state, ENUM_PHASE)
            max_rels = int(state.get("search_max_rels") or 12)
            enum_feasible = 2 <= base_rels <= max_rels
            mask = [1.0] + [1.0 if enum_feasible else 0.0] * (len(hrl.ENUM_LABELS) - 1)
            mask = hrl.apply_action_ablation_mask(
                mask, ENUM_PHASE, self.action_ablation
            ).tolist()
            with self._torch.no_grad():
                encoded = self._encode(structured_state)
                logits = model.enum_actor(encoded)
                predicted_value = float(model.enum_critic(encoded).squeeze().item())
                enum_idx, policy_meta = self._masked_action(
                    logits, mask, ENUM_PHASE, state
                )
            enum_action = hrl.ENUM_LABELS[enum_idx]
            _, enum_k = enum_action_to_strategy(enum_action)
            return {
                "action": ENUM_PHASE,
                "stop": False,
                "enum_action": enum_action,
                "enum_k": enum_k,
                "predicted_value": predicted_value,
                "model_source": self.source,
                "note": f"model Enum inference: enum={enum_action}",
                **policy_meta,
            }

        if request_type == ADAPT_PHASE:
            structured_state = self._plan_state(state)
            plan_json = state.get("plan_json") or state.get("plan") or {}
            has_join = _plan_contains_join(plan_json)
            has_hash_join = _plan_contains_join(plan_json, "hashjoin")
            mask = []
            for label in hrl.ADAPT_LABELS:
                needs_filter = label.startswith("filter_")
                needs_ajoin = "ajoin" in label
                valid = (not needs_filter or has_join) and (
                    not needs_ajoin or has_hash_join
                )
                mask.append(1.0 if valid else 0.0)
            mask = hrl.apply_action_ablation_mask(
                mask, ADAPT_PHASE, self.action_ablation
            ).tolist()
            with self._torch.no_grad():
                encoded = self._encode(structured_state)
                logits = model.adapt_actor(encoded)
                predicted_value = float(model.adapt_critic(encoded).squeeze().item())
                adapt_idx, policy_meta = self._masked_action(
                    logits, mask, ADAPT_PHASE, state
                )
            adapt_action = hrl.ADAPT_LABELS[adapt_idx]
            filter_action, ajoin_action = split_adapt_label(adapt_action)
            return {
                "action": ADAPT_PHASE,
                "stop": False,
                "filter_action": filter_action,
                "ajoin_action": ajoin_action,
                "adapt_action": adapt_action,
                "predicted_value": predicted_value,
                "model_source": self.source,
                "note": f"model Adapt inference: adapt={adapt_action}",
                **policy_meta,
            }

        raise ValueError(f"unknown request_type={request_type!r}")


def _normalize_prediction(
    pred: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    return normalize_policy_action(pred, phase=phase)


def _decide_dec(state: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    base_rels = int(state.get("base_rels", 0))
    remaining = int(state.get("remaining_splits", 0))

    if pred:
        pred = _normalize_prediction(pred, phase=DEC_PHASE)
        raw_action = pred.get("dec_action") or pred.get("action")
        if raw_action is None:
            raw_action = "skip" if _truthy(pred.get("stop", True)) else "apply"
        dec_action = canonical_dec_action(raw_action)
        return {
            "action": dec_action,
            "stop": dec_action == "skip",
            "order_decision": pred.get("order_decision", "only_cost"),
            "dec_action": dec_action,
            "note": pred.get("note", "model Dec action"),
            "model_source": pred.get("model_source"),
        }

    can_apply = base_rels > 2 and remaining > 0
    dec_action = "apply" if can_apply else "skip"
    return {
        "action": dec_action,
        "stop": not can_apply,
        "order_decision": "only_cost",
        "dec_action": dec_action,
        "note": f"stub Dec: {dec_action}",
    }


def _decide_enum(state: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    if pred:
        pred = _normalize_prediction(pred, phase=ENUM_PHASE)
        enum_action = canonical_enum_action(
            pred.get("enum_action"),
            pred.get("enum_k"),
        )
        _, default_k = enum_action_to_strategy(enum_action)
        return {
            "action": ENUM_PHASE,
            "stop": False,
            "enum_action": enum_action,
            "enum_k": int(pred.get("enum_k") or default_k),
            "note": pred.get("note", "model Enum action"),
            "model_source": pred.get("model_source"),
        }
    return {
        "action": ENUM_PHASE,
        "stop": False,
        "enum_action": "native",
        "enum_k": 1,
        "note": "stub Enum: native",
    }


def _decide_sched(state: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    candidates = state.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Sched request requires a non-empty candidates list")

    pred = _normalize_prediction(pred, phase=SCHED_PHASE) if pred else {}

    valid_ids = {
        int(candidate.get("candidate_id", idx))
        for idx, candidate in enumerate(candidates)
        if isinstance(candidate, dict)
    }
    requested = pred.get("candidate_id") if pred else None
    if requested is not None:
        candidate_id = int(requested)
        if candidate_id not in valid_ids:
            raise ValueError(
                f"model selected unknown candidate_id={candidate_id}; "
                f"valid={sorted(valid_ids)}"
            )
        action = {
            "action": SCHED_PHASE,
            "stop": False,
            "candidate_id": candidate_id,
            "selection_strategy": pred.get("selection_strategy", "model"),
            "note": pred.get("note", "model subquery selection"),
            "model_source": pred.get("model_source"),
        }
        if pred.get("sched_alpha") is not None:
            alpha = float(pred["sched_alpha"])
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"sched_alpha must be in [0,1], got {alpha}")
            action["sched_alpha"] = alpha
        if pred.get("sched_idx") is not None:
            action["sched_idx"] = int(pred["sched_idx"])
        return action

    requested_alpha = pred.get("sched_alpha")
    if requested_alpha is not None:
        alpha = float(requested_alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"sched_alpha must be in [0,1], got {alpha}")

        def alpha_key(item: tuple[int, Any]) -> tuple[float, int]:
            idx, candidate = item
            if not isinstance(candidate, dict):
                return float("inf"), idx
            cost = max(float(candidate.get("plan_total_cost") or 0.0), 1e-12)
            rows = max(float(candidate.get("plan_rows") or 0.0), 1e-12)
            candidate_id = int(candidate.get("candidate_id", idx))
            # Compare in log space to avoid overflow from C^alpha*S^(1-alpha).
            score = alpha * math.log(cost) + (1.0 - alpha) * math.log(rows)
            return score, candidate_id

        selected_idx, selected = min(enumerate(candidates), key=alpha_key)
        candidate_id = int(
            selected.get("candidate_id", selected_idx)
            if isinstance(selected, dict)
            else selected_idx
        )
        return {
            "action": SCHED_PHASE,
            "stop": False,
            "candidate_id": candidate_id,
            "sched_idx": pred.get("sched_idx"),
            "sched_alpha": alpha,
            "selection_strategy": pred.get("selection_strategy", f"alpha_{alpha:.2f}"),
            "note": pred.get("note", f"model scheduler alpha={alpha:.2f}"),
            "model_source": pred.get("model_source"),
        }

    def phi4_key(item: tuple[int, Any]) -> tuple[float, int]:
        idx, candidate = item
        if not isinstance(candidate, dict):
            return float("inf"), idx
        cost = float(candidate.get("plan_total_cost") or float("inf"))
        rows = max(float(candidate.get("plan_rows") or 1.0), 1.0)
        candidate_id = int(candidate.get("candidate_id", idx))
        return cost * rows, candidate_id

    selected_idx, selected = min(enumerate(candidates), key=phi4_key)
    candidate_id = int(
        selected.get("candidate_id", selected_idx)
        if isinstance(selected, dict)
        else selected_idx
    )
    return {
        "action": SCHED_PHASE,
        "stop": False,
        "candidate_id": candidate_id,
        "selection_strategy": "phi4",
        "note": "stub SSA: phi4=min(cost*rows)",
    }


def _decide_adapt(state: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    if pred:
        pred = _normalize_prediction(pred, phase=ADAPT_PHASE)
        filter_action = canonical_filter_action(pred.get("filter_action"))
        ajoin_action = canonical_ajoin_action(pred.get("ajoin_action"))
        action = {
            "action": ADAPT_PHASE,
            "stop": False,
            "filter_action": filter_action,
            "ajoin_action": ajoin_action,
            "adapt_action": adapt_label(filter_action, ajoin_action),
            "note": pred.get("note", "model Adapt action"),
            "model_source": pred.get("model_source"),
        }
        static_join = ajoin_action in {"hashjoin", "nestloop", "mergejoin"}
        if static_join and pred.get("aja_hint"):
            action["aja_hint"] = pred["aja_hint"]
        if static_join and pred.get("join_method"):
            action["join_method"] = pred["join_method"]
        return action

    return {
        "action": ADAPT_PHASE,
        "stop": False,
        "filter_action": "none",
        "ajoin_action": "off",
        "adapt_action": "none",
        "note": "stub Adapt: none",
    }


CONTROLLER: HierarchicalPolicyController | None = None
CONTROLLER_CONFIG: dict[str, Any] = {}
POLICY_LOCK = threading.RLock()
REQUIRE_MODEL = False
TRAJECTORY_LOG: str | None = None


def model_predict(state: dict[str, Any]) -> dict[str, Any]:
    """Call the configured learned policy, returning normalized action fields."""
    with POLICY_LOCK:
        if CONTROLLER is None:
            return {}
        return CONTROLLER.predict(state)


def decide_action(state: dict[str, Any]) -> dict[str, Any]:
    state = normalize_policy_state(state)
    try:
        pred = model_predict(state) or {}
    except Exception as exc:  # noqa: BLE001
        log(f"model prediction failed: {exc!r}")
        traceback.print_exc()
        pred = {}

    request_type = canonical_phase(state.get("request_type") or DEC_PHASE)
    if request_type == DEC_PHASE:
        action = _decide_dec(state, pred)
    elif request_type == SCHED_PHASE:
        action = _decide_sched(state, pred)
    elif request_type == ENUM_PHASE:
        action = _decide_enum(state, pred)
    elif request_type == ADAPT_PHASE:
        action = _decide_adapt(state, pred)
    else:
        raise ValueError(f"unknown request_type={request_type!r}")
    with POLICY_LOCK:
        source = CONTROLLER.source if CONTROLLER else "stub"
    if not action.get("model_source"):
        action["model_source"] = source
    for key in (
        "action_index",
        "action_mask",
        "action_probability",
        "log_probability",
        "policy_entropy",
        "predicted_value",
        "policy_version",
        "inference_mode",
        "temperature",
        "exploration_epsilon",
        "coverage_mix",
        "coverage_power",
        "coverage_probabilities",
        "coverage_state_hash",
        "stochastic_heads",
        "sampling_seed",
        "dec_apply_probability",
        "sched_idx",
        "sched_alpha",
    ):
        if key in pred and key not in action:
            action[key] = pred[key]
    if pred:
        action.setdefault("policy_version", source)
        action.setdefault("action_probability", 1.0)
        action.setdefault("log_probability", 0.0)
    return action


def render_action(
    action: dict[str, Any],
    *,
    request_type: str | None = None,
) -> bytes:
    """Serialize a canonical action, adapting legacy DB wire names if needed."""
    wire_phase = str(request_type or "").strip().lower()
    wire_action = action.get("action", "none")
    legacy_fields: list[str] = []
    if wire_phase == "high":
        dec_action = canonical_dec_action(
            action.get("dec_action", action.get("action"))
        )
        wire_action = "split" if dec_action == "apply" else "stop"
    elif wire_phase == "select":
        wire_action = "select"
    elif wire_phase == "search":
        wire_action = "search"
        strategy, search_k = enum_action_to_strategy(
            action.get("enum_action"),
            action.get("enum_k"),
        )
        legacy_fields.extend((f"search_strategy={strategy}", f"search_k={search_k}"))
    elif wire_phase == "low":
        wire_action = "low"
        filter_action = canonical_filter_action(action.get("filter_action"))
        ajoin_action = canonical_ajoin_action(action.get("ajoin_action"))
        legacy_fields.extend(
            (
                f"lip_action={filter_action}",
                "execution_action="
                f"{('none' if ajoin_action == 'off' else ajoin_action)}",
            )
        )

    lines = [
        f"action={wire_action}",
        f"stop={1 if action.get('stop', True) else 0}",
    ]
    if action.get("order_decision"):
        lines.append(f"order_decision={action['order_decision']}")
    if action.get("candidate_id") is not None:
        lines.append(f"candidate_id={int(action['candidate_id'])}")
    if action.get("selection_strategy"):
        lines.append(f"selection_strategy={action['selection_strategy']}")
    if action.get("dec_action"):
        lines.append(f"dec_action={action['dec_action']}")
    if action.get("sched_alpha") is not None:
        lines.append(f"sched_alpha={float(action['sched_alpha'])}")
    if action.get("enum_action"):
        lines.append(f"enum_action={action['enum_action']}")
    if action.get("enum_k"):
        lines.append(f"enum_k={int(action['enum_k'])}")
    if action.get("ajoin_action"):
        lines.append(f"ajoin_action={action['ajoin_action']}")
    if action.get("filter_action"):
        lines.append(f"filter_action={action['filter_action']}")
    if action.get("aja_hint"):
        lines.append(f"aja_hint={action['aja_hint']}")
    if action.get("join_method"):
        lines.append(f"join_method={action['join_method']}")
    lines.extend(legacy_fields)
    note = action.get("note")
    if note:
        lines.append(f"note={note}")
    return ("\n".join(lines) + "\n").encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def _respond(self, body: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _handle_reload(self, request: dict[str, Any]) -> None:
        global CONTROLLER, CONTROLLER_CONFIG

        started = time.perf_counter()
        new_config = dict(CONTROLLER_CONFIG)
        for key in (
            "model_module",
            "model_path",
            "model_method",
            "model_hidden",
            "workload",
            "catalog_path",
            "device",
            "nqo_src",
            "inference_mode",
            "temperature",
            "exploration_epsilon",
            "coverage_counts_path",
            "coverage_mix",
            "coverage_power",
            "stochastic_heads",
            "sampling_seed",
            "policy_version",
            "torch_threads",
            "action_ablation",
            "fixed_sched_alpha",
        ):
            if key in request and request[key] is not None:
                new_config[key] = request[key]

        if (
            COLLECTOR is not None
            and str(new_config["workload"]).lower() != COLLECTOR.dataset
        ):
            self._respond(
                b"reloaded=0\nnote=restart the collector to change datasets\n", 400
            )
            return

        try:
            controller = HierarchicalPolicyController(**new_config)
        except Exception as exc:  # noqa: BLE001
            log(f"policy reload failed: {exc!r}")
            traceback.print_exc()
            self._respond(f"reloaded=0\nnote={exc!r}\n".encode("utf-8"), 500)
            return

        if REQUIRE_MODEL and controller.source == "stub":
            note = "required model was not loaded during reload"
            log(note)
            self._respond(f"reloaded=0\nnote={note}\n".encode("utf-8"), 500)
            return

        with POLICY_LOCK:
            CONTROLLER = controller
            CONTROLLER_CONFIG = new_config

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        payload = {
            "ts": _now_iso(),
            "phase": "policy_reload",
            "latency_ms": elapsed_ms,
            "request": request,
            "model_source": controller.source,
        }
        _append_jsonl(TRAJECTORY_LOG, payload)
        log(f"policy reloaded source={controller.source} latency_ms={elapsed_ms:.2f}")
        self._respond(
            (
                "reloaded=1\n"
                f"model_source={controller.source}\n"
                f"latency_ms={elapsed_ms:.2f}\n"
            ).encode("utf-8")
        )

    def do_POST(self):
        t0 = time.perf_counter()
        if self.path.startswith("/shutdown"):
            self._respond(b"shutdown=1\n")
            threading.Thread(
                target=self.server.shutdown,
                name="nqo-server-shutdown",
                daemon=True,
            ).start()
            return
        try:
            raw_state = self._read_json_body()
            wire_request_type = str(raw_state.get("request_type") or "")
            state = normalize_policy_state(raw_state)
        except Exception as exc:  # noqa: BLE001
            log(f"bad request body: {exc!r}")
            self._respond(b"action=none\nstop=1\nnote=bad request\n", 400)
            return

        if self.path.startswith("/reload"):
            self._handle_reload(state)
            return

        try:
            action = decide_action(state)
        except ValueError as exc:
            log(f"invalid policy request: {exc}")
            self._respond(
                ("action=none\nstop=1\nerror=invalid_request\n" f"note={exc}\n").encode(
                    "utf-8"
                ),
                400,
            )
            return
        except Exception as exc:  # noqa: BLE001
            log(f"policy request failed: {exc!r}")
            traceback.print_exc()
            self._respond(
                b"action=none\nstop=1\nerror=server_error\n"
                b"note=policy request failed\n",
                500,
            )
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        body = render_action(action, request_type=wire_request_type)
        log(
            f"request={state.get('request_type', DEC_PHASE)} "
            f"run={state.get('run_id')} round={state.get('round')} "
            f"source={action.get('model_source')} action={action['action']} "
            f"stop={action['stop']} enum={action.get('enum_action')} "
            f"k={action.get('enum_k')} ajoin={action.get('ajoin_action')} "
            f"filter={action.get('filter_action')} aja_hint={action.get('aja_hint')} "
            f"latency_ms={elapsed_ms:.2f}"
        )
        _append_jsonl(
            TRAJECTORY_LOG,
            {
                "ts": _now_iso(),
                "phase": "policy_decision",
                "latency_ms": elapsed_ms,
                "state": state,
                "action": action,
            },
        )
        self._respond(body)

    def do_GET(self):
        with POLICY_LOCK:
            source = CONTROLLER.source if CONTROLLER else "stub"
        collector_status = "disabled"
        if COLLECTOR is not None:
            collector_status = (
                "error"
                if COLLECTOR.last_error or not COLLECTOR.is_alive()
                else "running"
            )
        self._respond(
            (
                f"action=none\nstop=1\nnote=health ok\nmodel_source={source}\n"
                f"collector={collector_status}\n"
            ).encode("utf-8")
        )

    def log_message(self, *args):  # silence default per-request stderr noise
        pass


def main() -> int:
    global CONTROLLER, CONTROLLER_CONFIG, REQUIRE_MODEL, TRAJECTORY_LOG, COLLECTOR

    ap = argparse.ArgumentParser(description="NQO hierarchical action server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--model-module", default=os.environ.get("NQO_MODEL_MODULE"))
    ap.add_argument("--model-path", default=os.environ.get("NQO_MODEL_PATH"))
    ap.add_argument(
        "--model-method",
        default=os.environ.get("NQO_MODEL_METHOD", "standardmdp_rl"),
    )
    ap.add_argument(
        "--model-hidden",
        type=int,
        default=int(os.environ.get("NQO_MODEL_HIDDEN", "128")),
    )
    ap.add_argument("--workload", default=os.environ.get("NQO_WORKLOAD", "job"))
    ap.add_argument(
        "--catalog-path",
        default=os.environ.get("NQO_CATALOG_PATH"),
        help="database-derived catalog snapshot created at run startup",
    )
    ap.add_argument("--device", default=os.environ.get("NQO_DEVICE", "cpu"))
    ap.add_argument("--nqo-src", default=os.environ.get("NQO_SRC"))
    ap.add_argument(
        "--inference-mode",
        choices=("deterministic", "stochastic"),
        default=os.environ.get("NQO_INFERENCE_MODE", "deterministic"),
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("NQO_TEMPERATURE", "1.0")),
    )
    ap.add_argument(
        "--exploration-epsilon",
        type=float,
        default=float(os.environ.get("NQO_EXPLORATION_EPSILON", "0.0")),
    )
    ap.add_argument(
        "--coverage-counts-path",
        default=os.environ.get("NQO_COVERAGE_COUNTS_PATH"),
    )
    ap.add_argument(
        "--coverage-mix",
        type=float,
        default=float(os.environ.get("NQO_COVERAGE_MIX", "0.0")),
    )
    ap.add_argument(
        "--coverage-power",
        type=float,
        default=float(os.environ.get("NQO_COVERAGE_POWER", "0.5")),
    )
    ap.add_argument(
        "--stochastic-heads",
        default=os.environ.get("NQO_STOCHASTIC_HEADS"),
        help=(
            "comma-separated policy phases sampled stochastically; "
            "other phases use deterministic argmax"
        ),
    )
    ap.add_argument(
        "--sampling-seed",
        type=int,
        default=int(os.environ.get("NQO_SAMPLING_SEED", "42")),
    )
    ap.add_argument(
        "--policy-version",
        default=os.environ.get("NQO_POLICY_VERSION"),
    )
    ap.add_argument(
        "--torch-threads",
        type=int,
        default=int(os.environ.get("NQO_TORCH_THREADS", "1")),
    )
    ap.add_argument(
        "--action-ablation",
        choices=("none", "no_dec", "no_enum", "no_filter", "no_ajoin"),
        default=os.environ.get("NQO_ACTION_ABLATION", "none"),
    )
    ap.add_argument(
        "--fixed-sched-alpha",
        type=float,
        default=(
            float(os.environ["NQO_FIXED_SCHED_ALPHA"])
            if os.environ.get("NQO_FIXED_SCHED_ALPHA") is not None
            else (
                float(os.environ["NQO_FIXED_SCHEDULE_ALPHA"])
                if os.environ.get("NQO_FIXED_SCHEDULE_ALPHA") is not None
                else None
            )
        ),
    )
    ap.add_argument("--trajectory-log", default=os.environ.get("NQO_TRAJECTORY_LOG"))
    ap.add_argument(
        "--collect-experience",
        action="store_true",
        help="enable the log collector, never training",
    )
    ap.add_argument("--data-dir", type=Path, default=default_data_dir())
    ap.add_argument(
        "--experience-database", help="exact PostgreSQL database name to collect"
    )
    ap.add_argument(
        "--db-trajectory-log",
        type=Path,
        help="PG-written JSONL path visible to this server",
    )
    ap.add_argument("--collector-interval", type=float, default=2.0)
    ap.add_argument(
        "--require-model",
        action="store_true",
        default=_truthy(os.environ.get("NQO_REQUIRE_MODEL")),
        help="fail startup if --model-module/--model-path cannot be loaded",
    )
    args = ap.parse_args()

    collector = None
    if args.collect_experience:
        if not args.experience_database:
            ap.error("--collect-experience requires --experience-database")
        dataset = args.workload.strip().lower()
        if not dataset or not all(c.isalnum() or c in "_-" for c in dataset):
            ap.error("workload must be a simple dataset name")
        data_dir = args.data_dir.expanduser().resolve()
        args.trajectory_log = args.trajectory_log or str(
            data_dir / "logs" / f"{dataset}.policy.jsonl"
        )
        db_log = args.db_trajectory_log or data_dir / "logs" / f"{dataset}.db.jsonl"
        try:
            collector = ExperienceCollector(
                dataset=dataset,
                database=args.experience_database,
                policy_log=Path(args.trajectory_log),
                db_log=db_log,
                buffer=data_dir / "experience" / f"{dataset}.sqlite",
                checkpoint=data_dir / "collector" / f"{dataset}.json",
                bootstrap=data_dir / "bootstrap" / f"{dataset}.sqlite",
                interval=args.collector_interval,
                log=log,
            )
        except ValueError as exc:
            ap.error(str(exc))
        Path(args.trajectory_log).parent.mkdir(parents=True, exist_ok=True)
        db_log.parent.mkdir(parents=True, exist_ok=True)

    REQUIRE_MODEL = bool(args.require_model)
    TRAJECTORY_LOG = args.trajectory_log
    CONTROLLER_CONFIG = {
        "model_module": args.model_module,
        "model_path": args.model_path,
        "model_method": args.model_method,
        "model_hidden": args.model_hidden,
        "workload": args.workload,
        "catalog_path": args.catalog_path,
        "device": args.device,
        "nqo_src": args.nqo_src,
        "inference_mode": args.inference_mode,
        "temperature": args.temperature,
        "exploration_epsilon": args.exploration_epsilon,
        "coverage_counts_path": args.coverage_counts_path,
        "coverage_mix": args.coverage_mix,
        "coverage_power": args.coverage_power,
        "stochastic_heads": args.stochastic_heads,
        "sampling_seed": args.sampling_seed,
        "policy_version": args.policy_version,
        "torch_threads": args.torch_threads,
        "action_ablation": args.action_ablation,
        "fixed_sched_alpha": args.fixed_sched_alpha,
    }
    try:
        CONTROLLER = HierarchicalPolicyController(**CONTROLLER_CONFIG)
    except Exception as exc:  # noqa: BLE001
        if args.require_model:
            log(f"policy controller initialization failed: {exc!r}")
            traceback.print_exc()
            return 2
        log(f"policy controller initialization failed: {exc!r}; using stub policy")
        traceback.print_exc()
        CONTROLLER = HierarchicalPolicyController()

    if args.require_model:
        requested_model = bool(args.model_module or args.model_path)
        if not requested_model or CONTROLLER.source == "stub":
            log(
                "required model was not loaded; pass --model-module or a valid "
                "--model-path with --nqo-src"
            )
            return 2

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    if collector is not None:
        COLLECTOR = collector
        collector.start()
        if not collector.ready.wait(timeout=30) or collector.startup_error:
            log(
                f"collector initialization failed: {collector.startup_error or 'startup timeout'}"
            )
            collector.stop()
            srv.server_close()
            return 2
        log(
            f"collector enabled buffer={collector.buffer} db_log={collector.paths['db']} interval={collector.interval}s"
        )
    log(
        f"NQO hierarchical action server listening on http://{args.host}:{args.port}/action "
        f"source={CONTROLLER.source} trajectory_log={TRAJECTORY_LOG or 'off'}"
    )
    try:

        def stop_on_signal(_signum, _frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop_on_signal)
        srv.serve_forever()
    except KeyboardInterrupt:
        log("shutting down")
    finally:
        srv.server_close()
        if collector is not None:
            collector.stop()
            COLLECTOR = None
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
NeurQO AI action server.

The DB-side NeurQO path calls this service sequentially:
  1. high policy: choose split or stop from the current residual query
  2. select policy: after split, choose one QSA candidate subquery
  3. search policy: choose search for the selected execution query
  4. low policy: choose LIP/AJA from the plan produced by search

The wire protocol remains intentionally simple. The request body is JSON, and
the response is line-oriented key=value fields that the C code can parse:

    action=stop
    stop=1

or, for a search request:

    action=search
    search_strategy=topk
    search_k=5

or, for a subquery-selection request:

    action=select
    candidate_id=2
    selection_strategy=phi4

or, for a low request:

    action=low
    execution_action=aggressive
    lip_action=full
    note=model: ...

Inference is layered:
  * --model-module path_or_module:callable lets a real controller plug in now.
  * --model-path loads the existing HRL checkpoint format when available.
  * without either, the server falls back to the deterministic stub policy.

Decision events can be appended to JSONL via --trajectory-log. DB-side timing
events are logged separately by the C path when neurqo.trajectory_log is set.

Run inside the container so the DB can reach it on localhost:
    python3 /code/neurdb-dev/neurqo/server/ai_server.py --host 127.0.0.1 --port 8088
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
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

ACTION_LABEL_TO_LIP_AJA = {
    "none": ("none", "none"),
    "lip_full": ("full", "none"),
    "lip_sel": ("selective", "none"),
    "aja": ("none", "aggressive"),
    "lip_full+aja": ("full", "aggressive"),
    "lip_sel+aja": ("selective", "aggressive"),
    "aja_conservative": ("none", "conservative"),
    "lip_full+aja_conservative": ("full", "conservative"),
    "lip_sel+aja_conservative": ("selective", "conservative"),
}

SEARCH_LABEL_TO_DB = {
    "default": ("default", 1),
    "none": ("default", 1),
    "split": ("topk", 1),
    "top5": ("topk", 5),
    "top10": ("topk", 10),
    "topk": ("topk", 5),
    "left_deep": ("left_deep", 1),
}

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
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _explain_node_name(name: Any) -> str:
    text = str(name or "Other")
    return PLAN_NODE_NAME_TO_EXPLAIN.get(text, text)


def _normalize_plan_json(plan: Any) -> Any:
    """Convert DB-side lightweight plan JSON into transfer_state's EXPLAIN schema."""
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


def _runtime_relation_plan(state: dict[str, Any]) -> dict[str, Any] | None:
    """Expose live catalog estimates, including analyzed temporary tables."""
    relations = state.get("relations") or []
    children = []
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        rows = max(float(relation.get("estimated_rows") or 0.0), 0.0)
        children.append(
            {
                "Node Type": "Seq Scan",
                "Relation Name": relation.get("relname"),
                "Alias": relation.get("alias"),
                "Plan Rows": rows,
                "Plan Width": 0,
                "Startup Cost": 0.0,
                "Total Cost": float(relation.get("pages") or 0.0),
            }
        )
    if not children:
        return None
    return {
        "Plan": {
            "Node Type": "Append",
            "Plan Rows": sum(child["Plan Rows"] for child in children),
            "Plan Width": 0,
            "Startup Cost": 0.0,
            "Total Cost": sum(child["Total Cost"] for child in children),
            "Plans": children,
        }
    }


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


class PolicyAdapter:
    """Optional learned-controller adapter with deterministic fallback."""

    def __init__(
        self,
        *,
        model_module: str | None = None,
        model_path: str | None = None,
        model_method: str = "standardmdp_rl",
        model_hidden: int = 128,
        workload: str = "job",
        device: str = "cpu",
        neurqo_src: str | None = None,
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
    ) -> None:
        self.model_module = model_module
        self.model_path = model_path
        self.model_method = model_method
        self.model_hidden = model_hidden
        self.workload = workload
        self.device_name = device
        self.neurqo_src = neurqo_src
        self.inference_mode = inference_mode.strip().lower()
        self.temperature = float(temperature)
        self.exploration_epsilon = float(exploration_epsilon)
        self.coverage_counts_path = coverage_counts_path
        self.coverage_mix = float(coverage_mix)
        self.coverage_power = float(coverage_power)
        self.coverage_counts: dict[str, dict[str, list[int]]] = {}
        self.sampling_seed = int(sampling_seed)
        valid_heads = {"high", "select", "search", "low"}
        if stochastic_heads is None:
            parsed_heads = valid_heads
        elif isinstance(stochastic_heads, str):
            parsed_heads = {
                item.strip().lower()
                for item in stochastic_heads.split(",")
                if item.strip()
            }
        else:
            parsed_heads = {
                str(item).strip().lower()
                for item in stochastic_heads
                if str(item).strip()
            }
        unknown_heads = parsed_heads - valid_heads
        if unknown_heads:
            raise ValueError(f"unknown stochastic heads: {sorted(unknown_heads)}")
        self.stochastic_heads = frozenset(parsed_heads)
        self.policy_version = policy_version
        self.torch_threads = int(torch_threads)
        self.action_ablation = str(action_ablation).strip().lower()
        if self.action_ablation not in {
            "none",
            "no_split",
            "no_topk",
            "no_filter",
            "no_ajoin",
        }:
            raise ValueError(f"unknown action ablation {self.action_ablation!r}")
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
                raise ValueError("coverage_counts_path is required when coverage_mix > 0")
            coverage_path = Path(self.coverage_counts_path)
            payload = json.loads(coverage_path.read_text(encoding="utf-8"))
            if int(payload.get("schema_version") or 0) != 1:
                raise ValueError(f"unsupported coverage snapshot: {coverage_path}")
            counts = payload.get("counts") or {}
            if not isinstance(counts, dict):
                raise ValueError("coverage snapshot counts must be an object")
            self.coverage_counts = counts
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
        self._schedule_trained = False
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
            import_name = f"neurqo_online_policy_{abs(hash(str(module_path)))}"
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

        candidate_src = [
            self.neurqo_src,
            os.environ.get("NEURQO_SRC"),
            "/code/neurqo/src",
            "/home/naili/neurqo/src",
        ]
        for src in candidate_src:
            if src and Path(src).exists() and src not in sys.path:
                sys.path.insert(0, src)

        try:
            import torch  # type: ignore
            from model.hrl import hrl_shared  # type: ignore
            from model.hrl import transfer_state  # type: ignore
            from model.hrl.hrl_train import (  # type: ignore
                HACNetwork,
                MAXQNetwork,
                OptionCriticNetwork,
            )
            from model.hrl.query_graph_encoder import CatalogInfo  # type: ignore
            from model.hrl.workload_config import get_workload_spec  # type: ignore
        except Exception as exc:  # noqa: BLE001
            log(f"failed to import HRL model code ({exc!r}); using stub policy")
            traceback.print_exc()
            return
        torch.set_num_threads(self.torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        if self.model_method in ("hac", "smdp", "standardmdp", "standardmdp_rl"):
            model = HACNetwork(hidden=self.model_hidden)
        elif self.model_method == "option":
            model = OptionCriticNetwork(hidden=self.model_hidden)
        elif self.model_method == "maxq":
            model = MAXQNetwork(hidden=self.model_hidden)
        else:
            log(f"unknown model method {self.model_method}; using stub policy")
            return

        device = torch.device(
            self.device_name
            if self.device_name != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        try:
            # NeurQO checkpoints are generated by the local trainer and carry
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

        state_dict, migrated_tensors = (
            hrl_shared.migrate_action_space_checkpoint_tensors(model, state_dict)
        )
        if migrated_tensors:
            log(
                "migrated legacy action-space checkpoint tensors: "
                + ",".join(migrated_tensors)
            )
        incompatible = model.load_state_dict(state_dict, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing_non_schedule = [
            key
            for key in incompatible.missing_keys
            if not key.startswith("schedule_")
            and key != "q_schedule.weight"
            and key != "q_schedule.bias"
        ]
        if unexpected or missing_non_schedule:
            raise RuntimeError(
                "checkpoint architecture mismatch: "
                f"missing={missing_non_schedule} unexpected={unexpected}"
            )

        trained_heads = set(metadata.get("trained_heads") or [])
        self._schedule_trained = "schedule" in trained_heads
        self.state_ablation = str(
            metadata.get("state_ablation") or "none"
        ).strip().lower()
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

        try:
            spec = get_workload_spec(self.workload)
            catalog = CatalogInfo(str(spec.catalog_path))
        except Exception:
            catalog = None

        self._torch = torch
        self._model = model
        self._device = device
        self._hrl = hrl_shared
        self._transfer = transfer_state
        self._catalog = catalog
        self.source = f"checkpoint:{path}"
        self._warmup_hrl()
        torch.manual_seed(self.sampling_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.sampling_seed)
        log(f"loaded HRL checkpoint {path} method={self.model_method} device={device}")

    def predict(self, state: dict[str, Any]) -> dict[str, Any]:
        state = _state_for_model(state)

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
            return raw

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
        transfer = self._transfer
        sql = state.get("sql") or state.get("original_sql") or ""
        plan_json = None
        if level != "high":
            plan_json = (
                state.get("plan_json")
                or state.get("plan")
                or _runtime_relation_plan(state)
            )
        if not sql or self._catalog is None:
            ctx_dim = transfer.HIGH_CTX_DIM if level == "high" else 0
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
                if (
                    getattr(self, "state_ablation", "none")
                    == "no_query_topology"
                ):
                    qgraph = transfer.remove_query_graph_topology(qgraph)
            except Exception:
                ctx_dim = transfer.HIGH_CTX_DIM if level == "high" else 0
                return transfer.empty_structured_state(level=level, ctx_dim=ctx_dim)
            if len(self._query_graph_cache) >= 512:
                self._query_graph_cache.pop(next(iter(self._query_graph_cache)))
            self._query_graph_cache[cache_key] = qgraph

        ctx = transfer.np.zeros(
            transfer.HIGH_CTX_DIM if level == "high" else 0,
            dtype=transfer.np.float32,
        )
        if level == "high":
            ctx = transfer.build_high_context(
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
        transfer = self._transfer
        plan_json = state.get("plan_json") or state.get("plan")
        ctx = transfer.build_low_context(
            cumulative_ms=float(state.get("cumulative_cost_ms") or 0.0),
            round_index=float(state.get("round") or 0.0),
            max_rounds=float(state.get("max_split_rounds") or 1.0),
            is_split_execution=bool(state.get("is_split_execution", False)),
            search_strategy=str(state.get("search_strategy") or "default"),
            search_k=int(state.get("search_k") or 0),
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
            level="low",
            query_graph=transfer.empty_query_graph_state(),
            current_plan=plan_tree,
            ctx=ctx,
            cache_key=(
                "online",
                "low",
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
            high = transfer.empty_structured_state(
                level="high",
                ctx_dim=transfer.HIGH_CTX_DIM,
            )
            high_encoded = self._encode(high)
            search = transfer.empty_structured_state(level="search", ctx_dim=0)
            search_encoded = self._encode(search)
            low = transfer.empty_structured_state(
                level="low",
                ctx_dim=transfer.LOW_CTX_DIM,
            )
            low_encoded = self._encode(low)
            if self.model_method in (
                "hac",
                "smdp",
                "standardmdp",
                "standardmdp_rl",
            ):
                model.high_actor(high_encoded)
                model.high_critic(high_encoded)
                schedule_encoded = model.schedule_features(high_encoded)
                model.schedule_actor(schedule_encoded)
                model.schedule_critic(schedule_encoded)
                model.search_actor(search_encoded)
                model.search_critic(search_encoded)
                model.low_actor(low_encoded)
                model.low_critic(low_encoded)
            elif self.model_method == "option":
                model.option_policy(high_encoded)
                model.q_options(high_encoded)
                schedule_encoded = model.schedule_features(high_encoded)
                model.schedule_actor(schedule_encoded)
                model.schedule_critic(schedule_encoded)
                model.search_actor(search_encoded)
                model.search_critic(search_encoded)
                for policy in model.intra_policies:
                    policy(low_encoded)
            else:
                model.q_high(high_encoded)
                model.q_schedule(model.schedule_features(high_encoded))
                model.q_search(search_encoded)
                model.q_low(low_encoded)

    def _masked_action(
        self,
        logits,
        mask,
        phase: str,
        state: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
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
            raw_counts = (
                self.coverage_counts.get(phase, {}).get(coverage_state_hash, [])
            )
            counts = torch.zeros_like(mask_t)
            for index, count in enumerate(raw_counts[: len(mask)]):
                counts[index] = max(float(count), 0.0)
            weights = torch.pow(counts + 1.0, -self.coverage_power) * mask_t
            coverage_probs = weights / weights.sum().clamp_min(1.0)
        mixed_probs = (
            (1.0 - effective_coverage_mix) * policy_probs
            + effective_coverage_mix * coverage_probs
        )
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
        if phase == "high" and len(mask) >= 2:
            metadata["high_split_probability"] = float(dist.probs[1].item())
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
            can_reuse_shared = (
                encoder is not None
                and callable(getattr(structured_state, "tensor", None))
            )
            if can_reuse_shared and level == "low":
                shared_key = ("plan", id(structured_state.current_plan))
                plan_embedding = self._shared_embedding_cache.get(shared_key)
                if plan_embedding is None:
                    plan_embedding = encoder.plan_encoder.encode_tree(
                        structured_state.current_plan,
                        self._device,
                    )
                    self._cache_shared_embedding(shared_key, plan_embedding)
                ctx = structured_state.tensor(
                    "ctx", structured_state.ctx, self._device
                )
                encoded = encoder.low_trunk(
                    self._torch.cat([plan_embedding, ctx], dim=0)
                )
            elif can_reuse_shared and level in {"high", "search"}:
                shared_key = ("graph", id(structured_state.query_graph))
                graph_embedding = self._shared_embedding_cache.get(shared_key)
                if graph_embedding is None:
                    graph_embedding = encoder.graph_encoder(
                        structured_state.query_graph,
                        self._device,
                    )
                    self._cache_shared_embedding(shared_key, graph_embedding)
                if level == "high":
                    ctx = structured_state.tensor(
                        "ctx", structured_state.ctx, self._device
                    )
                    encoded = encoder.high_trunk(
                        self._torch.cat([graph_embedding, ctx], dim=0)
                    )
                else:
                    encoded = encoder.search_trunk(graph_embedding)
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
            self._shared_embedding_cache.pop(
                next(iter(self._shared_embedding_cache))
            )
        self._shared_embedding_cache[key] = embedding

    def _predict_hrl(self, state: dict[str, Any]) -> dict[str, Any]:
        hrl = self._hrl
        model = self._model
        request_type = str(state.get("request_type") or "high").lower()
        base_rels = int(state.get("base_rels") or 0)
        remaining = int(state.get("remaining_splits") or 0)

        if request_type == "high":
            structured_state = self._query_graph_state(state, "high")
            split_allowed = (
                self.workload.strip().lower() != "tpch"
                and base_rels > 2
                and remaining > 0
            )
            mask = [1.0, 1.0 if split_allowed else 0.0]
            mask = hrl.apply_action_ablation_mask(
                mask, "high", self.action_ablation
            ).tolist()
            with self._torch.no_grad():
                encoded = self._encode(structured_state)
                if self.model_method in (
                    "hac",
                    "smdp",
                    "standardmdp",
                    "standardmdp_rl",
                ):
                    logits = model.high_actor(encoded)
                    predicted_value = float(model.high_critic(encoded).squeeze().item())
                elif self.model_method == "option":
                    logits = model.option_policy(encoded)
                    option_values = model.q_options(encoded)
                    predicted_value = None
                else:
                    logits = model.q_high(encoded)
                    predicted_value = None
                high_idx, policy_meta = self._masked_action(logits, mask, "high", state)
                if predicted_value is None:
                    values = option_values if self.model_method == "option" else logits
                    predicted_value = float(values.reshape(-1)[high_idx].item())
            high_action = "split" if high_idx == 1 else "stop"
            return {
                "action": high_action,
                "stop": high_idx == 0,
                "order_decision": "only_cost",
                "high_action": high_action,
                "predicted_value": predicted_value,
                "model_source": self.source,
                "note": f"model high inference: high={high_idx}",
                **policy_meta,
            }

        if request_type == "select":
            if not self._schedule_trained:
                # Legacy checkpoints keep the established phi4/alpha=0.5
                # scheduler instead of activating a randomly initialized head.
                return {}
            structured_state = self._query_graph_state(state, "high")
            mask = [1.0] * len(hrl.SCHEDULE_ALPHA_VALUES)
            mask = hrl.apply_action_ablation_mask(
                mask, "select", self.action_ablation
            ).tolist()
            with self._torch.no_grad():
                encoded = self._encode(structured_state)
                if self.model_method in (
                    "hac",
                    "smdp",
                    "standardmdp",
                    "standardmdp_rl",
                    "option",
                ):
                    schedule_encoded = model.schedule_features(encoded)
                    logits = model.schedule_actor(schedule_encoded)
                    predicted_value = float(
                        model.schedule_critic(schedule_encoded).squeeze().item()
                    )
                else:
                    logits = model.q_schedule(model.schedule_features(encoded))
                    predicted_value = None
                schedule_idx, policy_meta = self._masked_action(
                    logits, mask, "select", state
                )
                if predicted_value is None:
                    predicted_value = float(logits.reshape(-1)[schedule_idx].item())
            alpha = hrl.SCHEDULE_ALPHA_VALUES[schedule_idx]
            return {
                "action": "select",
                "stop": False,
                "schedule_idx": schedule_idx,
                "schedule_alpha": alpha,
                "selection_strategy": f"alpha_{alpha:.2f}",
                "predicted_value": predicted_value,
                "model_source": self.source,
                "note": f"model schedule inference: alpha={alpha:.2f}",
                **policy_meta,
            }

        if request_type == "search":
            structured_state = self._query_graph_state(state, "search")
            max_rels = int(state.get("search_max_rels") or 12)
            search_feasible = 2 <= base_rels <= max_rels
            mask = [1.0] + [1.0 if search_feasible else 0.0] * (
                len(hrl.SEARCH_LABELS) - 1
            )
            mask = hrl.apply_action_ablation_mask(
                mask, "search", self.action_ablation
            ).tolist()
            with self._torch.no_grad():
                encoded = self._encode(structured_state)
                if self.model_method in (
                    "hac",
                    "smdp",
                    "standardmdp",
                    "standardmdp_rl",
                    "option",
                ):
                    logits = model.search_actor(encoded)
                    predicted_value = float(
                        model.search_critic(encoded).squeeze().item()
                    )
                else:
                    logits = model.q_search(encoded)
                    predicted_value = None
                search_idx, policy_meta = self._masked_action(
                    logits, mask, "search", state
                )
                if predicted_value is None:
                    predicted_value = float(logits.reshape(-1)[search_idx].item())
            search_label = hrl.SEARCH_LABELS[search_idx]
            search_strategy, search_k = _map_search_label(search_label)
            return {
                "action": "search",
                "stop": False,
                "search_strategy": search_strategy,
                "search_k": search_k,
                "search_label": search_label,
                "predicted_value": predicted_value,
                "model_source": self.source,
                "note": f"model search inference: search={search_label}",
                **policy_meta,
            }

        if request_type == "low":
            structured_state = self._plan_state(state)
            plan_json = state.get("plan_json") or state.get("plan") or {}
            has_join = _plan_contains_join(plan_json)
            has_hash_join = _plan_contains_join(plan_json, "hashjoin")
            mask = []
            for label in hrl.ACTION_LABELS:
                needs_lip = label.startswith("lip_")
                needs_aja = "aja" in label
                valid = (not needs_lip or has_join) and (not needs_aja or has_hash_join)
                mask.append(1.0 if valid else 0.0)
            mask = hrl.apply_action_ablation_mask(
                mask, "low", self.action_ablation
            ).tolist()
            with self._torch.no_grad():
                encoded = self._encode(structured_state)
                if self.model_method in (
                    "hac",
                    "smdp",
                    "standardmdp",
                    "standardmdp_rl",
                ):
                    logits = model.low_actor(encoded)
                    predicted_value = float(model.low_critic(encoded).squeeze().item())
                elif self.model_method == "option":
                    high_action = str(state.get("high_action") or "stop").lower()
                    option_idx = 1 if high_action == "split" else 0
                    option_idx = min(option_idx, len(model.intra_policies) - 1)
                    logits = model.intra_policies[option_idx](encoded)
                    predicted_value = float(
                        model.q_options(encoded).reshape(-1)[option_idx].item()
                    )
                else:
                    logits = model.q_low(encoded)
                    predicted_value = None
                low_idx, policy_meta = self._masked_action(logits, mask, "low", state)
                if predicted_value is None:
                    predicted_value = float(logits.reshape(-1)[low_idx].item())
            low_label = hrl.ACTION_LABELS[low_idx]
            lip_action, execution_action = _map_low_label(low_label)
            return {
                "action": "low",
                "stop": False,
                "execution_action": execution_action,
                "lip_action": lip_action,
                "low_label": low_label,
                "predicted_value": predicted_value,
                "model_source": self.source,
                "note": f"model low inference: low={low_label}",
                **policy_meta,
            }

        raise ValueError(f"unknown request_type={request_type!r}")


def _map_search_label(label: str | None) -> tuple[str, int]:
    if not label:
        return "default", 1
    key = str(label).strip().lower()
    if key.startswith("top") and key[3:].isdigit():
        return "topk", int(key[3:])
    return SEARCH_LABEL_TO_DB.get(key, (key, 5 if key == "topk" else 1))


def _map_low_label(label: str | None) -> tuple[str, str]:
    if not label:
        return "none", "none"
    key = str(label).strip().lower()
    if key in ACTION_LABEL_TO_LIP_AJA:
        return ACTION_LABEL_TO_LIP_AJA[key]
    return "none", key


def _normalize_prediction(pred: dict[str, Any]) -> dict[str, Any]:
    pred = dict(pred)
    if "search_label" in pred and "search_strategy" not in pred:
        strategy, k = _map_search_label(str(pred["search_label"]))
        pred["search_strategy"] = strategy
        pred.setdefault("search_k", k)
    if "low_label" in pred and (
        "lip_action" not in pred or "execution_action" not in pred
    ):
        lip, aja = _map_low_label(str(pred["low_label"]))
        pred.setdefault("lip_action", lip)
        pred.setdefault("execution_action", aja)
    if "aja_level" in pred and "execution_action" not in pred:
        pred["execution_action"] = pred["aja_level"]
    if "execution_action" in pred:
        execution_action = str(pred["execution_action"]).strip().lower()
        # Existing checkpoints use a binary AJA label generated with v10pct.
        if execution_action == "aja":
            execution_action = "aggressive"
        pred["execution_action"] = execution_action
    if "high_action" in pred and "stop" not in pred:
        pred["stop"] = str(pred["high_action"]).lower() in {"stop", "none"}
    return pred


def _decide_high(state: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    base_rels = int(state.get("base_rels", 0))
    remaining = int(state.get("remaining_splits", 0))

    if pred:
        pred = _normalize_prediction(pred)
        high_action = str(pred.get("high_action") or pred.get("action") or "")
        high_action = high_action.strip().lower()
        stop = _truthy(pred.get("stop", high_action == "stop"))
        if high_action not in {"split", "stop"}:
            high_action = "stop" if stop else "split"
        return {
            "action": high_action,
            "stop": high_action == "stop",
            "order_decision": pred.get("order_decision", "only_cost"),
            "high_action": high_action,
            "note": pred.get("note", "model high action"),
            "model_source": pred.get("model_source"),
        }

    can_split = base_rels > 2 and remaining > 0
    return {
        "action": "split" if can_split else "stop",
        "stop": not can_split,
        "order_decision": "only_cost",
        "high_action": "split" if can_split else "stop",
        "note": f"stub high: {'split' if can_split else 'stop'}",
    }


def _decide_search(state: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    if pred:
        pred = _normalize_prediction(pred)
        strategy, default_k = _map_search_label(
            pred.get("search_label") or pred.get("search_strategy")
        )
        return {
            "action": "search",
            "stop": False,
            "search_strategy": strategy,
            "search_k": int(pred.get("search_k") or default_k),
            "search_label": pred.get("search_label"),
            "note": pred.get("note", "model search action"),
            "model_source": pred.get("model_source"),
        }
    return {
        "action": "search",
        "stop": False,
        "search_strategy": "default",
        "search_k": 1,
        "search_label": "default",
        "note": "stub search: default",
    }


def _decide_select(state: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    candidates = state.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("select request requires a non-empty candidates list")

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
        return {
            "action": "select",
            "stop": False,
            "candidate_id": candidate_id,
            "selection_strategy": pred.get("selection_strategy", "model"),
            "note": pred.get("note", "model subquery selection"),
            "model_source": pred.get("model_source"),
        }

    requested_alpha = pred.get("schedule_alpha") if pred else None
    if requested_alpha is not None:
        alpha = float(requested_alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"schedule_alpha must be in [0,1], got {alpha}")

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
            "action": "select",
            "stop": False,
            "candidate_id": candidate_id,
            "schedule_idx": pred.get("schedule_idx"),
            "schedule_alpha": alpha,
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
        "action": "select",
        "stop": False,
        "candidate_id": candidate_id,
        "selection_strategy": "phi4",
        "note": "stub SSA: phi4=min(cost*rows)",
    }


def _decide_low(state: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    if pred:
        pred = _normalize_prediction(pred)
        lip_action = str(pred.get("lip_action") or "none")
        execution_action = str(pred.get("execution_action") or "none")
        action = {
            "action": "low",
            "stop": False,
            "execution_action": execution_action,
            "lip_action": lip_action,
            "low_label": pred.get("low_label"),
            "note": pred.get("note", "model low action"),
            "model_source": pred.get("model_source"),
        }
        static_join = execution_action in {"hashjoin", "nestloop", "mergejoin"}
        if static_join and pred.get("aja_hint"):
            action["aja_hint"] = pred["aja_hint"]
        if static_join and pred.get("join_method"):
            action["join_method"] = pred["join_method"]
        return action

    return {
        "action": "low",
        "stop": False,
        "execution_action": "none",
        "lip_action": "none",
        "low_label": "none",
        "note": "stub low: none",
    }


ADAPTER: PolicyAdapter | None = None
ADAPTER_CONFIG: dict[str, Any] = {}
POLICY_LOCK = threading.RLock()
REQUIRE_MODEL = False
TRAJECTORY_LOG: str | None = None


def model_predict(state: dict[str, Any]) -> dict[str, Any]:
    """Call the configured learned policy, returning normalized action fields."""
    with POLICY_LOCK:
        if ADAPTER is None:
            return {}
        return ADAPTER.predict(state)


def decide_action(state: dict[str, Any]) -> dict[str, Any]:
    try:
        pred = model_predict(state) or {}
    except Exception as exc:  # noqa: BLE001
        log(f"model prediction failed: {exc!r}")
        traceback.print_exc()
        pred = {}

    request_type = str(state.get("request_type") or "high").lower()
    if request_type == "high":
        action = _decide_high(state, pred)
    elif request_type == "select":
        action = _decide_select(state, pred)
    elif request_type == "search":
        action = _decide_search(state, pred)
    elif request_type == "low":
        action = _decide_low(state, pred)
    else:
        raise ValueError(f"unknown request_type={request_type!r}")
    with POLICY_LOCK:
        source = ADAPTER.source if ADAPTER else "stub"
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
        "high_split_probability",
        "schedule_idx",
        "schedule_alpha",
    ):
        if key in pred and key not in action:
            action[key] = pred[key]
    if pred:
        action.setdefault("policy_version", source)
        action.setdefault("action_probability", 1.0)
        action.setdefault("log_probability", 0.0)
    return action


def render_action(action: dict[str, Any]) -> bytes:
    """Serialize an action dict into the line-oriented wire format."""
    lines = [
        f"action={action.get('action', 'none')}",
        f"stop={1 if action.get('stop', True) else 0}",
    ]
    if action.get("order_decision"):
        lines.append(f"order_decision={action['order_decision']}")
    if action.get("candidate_id") is not None:
        lines.append(f"candidate_id={int(action['candidate_id'])}")
    if action.get("selection_strategy"):
        lines.append(f"selection_strategy={action['selection_strategy']}")
    if action.get("search_strategy"):
        lines.append(f"search_strategy={action['search_strategy']}")
    if action.get("search_k"):
        lines.append(f"search_k={int(action['search_k'])}")
    if action.get("execution_action"):
        lines.append(f"execution_action={action['execution_action']}")
    if action.get("lip_action"):
        lines.append(f"lip_action={action['lip_action']}")
    if action.get("aja_hint"):
        lines.append(f"aja_hint={action['aja_hint']}")
    if action.get("join_method"):
        lines.append(f"join_method={action['join_method']}")
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
        global ADAPTER, ADAPTER_CONFIG

        started = time.perf_counter()
        new_config = dict(ADAPTER_CONFIG)
        for key in (
            "model_module",
            "model_path",
            "model_method",
            "model_hidden",
            "workload",
            "device",
            "neurqo_src",
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
        ):
            if key in request and request[key] is not None:
                new_config[key] = request[key]

        try:
            adapter = PolicyAdapter(**new_config)
        except Exception as exc:  # noqa: BLE001
            log(f"policy reload failed: {exc!r}")
            traceback.print_exc()
            self._respond(f"reloaded=0\nnote={exc!r}\n".encode("utf-8"), 500)
            return

        if REQUIRE_MODEL and adapter.source == "stub":
            note = "required model was not loaded during reload"
            log(note)
            self._respond(f"reloaded=0\nnote={note}\n".encode("utf-8"), 500)
            return

        with POLICY_LOCK:
            ADAPTER = adapter
            ADAPTER_CONFIG = new_config

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        payload = {
            "ts": _now_iso(),
            "phase": "policy_reload",
            "latency_ms": elapsed_ms,
            "request": request,
            "model_source": adapter.source,
        }
        _append_jsonl(TRAJECTORY_LOG, payload)
        log(f"policy reloaded source={adapter.source} latency_ms={elapsed_ms:.2f}")
        self._respond(
            (
                "reloaded=1\n"
                f"model_source={adapter.source}\n"
                f"latency_ms={elapsed_ms:.2f}\n"
            ).encode("utf-8")
        )

    def do_POST(self):
        t0 = time.perf_counter()
        if self.path.startswith("/shutdown"):
            self._respond(b"shutdown=1\n")
            threading.Thread(
                target=self.server.shutdown,
                name="neurqo-server-shutdown",
                daemon=True,
            ).start()
            return
        try:
            state = self._read_json_body()
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
        body = render_action(action)
        log(
            f"request={state.get('request_type', 'high')} "
            f"run={state.get('run_id')} round={state.get('round')} "
            f"source={action.get('model_source')} action={action['action']} "
            f"stop={action['stop']} search={action.get('search_strategy')} "
            f"k={action.get('search_k')} exec={action.get('execution_action')} "
            f"lip={action.get('lip_action')} aja_hint={action.get('aja_hint')} "
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
            source = ADAPTER.source if ADAPTER else "stub"
        self._respond(
            f"action=none\nstop=1\nnote=health ok\nmodel_source={source}\n".encode(
                "utf-8"
            )
        )

    def log_message(self, *args):  # silence default per-request stderr noise
        pass


def main() -> int:
    global ADAPTER, ADAPTER_CONFIG, REQUIRE_MODEL, TRAJECTORY_LOG

    ap = argparse.ArgumentParser(description="NeurQO AI action server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--model-module", default=os.environ.get("NEURQO_MODEL_MODULE"))
    ap.add_argument("--model-path", default=os.environ.get("NEURQO_MODEL_PATH"))
    ap.add_argument(
        "--model-method",
        default=os.environ.get("NEURQO_MODEL_METHOD", "standardmdp_rl"),
    )
    ap.add_argument(
        "--model-hidden",
        type=int,
        default=int(os.environ.get("NEURQO_MODEL_HIDDEN", "128")),
    )
    ap.add_argument("--workload", default=os.environ.get("NEURQO_WORKLOAD", "job"))
    ap.add_argument("--device", default=os.environ.get("NEURQO_DEVICE", "cpu"))
    ap.add_argument("--neurqo-src", default=os.environ.get("NEURQO_SRC"))
    ap.add_argument(
        "--inference-mode",
        choices=("deterministic", "stochastic"),
        default=os.environ.get("NEURQO_INFERENCE_MODE", "deterministic"),
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("NEURQO_TEMPERATURE", "1.0")),
    )
    ap.add_argument(
        "--exploration-epsilon",
        type=float,
        default=float(os.environ.get("NEURQO_EXPLORATION_EPSILON", "0.0")),
    )
    ap.add_argument(
        "--coverage-counts-path",
        default=os.environ.get("NEURQO_COVERAGE_COUNTS_PATH"),
    )
    ap.add_argument(
        "--coverage-mix",
        type=float,
        default=float(os.environ.get("NEURQO_COVERAGE_MIX", "0.0")),
    )
    ap.add_argument(
        "--coverage-power",
        type=float,
        default=float(os.environ.get("NEURQO_COVERAGE_POWER", "0.5")),
    )
    ap.add_argument(
        "--stochastic-heads",
        default=os.environ.get("NEURQO_STOCHASTIC_HEADS"),
        help=(
            "comma-separated policy phases sampled stochastically; "
            "other phases use deterministic argmax"
        ),
    )
    ap.add_argument(
        "--sampling-seed",
        type=int,
        default=int(os.environ.get("NEURQO_SAMPLING_SEED", "42")),
    )
    ap.add_argument(
        "--policy-version",
        default=os.environ.get("NEURQO_POLICY_VERSION"),
    )
    ap.add_argument(
        "--torch-threads",
        type=int,
        default=int(os.environ.get("NEURQO_TORCH_THREADS", "1")),
    )
    ap.add_argument(
        "--action-ablation",
        choices=("none", "no_split", "no_topk", "no_filter", "no_ajoin"),
        default=os.environ.get("NEURQO_ACTION_ABLATION", "none"),
    )
    ap.add_argument("--trajectory-log", default=os.environ.get("NEURQO_TRAJECTORY_LOG"))
    ap.add_argument(
        "--require-model",
        action="store_true",
        default=_truthy(os.environ.get("NEURQO_REQUIRE_MODEL")),
        help="fail startup if --model-module/--model-path cannot be loaded",
    )
    args = ap.parse_args()

    REQUIRE_MODEL = bool(args.require_model)
    TRAJECTORY_LOG = args.trajectory_log
    ADAPTER_CONFIG = {
        "model_module": args.model_module,
        "model_path": args.model_path,
        "model_method": args.model_method,
        "model_hidden": args.model_hidden,
        "workload": args.workload,
        "device": args.device,
        "neurqo_src": args.neurqo_src,
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
    }
    try:
        ADAPTER = PolicyAdapter(**ADAPTER_CONFIG)
    except Exception as exc:  # noqa: BLE001
        if args.require_model:
            log(f"policy adapter initialization failed: {exc!r}")
            traceback.print_exc()
            return 2
        log(f"policy adapter initialization failed: {exc!r}; using stub policy")
        traceback.print_exc()
        ADAPTER = PolicyAdapter()

    if args.require_model:
        requested_model = bool(args.model_module or args.model_path)
        if not requested_model or ADAPTER.source == "stub":
            log(
                "required model was not loaded; pass --model-module or a valid "
                "--model-path with --neurqo-src"
            )
            return 2

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    log(
        f"NeurQO AI action server listening on http://{args.host}:{args.port}/action "
        f"source={ADAPTER.source} trajectory_log={TRAJECTORY_LOG or 'off'}"
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log("shutting down")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

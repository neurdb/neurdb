#!/usr/bin/env python3
"""
NeurQO AI action server.

The DB-side NeurQO path calls this service at two decision points:
  1. round policy: choose split/search/LIP/AJA knobs for the current round
  2. AJA policy: map a baseline plan state to pg_hint_plan join-method hints

The wire protocol remains intentionally simple. The request body is JSON, and
the response is line-oriented key=value fields that the C code can parse:

    action=search
    stop=0
    search_strategy=topk
    search_k=5
    execution_action=aja
    lip_action=full
    order_decision=only_cost
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
import importlib
import importlib.util
import json
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
    "aja": ("none", "aja"),
    "lip_full+aja": ("full", "aja"),
    "lip_sel+aja": ("selective", "aja"),
}

SEARCH_LABEL_TO_DB = {
    "default": ("default", 1),
    "none": ("default", 1),
    "split": ("topk", 5),
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
    ) -> None:
        self.model_module = model_module
        self.model_path = model_path
        self.model_method = model_method
        self.model_hidden = model_hidden
        self.workload = workload
        self.device_name = device
        self.neurqo_src = neurqo_src
        self.source = "stub"
        self._callable: Callable[[dict[str, Any]], Any] | None = None
        self._torch = None
        self._model = None
        self._device = None
        self._hrl = None
        self._transfer = None
        self._catalog = None

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
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
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
            return raw

        if self._model is not None:
            return self._predict_hrl(state)

        return {}

    def _query_graph_state(self, state: dict[str, Any], level: str):
        transfer = self._transfer
        sql = state.get("sql") or state.get("original_sql") or ""
        plan_json = state.get("plan_json") or state.get("plan")
        if not sql or self._catalog is None:
            ctx_dim = transfer.HIGH_CTX_DIM if level == "high" else 0
            return transfer.empty_structured_state(level=level, ctx_dim=ctx_dim)

        try:
            graph = transfer.parse_query_graph(sql)
            qgraph, _stats = transfer.build_transfer_graph_state(
                sql, graph, self._catalog, plan_json=plan_json
            )
        except Exception:
            ctx_dim = transfer.HIGH_CTX_DIM if level == "high" else 0
            return transfer.empty_structured_state(level=level, ctx_dim=ctx_dim)

        ctx = []
        if level == "high":
            base_rels = float(state.get("base_rels") or 0.0)
            round_no = float(state.get("round") or 0.0)
            ctx = [min(base_rels / 16.0, 1.0), min(round_no / 16.0, 1.0)]
        return transfer.StructuredState(
            level=level,
            query_graph=qgraph,
            current_plan=transfer.empty_plan_tree(),
            ctx=transfer.np.asarray(ctx, dtype=transfer.np.float32),
            cache_key=("online", level, hash(sql), tuple(ctx)),
        )

    def _plan_state(self, state: dict[str, Any]):
        transfer = self._transfer
        plan_json = state.get("plan_json") or state.get("plan")
        if plan_json is not None and self._catalog is not None:
            try:
                plan_tree = transfer.plan_to_tree(plan_json, catalog=self._catalog)
            except Exception:
                plan_tree = transfer.empty_plan_tree()
        else:
            plan_tree = transfer.empty_plan_tree()
        empty = transfer.empty_structured_state(
            level="low", ctx_dim=transfer.LOW_CTX_DIM
        )
        return transfer.StructuredState(
            level="low",
            query_graph=empty.query_graph,
            current_plan=plan_tree,
            ctx=transfer.np.zeros(transfer.LOW_CTX_DIM, dtype=transfer.np.float32),
            cache_key=("online", "low", state.get("round"), state.get("plan_rows")),
        )

    def _masked_argmax(self, logits, mask) -> int:
        torch = self._torch
        mask_t = torch.tensor(mask, dtype=torch.float32, device=self._device)
        return int((logits + (mask_t - 1.0) * 1e9).argmax().item())

    def _encode(self, structured_state):
        with self._torch.no_grad():
            return self._model.encode_state_obj(structured_state, self._device)

    def _predict_hrl(self, state: dict[str, Any]) -> dict[str, Any]:
        hrl = self._hrl
        model = self._model
        base_rels = int(state.get("base_rels") or 0)
        remaining = int(state.get("remaining_splits") or 0)

        high_state = self._query_graph_state(state, "high")
        search_state = self._query_graph_state(state, "search")
        low_state = self._plan_state(state)

        high_mask = [1.0, 1.0 if base_rels > 2 and remaining > 0 else 0.0]
        search_mask = [1.0, 1.0, 1.0, 1.0]
        low_mask = [1.0] * len(hrl.ACTION_LABELS)

        with self._torch.no_grad():
            high_h = self._encode(high_state)
            search_h = self._encode(search_state)
            low_h = self._encode(low_state)

            if self.model_method in ("hac", "smdp", "standardmdp", "standardmdp_rl"):
                high_logits = model.high_actor(high_h)
                search_logits = model.search_actor(search_h)
                low_logits = model.low_actor(low_h)
            elif self.model_method == "option":
                high_logits = model.option_policy(high_h)
                search_logits = model.search_actor(search_h)
                option_idx = self._masked_argmax(high_logits, high_mask)
                low_logits = model.intra_policies[option_idx](low_h)
            else:
                high_logits = model.q_high(high_h)
                search_logits = model.q_search(search_h)
                low_logits = model.q_low(low_h)

            high_idx = self._masked_argmax(high_logits, high_mask)
            search_idx = self._masked_argmax(search_logits, search_mask)
            low_idx = self._masked_argmax(low_logits, low_mask)

        search_label = hrl.SEARCH_LABELS[search_idx]
        low_label = hrl.ACTION_LABELS[low_idx]
        search_strategy, search_k = _map_search_label(search_label)
        lip_action, execution_action = _map_low_label(low_label)

        return {
            "action": "split" if high_idx == 1 else "search",
            "stop": high_idx == 0,
            "search_strategy": search_strategy,
            "search_k": search_k,
            "execution_action": execution_action,
            "lip_action": lip_action,
            "order_decision": "only_cost",
            "high_action": "split" if high_idx == 1 else "stop",
            "search_label": search_label,
            "low_label": low_label,
            "model_source": self.source,
            "note": (
                f"model inference: high={high_idx} search={search_label} "
                f"low={low_label}"
            ),
        }


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
    if "high_action" in pred and "stop" not in pred:
        pred["stop"] = str(pred["high_action"]).lower() in {"stop", "none"}
    return pred


def _aliases_hint(method: str, aliases: list[str]) -> str:
    aliases = [str(a) for a in aliases if str(a)]
    if len(aliases) < 2:
        return "none"
    return f"{method}({' '.join(aliases)})"


def _state_aliases(state: dict[str, Any]) -> list[str]:
    aliases = state.get("aliases") or []
    if aliases:
        return [str(a) for a in aliases]
    rels = state.get("relations") or []
    out = []
    for rel in rels:
        if isinstance(rel, dict):
            out.append(str(rel.get("alias") or rel.get("relname") or ""))
    return [a for a in out if a]


def _decide_aja(state: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    aliases = _state_aliases(state)
    summary = state.get("plan_summary") or {}
    rows = float(state.get("plan_rows") or 0)
    joins = int(summary.get("joins") or 0)

    if pred.get("aja_hint") or pred.get("join_method"):
        method = pred.get("join_method", "")
        return {
            "action": "aja",
            "stop": False,
            "aja_hint": pred.get("aja_hint", _aliases_hint(method, aliases)),
            "join_method": method,
            "note": pred.get("note", "model AJA hint"),
        }

    method = "NestLoop" if joins <= 1 and 0 < rows < 10_000 else "HashJoin"
    return {
        "action": "aja",
        "stop": False,
        "aja_hint": _aliases_hint(method, aliases),
        "join_method": method,
        "note": (
            f"stub AJA: {method} from baseline plan "
            f"(joins={joins}, rows={rows:.0f})"
        ),
    }


def _decide_round(state: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    base_rels = int(state.get("base_rels", 0))

    if pred:
        pred = _normalize_prediction(pred)
        search_strategy, default_k = _map_search_label(pred.get("search_strategy"))
        action = {
            "action": pred.get("action", "search"),
            "stop": _truthy(pred.get("stop", False)),
            "search_strategy": search_strategy,
            "search_k": int(pred.get("search_k") or default_k or 5),
            "execution_action": pred.get("execution_action", "aja"),
            "lip_action": pred.get("lip_action", "full"),
            "order_decision": pred.get("order_decision", "only_cost"),
            "note": pred.get("note", "model round action"),
        }
        if pred.get("model_source"):
            action["model_source"] = pred["model_source"]
        if pred.get("high_action"):
            action["high_action"] = pred["high_action"]
        if pred.get("search_label"):
            action["search_label"] = pred["search_label"]
        if pred.get("low_label"):
            action["low_label"] = pred["low_label"]
        return action

    if base_rels > 2:
        return {
            "action": "search",
            "stop": False,
            "search_strategy": "topk",
            "search_k": 5,
            "execution_action": "aja",
            "lip_action": "full",
            "order_decision": "only_cost",
            "note": (
                "stub: in-DB top-k Leading search + "
                f"AJA plan-feature replan + LIP Bloom probes (base_rels={base_rels})"
            ),
        }

    return {
        "action": "none",
        "stop": True,
        "search_strategy": "default",
        "search_k": 1,
        "execution_action": "none",
        "lip_action": "none",
        "order_decision": "only_cost",
        "note": f"stub: no-op (base_rels={base_rels})",
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

    if state.get("request_type") == "aja" or state.get("action") == "aja":
        action = _decide_aja(state, pred)
    else:
        action = _decide_round(state, pred)
    with POLICY_LOCK:
        source = ADAPTER.source if ADAPTER else "stub"
    action.setdefault("model_source", source)
    return action


def render_action(action: dict[str, Any]) -> bytes:
    """Serialize an action dict into the line-oriented wire format."""
    lines = [
        f"action={action.get('action', 'none')}",
        f"stop={1 if action.get('stop', True) else 0}",
    ]
    if action.get("order_decision"):
        lines.append(f"order_decision={action['order_decision']}")
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
        try:
            state = self._read_json_body()
        except Exception as exc:  # noqa: BLE001
            log(f"bad request body: {exc!r}")
            self._respond(b"action=none\nstop=1\nnote=bad request\n", 400)
            return

        if self.path.startswith("/reload"):
            self._handle_reload(state)
            return

        action = decide_action(state)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        body = render_action(action)
        log(
            f"request={state.get('request_type', 'round')} "
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

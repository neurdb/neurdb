#!/usr/bin/env python3
"""
NeurQO AI action server.

A deliberately small, dependency-free HTTP service that the in-DB neurqo
planner path calls for two things:
  1. round policy: choose split/search/LIP/AJA knobs for the next DB round
  2. AJA policy: map a baseline plan state to pg_hint_plan join-method hints

  request  (DB -> server):  JSON state, e.g.
      {"pid":123,"round":0,"base_rels":5,"num_joins":4,"cmd":"select"}

  response (server -> DB):   line-oriented key=value, e.g.
      action=search
      stop=0
      search_strategy=topk
      search_k=5
      execution_action=aja
      lip_action=full
      order_decision=only_cost
      note=stub: in-DB top-k Search + AJA replan + LIP probes

The contract is "input state -> action/hint fields". Replace
`model_predict()` with a call into the real NeurQO policy later; keep returning
the normalized dict fields below so the DB-side parser does not change.

Run inside the container so the DB can reach it on localhost:
    python3 /code/neurdb-dev/neurqo/server/ai_server.py --host 127.0.0.1 --port 8088
"""
import argparse
import datetime
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


# --------------------------------------------------------------------------
# Policy adapter.  The hardcoded policy is intentionally tiny, but the adapter
# accepts the same normalized dict shape that a learned model can return later.
# --------------------------------------------------------------------------
def model_predict(state: dict) -> dict:
    """Placeholder for the learned model.

    Expected output fields are optional and normalized by `decide_action()`:
      action, stop, search_strategy, search_k, execution_action, lip_action,
      order_decision, aja_hint, join_method, note
    """
    return {}


def _aliases_hint(method: str, aliases: list[str]) -> str:
    aliases = [str(a) for a in aliases if str(a)]
    if len(aliases) < 2:
        return "none"
    return f"{method}({' '.join(aliases)})"


def _decide_aja(state: dict, pred: dict) -> dict:
    aliases = state.get("aliases") or []
    summary = state.get("plan_summary") or {}
    rows = float(state.get("plan_rows") or 0)
    joins = int(summary.get("joins") or 0)

    if pred.get("aja_hint") or pred.get("join_method"):
        return {
            "action": "aja",
            "stop": False,
            "aja_hint": pred.get("aja_hint", ""),
            "join_method": pred.get("join_method", ""),
            "note": pred.get("note", "model AJA hint"),
        }

    # Stub heuristic: hash joins for non-trivial joins; nested loop only for
    # tiny baseline outputs. This is deliberately easy to replace with model
    # output while still producing a real replan hint today.
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


def _decide_round(state: dict, pred: dict) -> dict:
    """Map a DB round state to ONE action dict.

    Returned action dict keys:
      action  : str label (search / split / lip / aja / none)
      stop    : bool  -> tell the DB loop to stop after applying this action
      search_strategy : join-order hint strategy (topk/left_deep/default)
      search_k : top-k join orders to keep for in-DB Search
      execution_action : execution hint action (none/aja/hashjoin/nestloop)
      lip_action : LIP Bloom-filter mode (none/selective/full)
      order_decision : RCenter alpha mode (only_cost / only_row / hybrids)
      note    : human-readable explanation
    """
    base_rels = int(state.get("base_rels", 0))

    if pred:
        action = {
            "action": pred.get("action", "search"),
            "stop": bool(pred.get("stop", False)),
            "search_strategy": pred.get("search_strategy", "topk"),
            "search_k": int(pred.get("search_k", 5)),
            "execution_action": pred.get("execution_action", "aja"),
            "lip_action": pred.get("lip_action", "full"),
            "order_decision": pred.get("order_decision", "only_cost"),
            "note": pred.get("note", "model round action"),
        }
        return action

    # Stub decision. The actual join-order DP and AJA replan happen inside the
    # DB; this only chooses which online action channels are active this round.
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


def decide_action(state: dict) -> dict:
    pred = model_predict(state) or {}
    if state.get("request_type") == "aja" or state.get("action") == "aja":
        return _decide_aja(state, pred)
    return _decide_round(state, pred)


def render_action(action: dict) -> bytes:
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


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}", flush=True)


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.0 -> connection closes after each response (matches the C client).
    protocol_version = "HTTP/1.0"

    def _respond(self, body: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            state = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception as exc:  # noqa: BLE001
            log(f"bad request body: {exc!r} raw={raw!r}")
            self._respond(b"action=none\nstop=1\nnote=bad request\n", 400)
            return

        action = decide_action(state)
        body = render_action(action)
        log(
            f"request={state.get('request_type', 'round')} -> action={action['action']} "
            f"stop={action['stop']} order_decision={action.get('order_decision')} "
            f"search_strategy={action.get('search_strategy')} "
            f"search_k={action.get('search_k')} "
            f"execution_action={action.get('execution_action')} "
            f"lip_action={action.get('lip_action')} "
            f"aja_hint={action.get('aja_hint')}"
        )
        self._respond(body)

    def do_GET(self):
        # simple health check
        self._respond(b"action=none\nstop=1\nnote=health ok\n")

    def log_message(self, *args):  # silence default per-request stderr noise
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="NeurQO AI action server (stub)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8088)
    args = ap.parse_args()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    log(
        f"NeurQO AI action server (stub) listening on "
        f"http://{args.host}:{args.port}/action"
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

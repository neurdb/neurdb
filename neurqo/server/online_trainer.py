#!/usr/bin/env python3
"""
NeurQO online training bridge.

This process consumes DB-side trajectory JSONL produced by:

    SET neurqo.trajectory_log = '/tmp/neurqo_runtime.jsonl';

DB events are converted into online RL transitions:

    state, action, reward, done, next_state=None, timing_ms

The bridge keeps the previous round for each run in memory and fills
next_state when the next split/final event arrives. The default reward is
negative round total time. This is intentionally simple: it gives us an online
data path immediately while keeping model-specific training outside the DB
critical path.

Optionally pass --trainer-module module_or_path:callable. The callable receives
a list[dict] batch and may update a model checkpoint however it wants.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

LOW_LABEL_BY_ACTION = {
    ("none", "none"): "none",
    ("full", "none"): "lip_full",
    ("selective", "none"): "lip_sel",
    ("none", "aja"): "aja",
    ("full", "aja"): "lip_full+aja",
    ("selective", "aja"): "lip_sel+aja",
}


def load_callable(spec: str | None) -> Callable[[list[dict[str, Any]]], Any] | None:
    if not spec:
        return None
    module_name, sep, attr = spec.partition(":")
    if not sep or not attr:
        raise ValueError("--trainer-module must be 'module_or_path:callable'")

    module_path = Path(module_name)
    if module_path.exists():
        import_name = f"neurqo_online_trainer_{abs(hash(str(module_path)))}"
        mod_spec = importlib.util.spec_from_file_location(import_name, module_path)
        if mod_spec is None or mod_spec.loader is None:
            raise ImportError(f"cannot import trainer module from {module_path}")
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)

    fn = getattr(module, attr)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _norm_lip(value: Any) -> str:
    key = str(value or "none").strip().lower()
    if key in {"off", "false", "0"}:
        return "none"
    if key in {"sel", "lip_sel"}:
        return "selective"
    if key in {"lip_full"}:
        return "full"
    return key


def _norm_aja(value: Any) -> str:
    key = str(value or "none").strip().lower()
    if key in {"off", "false", "0"}:
        return "none"
    if key in {"on", "true", "1", "v10pct"}:
        return "aja"
    return key


def _search_label(action: dict[str, Any]) -> str:
    label = action.get("search_label")
    if label:
        return str(label)

    strategy = str(action.get("search_strategy") or "default").strip().lower()
    if strategy == "default":
        return "default"
    if strategy in {"top10"}:
        return "top10"
    if strategy in {"top5"}:
        return "top5"
    if strategy in {"topk", "split"}:
        try:
            k = int(action.get("search_k") or 0)
        except Exception:
            k = 0
        if k >= 10:
            return "top10"
        if k >= 5:
            return "top5"
        return "split"
    return "default"


def _low_label(action: dict[str, Any]) -> str:
    label = action.get("low_label")
    if label:
        return str(label)
    key = (
        _norm_lip(action.get("lip_action")),
        _norm_aja(action.get("execution_action")),
    )
    return LOW_LABEL_BY_ACTION.get(key, "none")


def _high_label(transition: dict[str, Any]) -> str:
    action = transition.get("action") or {}
    explicit = (
        str(action.get("high_action") or action.get("action") or "").strip().lower()
    )
    if explicit in {"split", "stop"}:
        return explicit
    if transition.get("source_event") == "split" and not transition.get("done"):
        return "split"
    return "stop"


def _timing_total_ms(transition: dict[str, Any]) -> float:
    timing = transition.get("timing_ms") or {}
    total = timing.get("total")
    if total is None:
        total = -(float(transition.get("reward") or 0.0))
    try:
        return max(float(total), 1.0)
    except Exception:
        return 1.0


class OnlineCheckpointTrainer:
    """Small reward-weighted online update for the existing HRL checkpoint.

    This is intentionally conservative. The PG runtime observes one chosen
    high/search/low action per round, not a full offline rollout with all
    counterfactual costs. We therefore treat the trajectory as behavior data
    and reinforce faster observed rounds with a larger supervised weight.
    """

    def __init__(
        self,
        *,
        model_path: str,
        updated_model_path: str,
        metadata_out: str | None,
        model_method: str,
        model_hidden: int,
        workload: str,
        device: str,
        neurqo_src: str | None,
        learning_rate: float,
        epochs: int,
    ) -> None:
        from ai_server import PolicyAdapter  # local server module

        self.model_path = Path(model_path)
        self.updated_model_path = Path(updated_model_path)
        self.metadata_out = Path(metadata_out) if metadata_out else None
        self.model_method = model_method
        self.epochs = max(int(epochs), 1)
        self.learning_rate = float(learning_rate)
        self.adapter = PolicyAdapter(
            model_path=str(self.model_path),
            model_method=model_method,
            model_hidden=model_hidden,
            workload=workload,
            device=device,
            neurqo_src=neurqo_src,
        )
        if self.adapter._model is None or self.adapter._torch is None:
            raise RuntimeError(f"could not load model checkpoint {self.model_path}")

        self.torch = self.adapter._torch
        self.device = self.adapter._device
        self.model = self.adapter._model
        self.hrl = self.adapter._hrl
        self.optimizer = self.torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.history: list[dict[str, Any]] = []
        self.total_samples = 0
        self.total_updates = 0
        self.last_saved_path: str | None = None

    def _target_indices(
        self, transition: dict[str, Any]
    ) -> tuple[int, int, int] | None:
        action = transition.get("action") or {}
        try:
            high_idx = 1 if _high_label(transition) == "split" else 0
            search_idx = self.hrl.SEARCH_LABELS.index(_search_label(action))
            low_idx = self.hrl.ACTION_LABELS.index(_low_label(action))
        except ValueError:
            return None
        return high_idx, search_idx, low_idx

    def _logits_for_targets(
        self, state: dict[str, Any], high_idx: int
    ) -> tuple[Any, Any, Any]:
        model = self.model
        high_state = self.adapter._query_graph_state(state, "high")
        search_state = self.adapter._query_graph_state(state, "search")
        low_state = self.adapter._plan_state(state)

        high_h = model.encode_state_obj(high_state, self.device)
        search_h = model.encode_state_obj(search_state, self.device)
        low_h = model.encode_state_obj(low_state, self.device)

        if self.model_method in ("hac", "smdp", "standardmdp", "standardmdp_rl"):
            return (
                model.high_actor(high_h),
                model.search_actor(search_h),
                model.low_actor(low_h),
            )
        if self.model_method == "option":
            option_idx = max(0, min(int(high_idx), len(model.intra_policies) - 1))
            return (
                model.option_policy(high_h),
                model.search_actor(search_h),
                model.intra_policies[option_idx](low_h),
            )
        if self.model_method == "maxq":
            return model.q_high(high_h), model.q_search(search_h), model.q_low(low_h)
        raise RuntimeError(f"unsupported model method: {self.model_method}")

    def _ce(self, logits: Any, target: int) -> Any:
        torch = self.torch
        return torch.nn.functional.cross_entropy(
            logits.unsqueeze(0),
            torch.tensor([target], dtype=torch.long, device=self.device),
        )

    def __call__(self, transitions: list[dict[str, Any]]) -> dict[str, Any]:
        samples = []
        for transition in transitions:
            state = transition.get("state")
            if not isinstance(state, dict):
                continue
            targets = self._target_indices(transition)
            if targets is None:
                continue
            samples.append((transition, state, targets, _timing_total_ms(transition)))

        if not samples:
            result = {"samples": 0, "updates": 0, "loss": 0.0}
            self.history.append(result)
            self._save()
            return result

        mean_ms = sum(item[3] for item in samples) / len(samples)
        weights = [max(0.25, min(4.0, mean_ms / item[3])) for item in samples]

        self.model.train()
        losses: list[float] = []
        for _epoch in range(self.epochs):
            for (_, state, targets, _timing_ms), weight in zip(samples, weights):
                high_idx, search_idx, low_idx = targets
                high_logits, search_logits, low_logits = self._logits_for_targets(
                    state, high_idx
                )
                loss = (
                    self._ce(high_logits, high_idx)
                    + self._ce(search_logits, search_idx)
                    + self._ce(low_logits, low_idx)
                ) * float(weight)
                self.optimizer.zero_grad()
                loss.backward()
                self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                losses.append(float(loss.detach().cpu().item()))

        self.model.eval()
        self.total_samples += len(samples)
        self.total_updates += len(losses)
        result = {
            "samples": len(samples),
            "updates": len(losses),
            "loss": sum(losses) / len(losses) if losses else 0.0,
            "mean_timing_ms": mean_ms,
            "min_weight": min(weights),
            "max_weight": max(weights),
            "method": "reward_weighted_behavioral_update",
        }
        self.history.append(result)
        self._save()
        return result

    def _metadata(self) -> dict[str, Any]:
        return {
            "source_checkpoint": str(self.model_path),
            "updated_checkpoint": str(self.updated_model_path),
            "model_source": self.adapter.source,
            "model_method": self.model_method,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "total_samples": self.total_samples,
            "total_updates": self.total_updates,
            "history": self.history,
        }

    def _save(self) -> None:
        self.updated_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(self.model.state_dict(), self.updated_model_path)
        self.last_saved_path = str(self.updated_model_path)
        if self.metadata_out:
            self.metadata_out.parent.mkdir(parents=True, exist_ok=True)
            self.metadata_out.write_text(
                json.dumps(self._metadata(), indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )

    def flush(self) -> dict[str, Any]:
        self._save()
        return self._metadata()


def event_to_transition(event: dict[str, Any]) -> dict[str, Any] | None:
    phase = event.get("phase")
    if phase not in {"split", "final"}:
        return None
    timing = event.get("timing_ms") or {}
    total_ms = float(timing.get("total") or 0.0)
    return {
        "pid": event.get("pid"),
        "run_id": event.get("run_id"),
        "round": event.get("round"),
        "state": event.get("state"),
        "action": event.get("action"),
        "reward": -total_ms,
        "done": bool(event.get("stop")) or phase == "final",
        "next_state": None,
        "timing_ms": timing,
        "result": event.get("result"),
        "source_event": phase,
    }


def emit_transition(
    transition: dict[str, Any],
    *,
    out_path: Path,
    trainer: Callable[[list[dict[str, Any]]], Any] | None,
    batch: list[dict[str, Any]],
    batch_size: int,
) -> None:
    append_jsonl(out_path, transition)
    batch.append(transition)
    if trainer is not None and len(batch) >= batch_size:
        trainer(list(batch))
        batch.clear()


def read_new_lines(path: Path, offset: int) -> tuple[list[str], int]:
    if not path.exists():
        return [], offset
    with path.open("r", encoding="utf-8") as f:
        f.seek(offset)
        lines = f.readlines()
        return lines, f.tell()


def process_lines(
    lines: list[str],
    *,
    out_path: Path,
    trainer: Callable[[list[dict[str, Any]]], Any] | None,
    batch: list[dict[str, Any]],
    batch_size: int,
    seen: set[tuple[Any, Any, Any, str]],
    pending: dict[Any, dict[str, Any]],
) -> int:
    wrote = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        event = json.loads(line)
        key = (
            event.get("pid"),
            event.get("run_id"),
            event.get("round"),
            event.get("phase"),
        )
        if key in seen:
            continue
        seen.add(key)
        transition = event_to_transition(event)
        if transition is None:
            continue
        run_key = (transition.get("pid"), transition.get("run_id"))
        previous = pending.pop(run_key, None)
        if previous is not None:
            previous["next_state"] = transition.get("state")
            previous["done"] = False
            emit_transition(
                previous,
                out_path=out_path,
                trainer=trainer,
                batch=batch,
                batch_size=batch_size,
            )
            wrote += 1

        if transition.get("done"):
            emit_transition(
                transition,
                out_path=out_path,
                trainer=trainer,
                batch=batch,
                batch_size=batch_size,
            )
            wrote += 1
        else:
            pending[run_key] = transition
    return wrote


def main() -> int:
    ap = argparse.ArgumentParser(description="NeurQO online training bridge")
    ap.add_argument(
        "--db-log", required=True, help="DB-side neurqo.trajectory_log JSONL"
    )
    ap.add_argument("--out", required=True, help="transition JSONL output path")
    ap.add_argument("--trainer-module", default=None)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--updated-model-path", default=None)
    ap.add_argument("--metadata-out", default=None)
    ap.add_argument("--model-method", default="standardmdp_rl")
    ap.add_argument("--model-hidden", type=int, default=128)
    ap.add_argument("--workload", default="job")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--neurqo-src", default=None)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--poll-interval", type=float, default=1.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    db_log = Path(args.db_log)
    out_path = Path(args.out)
    trainer: Callable[[list[dict[str, Any]]], Any] | None
    trainer = load_callable(args.trainer_module)
    checkpoint_trainer: OnlineCheckpointTrainer | None = None
    if trainer is None and args.model_path:
        if not args.updated_model_path:
            raise ValueError("--updated-model-path is required with --model-path")
        checkpoint_trainer = OnlineCheckpointTrainer(
            model_path=args.model_path,
            updated_model_path=args.updated_model_path,
            metadata_out=args.metadata_out,
            model_method=args.model_method,
            model_hidden=args.model_hidden,
            workload=args.workload,
            device=args.device,
            neurqo_src=args.neurqo_src,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
        )
        trainer = checkpoint_trainer
    offset = 0
    seen: set[tuple[Any, Any, Any, str]] = set()
    pending: dict[Any, dict[str, Any]] = {}
    batch: list[dict[str, Any]] = []

    while True:
        lines, offset = read_new_lines(db_log, offset)
        wrote = process_lines(
            lines,
            out_path=out_path,
            trainer=trainer,
            batch=batch,
            batch_size=max(args.batch_size, 1),
            seen=seen,
            pending=pending,
        )
        if args.once:
            if trainer is not None and batch:
                trainer(list(batch))
                batch.clear()
            if checkpoint_trainer is not None:
                summary = checkpoint_trainer.flush()
                print(
                    "checkpoint_update "
                    f"samples={summary['total_samples']} "
                    f"updates={summary['total_updates']} "
                    f"path={summary['updated_checkpoint']}",
                    flush=True,
                )
            print(f"processed {wrote} new transitions", flush=True)
            return 0
        if wrote:
            print(f"processed {wrote} new transitions", flush=True)
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

from runtime.action_server import _decide_sched, render_action


def _fields(body: bytes) -> dict[str, str]:
    return dict(line.split("=", 1) for line in body.decode().splitlines())


def test_canonical_dec_wire_name_is_unchanged() -> None:
    fields = _fields(
        render_action(
            {"action": "apply", "stop": False, "dec_action": "apply"},
            request_type="dec",
        )
    )
    assert fields["action"] == "apply"
    assert fields["dec_action"] == "apply"


def test_legacy_high_wire_name_is_adapted() -> None:
    apply = _fields(
        render_action(
            {"action": "apply", "stop": False, "dec_action": "apply"},
            request_type="high",
        )
    )
    skip = _fields(
        render_action(
            {"action": "skip", "stop": True, "dec_action": "skip"},
            request_type="high",
        )
    )
    assert (apply["action"], apply["stop"]) == ("split", "0")
    assert (skip["action"], skip["stop"]) == ("stop", "1")


def test_legacy_search_and_low_fields_are_adapted() -> None:
    search = _fields(
        render_action(
            {
                "action": "enum",
                "stop": False,
                "enum_action": "top5",
                "enum_k": 5,
            },
            request_type="search",
        )
    )
    low = _fields(
        render_action(
            {
                "action": "adapt",
                "stop": False,
                "filter_action": "selective",
                "ajoin_action": "conservative",
            },
            request_type="low",
        )
    )
    assert (search["action"], search["search_strategy"], search["search_k"]) == (
        "search",
        "topk",
        "5",
    )
    assert (low["action"], low["lip_action"], low["execution_action"]) == (
        "low",
        "selective",
        "conservative",
    )


def test_legacy_direct_scheduler_keeps_alpha() -> None:
    action = _decide_sched(
        {"candidates": [{"candidate_id": 2}]},
        {"candidate_id": 2, "schedule_alpha": 1.0, "schedule_idx": 2},
    )
    assert action["candidate_id"] == 2
    assert action["sched_alpha"] == 1.0
    assert action["sched_idx"] == 2

from optimization.action_vocabulary import normalize_policy_action


def test_legacy_sched_without_alpha_uses_released_default() -> None:
    action = normalize_policy_action({"candidate_id": 2}, phase="select")
    assert action["candidate_id"] == 2
    assert action["sched_alpha"] == 0.5


def test_legacy_native_search_has_canonical_breadth() -> None:
    action = normalize_policy_action(
        {"search_strategy": "default", "search_k": 5},
        phase="search",
    )
    assert action["enum_action"] == "native"
    assert action["enum_k"] == 1

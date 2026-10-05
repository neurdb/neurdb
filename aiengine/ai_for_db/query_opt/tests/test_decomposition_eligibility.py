import json
from pathlib import Path

import pytest
from optimization.decomposition_eligibility import (
    decomposition_eligibility,
    query_supports_decomposition,
    summarize_query_compatibility,
)


def test_checked_analysis_decisions() -> None:
    expected = {
        "JOB": (113, 113, True),
        "STACK": (112, 112, True),
        "TPCH": (22, 0, False),
    }
    for workload, values in expected.items():
        result = decomposition_eligibility(workload.lower())
        assert (
            result.query_count,
            result.compatible_query_count,
            result.enabled,
        ) == values


def test_any_compatible_query_enables_query_level_gate() -> None:
    result = summarize_query_compatibility(
        "test",
        {
            "compatible": {"is_spj_compatible": True},
            "incompatible": {"is_spj_compatible": False},
        },
    )
    assert result.compatible_query_count == 1
    assert result.compatible_query_ratio == 0.5
    assert result.enabled

    result = summarize_query_compatibility(
        "test",
        {"incompatible": {"is_spj_compatible": False}},
    )
    assert not result.enabled


def test_json_compatibility_rule_is_applied(tmp_path: Path) -> None:
    path = tmp_path / "analysis.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "workloads": {
                    "CUSTOM": {
                        "query_count": 2,
                        "spj_compatible_query_count": 1,
                        "spj_compatible_query_ratio": 0.5,
                        "recommend_decomposition_enabled": True,
                        "queries": {
                            "q1": {
                                "centers": ["a", "b"],
                                "is_spj_compatible": True,
                            },
                            "q2": {
                                "centers": None,
                                "is_spj_compatible": False,
                            },
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    result = decomposition_eligibility("custom", analysis_path=path)
    assert result.compatible_query_count == 1
    assert result.enabled
    assert query_supports_decomposition("custom", "q1", analysis_path=path)
    assert not query_supports_decomposition("custom", "q2", analysis_path=path)


def test_empty_analysis_is_rejected() -> None:
    with pytest.raises(ValueError, match="no analyzed SQL queries"):
        summarize_query_compatibility("empty", {})

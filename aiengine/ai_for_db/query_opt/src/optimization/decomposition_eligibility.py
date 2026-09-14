"""Query-decomposition eligibility from static SPJ compatibility analysis."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

DEFAULT_ANALYSIS_PATH = (
    Path(__file__).resolve().parent / "data" / "workload_fk_center_analysis.json"
)


@dataclass(frozen=True)
class DecompositionEligibility:
    workload: str
    query_count: int
    compatible_query_count: int
    compatible_query_ratio: float
    enabled: bool


def _analysis_path(path: str | Path | None = None) -> Path:
    configured = path or os.environ.get("NQO_WORKLOAD_CENTER_ANALYSIS")
    return Path(configured).resolve() if configured else DEFAULT_ANALYSIS_PATH


@lru_cache(maxsize=None)
def _load_analysis(path: str) -> dict[str, Any]:
    analysis_path = Path(path)
    if not analysis_path.is_file():
        raise FileNotFoundError(
            f"query compatibility analysis does not exist: {analysis_path}"
        )
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version") or 0) != 2:
        raise ValueError(f"unsupported compatibility-analysis schema: {analysis_path}")
    workloads = payload.get("workloads")
    if not isinstance(workloads, dict):
        raise ValueError(f"invalid query compatibility analysis: {analysis_path}")
    return payload


def summarize_query_compatibility(
    workload: str,
    query_records: Mapping[str, object],
) -> DecompositionEligibility:
    query_count = len(query_records)
    if query_count == 0:
        raise ValueError(f"{workload} has no analyzed SQL queries")
    compatible = sum(
        isinstance(record, dict) and record.get("is_spj_compatible") is True
        for record in query_records.values()
    )
    ratio = compatible / query_count
    return DecompositionEligibility(
        workload=workload.upper(),
        query_count=query_count,
        compatible_query_count=compatible,
        compatible_query_ratio=ratio,
        enabled=compatible > 0,
    )


def _workload_payload(
    workload: str,
    *,
    analysis_path: str | Path | None = None,
) -> tuple[Path, str, dict[str, Any]]:
    path = _analysis_path(analysis_path)
    payload = _load_analysis(str(path))
    workload_key = workload.strip().upper()
    workload_payload = payload["workloads"].get(workload_key)
    if not isinstance(workload_payload, dict):
        raise KeyError(f"workload {workload_key!r} is missing from {path}")
    return path, workload_key, workload_payload


def decomposition_eligibility(
    workload: str,
    *,
    analysis_path: str | Path | None = None,
) -> DecompositionEligibility:
    path, workload_key, workload_payload = _workload_payload(
        workload,
        analysis_path=analysis_path,
    )
    query_records = workload_payload.get("queries")
    if not isinstance(query_records, dict):
        raise ValueError(f"workload {workload_key!r} has no query records in {path}")
    result = summarize_query_compatibility(workload_key, query_records)
    expected = (
        int(workload_payload.get("query_count") or 0),
        int(workload_payload.get("spj_compatible_query_count") or 0),
        float(workload_payload.get("spj_compatible_query_ratio") or 0.0),
        workload_payload.get("recommend_decomposition_enabled"),
    )
    if (
        expected[0] != result.query_count
        or expected[1] != result.compatible_query_count
        or not math.isclose(
            expected[2],
            result.compatible_query_ratio,
            abs_tol=1e-6,
        )
        or not isinstance(expected[3], bool)
        or expected[3] is not result.enabled
    ):
        raise ValueError(f"stale query compatibility summary for {workload_key}")
    return result


def query_supports_decomposition(
    workload: str,
    query_id: str,
    *,
    analysis_path: str | Path | None = None,
) -> bool:
    path, workload_key, workload_payload = _workload_payload(
        workload,
        analysis_path=analysis_path,
    )
    records = workload_payload.get("queries")
    if not isinstance(records, dict):
        raise ValueError(f"workload {workload_key!r} has no query records in {path}")
    record = records.get(str(query_id))
    if not isinstance(record, dict):
        raise KeyError(f"query {query_id!r} is missing from {workload_key} in {path}")
    compatible = record.get("is_spj_compatible")
    if not isinstance(compatible, bool):
        raise ValueError(
            f"query {query_id!r} has no compatibility flag in {workload_key}"
        )
    return compatible


def workload_supports_decomposition(
    workload: str,
    *,
    analysis_path: str | Path | None = None,
) -> bool:
    return decomposition_eligibility(
        workload,
        analysis_path=analysis_path,
    ).enabled

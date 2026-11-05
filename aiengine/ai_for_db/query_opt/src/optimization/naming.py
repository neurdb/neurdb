"""NeurQO names and read-only adapters for released NQO artifacts."""

from __future__ import annotations

import csv
import os
import re
from collections.abc import Iterator, Mapping
from typing import Any


class CompatibleEnvironment(Mapping[str, str]):
    """Prefer NEURQO_* while accepting NQO_* without changing os.environ."""

    def __getitem__(self, name: str) -> str:
        if name in os.environ:
            return os.environ[name]
        if name.startswith("NEURQO_"):
            return os.environ["NQO_" + name[len("NEURQO_") :]]
        raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        return iter(
            dict.fromkeys(
                "NEURQO_" + key[4:] if key.startswith("NQO_") else key
                for key in os.environ
            )
        )

    def __len__(self) -> int:
        return sum(1 for _ in self)


environ = CompatibleEnvironment()


def method_name(value: str) -> str:
    """Normalize method labels read from old CSVs, not stored data or SQL."""
    if value in {"NQO", "nqo", "neurqo", "NeurQO"}:
        return "NeurQO"
    return re.sub(r"\bNQO\b", "NeurQO", value)


class ResultDictReader(csv.DictReader):
    """Read legacy benchmark method labels without rewriting historical CSVs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.fieldnames is not None:
            self.fieldnames = [
                name.replace("nqo_", "neurqo_") for name in self.fieldnames
            ]

    def __next__(self) -> dict[str, Any]:
        row = super().__next__()
        for key in ("method", "Method"):
            if isinstance(row.get(key), str):
                row[key] = method_name(row[key])
        return row


def legacy_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Retain pre-rename profile hashes for existing execution-cache entries."""
    result = dict(profile)
    if "neurqo_enabled" in result:
        result["nqo_enabled"] = result.pop("neurqo_enabled")
    if result.get("name") == "neurqo_none":
        result["name"] = "nqo_none"
    return result


def normalize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    result = dict(profile)
    if "nqo_enabled" in result:
        result.setdefault("neurqo_enabled", result.pop("nqo_enabled"))
    return result

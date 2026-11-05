#!/usr/bin/env python3
"""Workload-agnostic SQL-to-query-graph construction."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

# ── Constants ────────────────────────────────────────────────────────────────

GRAPH_EMB_DIM = 64

# ── Catalog loader ───────────────────────────────────────────────────────────


class CatalogInfo:
    """Preloaded database catalog for feature lookups."""

    def __init__(self, catalog: str | Path | Mapping[str, Any]):
        if isinstance(catalog, Mapping):
            cat = dict(catalog)
        else:
            with Path(catalog).open(encoding="utf-8") as f:
                cat = json.load(f)

        self.table_stats: Dict[str, dict] = {}
        for t in cat.get("tables", []):
            name = t["table_name"]
            self.table_stats[name] = {
                "row_count": t.get("row_count", 0),
                "table_size": t.get("table_size", 0),
                "index_size": t.get("index_size", 0),
            }

        self.col_stats: Dict[Tuple[str, str], dict] = {}
        self.table_columns_map: Dict[str, List[str]] = {}
        for cs in cat.get("column_statistics", []):
            key = (cs["table_name"], cs["column_name"])
            self.col_stats[key] = {
                "n_distinct": cs.get("n_distinct", 0),
                "null_frac": cs.get("null_frac", 0),
                "avg_width": cs.get("avg_width", 0),
                "data_type": cs.get("data_type", ""),
                "most_common_vals": cs.get("most_common_vals", None),
                "most_common_freqs": cs.get("most_common_freqs", None),
                "histogram_bounds": cs.get("histogram_bounds", None),
            }
            self.table_columns_map.setdefault(cs["table_name"], []).append(
                cs["column_name"]
            )

        self.fk_set: set = set()
        for fk in cat.get("foreign_keys", []):
            self.fk_set.add((fk["table_name"], fk["column_name"]))

        self.pk_set: set = set()
        for pk in cat.get("primary_keys", []):
            self.pk_set.add((pk["table_name"], pk["column_name"]))

        self.indexes: Dict[Tuple[str, str], bool] = {}
        for idx in cat.get("indexes", []):
            key = (idx["table_name"], idx["column_name"])
            self.indexes[key] = True

    @classmethod
    def from_database(cls, connection: Any, schema: str = "public") -> "CatalogInfo":
        from database.catalog import read_postgres_catalog

        return cls(read_postgres_catalog(connection, schema=schema))

    def table_row_count(self, table_name: str) -> float:
        return self.table_stats.get(table_name, {}).get("row_count", 0)

    def table_size(self, table_name: str) -> float:
        return self.table_stats.get(table_name, {}).get("table_size", 0)

    def index_size(self, table_name: str) -> float:
        return self.table_stats.get(table_name, {}).get("index_size", 0)

    def col_n_distinct(self, table_name: str, col_name: str) -> float:
        return self.col_stats.get((table_name, col_name), {}).get("n_distinct", 0)

    def col_null_frac(self, table_name: str, col_name: str) -> float:
        return self.col_stats.get((table_name, col_name), {}).get("null_frac", 0)

    def col_avg_width(self, table_name: str, col_name: str) -> float:
        return self.col_stats.get((table_name, col_name), {}).get("avg_width", 0)

    def col_data_type(self, table_name: str, col_name: str) -> str:
        return self.col_stats.get((table_name, col_name), {}).get("data_type", "")

    def col_most_common_freqs(self, table_name: str, col_name: str) -> list:
        freqs = self.col_stats.get((table_name, col_name), {}).get(
            "most_common_freqs", None
        )
        return freqs if isinstance(freqs, list) else []

    def col_most_common_values(self, table_name: str, col_name: str):
        return self.col_stats.get((table_name, col_name), {}).get(
            "most_common_vals", None
        )

    def col_histogram_bounds(self, table_name: str, col_name: str):
        return self.col_stats.get((table_name, col_name), {}).get(
            "histogram_bounds", None
        )

    def is_fk(self, table_name: str, col_name: str) -> bool:
        return (table_name, col_name) in self.fk_set

    def is_pk(self, table_name: str, col_name: str) -> bool:
        return (table_name, col_name) in self.pk_set

    def has_index(self, table_name: str, col_name: str) -> bool:
        return (table_name, col_name) in self.indexes

    def table_columns(self, table_name: str) -> List[str]:
        return sorted(self.table_columns_map.get(table_name, []))

    def key_role_id(self, table_name: str, col_name: str) -> int:
        is_pk = self.is_pk(table_name, col_name)
        is_fk = self.is_fk(table_name, col_name)
        if is_pk and is_fk:
            return 3
        if is_pk:
            return 1
        if is_fk:
            return 2
        return 0


# ── SQL parser → join graph ──────────────────────────────────────────────────

_JOIN_COND_RE = re.compile(r"(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)", re.IGNORECASE)
_IDENTIFIER = r'(?:"(?:[^"]|"")+"|[A-Za-z_][A-Za-z0-9_$]*)'
_RELATION_START_RE = re.compile(
    rf"^\s*({_IDENTIFIER}(?:\s*\.\s*{_IDENTIFIER})?)"
    rf"(?:\s+(?:AS\s+)?({_IDENTIFIER}))?",
    re.IGNORECASE,
)
_RESERVED_RELATION_WORDS = {
    "cross",
    "full",
    "inner",
    "join",
    "left",
    "natural",
    "on",
    "outer",
    "right",
    "using",
}


def _top_level_from_clause(sql: str) -> str:
    """Return the outer FROM clause without matching nested subqueries."""
    depth = 0
    start: Optional[int] = None
    token_start: Optional[int] = None
    in_single_quote = False
    in_double_quote = False
    in_line_comment = False
    in_block_comment = False
    index = 0
    end_keywords = {"WHERE", "GROUP", "HAVING", "ORDER", "LIMIT", "UNION"}

    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""
        if in_line_comment:
            if char == "\n":
                in_line_comment = False
            index += 1
            continue
        if in_block_comment:
            if char == "*" and next_char == "/":
                in_block_comment = False
                index += 2
            else:
                index += 1
            continue
        if in_single_quote:
            if char == "'" and next_char == "'":
                index += 2
            elif char == "'":
                in_single_quote = False
                index += 1
            else:
                index += 1
            continue
        if in_double_quote:
            if char == '"' and next_char == '"':
                index += 2
            elif char == '"':
                in_double_quote = False
                index += 1
            else:
                index += 1
            continue
        if char == "-" and next_char == "-":
            in_line_comment = True
            index += 2
            continue
        if char == "/" and next_char == "*":
            in_block_comment = True
            index += 2
            continue
        if char == "'":
            in_single_quote = True
            index += 1
            continue
        if char == '"':
            in_double_quote = True
            index += 1
            continue
        if char == "(":
            depth += 1
            index += 1
            continue
        if char == ")":
            depth = max(depth - 1, 0)
            index += 1
            continue
        if depth == 0 and (char.isalpha() or char == "_"):
            token_start = index
            index += 1
            while index < len(sql) and (
                sql[index].isalnum() or sql[index] in {"_", "$"}
            ):
                index += 1
            token = sql[token_start:index].upper()
            if start is None and token == "FROM":
                start = index
            elif start is not None and token in end_keywords:
                return sql[start:token_start]
            continue
        index += 1
    return sql[start:] if start is not None else ""


def _split_top_level_relations(from_clause: str) -> List[str]:
    """Split a FROM clause at top-level commas and JOIN keywords."""
    segments: List[str] = []
    depth = 0
    start = 0
    in_single_quote = False
    in_double_quote = False
    index = 0
    while index < len(from_clause):
        char = from_clause[index]
        next_char = from_clause[index + 1] if index + 1 < len(from_clause) else ""
        if in_single_quote:
            if char == "'" and next_char == "'":
                index += 2
            elif char == "'":
                in_single_quote = False
                index += 1
            else:
                index += 1
            continue
        if in_double_quote:
            if char == '"' and next_char == '"':
                index += 2
            elif char == '"':
                in_double_quote = False
                index += 1
            else:
                index += 1
            continue
        if char == "'":
            in_single_quote = True
            index += 1
            continue
        if char == '"':
            in_double_quote = True
            index += 1
            continue
        if char == "(":
            depth += 1
            index += 1
            continue
        if char == ")":
            depth = max(depth - 1, 0)
            index += 1
            continue
        if depth == 0 and char == ",":
            segments.append(from_clause[start:index])
            start = index + 1
            index += 1
            continue
        if depth == 0 and (char.isalpha() or char == "_"):
            token_start = index
            index += 1
            while index < len(from_clause) and (
                from_clause[index].isalnum() or from_clause[index] in {"_", "$"}
            ):
                index += 1
            if from_clause[token_start:index].upper() == "JOIN":
                segments.append(from_clause[start:token_start])
                start = index
            continue
        index += 1
    segments.append(from_clause[start:])
    return segments


def _unquote_identifier(identifier: str) -> str:
    identifier = identifier.strip()
    if identifier.startswith('"') and identifier.endswith('"'):
        identifier = identifier[1:-1].replace('""', '"')
    return identifier.lower()


def _parse_from_relations(sql: str) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for segment in _split_top_level_relations(_top_level_from_clause(sql)):
        match = _RELATION_START_RE.match(segment)
        if match is None:
            continue
        raw_table, raw_alias = match.groups()
        table = _unquote_identifier(re.split(r"\s*\.\s*", raw_table)[-1])
        alias = _unquote_identifier(raw_alias) if raw_alias else table
        if alias in _RESERVED_RELATION_WORDS:
            alias = table
        aliases[alias] = table
    return aliases


class JoinGraph:
    """Table nodes + join edges parsed from SQL."""

    def __init__(self):
        self.aliases: Dict[str, str] = {}  # alias -> table_name
        self.edges: List[Tuple[str, str, str, str]] = (
            []
        )  # (l_alias, r_alias, l_col, r_col)
        self.predicates: Dict[str, int] = {}  # alias -> count of local predicates

    @property
    def n_nodes(self) -> int:
        return len(self.aliases)

    def alias_list(self) -> List[str]:
        return sorted(self.aliases.keys())


def parse_subquery_graph(sql: str) -> JoinGraph:
    """Parse a split subquery (may use JOIN...ON syntax)."""
    g = JoinGraph()
    g.aliases = _parse_from_relations(sql)

    for m in _JOIN_COND_RE.finditer(sql):
        la, lc = m.group(1).lower(), m.group(2).lower()
        ra, rc = m.group(3).lower(), m.group(4).lower()
        if la in g.aliases and ra in g.aliases:
            edge = (la, ra, lc, rc)
            if edge not in g.edges:
                g.edges.append(edge)

    where_match = re.search(
        r"\bWHERE\b(.*?)(?:GROUP|ORDER|LIMIT|;|$)", sql, re.DOTALL | re.IGNORECASE
    )
    if where_match:
        where_text = where_match.group(1)
        for alias in g.aliases:
            count = 0
            pattern = re.compile(
                rf"\b{re.escape(alias)}\.\w+\s*(?:=|!=|<|>|<=|>=|LIKE|IN|NOT)\s",
                re.IGNORECASE,
            )
            for _ in pattern.finditer(where_text):
                count += 1
            pred_in_join = sum(1 for e in g.edges if e[0] == alias or e[1] == alias)
            g.predicates[alias] = max(0, count - pred_in_join)

    return g

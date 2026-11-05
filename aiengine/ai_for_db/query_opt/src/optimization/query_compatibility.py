"""Static compatibility checks for QuerySplit's SPJ join-core rewrite."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

AGGREGATE_FUNCTIONS = {
    "array_agg",
    "avg",
    "bit_and",
    "bit_or",
    "bool_and",
    "bool_or",
    "count",
    "every",
    "json_agg",
    "json_object_agg",
    "jsonb_agg",
    "jsonb_object_agg",
    "max",
    "min",
    "string_agg",
    "sum",
    "xmlagg",
}
SQL_IGNORED_TEXT = re.compile(
    r"--[^\n]*(?:\n|$)|/\*.*?\*/|'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"",
    flags=re.DOTALL,
)
SQL_TOKEN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_$]*|\d+(?:\.\d+)?|::|<=|>=|<>|!=" r"|[-+*/%.=<>(),;]|[^\s]"
)
SQL_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_$]*$")
SQL_CAST_TYPE = re.compile(
    r"[a-z_][a-z0-9_$]*(?: \. [a-z_][a-z0-9_$]*)?"
    r"(?: varying| precision)?"
    r"(?: \( \d+(?: , \d+)* \))?"
    r"(?: (?:with|without) time zone)?"
)


def _simple_column_expression(argument: list[str]) -> bool:
    """Recognize a column wrapped only in parentheses and type casts."""
    depth = 0
    outer_end = None
    cast_index = None
    for index, token in enumerate(argument):
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
            if depth == 0 and outer_end is None:
                outer_end = index
        elif token == "::" and depth == 0:
            cast_index = index
        if depth < 0:
            return False
    if depth != 0:
        return False

    # Deparsed PG aggregates include casts such as MIN((lt.link)::text).
    # Unwrap only the cast, so arithmetic/functions underneath remain rejected.
    if cast_index is not None:
        return bool(SQL_CAST_TYPE.fullmatch(" ".join(argument[cast_index + 1 :]))) and (
            _simple_column_expression(argument[:cast_index])
        )
    if outer_end == len(argument) - 1:
        if argument[0] == "(":
            return _simple_column_expression(argument[1:-1])
        if argument[:2] == ["cast", "("]:
            depth = 0
            for index in range(2, len(argument) - 1):
                token = argument[index]
                if token == "(":
                    depth += 1
                elif token == ")":
                    depth -= 1
                elif token == "as" and depth == 0:
                    return bool(
                        SQL_CAST_TYPE.fullmatch(" ".join(argument[index + 1 : -1]))
                    ) and _simple_column_expression(argument[2:index])
            return False
    if len(argument) == 1:
        return bool(SQL_IDENTIFIER.match(argument[0]))
    return (
        len(argument) == 3
        and bool(SQL_IDENTIFIER.match(argument[0]))
        and argument[1] == "."
        and bool(SQL_IDENTIFIER.match(argument[2]))
    )


def _simple_result_aggregate(argument: list[str]) -> bool:
    """Return whether an aggregate is a simple JOB/STACK result reducer."""
    if argument == ["*"]:
        return True
    if argument and argument[0] == "distinct":
        argument = argument[1:]
    return _simple_column_expression(argument)


def _has_complex_aggregate(tokens: list[str]) -> bool:
    for index, token in enumerate(tokens[:-1]):
        if token not in AGGREGATE_FUNCTIONS or tokens[index + 1] != "(":
            continue
        depth = 0
        for end in range(index + 1, len(tokens)):
            if tokens[end] == "(":
                depth += 1
            elif tokens[end] == ")":
                depth -= 1
                if depth == 0:
                    if not _simple_result_aggregate(tokens[index + 2 : end]):
                        return True
                    break
    return False


@lru_cache(maxsize=4096)
def _incompatibility_reasons(sql: str) -> tuple[str, ...]:
    normalized = SQL_IGNORED_TEXT.sub(" ", sql)
    tokens = [token.lower() for token in SQL_TOKEN.findall(normalized)]
    token_set = set(tokens)

    def contains_pair(first: str, second: str) -> bool:
        return any(
            left == first and right == second for left, right in zip(tokens, tokens[1:])
        )

    reasons: list[str] = []
    if tokens.count("select") > 1:
        reasons.append("subquery")
    if _has_complex_aggregate(tokens):
        reasons.append("complex_aggregation")
    if "window" in token_set or "over" in token_set:
        reasons.append("window")
    if token_set.intersection({"union", "intersect", "except"}):
        reasons.append("set_operation")
    if any(
        contains_pair(join_type, "join")
        for join_type in ("left", "right", "full", "cross", "natural")
    ):
        reasons.append("non_inner_join")
    if "with" in token_set:
        reasons.append("cte")
    if "lateral" in token_set:
        reasons.append("lateral")
    return tuple(reasons)


def classify_spj_compatibility(sql: str) -> dict[str, Any]:
    """Classify whether SQL has a QuerySplit-compatible SPJ join core."""
    reasons = _incompatibility_reasons(sql)
    return {
        "is_spj_compatible": not reasons,
        "non_spj_features": list(reasons),
    }


def querysplit_compatible(sql: str) -> bool:
    """Return whether QuerySplit may be exposed for this SQL statement."""
    return not _incompatibility_reasons(sql)

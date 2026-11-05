import pytest
from optimization.query_compatibility import classify_spj_compatibility


@pytest.mark.parametrize(
    "expression",
    [
        "MIN(a.x)",
        "MIN((a.x))",
        "MIN(((a.x)))",
        "MIN((lt.link)::text)",
        "MIN(((lt.link)::text))",
        "MIN(lt.link::pg_catalog.text)",
        "MIN((lt.link)::character varying(40))",
        "MIN(CAST(lt.link AS text))",
        "MIN(CAST((lt.link)::varchar AS text))",
        "MIN((a.x)::numeric(10, 2))",
        "MIN(a.x::double precision)",
        "MIN(a.x::timestamp without time zone)",
        "COUNT(*)",
        "COUNT(DISTINCT ((a.x)::text))",
        "MIN(a.x::text::varchar)",
    ],
)
def test_allows_column_aggregate_wrappers(expression):
    assert classify_spj_compatibility(f"SELECT {expression} FROM a, link_type lt") == {
        "is_spj_compatible": True,
        "non_spj_features": [],
    }


@pytest.mark.parametrize(
    "expression",
    [
        "SUM(a.x * a.y)",
        "SUM((a.x * a.y)::numeric)",
        "MIN(CAST(a.x + a.y AS text))",
        "MIN(lower(a.x)::text)",
        "MIN(COALESCE(a.x, a.y)::text)",
        "MIN((a.x)::numeric + a.y)",
        "MIN(a.x::text || a.y)",
        "MIN(a.x::text || 'suffix')",
        "MIN(CAST(a.x AS text) || a.y)",
        "MIN(CASE WHEN a.x > 0 THEN a.x ELSE a.y END)",
        "COUNT(DISTINCT (a.x, a.y))",
        "STRING_AGG(a.x, ',')",
        "MIN('literal'::text)",
        "MIN(a.x[1])",
    ],
)
def test_casts_do_not_hide_complex_aggregates(expression):
    assert classify_spj_compatibility(f"SELECT {expression} FROM a") == {
        "is_spj_compatible": False,
        "non_spj_features": ["complex_aggregation"],
    }


@pytest.mark.parametrize(
    ("tail", "reason"),
    [
        ("FROM a WHERE a.x IN (SELECT b.x FROM b)", "subquery"),
        ("FROM a LEFT JOIN b ON a.x = b.x", "non_inner_join"),
        ("FROM a UNION SELECT MIN(b.x) FROM b", "set_operation"),
        ("OVER () FROM a", "window"),
        ("FROM a, LATERAL (SELECT b.x FROM b) q", "lateral"),
    ],
)
def test_wrapped_column_does_not_enable_unsupported_queries(tail, reason):
    result = classify_spj_compatibility(f"SELECT MIN((a.x)::text) {tail}")
    assert result["is_spj_compatible"] is False
    assert reason in result["non_spj_features"]
    assert "complex_aggregation" not in result["non_spj_features"]

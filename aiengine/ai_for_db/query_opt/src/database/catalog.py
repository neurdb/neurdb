#!/usr/bin/env python3
"""Load workload-agnostic database metadata from PostgreSQL.

The model uses table and column names only as lookup keys.  The values fed to
the model are generic structural and statistical features obtained from the
target database's system catalogs.  A snapshot is taken once at process/run
startup so inference never issues catalog queries on the per-action path.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CATALOG_SNAPSHOT_SCHEMA_VERSION = 1


def _dict_rows(cursor: Any) -> list[dict[str, Any]]:
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def read_postgres_catalog(connection: Any, schema: str = "public") -> dict[str, Any]:
    """Read one model catalog snapshot using set-based catalog queries."""
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT current_database() AS database_name")
        database_name = str(cursor.fetchone()[0])

        cursor.execute(
            """
            SELECT
                n.nspname AS table_schema,
                c.relname AS table_name,
                GREATEST(c.reltuples, 0)::bigint AS row_count,
                c.relpages::bigint
                    * current_setting('block_size')::bigint AS table_size,
                COALESCE((
                    SELECT sum(ic.relpages)::bigint
                    FROM pg_catalog.pg_index AS ix
                    JOIN pg_catalog.pg_class AS ic ON ic.oid = ix.indexrelid
                    WHERE ix.indrelid = c.oid
                ), 0) * current_setting('block_size')::bigint AS index_size
            FROM pg_catalog.pg_class AS c
            JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
            WHERE n.nspname = %s
              AND c.relkind IN ('r', 'p')
              AND c.relpersistence <> 't'
            ORDER BY c.relname
            """,
            (schema,),
        )
        tables = _dict_rows(cursor)

        cursor.execute(
            """
            SELECT
                n.nspname AS table_schema,
                c.relname AS table_name,
                a.attname AS column_name,
                pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type,
                0::float8 AS null_frac,
                0::integer AS avg_width,
                0::float8 AS n_distinct,
                NULL::text AS most_common_vals,
                NULL AS most_common_freqs,
                NULL::text AS histogram_bounds
            FROM pg_catalog.pg_attribute AS a
            JOIN pg_catalog.pg_class AS c ON c.oid = a.attrelid
            JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
            WHERE n.nspname = %s
              AND c.relkind IN ('r', 'p')
              AND c.relpersistence <> 't'
              AND a.attnum > 0
              AND NOT a.attisdropped
            ORDER BY c.relname, a.attnum
            """,
            (schema,),
        )
        column_statistics = _dict_rows(cursor)

        # Read pg_statistic directly.  Besides avoiding one row-at-a-time
        # queries, this also works on the PostgreSQL fork where the
        # standard pg_stats view cannot currently be expanded by the parser.
        cursor.execute(
            """
            SELECT
                c.relname AS table_name,
                a.attname AS column_name,
                s.stanullfrac::float8 AS null_frac,
                s.stawidth::integer AS avg_width,
                s.stadistinct::float8 AS n_distinct,
                s.stakind1, s.stakind2, s.stakind3, s.stakind4, s.stakind5,
                s.stavalues1::text AS stavalues1,
                s.stavalues2::text AS stavalues2,
                s.stavalues3::text AS stavalues3,
                s.stavalues4::text AS stavalues4,
                s.stavalues5::text AS stavalues5,
                s.stanumbers1, s.stanumbers2, s.stanumbers3,
                s.stanumbers4, s.stanumbers5
            FROM pg_catalog.pg_statistic AS s
            JOIN pg_catalog.pg_class AS c ON c.oid = s.starelid
            JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
            JOIN pg_catalog.pg_attribute AS a
              ON a.attrelid = c.oid AND a.attnum = s.staattnum
            WHERE n.nspname = %s
              AND c.relkind IN ('r', 'p')
              AND c.relpersistence <> 't'
              AND NOT a.attisdropped
            ORDER BY c.relname, a.attnum
            """,
            (schema,),
        )
        statistic_slots = {
            (row["table_name"], row["column_name"]): row for row in _dict_rows(cursor)
        }
        for column in column_statistics:
            slots = statistic_slots.get((column["table_name"], column["column_name"]))
            if slots is None:
                continue
            column["null_frac"] = slots["null_frac"]
            column["avg_width"] = slots["avg_width"]
            column["n_distinct"] = slots["n_distinct"]
            for slot_number in range(1, 6):
                kind = slots[f"stakind{slot_number}"]
                if kind == 1:  # STATISTIC_KIND_MCV
                    column["most_common_vals"] = slots[f"stavalues{slot_number}"]
                    column["most_common_freqs"] = slots[f"stanumbers{slot_number}"]
                elif kind == 2:  # STATISTIC_KIND_HISTOGRAM
                    column["histogram_bounds"] = slots[f"stavalues{slot_number}"]

        cursor.execute(
            """
            SELECT
                n.nspname AS table_schema,
                c.relname AS table_name,
                a.attname AS column_name,
                con.conname AS constraint_name,
                key.ordinal_position
            FROM pg_catalog.pg_constraint AS con
            JOIN pg_catalog.pg_class AS c ON c.oid = con.conrelid
            JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
            CROSS JOIN LATERAL unnest(con.conkey)
                WITH ORDINALITY AS key(attnum, ordinal_position)
            JOIN pg_catalog.pg_attribute AS a
              ON a.attrelid = c.oid AND a.attnum = key.attnum
            WHERE con.contype = 'p' AND n.nspname = %s
            ORDER BY c.relname, key.ordinal_position
            """,
            (schema,),
        )
        primary_keys = _dict_rows(cursor)

        cursor.execute(
            """
            SELECT
                con.conname AS constraint_name,
                n.nspname AS table_schema,
                c.relname AS table_name,
                a.attname AS column_name,
                rn.nspname AS referenced_table_schema,
                rc.relname AS referenced_table_name,
                ra.attname AS referenced_column_name,
                key.ordinal_position
            FROM pg_catalog.pg_constraint AS con
            JOIN pg_catalog.pg_class AS c ON c.oid = con.conrelid
            JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
            JOIN pg_catalog.pg_class AS rc ON rc.oid = con.confrelid
            JOIN pg_catalog.pg_namespace AS rn ON rn.oid = rc.relnamespace
            CROSS JOIN LATERAL unnest(con.conkey, con.confkey)
                WITH ORDINALITY AS key(attnum, ref_attnum, ordinal_position)
            JOIN pg_catalog.pg_attribute AS a
              ON a.attrelid = c.oid AND a.attnum = key.attnum
            JOIN pg_catalog.pg_attribute AS ra
              ON ra.attrelid = rc.oid AND ra.attnum = key.ref_attnum
            WHERE con.contype = 'f' AND n.nspname = %s
            ORDER BY c.relname, con.conname, key.ordinal_position
            """,
            (schema,),
        )
        foreign_keys = _dict_rows(cursor)

        cursor.execute(
            """
            SELECT
                ni.nspname AS index_schema,
                ic.relname AS index_name,
                nt.nspname AS table_schema,
                tc.relname AS table_name,
                a.attname AS column_name,
                key.ordinal_position AS column_position,
                ix.indisunique AS is_unique,
                ix.indisprimary AS is_primary,
                am.amname AS index_type
            FROM pg_catalog.pg_index AS ix
            JOIN pg_catalog.pg_class AS ic ON ic.oid = ix.indexrelid
            JOIN pg_catalog.pg_namespace AS ni ON ni.oid = ic.relnamespace
            JOIN pg_catalog.pg_class AS tc ON tc.oid = ix.indrelid
            JOIN pg_catalog.pg_namespace AS nt ON nt.oid = tc.relnamespace
            JOIN pg_catalog.pg_am AS am ON am.oid = ic.relam
            CROSS JOIN LATERAL unnest(ix.indkey)
                WITH ORDINALITY AS key(attnum, ordinal_position)
            JOIN pg_catalog.pg_attribute AS a
              ON a.attrelid = tc.oid AND a.attnum = key.attnum
            WHERE nt.nspname = %s AND key.attnum > 0
            ORDER BY tc.relname, ic.relname, key.ordinal_position
            """,
            (schema,),
        )
        indexes = _dict_rows(cursor)
    finally:
        cursor.close()

    return {
        "snapshot_schema_version": CATALOG_SNAPSHOT_SCHEMA_VERSION,
        "database": database_name,
        "schema": schema,
        "tables": tables,
        "column_statistics": column_statistics,
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
        "indexes": indexes,
    }


def catalog_snapshot_hash(snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_catalog_snapshot(path: Path, snapshot: dict[str, Any]) -> str:
    """Atomically write a runtime catalog snapshot and return its hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return catalog_snapshot_hash(snapshot)

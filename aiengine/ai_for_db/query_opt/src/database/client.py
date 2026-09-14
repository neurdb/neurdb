"""Execute SQL through NeurDB, with session-local NQO configuration."""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib.resources import files
from pathlib import Path
from typing import Any

import psycopg2
from database.catalog import read_postgres_catalog, write_catalog_snapshot
from optimization.actions import ActionProfile
from psycopg2 import sql


def dataset_settings(dataset: str) -> dict[str, Any]:
    config = json.loads(
        files("optimization").joinpath("data/action_config.json").read_text()
    )
    profile = ActionProfile.from_mapping(
        name=dataset, values=config["datasets"][dataset.upper()]["base_profile"]
    )
    return profile.guc_settings()


def configure_session(
    cursor: Any,
    *,
    enabled: bool,
    server_url: str,
    timeout_ms: int,
    dataset: str | None = None,
    trajectory_log: str | None = None,
) -> None:
    if timeout_ms <= 0:
        raise ValueError("timeout_ms must be positive")
    # Disable NQO while configuring the session; never alter global settings.
    settings: dict[str, Any] = {"nqo": "off", "statement_timeout": timeout_ms}
    if enabled:
        if dataset:
            settings.update(dataset_settings(dataset))
        settings["nqo.server_url"] = server_url
        if trajectory_log:
            settings["nqo.trajectory_log"] = trajectory_log
    for name, value in settings.items():
        if isinstance(value, bool):
            value = "on" if value else "off"
        cursor.execute(sql.SQL("SET {} TO %s").format(sql.Identifier(name)), (value,))
    if enabled:
        cursor.execute("SET nqo TO on")


def execute_sql(cursor: Any, query: str) -> dict[str, Any]:
    start = time.perf_counter()
    cursor.execute(query)
    columns = [description[0] for description in cursor.description or ()]
    rows = cursor.fetchall() if cursor.description else []
    return {
        "wall_ms": (time.perf_counter() - start) * 1000.0,
        "columns": columns,
        "rows": rows,
        "row_count": cursor.rowcount,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--sql", help="SQL sent to PostgreSQL, not to the AI server")
    source.add_argument(
        "--file", type=Path, help="SQL file, without psql meta-commands"
    )
    parser.add_argument("--host", help="libpq PGHOST default when omitted")
    parser.add_argument("--port", type=int, help="libpq PGPORT default when omitted")
    parser.add_argument("--user", help="libpq PGUSER default when omitted")
    parser.add_argument("--dbname", help="libpq PGDATABASE default when omitted")
    parser.add_argument(
        "--pg", action="store_true", help="disable NQO for this session"
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:8088/action")
    parser.add_argument("--dataset", choices=("job", "stack", "tpch"))
    parser.add_argument("--timeout-ms", type=int, default=60_000)
    parser.add_argument("--trajectory-log", help="absolute path writable by PostgreSQL")
    parser.add_argument(
        "--export-catalog", type=Path, help="write catalog with NQO off"
    )
    args = parser.parse_args(argv)
    if args.sql is None and args.file is None and args.export_catalog is None:
        parser.error("provide --sql, --file, or --export-catalog")
    if args.timeout_ms <= 0:
        parser.error("--timeout-ms must be positive")

    connection = None
    try:
        query = args.file.read_text(encoding="utf-8") if args.file else args.sql
        if query is not None and not query.strip():
            raise ValueError("SQL must not be empty")
        connection = psycopg2.connect(
            **{
                key: value
                for key in ("host", "port", "user", "dbname")
                if (value := getattr(args, key)) is not None
            },
            application_name="nqo-sql",
            connect_timeout=10,
        )
        connection.autocommit = True
        with connection.cursor() as cursor:
            configure_session(
                cursor,
                enabled=False,
                server_url=args.server_url,
                timeout_ms=args.timeout_ms,
            )
            if args.export_catalog:
                write_catalog_snapshot(
                    args.export_catalog, read_postgres_catalog(connection)
                )
                print(f"Catalog written to {args.export_catalog}", file=sys.stderr)
            if query is not None:
                configure_session(
                    cursor,
                    enabled=not args.pg,
                    server_url=args.server_url,
                    timeout_ms=args.timeout_ms,
                    dataset=args.dataset,
                    trajectory_log=args.trajectory_log,
                )
                print(json.dumps(execute_sql(cursor, query), default=str))
        return 0
    except KeyboardInterrupt:
        if connection is not None:
            connection.cancel()
        print("Query interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError, psycopg2.Error) as exc:
        print(f"nqo-sql: {exc}", file=sys.stderr)
        return 1
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    raise SystemExit(main())

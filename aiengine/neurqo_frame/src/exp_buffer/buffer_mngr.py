# Standard library imports
import argparse
import os
from typing import Iterable, List, Optional, Tuple

# Local/project imports
from db.pg_conn import PostgresConnector
from exp_buffer.sqllite import ExperienceBuffer, PlanRecord


class BufferManager:
    def __init__(self, db_name: str):
        self.storage = ExperienceBuffer(db_name)
        self.stats = {
            "total_queries": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "unique_queries": set(),
        }

    def run_log_query_exec(
        self,
        conn: PostgresConnector,
        sql: str,
        hints: Optional[List[str]] = None,
        join_order_hint: str = None,
    ):
        self.stats["total_queries"] += 1

        # Normalize inputs to a single canonical form
        # hints: non-empty list or None
        # join_order_hint: no-empty str or None
        hints = hints if isinstance(hints, list) and len(hints) > 0 else None
        join_order_hint = (
            (join_order_hint.strip() or None)
            if isinstance(join_order_hint, str)
            else None
        )

        # Skip if already exists in buffer; exists returns existing id or None
        existing_id = self.storage.exists(
            sql=sql, hints=hints, join_order_hint=join_order_hint
        )
        if existing_id is not None:
            print("already exist, skip !")
            return existing_id

        conn.drop_buffer_cache()

        if hints is not None:
            conn.apply_hints(hints)

        if join_order_hint is not None:
            exe_sql = join_order_hint + " " + sql
        else:
            exe_sql = sql

        plan_json, actual_latency, plan_time = conn.explain_analysis(exe_sql)

        if not plan_json or actual_latency is None:
            self.stats["failed_executions"] += 1
            return None

        self.stats["successful_executions"] += 1

        rec_id = self.storage.save_plan(
            sql=sql,
            actual_plan_json=plan_json,
            actual_latency=actual_latency,
            actual_plan_time=plan_time,
            hint_json=hints,  # None or non-empty list
            join_order_hint=join_order_hint,
        )
        return rec_id

    def get_plan_latency_pairs(self, limit: int = None) -> List[PlanRecord]:
        """Flat list of PlanRecord."""
        return self.storage.get_plan_latency_pairs(limit=limit)

    def get_plan_latency_groups(self, limit: int = None) -> Iterable[List[PlanRecord]]:
        """Generator of groups (List[PlanRecord]) by query_hash (len >= 2)."""
        return self.storage.get_plan_latency_groups(limit=limit)


def read_sql_files(input_dir) -> List[Tuple]:
    queries = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".sql"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path) as f:
                sql = f.read()
            queries.append((filename, sql))
    return queries


def main():
    parser = argparse.ArgumentParser(
        description="Unified Data Collection for LQO Systems"
    )
    parser.add_argument(
        "--input_sql_dir",
        type=str,
        default="../datasets/query_on_imdb/job-light-mini",
        help="Input SQL dir (one query per file)",
    )
    parser.add_argument("--dbname", type=str, default="imdb_ori", help="Database name")

    args = parser.parse_args()

    collector = BufferManager(f"buffer_{args.dbname}.db")

    query_w_name = read_sql_files(args.input_sql_dir)
    print(f"Processing {len(query_w_name)} queries")
    print(f"Unified SQLite database: {collector.storage.db_path}")

    with PostgresConnector(args.dbname) as conn:
        for sql_name, sql in query_w_name:
            query_id = sql_name
            execution_id = collector.run_log_query_exec(conn=conn, sql=sql)
            if execution_id:
                print(f"✓ Saved execution {execution_id} for {query_id}: {sql[:60]}...")
            else:
                print(f"✗ Failed to collect data for {query_id}: {sql[:60]}...")

    print(f"All data stored in: {collector.storage.db_path}")


if __name__ == "__main__":
    main()

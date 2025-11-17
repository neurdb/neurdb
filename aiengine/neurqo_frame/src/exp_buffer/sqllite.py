# Standard library imports
import base64
import json
import os
import sqlite3
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Local/project imports
from utils.plan_utils import PlanHelper


@dataclass
class PlanRecord:
    """Single-table unified plan buffer record"""

    id: Optional[int] = None
    query_hash: str = ""
    query: str = ""
    actual_plan_json: Dict[str, Any] = None
    actual_plan_hash: Optional[str] = None
    hint_json: Optional[Dict[str, Any]] = (
        None  # parsed to dict for convenient consumption
    )
    join_order_hint: Optional[str] = None
    actual_latency: Optional[float] = None
    plan_time: Optional[float] = None


class ExperienceBuffer:
    """
    Unified SQLite storage for LQO data collection (single-table: plan_buffer).
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        # Only initialize schema when DB file does not exist yet.
        if not os.path.exists(self.db_path):
            self.initialize_database()

    def initialize_database(self):
        """Initialize the unified LQO database with a clean schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Single table: plan_buffer
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS plan_buffer (
            id INTEGER PRIMARY KEY,
            query_hash TEXT NOT NULL,
            query TEXT NOT NULL,
            actual_plan_json TEXT NOT NULL,
            actual_plan_hash TEXT,
            actual_latency REAL,
            plan_time REAL,
            hint_json TEXT,
            join_order_hint TEXT
        )"""
        )

        # Indexes for better performance
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_plan_buffer_query_hash ON plan_buffer(query_hash)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_plan_buffer_actual_plan_hash ON plan_buffer(actual_plan_hash)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_plan_buffer_latency ON plan_buffer(actual_latency)"
        )

        conn.commit()
        conn.close()

    def _get_connection(self):
        """Get database connection with proper configuration"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn

    @staticmethod
    def _normalize_sql(sql: str) -> str:
        """Lightweight SQL normalization prior to hashing:
        - strip line comments starting with '--'
        - collapse consecutive whitespace
        - trim surrounding whitespace
        - drop trailing semicolon
        """
        try:
            # remove -- comments
            lines = []
            for line in sql.splitlines():
                # keep portion before comment
                part = line.split("--", 1)[0]
                if part is not None:
                    lines.append(part)
            sql_no_comments = "\n".join(lines)
            # collapse whitespace
            collapsed = " ".join(sql_no_comments.split())
            # trim and remove trailing semicolon
            trimmed = collapsed.strip()
            if trimmed.endswith(";"):
                trimmed = trimmed[:-1].strip()
            return trimmed
        except Exception:
            # fallback: basic collapse
            return " ".join((sql or "").split()).strip()

    @staticmethod
    def _default_query_hash(sql: str) -> str:
        """Generate a stable query hash (compatible with RoutingHelper approach)."""
        normalized = ExperienceBuffer._normalize_sql(sql or "")
        compressed = zlib.compress(normalized.encode("utf-8"))
        encoded = base64.urlsafe_b64encode(compressed).decode("utf-8")
        return encoded

    def save_plan(
        self,
        sql: str,
        actual_plan_json: Dict[str, Any],
        actual_latency: float,
        actual_plan_time: Optional[float] = None,
        hint_json: Optional[List[str]] = None,
        join_order_hint: Optional[str] = None,
    ) -> int:
        """Save a plan record and return the ID.
        The method computes query_hash and actual_plan_hash internally."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
        INSERT INTO plan_buffer
        (query_hash, query, actual_plan_json, actual_plan_hash, actual_latency, plan_time, hint_json, join_order_hint)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                self._default_query_hash(sql),
                sql,
                json.dumps(actual_plan_json),
                PlanHelper.compute_plan_hash(actual_plan_json),
                actual_latency,
                actual_plan_time,
                json.dumps(hint_json) if hint_json else None,
                join_order_hint,
            ),
        )

        rec_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return rec_id

    def close(self):
        """Close database connections (if needed)"""
        # SQLite connections are automatically closed
        pass

    # -------- Basic queries --------
    def get_all(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()

        query = """
        SELECT * FROM plan_buffer
        ORDER BY id DESC
        """
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]

        conn.close()
        return results

    # -------- Mapping helpers to dataclass --------
    @staticmethod
    def _row_to_plan_record(r: sqlite3.Row) -> PlanRecord:
        if r["actual_plan_json"] is None:
            plan_val = None
        else:
            plan_val = json.loads(r["actual_plan_json"])
        if r["hint_json"] is None:
            hint_val = None
        else:
            hint_val = json.loads(r["hint_json"])

        return PlanRecord(
            id=r["id"],
            query_hash=r["query_hash"],
            query=r["query"],
            actual_plan_json=plan_val,
            actual_plan_hash=r["actual_plan_hash"],
            hint_json=hint_val,
            join_order_hint=r["join_order_hint"],
            actual_latency=float(r["actual_latency"]),
            plan_time=float(r["plan_time"]) if r["plan_time"] is not None else None,
        )

    # -------- Existence check --------
    def exists(
        self,
        sql: str,
        hints: Optional[List[str]] = None,
        join_order_hint: Optional[str] = None,
    ) -> Optional[int]:
        """
        Return existing record id matching (query_hash, hint_json, join_order_hint), or None if not found.
        """
        qhash = self._default_query_hash(sql)
        stored_hint = json.dumps(hints) if hints else None
        stored_join = join_order_hint
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id
            FROM plan_buffer
            WHERE query_hash = ?
              AND COALESCE(hint_json, '') = COALESCE(?, '')
              AND COALESCE(join_order_hint, '') = COALESCE(?, '')
            ORDER BY id DESC
            LIMIT 1
        """,
            (qhash, stored_hint, stored_join),
        )
        row = cur.fetchone()
        conn.close()
        return int(row["id"]) if row is not None else None

    # -------- Logs for training/regression (plan, latency) --------
    def get_plan_latency_pairs(self, limit: Optional[int] = None) -> List[PlanRecord]:
        """
        Flat list of PlanRecord (actual_plan_json as dict, actual_latency as float).
        Ordered by newest first.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        sql = """
            SELECT id, query_hash, query, actual_plan_json, actual_plan_hash, hint_json, join_order_hint, actual_latency, plan_time
            FROM plan_buffer
            WHERE actual_plan_json IS NOT NULL AND actual_latency IS NOT NULL
            ORDER BY id DESC
        """
        if limit:
            sql += f" LIMIT {limit}"
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        out: List[PlanRecord] = []
        for r in rows:
            out.append(self._row_to_plan_record(r))
        return out

    def get_plan_latency_groups(self, limit: Optional[int] = None):
        """
        Generator of groups; each group is a list of PlanRecord for the same query_hash,
        and only groups with len >= 2 are yielded.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        sql = """
            SELECT id, query_hash, query, actual_plan_json, actual_plan_hash, hint_json, join_order_hint, actual_latency, plan_time
            FROM plan_buffer
            WHERE actual_plan_json IS NOT NULL AND actual_latency IS NOT NULL
            ORDER BY id DESC
        """
        if limit:
            sql += f" LIMIT {limit}"
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        from collections import defaultdict

        groups = defaultdict(list)
        for r in rows:
            pr = self._row_to_plan_record(r)
            groups[pr.query_hash].append(pr)
        # only yield groups with multiple plans (meaningful for regression comparison)
        for g in groups.values():
            if len(g) >= 2:
                yield g

    def get_by_query_hash(
        self, query_hash: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()

        if limit:
            cursor.execute(
                """
                SELECT * FROM plan_buffer
                WHERE query_hash = ?
                ORDER BY id DESC
                LIMIT ?
            """,
                (query_hash, limit),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM plan_buffer
                WHERE query_hash = ?
                ORDER BY id DESC
            """,
                (query_hash,),
            )
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        conn.close()
        return results


if __name__ == "__main__":
    buf = ExperienceBuffer("buffer_imdb_ori_db")
    rows = buf.get_all()
    print("total rows:", len(rows))
    for r in rows[:5]:
        if r.get("actual_plan_json"):
            r["actual_plan_json"] = json.loads(r["actual_plan_json"])
        if r.get("hint_json"):
            r["hint_json"] = json.loads(r["hint_json"])
        print(json.dumps(r, indent=2))
